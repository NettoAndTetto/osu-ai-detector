from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from .advanced_features import WindowFeatures, extract_windows, hash_ngrams, iter_ngrams
from .parser import Beatmap


DEFAULT_MODEL = Path(__file__).with_name("models") / "statistical_v1.json"


@dataclasses.dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    contribution: float


@dataclasses.dataclass(frozen=True)
class SequenceContribution:
    bin: int
    value: float
    contribution: float
    examples: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SegmentScore:
    start_ms: int
    end_ms: int
    object_count: int
    numeric_score: float
    sequence_score: float
    combined_score: float
    concordance_score: float
    top_numeric: tuple[FeatureContribution, ...]
    top_sequence: tuple[SequenceContribution, ...]


@dataclasses.dataclass(frozen=True)
class StatisticalReport:
    available: bool
    model_id: str | None
    scope: str | None
    combined_score: float
    concordance_score: float
    numeric_score: float
    sequence_score: float
    high: bool
    suspicious: bool
    thresholds: dict[str, float]
    settings: dict[str, int]
    segments: tuple[SegmentScore, ...]
    limitations: tuple[str, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _top_two_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    count = min(2, len(ordered))
    return sum(ordered[:count]) / count


@functools.lru_cache(maxsize=4)
def load_model(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash_index(ngram: str, dimensions: int) -> int:
    digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8, person=b"osu-ai-v1").digest()
    return int.from_bytes(digest, "little") % dimensions


def _score_window(window: WindowFeatures, model: dict[str, object]) -> SegmentScore:
    numeric = model["numeric"]
    sequence = model["sequence"]
    names = numeric["feature_names"]
    contributions: list[FeatureContribution] = []
    numeric_logit = float(numeric["intercept"])
    for name, mean, scale, weight in zip(names, numeric["mean"], numeric["scale"], numeric["weights"]):
        value = float(window.values.get(name, 0.0))
        standardized = (value - float(mean)) / float(scale) if scale else 0.0
        contribution = standardized * float(weight)
        numeric_logit += contribution
        contributions.append(FeatureContribution(name, value, contribution))

    dimensions = int(sequence["dimensions"])
    hashed = hash_ngrams(window.sequence_tokens, dimensions)
    sequence_logit = float(sequence["intercept"])
    bin_contributions: list[tuple[int, float, float]] = []
    weights = sequence["weights"]
    for index, value in hashed.items():
        contribution = value * float(weights[index])
        sequence_logit += contribution
        bin_contributions.append((index, value, contribution))

    examples: dict[int, list[str]] = {}
    wanted = {index for index, _, _ in sorted(bin_contributions, key=lambda item: abs(item[2]), reverse=True)[:8]}
    if wanted:
        for ngram in iter_ngrams(window.sequence_tokens):
            index = _hash_index(ngram, dimensions)
            if index in wanted:
                bucket = examples.setdefault(index, [])
                if ngram not in bucket and len(bucket) < 3:
                    bucket.append(ngram)

    numeric_score = _sigmoid(numeric_logit)
    sequence_score = _sigmoid(sequence_logit)
    top_numeric = tuple(sorted(contributions, key=lambda item: abs(item.contribution), reverse=True)[:8])
    top_sequence = tuple(
        SequenceContribution(index, value, contribution, tuple(examples.get(index, ())))
        for index, value, contribution in sorted(bin_contributions, key=lambda item: abs(item[2]), reverse=True)[:8]
    )
    return SegmentScore(
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        object_count=window.object_count,
        numeric_score=numeric_score,
        sequence_score=sequence_score,
        combined_score=(numeric_score + sequence_score) / 2,
        concordance_score=min(numeric_score, sequence_score),
        top_numeric=top_numeric,
        top_sequence=top_sequence,
    )


def analyze_statistical(beatmap: Beatmap, model_path: str | Path | None = None) -> StatisticalReport:
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL
    if not path.is_file():
        return StatisticalReport(
            available=False, model_id=None, scope=None, combined_score=0.0, concordance_score=0.0,
            numeric_score=0.0, sequence_score=0.0, high=False, suspicious=False, thresholds={},
            settings={}, segments=(), limitations=(), reason=f"model artifact not found: {path}",
        )
    model = load_model(str(path.resolve()))
    windows = extract_windows(
        beatmap,
        window_ms=int(model.get("window_ms", 16_000)),
        stride_ms=int(model.get("stride_ms", 8_000)),
        min_objects=int(model.get("min_objects", 12)),
    )
    if not windows:
        return StatisticalReport(
            available=True, model_id=str(model.get("model_id")), scope=str(model.get("scope")),
            combined_score=0.0, concordance_score=0.0, numeric_score=0.0, sequence_score=0.0,
            high=False, suspicious=False, thresholds=dict(model["thresholds"]),
            settings={
                "window_ms": int(model.get("window_ms", 16_000)),
                "stride_ms": int(model.get("stride_ms", 8_000)),
                "min_objects": int(model.get("min_objects", 12)),
                "min_high_windows": int(model.get("min_high_windows", 2)),
            }, segments=(),
            limitations=tuple(model.get("limitations", ())), reason="no window has enough hit objects",
        )
    segments = tuple(_score_window(window, model) for window in windows)
    combined = _top_two_mean([item.combined_score for item in segments])
    concordance = _top_two_mean([item.concordance_score for item in segments])
    numeric = _top_two_mean([item.numeric_score for item in segments])
    sequence = _top_two_mean([item.sequence_score for item in segments])
    thresholds = {key: float(value) for key, value in model["thresholds"].items()}
    high = (
        len(segments) >= int(model.get("min_high_windows", 2))
        and combined >= thresholds["high_combined"]
        and concordance >= thresholds["high_concordance"]
    )
    suspicious = high or combined >= thresholds["suspicious_combined"]
    ordered = tuple(sorted(segments, key=lambda item: item.combined_score, reverse=True))
    return StatisticalReport(
        available=True, model_id=str(model.get("model_id")), scope=str(model.get("scope")),
        combined_score=combined, concordance_score=concordance, numeric_score=numeric,
        sequence_score=sequence, high=high, suspicious=suspicious, thresholds=thresholds,
        settings={
            "window_ms": int(model.get("window_ms", 16_000)),
            "stride_ms": int(model.get("stride_ms", 8_000)),
            "min_objects": int(model.get("min_objects", 12)),
            "min_high_windows": int(model.get("min_high_windows", 2)),
        }, segments=ordered, limitations=tuple(model.get("limitations", ())), reason=None,
    )
