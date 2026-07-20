"""Auditable, independently calibrated fusion of cheap detector channels.

The content and revision-forensic ensembles expose ranking scores on very
different scales.  This module first converts each score to an upper-tail
surprisal against *development OOF human* scores, then applies a frozen
combination rule.  A separate human-null split calibrates the complete map
statistic, including both channel searches.  Consequently a fused threshold
is not an informal OR of two individually calibrated tests.
"""

from __future__ import annotations

import bisect
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "osu-ai-detector.cheap-fusion/v1"
RESULT_SCHEMA_VERSION = "osu-ai-detector.cheap-fusion-result/v1"
DEFAULT_MODEL = Path(__file__).with_name("models") / "cheap_fusion_v1.json"
CHANNELS: tuple[str, ...] = ("content_v2", "revision_forensics_v3")
# Orders of magnitude smaller than the narrowest adjacent empirical tail-rank
# step in production. It only orders raw scores that share one finite-sample
# tail rank; independent human-null calibration still governs every threshold.
TAIL_TIE_BREAK_SCALE = 1e-6


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def upper_tail_p(reference: Sequence[float], score: float) -> float:
    """Smoothed upper-tail empirical rank used only as a scale transform."""

    if not reference:
        raise ValueError("development human reference is empty")
    index = bisect.bisect_left(reference, score)
    return (1 + len(reference) - index) / (len(reference) + 1)


def tail_surprisal(
    reference: Sequence[float],
    score: float,
    *,
    tie_break_scale: float = 0.0,
) -> float:
    """Empirical tail surprisal with a frozen within-rank monotone tie break."""

    if not math.isfinite(tie_break_scale) or not 0.0 <= tie_break_scale <= 1e-3:
        raise ValueError("tail tie-break scale must be finite and between 0 and 1e-3")
    return -math.log10(upper_tail_p(reference, score)) + tie_break_scale * float(score)


def combine_surprisals(
    surprisals: Mapping[str, float],
    weights: Mapping[str, float],
    method: str,
) -> tuple[float, dict[str, float]]:
    contributions = {
        channel: float(weights[channel]) * float(surprisals[channel])
        for channel in CHANNELS
    }
    if method == "weighted_sum_tail_surprisal":
        score = math.fsum(contributions.values())
    elif method == "max_weighted_tail_surprisal":
        score = max(contributions.values())
    else:
        raise ValueError(f"unsupported fusion method: {method!r}")
    return score, contributions


def _thresholds(calibration: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    raw_thresholds = calibration.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        return result
    for name, raw in raw_thresholds.items():
        value = raw
        if isinstance(raw, Mapping):
            if not raw.get("supported") or str(raw.get("operator") or ">") != ">":
                continue
            value = raw.get("threshold")
        number = _finite(value)
        if number is not None:
            result[str(name)] = number
    return result


@dataclasses.dataclass(frozen=True)
class ChannelFusionDetail:
    channel: str
    raw_score: float
    development_human_reference_size: int
    development_upper_tail_p: float
    development_tail_surprisal: float
    weight: float
    weighted_contribution: float


class CheapFusionModel:
    """Pure-Python runtime for the frozen two-channel fusion statistic."""

    def __init__(self, artifact: Mapping[str, Any]):
        if artifact.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported cheap-fusion schema: {artifact.get('schema_version')!r}")
        declared = artifact.get("channels")
        if declared != list(CHANNELS):
            raise ValueError(f"fusion channels must be exactly {list(CHANNELS)!r}")
        references = artifact.get("development_human_oof_reference")
        if not isinstance(references, Mapping):
            raise ValueError("development_human_oof_reference is missing")
        self.references: dict[str, list[float]] = {}
        for channel in CHANNELS:
            values = references.get(channel)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise ValueError(f"development reference for {channel} is empty")
            normalized = [_finite(value) for value in values]
            if any(value is None for value in normalized):
                raise ValueError(f"development reference for {channel} contains a non-finite value")
            ordered = sorted(float(value) for value in normalized if value is not None)
            if list(map(float, values)) != ordered:
                raise ValueError(f"development reference for {channel} must be sorted")
            self.references[channel] = ordered

        combination = artifact.get("combination")
        if not isinstance(combination, Mapping):
            raise ValueError("combination is missing")
        self.method = str(combination.get("method") or "")
        if self.method not in {"weighted_sum_tail_surprisal", "max_weighted_tail_surprisal"}:
            raise ValueError(f"unsupported fusion method: {self.method!r}")
        raw_weights = combination.get("weights")
        if not isinstance(raw_weights, Mapping):
            raise ValueError("combination.weights is missing")
        self.weights: dict[str, float] = {}
        for channel in CHANNELS:
            value = _finite(raw_weights.get(channel))
            if value is None or value < 0:
                raise ValueError(f"fusion weight for {channel} must be finite and non-negative")
            self.weights[channel] = value
        if not any(self.weights.values()):
            raise ValueError("at least one fusion weight must be positive")
        tie_break_value = combination.get("tail_tie_break_scale", 0.0)
        self.tail_tie_break_scale = float(tie_break_value)
        if (
            not math.isfinite(self.tail_tie_break_scale)
            or not 0.0 <= self.tail_tie_break_scale <= 1e-3
        ):
            raise ValueError("combination.tail_tie_break_scale is invalid")

        binding = artifact.get("base_model_binding")
        self.binding = copy.deepcopy(dict(binding)) if isinstance(binding, Mapping) else {}
        self.calibration = (
            copy.deepcopy(dict(artifact.get("calibration")))
            if isinstance(artifact.get("calibration"), Mapping)
            else {}
        )
        self.model_id = str(artifact.get("model_id") or "unnamed-cheap-fusion")
        self.artifact = copy.deepcopy(dict(artifact))

    @classmethod
    def from_path(cls, path: str | Path) -> "CheapFusionModel":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("cheap-fusion artifact must be a JSON object")
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.artifact)

    def _binding_errors(
        self,
        channel_results: Mapping[str, Mapping[str, Any]],
        observed_model_sha256: Mapping[str, str] | None,
    ) -> list[str]:
        observed_model_sha256 = observed_model_sha256 or {}
        errors: list[str] = []
        for channel in CHANNELS:
            expected_value = self.binding.get(channel)
            expected = expected_value if isinstance(expected_value, Mapping) else {}
            expected_hash = str(expected.get("sha256") or "").casefold()
            observed_hash = str(observed_model_sha256.get(channel) or "").casefold()
            if expected_hash and observed_hash != expected_hash:
                errors.append(
                    f"{channel} artifact hash mismatch (expected {expected_hash}, observed {observed_hash or 'missing'})"
                )
            expected_id = str(expected.get("model_id") or "")
            observed_id = str(channel_results[channel].get("model_id") or "")
            if expected_id and observed_id != expected_id:
                errors.append(
                    f"{channel} model_id mismatch (expected {expected_id!r}, observed {observed_id!r})"
                )
        return errors

    def analyze(
        self,
        content: Mapping[str, Any],
        forensic: Mapping[str, Any],
        *,
        observed_model_sha256: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        channel_results: dict[str, Mapping[str, Any]] = {
            "content_v2": content,
            "revision_forensics_v3": forensic,
        }
        errors = self._binding_errors(channel_results, observed_model_sha256)
        raw_scores: dict[str, float] = {}
        for channel, result in channel_results.items():
            if not result.get("available"):
                errors.append(f"{channel} is unavailable")
                continue
            score = _finite(result.get("score"))
            if score is None:
                errors.append(f"{channel} has no finite map score")
                continue
            if not result.get("windows"):
                errors.append(f"{channel} has no eligible windows")
                continue
            raw_scores[channel] = score
        if bool(content.get("ood", {}).get("abstain")):
            errors.append("content_v2 is outside its calibrated representation support")

        base = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "available": True,
            "score_semantics": (
                "Frozen tail-surprisal fusion statistic; independently calibrated on complete human maps; "
                "not an AI-authorship probability."
            ),
            "combination": {"method": self.method, "weights": dict(self.weights)},
            "base_model_binding": copy.deepcopy(self.binding),
        }
        if errors:
            return base | {
                "status": "abstain",
                "decision_usable": False,
                "abstention_reasons": errors,
                "score": None,
                "human_null_p_value": None,
                "calibrated": False,
                "thresholds": {},
                "threshold_guarantees": copy.deepcopy(self.calibration.get("thresholds", {})),
                "threshold_flags": {"elevated": False, "high": False},
                "channels": [],
            }

        p_values = {
            channel: upper_tail_p(self.references[channel], raw_scores[channel])
            for channel in CHANNELS
        }
        surprisals = {
            channel: tail_surprisal(
                self.references[channel],
                raw_scores[channel],
                tie_break_scale=self.tail_tie_break_scale,
            )
            for channel in CHANNELS
        }
        score, contributions = combine_surprisals(surprisals, self.weights, self.method)
        details = [
            ChannelFusionDetail(
                channel=channel,
                raw_score=raw_scores[channel],
                development_human_reference_size=len(self.references[channel]),
                development_upper_tail_p=p_values[channel],
                development_tail_surprisal=surprisals[channel],
                weight=self.weights[channel],
                weighted_contribution=contributions[channel],
            )
            for channel in CHANNELS
        ]

        human_values = self.calibration.get("human_scores")
        human_scores = []
        if isinstance(human_values, Sequence) and not isinstance(human_values, (str, bytes)):
            human_scores = sorted(
                number
                for value in human_values
                if (number := _finite(value)) is not None
            )
        thresholds = _thresholds(self.calibration)
        elevated = thresholds.get("elevated_np_fpr_1pct_delta_5pct")
        high = thresholds.get("high_np_fpr_0_1pct_delta_5pct")
        calibrated = bool(human_scores and elevated is not None and high is not None)
        conformal_p = (
            (1 + sum(reference >= score for reference in human_scores)) / (len(human_scores) + 1)
            if human_scores
            else None
        )
        return base | {
            "status": "ok" if calibrated else "abstain",
            "decision_usable": calibrated,
            "abstention_reasons": [] if calibrated else [
                "The complete fusion statistic has no supported independent human-null calibration."
            ],
            "score": score,
            "human_null_p_value": conformal_p,
            "calibration_maps": len(human_scores),
            "minimum_human_null_p_value": self.calibration.get("minimum_conformal_p"),
            "calibrated": calibrated,
            "thresholds": thresholds,
            "threshold_guarantees": copy.deepcopy(self.calibration.get("thresholds", {})),
            "threshold_flags": {
                "elevated": bool(elevated is not None and score > elevated),
                "high": bool(high is not None and score > high),
            },
            "channels": [dataclasses.asdict(detail) for detail in details],
            "training": copy.deepcopy(self.artifact.get("training", {})),
            "limitations": copy.deepcopy(self.artifact.get("limitations", [])),
        }


def unavailable(reason: str, *, model_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "unavailable",
        "available": False,
        "reason": reason,
        "model_path": str(model_path) if model_path is not None else None,
        "decision_usable": False,
        "score": None,
        "human_null_p_value": None,
        "calibrated": False,
        "thresholds": {},
        "threshold_guarantees": {},
        "threshold_flags": {"elevated": False, "high": False},
        "channels": [],
    }


__all__ = [
    "CHANNELS",
    "DEFAULT_MODEL",
    "RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "CheapFusionModel",
    "combine_surprisals",
    "sha256_file",
    "tail_surprisal",
    "unavailable",
    "upper_tail_p",
]
