"""Pure-Python runtime for the serialization-independent content ensemble."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from .canonical import ContentWindow, extract_content_windows, hash_canonical_ngrams
from .parser import Beatmap


@dataclasses.dataclass(frozen=True)
class ContentWindowScore:
    start_ms: int
    end_ms: int
    object_count: int
    score: float
    ood_distance: float
    ood_feature_fraction: float
    top_evidence: tuple[dict[str, float | str], ...]


def _vector(window: ContentWindow, names: list[str], dimensions: int) -> list[float]:
    values = dict(window.semantic)
    values.update({f"ngram::{index}": value for index, value in hash_canonical_ngrams(window.sequence_tokens, dimensions).items()})
    return [float(values.get(name, 0.0)) for name in names]


def _safe_probability(values: list[float] | tuple[float, ...], index: int) -> float:
    value = float(values[index])
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.5


class ContentEnsemble:
    """Inference wrapper around an exported scikit-learn ExtraTrees model."""

    def __init__(self, artifact: dict[str, Any]):
        if int(artifact.get("schema_version", 0)) != 2:
            raise ValueError("Unsupported content model schema")
        self.artifact = artifact
        self.feature_names = [str(name) for name in artifact["feature_names"]]
        self.ngram_dimensions = int(artifact.get("ngram_dimensions", 512))
        self.trees = list(artifact["trees"])
        self.ood_median = [float(x) for x in artifact["ood_reference"]["median"]]
        self.ood_scale = [max(float(x), 1e-9) for x in artifact["ood_reference"]["robust_scale"]]
        if len(self.ood_median) != len(self.feature_names) or len(self.ood_scale) != len(self.feature_names):
            raise ValueError("Content model OOD reference length mismatch")

    @classmethod
    def from_path(cls, path: str | Path) -> "ContentEnsemble":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _tree_score(self, tree: dict[str, Any], vector: list[float]) -> tuple[float, dict[int, float]]:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        probability = tree["probability"]
        node = 0
        evidence: dict[int, float] = {}
        while int(feature[node]) >= 0:
            feature_index = int(feature[node])
            child = int(left[node]) if vector[feature_index] <= float(threshold[node]) else int(right[node])
            evidence[feature_index] = evidence.get(feature_index, 0.0) + (
                _safe_probability(probability, child) - _safe_probability(probability, node)
            )
            node = child
        return _safe_probability(probability, node), evidence

    def score_vector(self, vector: list[float]) -> tuple[float, dict[int, float]]:
        if len(vector) != len(self.feature_names):
            raise ValueError("Content model vector length mismatch")
        if not self.trees:
            return 0.5, {}
        scores: list[float] = []
        combined: dict[int, float] = {}
        for tree in self.trees:
            score, evidence = self._tree_score(tree, vector)
            scores.append(score)
            for index, value in evidence.items():
                combined[index] = combined.get(index, 0.0) + value / len(self.trees)
        return sum(scores) / len(scores), combined

    def ood(self, vector: list[float]) -> tuple[float, float]:
        if not self.ood_median:
            return 0.0, 0.0
        robust_z = [abs(value - center) / scale for value, center, scale in zip(vector, self.ood_median, self.ood_scale)]
        # A trimmed RMS is stable when a single legitimate rare slider makes
        # one feature extreme, while still exposing broad distribution shift.
        clipped = [min(value, 20.0) for value in robust_z]
        distance = math.sqrt(sum(value * value for value in clipped) / len(clipped))
        fraction = sum(value > 8.0 for value in robust_z) / len(robust_z)
        return distance, fraction

    def score_window(self, window: ContentWindow, top_k: int = 12) -> ContentWindowScore:
        vector = _vector(window, self.feature_names, self.ngram_dimensions)
        score, evidence = self.score_vector(vector)
        distance, fraction = self.ood(vector)
        ordered = sorted(evidence.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        details = tuple(
            {
                "feature": self.feature_names[index],
                "contribution": round(value, 8),
                "value": round(vector[index], 8),
                "reference_median": round(self.ood_median[index], 8),
                "robust_z": round(abs(vector[index] - self.ood_median[index]) / self.ood_scale[index], 5),
            }
            for index, value in ordered
        )
        return ContentWindowScore(
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            object_count=window.object_count,
            score=score,
            ood_distance=distance,
            ood_feature_fraction=fraction,
            top_evidence=details,
        )

    def analyze(self, beatmap: Beatmap) -> dict[str, Any]:
        windows = extract_content_windows(
            beatmap,
            window_ms=int(self.artifact.get("window_ms", 24_000)),
            stride_ms=int(self.artifact.get("stride_ms", 8_000)),
            min_objects=int(self.artifact.get("min_objects", 12)),
        )
        scored = [self.score_window(window) for window in windows]
        ordered = sorted((window.score for window in scored), reverse=True)
        take = min(int(self.artifact.get("aggregation_top_windows", 2)), len(ordered))
        score = sum(ordered[:take]) / take if take else 0.0
        calibration = self.artifact.get("calibration", {})
        human_scores = sorted(float(x) for x in calibration.get("human_scores", []))
        p_value = (
            (1 + sum(reference >= score for reference in human_scores)) / (len(human_scores) + 1)
            if human_scores
            else None
        )
        max_ood = max((window.ood_distance for window in scored), default=0.0)
        max_ood_fraction = max((window.ood_feature_fraction for window in scored), default=0.0)
        raw_ood_limit = self.artifact.get("ood_reference", {}).get("abstain_distance")
        ood_limit = float(raw_ood_limit) if raw_ood_limit is not None else math.inf
        threshold_details = calibration.get("thresholds", {})
        thresholds: dict[str, float] = {}
        for name, value in threshold_details.items():
            # Calibration artifacts retain the complete Neyman-Pearson order
            # statistic and guarantee.  Older/toy artifacts may contain a
            # scalar directly, so support both without weakening the rule.
            if isinstance(value, dict):
                if not value.get("supported") or value.get("threshold") is None:
                    continue
                raw_threshold = value["threshold"]
            else:
                raw_threshold = value
            try:
                numeric_threshold = float(raw_threshold)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_threshold):
                thresholds[str(name)] = numeric_threshold
        elevated_threshold = thresholds.get("elevated_np_fpr_1pct_delta_5pct")
        high_threshold = thresholds.get("high_np_fpr_0_1pct_delta_5pct")
        calibrated = bool(human_scores and elevated_threshold is not None and high_threshold is not None)
        abstain = max_ood > ood_limit
        top_evidence: list[dict[str, Any]] = []
        for window in scored:
            for evidence in window.top_evidence:
                top_evidence.append({
                    "window_start_ms": window.start_ms,
                    "window_end_ms": window.end_ms,
                    **evidence,
                })
        top_evidence.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
        return {
            "available": True,
            "model_id": self.artifact.get("model_id"),
            "score": score,
            "score_semantics": "discriminative content score; not an authorship probability",
            "human_null_p_value": p_value,
            "calibration_maps": len(human_scores),
            "minimum_human_null_p_value": calibration.get("minimum_conformal_p"),
            "calibrated": calibrated,
            "decision_usable": calibrated and not abstain and bool(scored),
            "thresholds": thresholds,
            "threshold_guarantees": threshold_details,
            "threshold_flags": {
                # The calibrated order-statistic rule is strictly greater;
                # using >= would turn a tied calibration maximum into a false
                # positive and invalidate the advertised finite-sample bound.
                "elevated": bool(elevated_threshold is not None and score > elevated_threshold),
                "high": bool(high_threshold is not None and score > high_threshold),
            },
            "ood": {
                "max_distance": max_ood,
                "max_extreme_feature_fraction": max_ood_fraction,
                "abstain_distance": None if not math.isfinite(ood_limit) else ood_limit,
                "abstain": abstain,
            },
            "windows": [dataclasses.asdict(window) for window in scored],
            "top_evidence": top_evidence[:24],
            "representation_audit": self.artifact.get("representation_audit", {}),
            "training": self.artifact.get("training", {}),
            "limitations": self.artifact.get("limitations", []),
        }


def unavailable(reason: str, *, model_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "model_path": str(model_path) if model_path is not None else None,
        "score": None,
        "human_null_p_value": None,
        "calibrated": False,
        "decision_usable": False,
        "thresholds": {},
        "threshold_guarantees": {},
        "threshold_flags": {"elevated": False, "high": False},
        "ood": {"abstain": True, "reason": "content model unavailable"},
        "top_evidence": [],
        "windows": [],
    }
