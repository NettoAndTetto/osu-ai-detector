"""Calibrated Mapperatorinator revision-forensic channel.

This module deliberately keeps generator/export mechanics separate from map
content.  Coordinates on decoder grids, integer-millisecond residues, exact
floating-point SV serialization and output-template fields can be very useful
for identifying an *unmodified revision*, but they are easy to remove with an
editor save and are not a universal definition of AI mapping.

The runtime is pure Python.  Training exports an ExtraTrees ensemble to JSON;
an independent human-null split later supplies conformal p-values and
Neyman--Pearson order-statistic thresholds.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from .advanced_features import WindowFeatures, extract_windows
from .parser import Beatmap


DEFAULT_MODEL = Path(__file__).with_name("models") / "forensic_v3.json"


_TIMING_FORENSIC = {
    "timing_sample_index_minus1",
    "timing_default_red_fields",
    "timing_red_green_same_offset",
    "timing_green_at_object",
    "timing_green_plus6",
    "timing_sv_quantized_001",
    "timing_sv_quantized_005",
    "timing_sv_epsilon_count",
    "timing_sv_epsilon_ratio",
    "timing_beat_length_decimals",
    "timing_beat_length_12plus",
    # Context/support variables.  They cannot be source-anchored evidence on
    # their own but let a tree distinguish one match from a stable pattern.
    "timing_points",
    "timing_red",
    "timing_green",
}

_SPATIAL_MARKERS = (
    "_both_even",
    "_both_mod4_",
    "_coord_even",
    "_coord_mod",
    "_entropy_mod",
    "_top_mod32",
    "_mod4_",
    "_mod32_",
    "_coordinates",
    "_pairs",
)


def forensic_feature_family(name: str) -> str | None:
    """Return the audit family for an allow-listed forensic feature.

    Unknown features fail closed.  In particular, semantic rhythm, pattern
    shape, hitsound and density features cannot silently enter this channel.
    """

    if name.startswith("file_"):
        return "serialization_template"
    if name.startswith("rhythm_time_mod10_"):
        return "integer_millisecond_residue"
    if name.startswith("slider_decimal_"):
        return "float_serialization"
    if name.startswith("space_") and any(marker in name for marker in _SPATIAL_MARKERS):
        return "decoder_coordinate_grid"
    if name in _TIMING_FORENSIC:
        if name.startswith("timing_sv_") or name.startswith("timing_beat_length_"):
            return "sv_float_postprocessor"
        if name in {"timing_points", "timing_red", "timing_green"}:
            return "support_count"
        return "timing_postprocessor"
    return None


def forensic_representation_audit(feature_names: list[str] | None = None) -> dict[str, Any]:
    names = feature_names or []
    return {
        "purpose": "revision provenance / generator-export mechanics",
        "separate_from_semantic_content": True,
        "raw_metadata_authorship_fields_used": False,
        "serialization_fields_used": True,
        "unknown_features_fail_closed": True,
        "allow_list_feature_count": len(names),
        "families": sorted({family for name in names if (family := forensic_feature_family(name))}),
        "interpretation": (
            "A positive match supports a Mapperatorinator-like unmodified revision; "
            "an editor save or manual cleanup can erase it, and a negative result cannot exclude AI use."
        ),
    }


def forensic_values(window: WindowFeatures) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in window.values.items()
        if forensic_feature_family(name) is not None and math.isfinite(float(value))
    }


@dataclasses.dataclass(frozen=True)
class ForensicWindowScore:
    start_ms: int
    end_ms: int
    object_count: int
    score: float
    top_evidence: tuple[dict[str, Any], ...]


def _probability(values: list[float] | tuple[float, ...], index: int) -> float:
    value = float(values[index])
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.5


def _supported_thresholds(calibration: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, raw in calibration.get("thresholds", {}).items():
        if isinstance(raw, dict):
            if not raw.get("supported") or raw.get("threshold") is None:
                continue
            raw = raw["threshold"]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[str(name)] = value
    return result


class ForensicEnsemble:
    """Pure-Python inference for the mechanical/serialization ensemble."""

    def __init__(self, artifact: dict[str, Any]):
        if int(artifact.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported forensic model schema")
        self.artifact = artifact
        self.feature_names = [str(name) for name in artifact["feature_names"]]
        invalid = [name for name in self.feature_names if forensic_feature_family(name) is None]
        if invalid:
            raise ValueError(f"Forensic artifact contains non-allow-listed features: {invalid[:5]}")
        self.trees = list(artifact["trees"])

    @classmethod
    def from_path(cls, path: str | Path) -> "ForensicEnsemble":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _tree_score(self, tree: dict[str, Any], vector: list[float]) -> tuple[float, dict[int, float]]:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        probabilities = tree["probability"]
        node = 0
        evidence: dict[int, float] = {}
        while int(feature[node]) >= 0:
            index = int(feature[node])
            child = int(left[node]) if vector[index] <= float(threshold[node]) else int(right[node])
            evidence[index] = evidence.get(index, 0.0) + (
                _probability(probabilities, child) - _probability(probabilities, node)
            )
            node = child
        return _probability(probabilities, node), evidence

    def score_vector(self, vector: list[float]) -> tuple[float, dict[int, float]]:
        """Score an already allow-listed feature vector.

        This small public primitive is also used by the offline exporter to
        fail closed if pure-Python JSON inference ever diverges from the fitted
        scikit-learn ensemble.
        """

        if len(vector) != len(self.feature_names):
            raise ValueError("Forensic model vector length mismatch")
        if not self.trees:
            return 0.5, {}
        scores: list[float] = []
        evidence: dict[int, float] = {}
        for tree in self.trees:
            score, contributions = self._tree_score(tree, vector)
            scores.append(score)
            for index, contribution in contributions.items():
                evidence[index] = evidence.get(index, 0.0) + contribution / len(self.trees)
        return sum(scores) / len(scores), evidence

    def score_window(self, window: WindowFeatures, top_k: int = 14) -> ForensicWindowScore:
        values = forensic_values(window)
        vector = [float(values.get(name, 0.0)) for name in self.feature_names]
        score, evidence = self.score_vector(vector)
        ordered = sorted(evidence.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        details = tuple(
            {
                "feature": self.feature_names[index],
                "family": forensic_feature_family(self.feature_names[index]),
                "value": round(vector[index], 9),
                "contribution": round(contribution, 9),
                "source_anchored": forensic_feature_family(self.feature_names[index]) != "support_count",
            }
            for index, contribution in ordered
        )
        return ForensicWindowScore(
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            object_count=window.object_count,
            score=score,
            top_evidence=details,
        )

    def analyze(self, beatmap: Beatmap) -> dict[str, Any]:
        windows = extract_windows(
            beatmap,
            window_ms=int(self.artifact.get("window_ms", 24_000)),
            stride_ms=int(self.artifact.get("stride_ms", 8_000)),
            min_objects=int(self.artifact.get("min_objects", 12)),
        )
        scored = [self.score_window(window) for window in windows]
        ordered = sorted((item.score for item in scored), reverse=True)
        take = min(int(self.artifact.get("aggregation_top_windows", 2)), len(ordered))
        score = sum(ordered[:take]) / take if take else 0.0

        calibration = self.artifact.get("calibration", {})
        human_scores = sorted(float(value) for value in calibration.get("human_scores", []))
        conformal_p = (
            (1 + sum(reference >= score for reference in human_scores)) / (len(human_scores) + 1)
            if human_scores
            else None
        )
        thresholds = _supported_thresholds(calibration)
        elevated = thresholds.get("elevated_np_fpr_1pct_delta_5pct")
        high = thresholds.get("high_np_fpr_0_1pct_delta_5pct")
        calibrated = bool(human_scores and elevated is not None and high is not None)

        all_evidence = [
            {"window_start_ms": item.start_ms, "window_end_ms": item.end_ms, **entry}
            for item in scored
            for entry in item.top_evidence
        ]
        all_evidence.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
        positive_anchored = [
            item for item in all_evidence
            if bool(item["source_anchored"]) and float(item["contribution"]) > 0
        ]
        if not scored:
            abstention_reason = "No forensic windows met the model's minimum object requirement."
        elif not human_scores:
            abstention_reason = "No independent human-null calibration scores are embedded."
        elif elevated is None or high is None:
            abstention_reason = "Required supported Neyman-Pearson thresholds are unavailable."
        else:
            abstention_reason = None
        return {
            "status": "ok" if calibrated and bool(scored) else "abstain",
            "available": True,
            "model_id": self.artifact.get("model_id"),
            "score": score,
            "score_semantics": "discriminative revision-forensic score; not an authorship probability",
            "human_null_p_value": conformal_p,
            "calibration_maps": len(human_scores),
            "minimum_human_null_p_value": calibration.get("minimum_conformal_p"),
            "calibrated": calibrated,
            "decision_usable": calibrated and bool(scored),
            "abstention_reason": abstention_reason,
            "thresholds": thresholds,
            "threshold_guarantees": calibration.get("thresholds", {}),
            "threshold_flags": {
                # Strict greater-than is part of the NP order-statistic rule.
                "elevated": bool(elevated is not None and score > elevated),
                "high": bool(high is not None and score > high),
            },
            "source_anchored_positive_evidence": positive_anchored[:24],
            "top_evidence": all_evidence[:30],
            "windows": [dataclasses.asdict(item) for item in scored],
            "representation_audit": self.artifact.get(
                "representation_audit", forensic_representation_audit(self.feature_names)
            ),
            "training": self.artifact.get("training", {}),
            "limitations": self.artifact.get("limitations", []),
        }


def unavailable(reason: str, *, model_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "available": False,
        "reason": reason,
        "abstention_reason": reason,
        "model_path": str(model_path) if model_path is not None else None,
        "score": None,
        "human_null_p_value": None,
        "calibrated": False,
        "decision_usable": False,
        "thresholds": {},
        "threshold_guarantees": {},
        "threshold_flags": {"elevated": False, "high": False},
        "source_anchored_positive_evidence": [],
        "top_evidence": [],
        "windows": [],
    }
