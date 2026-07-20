"""Pure-Python, fail-closed fusion of cheap and white-box map statistics.

The two input scores are deliberately treated as ranking statistics, not as
probabilities.  Each is first put on a common scale using a source-song
balanced, smoothed upper-tail rank from *development OOF humans*.  A frozen
weighted-sum or maximum rule then produces one whole-map statistic.  Its
thresholds and p-value are valid only after calibrating that complete search
on an independent, revision-pinned human-null split.

No training dependency is imported here.  Loading and scoring a frozen JSON
artifact needs only the Python standard library.
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


SCHEMA_VERSION = "osu-ai-detector.deep-fusion/v1"
RESULT_SCHEMA_VERSION = "osu-ai-detector.deep-fusion-result/v1"
DEFAULT_MODEL = Path(__file__).with_name("models") / "deep_fusion_v1.json"

CHANNELS: tuple[str, ...] = ("cheap_fusion_v1", "whitebox_discriminator_v1")
METHODS: tuple[str, ...] = (
    "weighted_sum_tail_surprisal",
    "max_weighted_tail_surprisal",
)
ELEVATED_THRESHOLD = "elevated_np_fpr_1pct_delta_5pct"
HIGH_THRESHOLD = "high_np_fpr_0_1pct_delta_5pct"


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


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def weighted_upper_tail_p(
    scores: Sequence[float],
    weights: Sequence[float],
    score: float,
    *,
    smoothing_mass: float = 1.0,
) -> float:
    """Smoothed weighted upper-tail rank, including reference ties.

    Training gives every source BeatmapSetID one unit of total mass and divides
    it equally over that source's human maps.  ``smoothing_mass=1`` therefore
    preserves the familiar minimum p-value of ``1 / (groups + 1)`` even when
    sources contain different numbers of difficulties.
    """

    if len(scores) != len(weights) or not scores:
        raise ValueError("reference scores and weights must have the same non-zero length")
    normalized_scores = tuple(float(value) for value in scores)
    normalized_weights = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) for value in normalized_scores):
        raise ValueError("reference scores must be finite")
    if tuple(sorted(normalized_scores)) != normalized_scores:
        raise ValueError("reference scores must be sorted")
    if any(not math.isfinite(value) or value <= 0 for value in normalized_weights):
        raise ValueError("reference weights must be finite and positive")
    if not math.isfinite(float(score)):
        raise ValueError("query score must be finite")
    if not math.isfinite(float(smoothing_mass)) or smoothing_mass <= 0:
        raise ValueError("smoothing_mass must be finite and positive")
    index = bisect.bisect_left(normalized_scores, float(score))
    total = math.fsum(normalized_weights)
    tail = math.fsum(normalized_weights[index:])
    if total <= 0:
        raise ValueError("reference weights must have positive total mass")
    return (float(smoothing_mass) + tail) / (float(smoothing_mass) + total)


def tail_surprisal(
    scores: Sequence[float],
    weights: Sequence[float],
    score: float,
    *,
    smoothing_mass: float = 1.0,
) -> float:
    return -math.log10(
        weighted_upper_tail_p(scores, weights, score, smoothing_mass=smoothing_mass)
    )


def combine_surprisals(
    surprisals: Mapping[str, float],
    weights: Mapping[str, float],
    method: str,
) -> tuple[float, dict[str, float]]:
    contributions = {
        channel: float(weights[channel]) * float(surprisals[channel]) for channel in CHANNELS
    }
    if method == "weighted_sum_tail_surprisal":
        score = math.fsum(contributions.values())
    elif method == "max_weighted_tail_surprisal":
        score = max(contributions.values())
    else:
        raise ValueError(f"unsupported deep-fusion method: {method!r}")
    return score, contributions


def _supported_thresholds(calibration: Mapping[str, Any]) -> dict[str, float]:
    raw_thresholds = calibration.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        return {}
    result: dict[str, float] = {}
    for name, raw in raw_thresholds.items():
        if not isinstance(raw, Mapping):
            # A formal deep-fusion threshold must retain its NP guarantee
            # metadata; bare numbers are intentionally not accepted.
            continue
        if not raw.get("supported") or str(raw.get("operator") or "") != ">":
            continue
        value = _finite(raw.get("threshold"))
        if value is not None:
            result[str(name)] = value
    return result


@dataclasses.dataclass(frozen=True)
class DevelopmentReference:
    scores: tuple[float, ...]
    weights: tuple[float, ...]
    smoothing_mass: float
    independent_groups: int
    maps: int
    weighting: str


@dataclasses.dataclass(frozen=True)
class ChannelDetail:
    channel: str
    raw_map_score: float
    development_reference_maps: int
    development_independent_groups: int
    development_total_weight: float
    development_smoothing_mass: float
    development_upper_tail_p: float
    development_tail_surprisal: float
    fusion_weight: float
    weighted_contribution: float
    expected_model_id: str
    expected_artifact_sha256: str
    observed_model_id: str
    observed_artifact_sha256: str
    base_calibrated: bool


class DeepFusionModel:
    """Apply a frozen cheap+white-box statistic and its independent null."""

    def __init__(self, artifact: Mapping[str, Any]):
        if artifact.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported deep-fusion schema: {artifact.get('schema_version')!r}")
        if artifact.get("channels") != list(CHANNELS):
            raise ValueError(f"deep-fusion channels must be exactly {list(CHANNELS)!r}")

        raw_references = artifact.get("development_human_oof_reference")
        if not isinstance(raw_references, Mapping):
            raise ValueError("development_human_oof_reference is missing")
        self.references: dict[str, DevelopmentReference] = {}
        for channel in CHANNELS:
            raw = raw_references.get(channel)
            if not isinstance(raw, Mapping):
                raise ValueError(f"development reference for {channel} is missing")
            raw_scores, raw_weights = raw.get("scores"), raw.get("weights")
            if not _is_sequence(raw_scores) or not _is_sequence(raw_weights):
                raise ValueError(f"development reference arrays for {channel} are missing")
            if not raw_scores or len(raw_scores) != len(raw_weights):
                raise ValueError(f"development reference arrays for {channel} have invalid lengths")
            scores = tuple(float(value) for value in raw_scores)
            weights = tuple(float(value) for value in raw_weights)
            if any(not math.isfinite(value) for value in scores):
                raise ValueError(f"development reference scores for {channel} are non-finite")
            if tuple(sorted(scores)) != scores:
                raise ValueError(f"development reference scores for {channel} must be sorted")
            if any(not math.isfinite(value) or value <= 0 for value in weights):
                raise ValueError(f"development reference weights for {channel} must be finite and positive")
            smoothing = _finite(raw.get("smoothing_mass"))
            groups = raw.get("independent_groups")
            maps = raw.get("maps")
            if smoothing is None or smoothing <= 0:
                raise ValueError(f"development smoothing mass for {channel} must be positive")
            try:
                group_count, map_count = int(groups), int(maps)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"development reference counts for {channel} are invalid") from exc
            if group_count <= 0 or map_count != len(scores):
                raise ValueError(f"development reference counts for {channel} do not match its arrays")
            # With the frozen equal-source weighting, total mass must reconcile
            # to the declared independent group count.
            if not math.isclose(math.fsum(weights), group_count, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"development reference weights for {channel} do not reconcile to groups")
            self.references[channel] = DevelopmentReference(
                scores=scores,
                weights=weights,
                smoothing_mass=smoothing,
                independent_groups=group_count,
                maps=map_count,
                weighting=str(raw.get("weighting") or ""),
            )

        combination = artifact.get("combination")
        if not isinstance(combination, Mapping):
            raise ValueError("combination is missing")
        self.method = str(combination.get("method") or "")
        if self.method not in METHODS:
            raise ValueError(f"unsupported deep-fusion method: {self.method!r}")
        raw_weights = combination.get("weights")
        if not isinstance(raw_weights, Mapping):
            raise ValueError("combination.weights is missing")
        self.weights: dict[str, float] = {}
        for channel in CHANNELS:
            value = _finite(raw_weights.get(channel))
            if value is None or value < 0:
                raise ValueError(f"fusion weight for {channel} must be finite and non-negative")
            self.weights[channel] = value
        if not any(value > 0 for value in self.weights.values()):
            raise ValueError("at least one deep-fusion weight must be positive")

        binding = artifact.get("base_model_binding")
        if not isinstance(binding, Mapping):
            raise ValueError("base_model_binding is missing")
        self.binding: dict[str, dict[str, Any]] = {}
        for channel in CHANNELS:
            raw = binding.get(channel)
            if not isinstance(raw, Mapping):
                raise ValueError(f"base-model binding for {channel} is missing")
            model_id = str(raw.get("model_id") or "").strip()
            digest = str(raw.get("sha256") or "").strip().casefold()
            if not model_id:
                raise ValueError(f"base-model binding for {channel} has no model_id")
            if not _valid_sha256(digest):
                raise ValueError(f"base-model binding for {channel} has an invalid SHA-256")
            self.binding[channel] = copy.deepcopy(dict(raw))
            self.binding[channel]["model_id"] = model_id
            self.binding[channel]["sha256"] = digest

        calibration = artifact.get("calibration")
        self.calibration = copy.deepcopy(dict(calibration)) if isinstance(calibration, Mapping) else {}
        raw_humans = self.calibration.get("human_scores", [])
        if raw_humans and not _is_sequence(raw_humans):
            raise ValueError("calibration.human_scores must be a sequence")
        if _is_sequence(raw_humans):
            human_scores = tuple(float(value) for value in raw_humans)
            if any(not math.isfinite(value) for value in human_scores):
                raise ValueError("calibration.human_scores contains a non-finite value")
            if tuple(sorted(human_scores)) != human_scores:
                raise ValueError("calibration.human_scores must be sorted")
            self.human_scores = human_scores
        else:
            self.human_scores = ()
        if self.human_scores:
            if self.calibration.get("status") != "available":
                raise ValueError("non-empty calibration must have status='available'")
            minimum_p = _finite(self.calibration.get("minimum_conformal_p"))
            expected_minimum = 1.0 / (len(self.human_scores) + 1)
            if minimum_p is None or not math.isclose(
                minimum_p, expected_minimum, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("calibration.minimum_conformal_p does not reconcile to human_scores")
            if "maps" in self.calibration:
                try:
                    declared_maps = int(self.calibration["maps"])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("calibration.maps must be an integer") from exc
                if declared_maps != len(self.human_scores):
                    raise ValueError("calibration.maps does not reconcile to human_scores")
        self.model_id = str(artifact.get("model_id") or "").strip()
        if not self.model_id:
            raise ValueError("deep-fusion model_id is missing")
        self.artifact = copy.deepcopy(dict(artifact))

    @classmethod
    def from_path(cls, path: str | Path) -> "DeepFusionModel":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("deep-fusion artifact must be a JSON object")
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.artifact)

    def _binding_errors(
        self,
        results: Mapping[str, Mapping[str, Any]],
        observed_model_sha256: Mapping[str, str] | None,
    ) -> list[str]:
        observed_hashes = observed_model_sha256 or {}
        errors: list[str] = []
        for channel in CHANNELS:
            expected = self.binding[channel]
            observed_hash = str(observed_hashes.get(channel) or "").strip().casefold()
            expected_hash = str(expected["sha256"])
            if observed_hash != expected_hash:
                errors.append(
                    f"{channel} artifact hash mismatch (expected {expected_hash}, "
                    f"observed {observed_hash or 'missing'})"
                )
            observed_id = str(results[channel].get("model_id") or "")
            if observed_id != expected["model_id"]:
                errors.append(
                    f"{channel} model_id mismatch (expected {expected['model_id']!r}, "
                    f"observed {observed_id or 'missing'!r})"
                )
        return errors

    def analyze(
        self,
        cheap_fusion: Mapping[str, Any],
        whitebox_discriminator: Mapping[str, Any],
        *,
        observed_model_sha256: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Mapping[str, Any]] = {
            "cheap_fusion_v1": cheap_fusion,
            "whitebox_discriminator_v1": whitebox_discriminator,
        }
        errors = self._binding_errors(results, observed_model_sha256)
        raw_scores: dict[str, float] = {}
        base_calibrated: dict[str, bool] = {}

        if not cheap_fusion.get("available"):
            errors.append("cheap_fusion_v1 is unavailable")
        if cheap_fusion.get("status") != "ok" or not cheap_fusion.get("decision_usable"):
            errors.append("cheap_fusion_v1 has no usable independently calibrated map decision")
        cheap_calibrated = bool(cheap_fusion.get("calibrated"))
        base_calibrated["cheap_fusion_v1"] = cheap_calibrated
        if not cheap_calibrated:
            errors.append("cheap_fusion_v1 complete statistic is not calibrated")
        cheap_score = _finite(cheap_fusion.get("score"))
        if cheap_score is None:
            errors.append("cheap_fusion_v1 has no finite map score")
        else:
            raw_scores["cheap_fusion_v1"] = cheap_score

        if not whitebox_discriminator.get("available"):
            errors.append("whitebox_discriminator_v1 is unavailable")
        if whitebox_discriminator.get("status") != "ok":
            errors.append("whitebox_discriminator_v1 has no successful aggregate")
        aggregate = whitebox_discriminator.get("aggregate")
        whitebox_score = _finite(aggregate.get("ranking_score")) if isinstance(aggregate, Mapping) else None
        if whitebox_score is None:
            errors.append("whitebox_discriminator_v1 has no finite aggregate ranking score")
        else:
            raw_scores["whitebox_discriminator_v1"] = whitebox_score
        whitebox_calibration = whitebox_discriminator.get("calibration")
        whitebox_calibrated = bool(
            isinstance(whitebox_calibration, Mapping) and whitebox_calibration.get("calibrated")
        )
        base_calibrated["whitebox_discriminator_v1"] = whitebox_calibrated
        if not whitebox_calibrated or whitebox_discriminator.get("uncalibrated"):
            errors.append("whitebox_discriminator_v1 complete statistic is not calibrated")

        observed_hashes = {
            channel: str((observed_model_sha256 or {}).get(channel) or "").casefold()
            for channel in CHANNELS
        }
        base = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "available": True,
            "score_semantics": (
                "Frozen source-balanced tail-surprisal fusion of cheap and white-box whole-map ranking "
                "statistics; independently calibrated as one complete search; not an AI-authorship probability."
            ),
            "combination": {
                "method": self.method,
                "weights": dict(self.weights),
                "formula": (
                    "weighted sum of -log10 smoothed development-human upper-tail ranks"
                    if self.method == "weighted_sum_tail_surprisal"
                    else "maximum weighted -log10 smoothed development-human upper-tail rank"
                ),
            },
            "base_model_binding": copy.deepcopy(self.binding),
            "observed_base_model_sha256": observed_hashes,
        }
        if errors:
            return base | {
                "status": "abstain",
                "decision_usable": False,
                "abstention_reasons": errors,
                "score": None,
                "human_null_p_value": None,
                "calibrated": False,
                "calibration_maps": len(self.human_scores),
                "thresholds": {},
                "threshold_guarantees": copy.deepcopy(self.calibration.get("thresholds", {})),
                "threshold_flags": {"elevated": False, "high": False},
                "channels": [],
                "limitations": copy.deepcopy(self.artifact.get("limitations", [])),
            }

        p_values: dict[str, float] = {}
        surprisals: dict[str, float] = {}
        for channel in CHANNELS:
            reference = self.references[channel]
            p_value = weighted_upper_tail_p(
                reference.scores,
                reference.weights,
                raw_scores[channel],
                smoothing_mass=reference.smoothing_mass,
            )
            p_values[channel] = p_value
            surprisals[channel] = -math.log10(p_value)
        score, contributions = combine_surprisals(surprisals, self.weights, self.method)

        details = []
        for channel in CHANNELS:
            reference = self.references[channel]
            details.append(
                dataclasses.asdict(
                    ChannelDetail(
                        channel=channel,
                        raw_map_score=raw_scores[channel],
                        development_reference_maps=reference.maps,
                        development_independent_groups=reference.independent_groups,
                        development_total_weight=math.fsum(reference.weights),
                        development_smoothing_mass=reference.smoothing_mass,
                        development_upper_tail_p=p_values[channel],
                        development_tail_surprisal=surprisals[channel],
                        fusion_weight=self.weights[channel],
                        weighted_contribution=contributions[channel],
                        expected_model_id=str(self.binding[channel]["model_id"]),
                        expected_artifact_sha256=str(self.binding[channel]["sha256"]),
                        observed_model_id=str(results[channel].get("model_id")),
                        observed_artifact_sha256=observed_hashes[channel],
                        base_calibrated=base_calibrated[channel],
                    )
                )
            )

        thresholds = _supported_thresholds(self.calibration)
        elevated = thresholds.get(ELEVATED_THRESHOLD)
        high = thresholds.get(HIGH_THRESHOLD)
        calibrated = bool(self.human_scores and elevated is not None and high is not None)
        conformal_p = (
            (1 + sum(reference >= score for reference in self.human_scores))
            / (len(self.human_scores) + 1)
            if self.human_scores
            else None
        )
        reasons = [] if calibrated else [
            "The complete cheap+white-box statistic lacks both supported strict NP thresholds on an independent human null."
        ]
        return base | {
            "status": "ok" if calibrated else "abstain",
            "decision_usable": calibrated,
            "abstention_reasons": reasons,
            "score": score,
            "human_null_p_value": conformal_p,
            "calibrated": calibrated,
            "calibration_maps": len(self.human_scores),
            "minimum_human_null_p_value": self.calibration.get("minimum_conformal_p"),
            "thresholds": thresholds,
            "threshold_guarantees": copy.deepcopy(self.calibration.get("thresholds", {})),
            "threshold_flags": {
                "elevated": bool(elevated is not None and score > elevated),
                "high": bool(high is not None and score > high),
            },
            "channels": details,
            "training": copy.deepcopy(self.artifact.get("training", {})),
            "calibration_audit": {
                "protocol": self.calibration.get("protocol"),
                "independence_unit": self.calibration.get("independence_unit"),
                "human_manifest_sha256": self.calibration.get("human_manifest_sha256"),
                "human_audio_manifest_sha256": self.calibration.get("human_audio_manifest_sha256"),
            },
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
    "ELEVATED_THRESHOLD",
    "HIGH_THRESHOLD",
    "METHODS",
    "RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "DeepFusionModel",
    "combine_surprisals",
    "sha256_file",
    "tail_surprisal",
    "unavailable",
    "weighted_upper_tail_p",
]
