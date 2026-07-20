from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

from .content_model import ContentEnsemble, unavailable as content_unavailable
from .deep_fusion import (
    DEFAULT_MODEL as DEFAULT_DEEP_FUSION_MODEL,
    DeepFusionModel,
    unavailable as deep_fusion_unavailable,
)
from .features import FeatureSet, extract_features
from .forensic import (
    DEFAULT_MODEL as DEFAULT_FORENSIC_MODEL,
    ForensicEnsemble,
    unavailable as forensic_unavailable,
)
from .fusion import (
    DEFAULT_MODEL as DEFAULT_FUSION_MODEL,
    CheapFusionModel,
    sha256_file,
    unavailable as fusion_unavailable,
)
from .parser import Beatmap, parse_beatmap
from .revision_registry import (
    DEFAULT_REGISTRY as DEFAULT_REVISION_REGISTRY,
    load_revision_registry,
    match_revision,
    unavailable as revision_registry_unavailable,
)
from .statistical import analyze_statistical


DEFAULT_CONTENT_MODEL = Path(__file__).resolve().parent / "models" / "content_v2.json"


class Verdict(str, enum.Enum):
    HIGH_CONFIDENCE_AI = "high_confidence_ai"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED_MODE = "unsupported_mode"


@dataclasses.dataclass(frozen=True)
class Evidence:
    family: str
    key: str
    strength: float
    reliability: float
    description: str
    observed: dict[str, float | int]

    @property
    def contribution(self) -> float:
        return self.strength * self.reliability


@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    """Deprecated v0.2 exploratory-channel thresholds.

    These settings are retained so earlier clients can reproduce and inspect
    the old rule/statistical output.  They never control the primary verdict.
    """

    profile_name: str = "deprecated-exploratory-v0.2"
    min_objects: int = 40
    min_positioned_objects: int = 30
    min_coordinates: int = 100
    evidence_activation: float = 0.25
    strong_family_contribution: float = 0.58
    exact_signature_contribution: float = 0.84
    high_confidence_score: float = 0.78
    suspicious_score: float = 0.48
    min_strong_families: int = 2
    statistical_enabled: int = 1

    template_reliability: float = 0.88
    v32_even_start: float = 0.985
    v32_even_full: float = 0.998
    v32_entropy_center: float = 0.795
    v32_entropy_max_distance: float = 0.040
    v32_entropy_full_distance: float = 0.015
    v32_sample_start: int = 100
    v32_sample_full: int = 180
    v32_spatial_reliability: float = 0.72
    # V28/V29 combined position tokens decode to the fixed centre of each
    # 32 px cell. Generated slider anchors can be postprocessed off-centre,
    # hence 0.92 (rather than 0.98) is the exact-signature saturation point.
    # The maximum observed in 470 human calibration/frozen maps is < 0.52.
    v29_center_start: float = 0.70
    v29_center_full: float = 0.92
    v29_center_reliability: float = 0.96
    legacy_mod4_start: float = 0.58
    legacy_mod4_full: float = 0.88
    legacy_offset_start: float = 0.56
    legacy_offset_full: float = 0.82
    legacy_entropy_ceiling: float = 0.82
    legacy_entropy_span: float = 0.22
    legacy_spatial_reliability: float = 0.42

    offgrid_min_objects: int = 18
    offgrid_halfstep_start: float = 0.42
    offgrid_halfstep_full: float = 0.78
    temporal_reliability: float = 0.79

    epsilon_min_matches: int = 3
    epsilon_min_ratio: float = 0.12
    epsilon_ratio_start: float = 0.10
    epsilon_ratio_full: float = 0.28
    epsilon_count_start: int = 2
    epsilon_count_full: int = 6
    epsilon_reliability: float = 0.97

    super_timing_redline_start: float = 16.0
    super_timing_redline_full: float = 32.0
    super_timing_overlap_start: float = 0.45
    super_timing_overlap_full: float = 0.85
    volume_reset_min_greenlines: int = 20
    volume_reset_start: float = 0.05
    volume_reset_full: float = 0.35
    timing_pattern_reliability: float = 0.46

    slider_min_count: int = 20
    slider_decimal_start: float = 0.25
    slider_decimal_full: float = 0.75
    slider_min_comparable_pairs: int = 15
    slider_geometry_start: float = 0.45
    slider_geometry_full: float = 0.80
    slider_geometry_weight: float = 0.75
    slider_reliability: float = 0.34

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "profile_name":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("profile_name must be a non-empty string")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{field.name} must be a finite number")
            if isinstance(field.default, int) and not isinstance(field.default, bool) and not isinstance(value, int):
                raise ValueError(f"{field.name} must be an integer")
            if value < 0:
                raise ValueError(f"{field.name} must not be negative")
            if isinstance(field.default, float) and not field.name.startswith("super_timing_redline_") and value > 1:
                raise ValueError(f"{field.name} must be between 0 and 1")

        ordered = (
            ("v32_even_start", "v32_even_full"),
            ("v32_sample_start", "v32_sample_full"),
            ("v29_center_start", "v29_center_full"),
            ("legacy_mod4_start", "legacy_mod4_full"),
            ("legacy_offset_start", "legacy_offset_full"),
            ("offgrid_halfstep_start", "offgrid_halfstep_full"),
            ("epsilon_ratio_start", "epsilon_ratio_full"),
            ("epsilon_count_start", "epsilon_count_full"),
            ("super_timing_redline_start", "super_timing_redline_full"),
            ("super_timing_overlap_start", "super_timing_overlap_full"),
            ("volume_reset_start", "volume_reset_full"),
            ("slider_decimal_start", "slider_decimal_full"),
            ("slider_geometry_start", "slider_geometry_full"),
        )
        for start, full in ordered:
            if getattr(self, start) > getattr(self, full):
                raise ValueError(f"{start} must not exceed {full}")
        if self.v32_entropy_full_distance > self.v32_entropy_max_distance:
            raise ValueError("v32_entropy_full_distance must not exceed v32_entropy_max_distance")
        if self.suspicious_score > self.high_confidence_score:
            raise ValueError("suspicious_score must not exceed high_confidence_score")
        if self.min_strong_families < 1:
            raise ValueError("min_strong_families must be at least 1")
        if self.statistical_enabled not in {0, 1}:
            raise ValueError("statistical_enabled must be 0 or 1")

    @classmethod
    def from_dict(cls, values: dict | None) -> "DetectorConfig":
        values = values or {}
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("Unknown detector configuration fields: " + ", ".join(unknown))
        return cls(**values)

    def to_dict(self) -> dict[str, str | float | int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DetectionReport:
    path: str
    verdict: Verdict
    evidence_score: float
    mapperatorinator_scope: str
    evidence: tuple[Evidence, ...]
    features: dict[str, float]
    counts: dict[str, int]
    caveats: tuple[str, ...]
    decision_trace: dict[str, object]
    map_identity: dict[str, object]
    analysis_channels: dict[str, object] = dataclasses.field(default_factory=dict)

    def to_dict(self, include_features: bool = True) -> dict:
        result = {
            "path": self.path,
            "verdict": self.verdict.value,
            "evidence_score": round(self.evidence_score, 4),
            "mapperatorinator_scope": self.mapperatorinator_scope,
            "evidence": [
                {**dataclasses.asdict(item), "contribution": round(item.contribution, 8)}
                for item in self.evidence
            ],
            "counts": self.counts,
            "caveats": list(self.caveats),
            "decision_trace": self.decision_trace,
            "map_identity": self.map_identity,
            "analysis_channels": self.analysis_channels,
        }
        if include_features:
            result["features"] = {key: round(value, 8) for key, value in self.features.items()}
        return result


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ramp(value: float, start: float, full: float) -> float:
    if full <= start:
        return float(value >= full)
    return _clamp((value - start) / (full - start))


def _map_identity(beatmap: Beatmap) -> dict[str, object]:
    try:
        raw = beatmap.path.read_bytes()
    except OSError:
        raw = beatmap.raw_text.encode("utf-8")
    metadata = beatmap.metadata
    return {
        "format_version": beatmap.format_version,
        "mode": beatmap.mode,
        "bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),  # nosec B324 - osu! revision checksum convention
        "sha256": hashlib.sha256(raw).hexdigest(),
        "beatmap_id": metadata.get("BeatmapID"),
        "beatmapset_id": metadata.get("BeatmapSetID"),
        "artist": metadata.get("Artist"),
        "title": metadata.get("Title"),
        "creator": metadata.get("Creator"),
        "version": metadata.get("Version"),
    }


class Detector:
    """Multi-channel detector with a calibrated content model as primary path.

    Version 0.2 rules and its statistical model remain available for audit and
    backwards-compatible detail, but are deprecated exploratory diagnostics
    and never promote the final verdict.  A direct disclosure is treated as a
    declaration, while model-driven verdicts require a present, calibrated and
    non-OOD ``content_v2`` artifact.
    """

    scope = (
        "osu!standard; calibrated serialization-independent content_v2 primary channel; "
        "independently calibrated content+revision_forensics cheap fusion when available; "
        "optional independently calibrated audio-backed cheap+white-box deep fusion; "
        "independent revision_forensics_v3 fallback corroboration/suspicious-cap channel; optional Mapperatorinator "
        "V29-V32 white-box channel; V28-V32 v0.2 diagnostics deprecated"
    )

    def __init__(
        self,
        config: DetectorConfig | None = None,
        *,
        content_model_path: str | Path | None = None,
        content_model_enabled: bool = True,
        forensic_model_path: str | Path | None = None,
        forensic_model_enabled: bool = True,
        revision_registry_path: str | Path | None = None,
        revision_registry_enabled: bool = True,
        fusion_model_path: str | Path | None = None,
        fusion_model_enabled: bool = True,
        deep_fusion_model_path: str | Path | None = None,
        deep_fusion_model_enabled: bool = False,
    ) -> None:
        self.config = config or DetectorConfig()
        configured_path = content_model_path or os.environ.get("OSU_AI_CONTENT_MODEL") or DEFAULT_CONTENT_MODEL
        self.content_model_path = Path(configured_path).expanduser().resolve()
        self.content_model_enabled = bool(content_model_enabled)
        self._content_model: ContentEnsemble | None = None
        self._content_model_error: str | None = None
        if self.content_model_enabled:
            if not self.content_model_path.is_file():
                self._content_model_error = (
                    f"content_v2 artifact is missing: {self.content_model_path}; "
                    "no content-model confidence is available"
                )
            else:
                try:
                    self._content_model = ContentEnsemble.from_path(self.content_model_path)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._content_model_error = f"cannot load content_v2 artifact: {type(exc).__name__}: {exc}"

        forensic_path = forensic_model_path or os.environ.get("OSU_AI_FORENSIC_MODEL") or DEFAULT_FORENSIC_MODEL
        self.forensic_model_path = Path(forensic_path).expanduser().resolve()
        self.forensic_model_enabled = bool(forensic_model_enabled)
        self._forensic_model: ForensicEnsemble | None = None
        self._forensic_model_error: str | None = None
        if self.forensic_model_enabled:
            if not self.forensic_model_path.is_file():
                self._forensic_model_error = (
                    f"revision_forensics_v3 artifact is missing: {self.forensic_model_path}; "
                    "the forensic channel abstains"
                )
            else:
                try:
                    self._forensic_model = ForensicEnsemble.from_path(self.forensic_model_path)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._forensic_model_error = (
                        f"cannot load revision_forensics_v3 artifact: {type(exc).__name__}: {exc}"
                    )

        registry_path = (
            revision_registry_path
            or os.environ.get("OSU_AI_REVISION_REGISTRY")
            or DEFAULT_REVISION_REGISTRY
        )
        self.revision_registry_path = Path(registry_path).expanduser().resolve()
        self.revision_registry_enabled = bool(revision_registry_enabled)
        self._revision_registry: dict[str, object] | None = None
        self._revision_registry_error: str | None = None
        if self.revision_registry_enabled:
            if not self.revision_registry_path.is_file():
                self._revision_registry_error = (
                    f"public revision registry is missing: {self.revision_registry_path}; "
                    "exact historical-revision lookup is unavailable"
                )
            else:
                try:
                    self._revision_registry = load_revision_registry(str(self.revision_registry_path))
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._revision_registry_error = (
                        f"cannot load public revision registry: {type(exc).__name__}: {exc}"
                    )

        self.observed_base_model_sha256: dict[str, str] = {}
        for channel, path in (
            ("content_v2", self.content_model_path),
            ("revision_forensics_v3", self.forensic_model_path),
        ):
            if path.is_file():
                try:
                    self.observed_base_model_sha256[channel] = sha256_file(path)
                except OSError:
                    pass

        fusion_path = fusion_model_path or os.environ.get("OSU_AI_FUSION_MODEL") or DEFAULT_FUSION_MODEL
        self.fusion_model_path = Path(fusion_path).expanduser().resolve()
        self.fusion_model_enabled = bool(fusion_model_enabled)
        self._fusion_model: CheapFusionModel | None = None
        self._fusion_model_error: str | None = None
        if self.fusion_model_enabled:
            if not self.fusion_model_path.is_file():
                self._fusion_model_error = (
                    f"cheap fusion artifact is missing: {self.fusion_model_path}; "
                    "the fusion channel abstains and base-channel fallback remains active"
                )
            else:
                try:
                    self._fusion_model = CheapFusionModel.from_path(self.fusion_model_path)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._fusion_model_error = (
                        f"cannot load cheap fusion artifact: {type(exc).__name__}: {exc}"
                    )

        deep_path = (
            deep_fusion_model_path
            or os.environ.get("OSU_AI_DEEP_FUSION_MODEL")
            or DEFAULT_DEEP_FUSION_MODEL
        )
        self.deep_fusion_model_path = Path(deep_path).expanduser().resolve()
        self.deep_fusion_model_enabled = bool(deep_fusion_model_enabled)
        self._deep_fusion_model: DeepFusionModel | None = None
        self._deep_fusion_model_error: str | None = None
        if self.deep_fusion_model_enabled:
            if not self.deep_fusion_model_path.is_file():
                self._deep_fusion_model_error = (
                    f"deep fusion artifact is missing: {self.deep_fusion_model_path}; "
                    "the optional audio-backed channel abstains"
                )
            else:
                try:
                    self._deep_fusion_model = DeepFusionModel.from_path(self.deep_fusion_model_path)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._deep_fusion_model_error = (
                        f"cannot load deep fusion artifact: {type(exc).__name__}: {exc}"
                    )

    def _analyze_content(self, beatmap: Beatmap) -> dict[str, object]:
        if not self.content_model_enabled:
            return content_unavailable(
                "content_v2 disabled by request; final model confidence is unavailable",
                model_path=self.content_model_path,
            )
        if self._content_model is None:
            return content_unavailable(
                self._content_model_error or "content_v2 is unavailable",
                model_path=self.content_model_path,
            )
        if beatmap.mode != 0:
            return content_unavailable(
                f"content_v2 supports osu!standard only; observed mode={beatmap.mode}",
                model_path=self.content_model_path,
            )
        try:
            result = self._content_model.analyze(beatmap)
            result["model_path"] = str(self.content_model_path)
            return result
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
            return content_unavailable(
                f"content_v2 analysis failed: {type(exc).__name__}: {exc}",
                model_path=self.content_model_path,
            )

    def _analyze_forensic(self, beatmap: Beatmap) -> dict[str, object]:
        if not self.forensic_model_enabled:
            result = forensic_unavailable(
                "revision_forensics_v3 disabled by request; the channel explicitly abstains",
                model_path=self.forensic_model_path,
            )
            result["status"] = "disabled"
            return result
        if self._forensic_model is None:
            return forensic_unavailable(
                self._forensic_model_error or "revision_forensics_v3 is unavailable",
                model_path=self.forensic_model_path,
            )
        if beatmap.mode != 0:
            return forensic_unavailable(
                f"revision_forensics_v3 supports osu!standard only; observed mode={beatmap.mode}",
                model_path=self.forensic_model_path,
            )
        try:
            result = self._forensic_model.analyze(beatmap)
            result["model_path"] = str(self.forensic_model_path)
            return result
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
            return forensic_unavailable(
                f"revision_forensics_v3 analysis failed: {type(exc).__name__}: {exc}",
                model_path=self.forensic_model_path,
            )

    def _analyze_revision_registry(self, beatmap: Beatmap) -> dict[str, object]:
        if not self.revision_registry_enabled:
            result = revision_registry_unavailable(
                "public revision registry disabled by request; no historical-revision identity lookup was performed",
                self.revision_registry_path,
            )
            result["status"] = "disabled"
            return result
        if self._revision_registry is None:
            return revision_registry_unavailable(
                self._revision_registry_error or "public revision registry is unavailable",
                self.revision_registry_path,
            )
        try:
            result = match_revision(beatmap, self._revision_registry)
            result["registry_path"] = str(self.revision_registry_path)
            return result
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
            return revision_registry_unavailable(
                f"public revision registry lookup failed: {type(exc).__name__}: {exc}",
                self.revision_registry_path,
            )

    def _analyze_fusion(
        self, content: dict[str, object], forensic: dict[str, object]
    ) -> dict[str, object]:
        if not self.fusion_model_enabled:
            result = fusion_unavailable(
                "cheap fusion disabled by request; base-channel fallback remains active",
                model_path=self.fusion_model_path,
            )
            result["status"] = "disabled"
        elif self._fusion_model is None:
            result = fusion_unavailable(
                self._fusion_model_error or "cheap fusion is unavailable",
                model_path=self.fusion_model_path,
            )
        else:
            try:
                result = self._fusion_model.analyze(
                    content,
                    forensic,
                    observed_model_sha256=self.observed_base_model_sha256,
                )
                result["model_path"] = str(self.fusion_model_path)
            except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
                result = fusion_unavailable(
                    f"cheap fusion analysis failed: {type(exc).__name__}: {exc}",
                    model_path=self.fusion_model_path,
                )
        result["observed_base_model_sha256"] = dict(self.observed_base_model_sha256)
        result["used_for_final_verdict"] = False
        result["whitebox_included"] = False
        return result

    def analyze_deep_fusion(
        self,
        cheap_fusion: Mapping[str, object],
        whitebox_discriminator: Mapping[str, object],
        *,
        whitebox_model_path: str | Path | None = None,
        observed_whitebox_model_sha256: str | None = None,
    ) -> dict[str, object]:
        """Score the optional audio-backed channel after white-box extraction.

        ``Detector.analyze`` remains map-only, so callers that possess audio
        invoke this method after attaching a :class:`WhiteboxDiscriminator`
        result.  Both exact base artifact hashes are recomputed/required; a
        caller cannot accidentally turn a missing model into numeric zero.
        """

        if not self.deep_fusion_model_enabled:
            result = deep_fusion_unavailable(
                "deep fusion was not requested; run the white-box discriminator and explicitly enable it",
                model_path=self.deep_fusion_model_path,
            )
            result["status"] = "disabled"
            result["enabled"] = False
            result["used_for_final_verdict"] = False
            result["independent_deep_conclusion"] = "not_requested"
            return result
        if self._deep_fusion_model is None:
            result = deep_fusion_unavailable(
                self._deep_fusion_model_error or "deep fusion is unavailable",
                model_path=self.deep_fusion_model_path,
            )
            result["enabled"] = True
            result["used_for_final_verdict"] = False
            result["independent_deep_conclusion"] = "abstain"
            return result

        observed_hashes: dict[str, str] = {}
        if self.fusion_model_path.is_file():
            try:
                observed_hashes["cheap_fusion_v1"] = sha256_file(self.fusion_model_path)
            except OSError:
                pass
        if observed_whitebox_model_sha256:
            observed_hashes["whitebox_discriminator_v1"] = str(
                observed_whitebox_model_sha256
            ).casefold()
        elif whitebox_model_path is not None:
            candidate = Path(whitebox_model_path).expanduser().resolve()
            if candidate.is_file():
                try:
                    observed_hashes["whitebox_discriminator_v1"] = sha256_file(candidate)
                except OSError:
                    pass
        try:
            result = self._deep_fusion_model.analyze(
                cheap_fusion,
                whitebox_discriminator,
                observed_model_sha256=observed_hashes,
            )
            result["model_path"] = str(self.deep_fusion_model_path)
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
            result = deep_fusion_unavailable(
                f"deep fusion analysis failed: {type(exc).__name__}: {exc}",
                model_path=self.deep_fusion_model_path,
            )
        result["enabled"] = True
        result["observed_base_model_sha256"] = observed_hashes
        # The frozen map-only verdict policy is not silently rewritten by an
        # optional post-analysis call.  Consumers can use the explicit,
        # independently calibrated recommendation once production evaluation
        # has frozen that policy.
        result["used_for_final_verdict"] = False
        flags = result.get("threshold_flags")
        flags = flags if isinstance(flags, Mapping) else {}
        if not result.get("decision_usable"):
            conclusion = "abstain"
        elif flags.get("high"):
            conclusion = "high_confidence_ai"
        elif flags.get("elevated"):
            conclusion = "suspicious"
        else:
            conclusion = "inconclusive"
        result["independent_deep_conclusion"] = conclusion
        result["eligible_for_frozen_verdict_policy"] = bool(result.get("decision_usable"))
        return result

    def analyze(self, path_or_map: str | Path | Beatmap) -> DetectionReport:
        beatmap = path_or_map if isinstance(path_or_map, Beatmap) else parse_beatmap(path_or_map)
        identity = _map_identity(beatmap)
        features = extract_features(beatmap)
        content = self._analyze_content(beatmap)
        forensic = self._analyze_forensic(beatmap)
        registry = self._analyze_revision_registry(beatmap)
        fusion = self._analyze_fusion(content, forensic)
        fusion_usable = bool(
            fusion.get("available")
            and fusion.get("status") == "ok"
            and fusion.get("calibrated")
            and fusion.get("decision_usable")
        )
        registry_exact_positive = bool(
            registry.get("available")
            and registry.get("status") == "exact_verified_public_positive"
            and registry.get("known_exact_positive") is True
            and registry.get("exact_match")
        )
        registry["used_for_final_verdict"] = False
        registry["verdict_basis"] = (
            "exact_sha256_historical_revision_identity"
            if registry_exact_positive
            else "none; identity/BeatmapID/BeatmapSetID matches never inherit a positive label"
        )
        forensic["used_for_final_verdict"] = False
        forensic["high_confidence_verdict_allowed"] = False
        forensic["channel_role"] = "abstain" if not forensic.get("decision_usable") else "corroboration_only"
        caveats = [
            "The score is not proof of authorship and must not be used alone for public accusations.",
            "An inconclusive result does not mean human-made; editing can erase generator fingerprints.",
            "The v0.2 rule/statistical channel is deprecated exploratory evidence and does not set the verdict.",
        ]
        if not content.get("available"):
            caveats.append(str(content.get("reason", "The primary content model is unavailable.")))
        if not forensic.get("decision_usable"):
            caveats.append(
                str(
                    forensic.get("abstention_reason")
                    or forensic.get("reason")
                    or "revision_forensics_v3 is not calibrated or has no usable windows; the channel abstains."
                )
            )
        if not registry.get("available"):
            caveats.append(str(registry.get("reason", "The public revision registry is unavailable.")))
        if not fusion_usable:
            fusion_reasons = fusion.get("abstention_reasons")
            if isinstance(fusion_reasons, list) and fusion_reasons:
                caveats.append("Cheap fusion abstained: " + "; ".join(map(str, fusion_reasons)))
            else:
                caveats.append(str(fusion.get("reason", "Cheap fusion is unavailable; base-channel fallback is used.")))

        if beatmap.mode != 0 and not registry_exact_positive:
            caveats.append("Only osu!standard thresholds have been calibrated in this release.")
            return DetectionReport(
                path=str(beatmap.path),
                verdict=Verdict.UNSUPPORTED_MODE,
                evidence_score=0.0,
                mapperatorinator_scope=self.scope,
                evidence=(),
                features=features.values,
                counts=features.counts,
                caveats=tuple(caveats),
                decision_trace={
                    "reason": "unsupported_mode",
                    "observed_mode": beatmap.mode,
                    "content_model": content,
                    "public_revision_registry": registry,
                    "cheap_fusion_v1": fusion,
                },
                map_identity=identity,
                analysis_channels={
                    "content_v2": content,
                    "public_revision_registry": registry,
                    "cheap_fusion_v1": fusion,
                    "revision_forensics_v3": forensic,
                    "whitebox": {
                        "enabled": False,
                        "status": "not_requested",
                        "discriminator": {
                            "enabled": False,
                            "status": "not_requested",
                            "available": False,
                            "used_for_final_verdict": False,
                        },
                    },
                    "deprecated_exploratory_v0_2": {
                        "status": "not_run_for_unsupported_mode",
                        "used_for_final_verdict": False,
                    },
                },
            )

        if not registry_exact_positive and (
            features.counts.get("objects", 0) < self.config.min_objects
            or features.counts.get("positioned_objects", 0) < self.config.min_positioned_objects
        ):
            caveats.append("Too few positioned objects for stable distributional checks.")
            return DetectionReport(
                path=str(beatmap.path),
                verdict=Verdict.INSUFFICIENT_DATA,
                evidence_score=0.0,
                mapperatorinator_scope=self.scope,
                evidence=(),
                features=features.values,
                counts=features.counts,
                caveats=tuple(caveats),
                decision_trace={
                    "reason": "insufficient_data",
                    "required_objects": self.config.min_objects,
                    "required_positioned_objects": self.config.min_positioned_objects,
                    "content_model": content,
                    "public_revision_registry": registry,
                    "cheap_fusion_v1": fusion,
                },
                map_identity=identity,
                analysis_channels={
                    "content_v2": content,
                    "public_revision_registry": registry,
                    "cheap_fusion_v1": fusion,
                    "revision_forensics_v3": forensic,
                    "whitebox": {
                        "enabled": False,
                        "status": "not_requested",
                        "discriminator": {
                            "enabled": False,
                            "status": "not_requested",
                            "available": False,
                            "used_for_final_verdict": False,
                        },
                    },
                    "deprecated_exploratory_v0_2": {
                        "status": "not_run_for_insufficient_data",
                        "used_for_final_verdict": False,
                    },
                },
            )

        evidence = self._evidence(features)
        rule_score = self._fuse(evidence)
        statistical = analyze_statistical(
            beatmap, os.environ.get("OSU_AI_STATISTICAL_MODEL")
        ) if self.config.statistical_enabled else None
        deprecated_score = max(
            rule_score,
            statistical.combined_score if statistical and statistical.available else 0.0,
        )
        family_strength: dict[str, float] = {}
        for item in evidence:
            family_strength[item.family] = max(family_strength.get(item.family, 0.0), item.contribution)
        strong_families = sum(value >= self.config.strong_family_contribution for value in family_strength.values())

        direct = any(item.key == "explicit_disclosure" for item in evidence)
        exact_signature = any(
            item.key in {"epsilon_sv_signature", "exact_template_bundle", "v29_cell_center_signature"}
            and item.contribution >= self.config.exact_signature_contribution
            for item in evidence
        )
        compound_high = (
            strong_families >= self.config.min_strong_families
            and rule_score >= self.config.high_confidence_score
        )
        statistical_high = bool(statistical and statistical.available and statistical.high)
        statistical_suspicious = bool(statistical and statistical.available and statistical.suspicious)
        deprecated_would_be_high = direct or exact_signature or compound_high or statistical_high
        deprecated_would_be_suspicious = (
            strong_families >= 1
            or deprecated_score >= self.config.suspicious_score
            or statistical_suspicious
        )
        if direct:
            verdict = Verdict.HIGH_CONFIDENCE_AI
            score = 1.0
            primary_reason = "explicit_self_disclosure"
        elif registry_exact_positive:
            verdict = Verdict.HIGH_CONFIDENCE_AI
            score = 1.0
            primary_reason = "known_exact_public_positive_revision"
            registry["used_for_final_verdict"] = True
        elif fusion_usable:
            fusion["used_for_final_verdict"] = True
            score = float(fusion.get("score") or 0.0)
            if bool(fusion.get("threshold_flags", {}).get("high")):
                verdict = Verdict.HIGH_CONFIDENCE_AI
                primary_reason = "cheap_fusion_v1_high_np_threshold"
            elif bool(fusion.get("threshold_flags", {}).get("elevated")):
                verdict = Verdict.SUSPICIOUS
                primary_reason = "cheap_fusion_v1_elevated_np_threshold"
            else:
                verdict = Verdict.INCONCLUSIVE
                primary_reason = "cheap_fusion_v1_below_elevated_threshold"
        else:
            if not content.get("available"):
                verdict = Verdict.INCONCLUSIVE
                score = 0.0
                primary_reason = "content_v2_unavailable"
            elif bool(content.get("ood", {}).get("abstain")):
                verdict = Verdict.INCONCLUSIVE
                score = float(content.get("score") or 0.0)
                primary_reason = "content_v2_ood_abstention"
            elif not content.get("decision_usable"):
                verdict = Verdict.INCONCLUSIVE
                score = float(content.get("score") or 0.0)
                primary_reason = "content_v2_not_calibrated_or_no_windows"
            elif bool(content.get("threshold_flags", {}).get("high")):
                verdict = Verdict.HIGH_CONFIDENCE_AI
                score = float(content.get("score") or 0.0)
                primary_reason = "content_v2_high_np_threshold"
            elif bool(content.get("threshold_flags", {}).get("elevated")):
                verdict = Verdict.SUSPICIOUS
                score = float(content.get("score") or 0.0)
                primary_reason = "content_v2_elevated_np_threshold"
            else:
                verdict = Verdict.INCONCLUSIVE
                score = float(content.get("score") or 0.0)
                primary_reason = "content_v2_below_elevated_threshold"

        forensic_high = bool(
            forensic.get("decision_usable") and forensic.get("threshold_flags", {}).get("high")
        )
        forensic_elevated = bool(
            forensic.get("decision_usable") and forensic.get("threshold_flags", {}).get("elevated")
        )
        forensic_promoted_suspicious = False
        fusion_used = bool(fusion.get("used_for_final_verdict"))
        content["used_via_cheap_fusion"] = fusion_used
        forensic["used_via_cheap_fusion"] = fusion_used
        if fusion_used:
            forensic["channel_role"] = "input_to_calibrated_cheap_fusion"
        elif forensic_high:
            if direct:
                forensic["channel_role"] = "corroborates_explicit_disclosure"
            elif registry_exact_positive:
                forensic["channel_role"] = "corroborates_known_exact_public_revision"
            elif verdict is Verdict.HIGH_CONFIDENCE_AI:
                forensic["channel_role"] = "corroborates_content_v2_high"
            elif verdict is Verdict.SUSPICIOUS:
                forensic["channel_role"] = "corroborates_existing_suspicious"
            else:
                # Until a pre-registered multi-channel fusion is calibrated,
                # even a calibrated forensic high may only request review.
                verdict = Verdict.SUSPICIOUS
                score = max(score, float(forensic.get("score") or 0.0))
                primary_reason = "revision_forensics_v3_high_suspicious_cap"
                forensic_promoted_suspicious = True
                forensic["channel_role"] = "suspicious_cap"
        elif forensic_elevated:
            forensic["channel_role"] = "elevated_corroboration_only"
        forensic["used_for_final_verdict"] = fusion_used or forensic_promoted_suspicious

        deprecated_channel = {
            "status": "deprecated_exploratory",
            "used_for_final_verdict": False,
            "warning": (
                "v0.2 is dominated by serialization/template shortcuts in its historical evaluation; "
                "these values are retained only for forensic inspection"
            ),
            "rule_evidence_score": round(rule_score, 8),
            "combined_legacy_score": round(deprecated_score, 8),
            "would_have_been_high": deprecated_would_be_high,
            "would_have_been_suspicious": deprecated_would_be_suspicious,
            "statistical_model": statistical.to_dict() if statistical else {
                "available": False,
                "reason": "disabled by statistical_enabled=0",
            },
            "family_contributions": {
                key: round(value, 8) for key, value in sorted(family_strength.items())
            },
        }

        return DetectionReport(
            path=str(beatmap.path),
            verdict=verdict,
            evidence_score=score,
            mapperatorinator_scope=self.scope,
            evidence=tuple(sorted(evidence, key=lambda item: item.contribution, reverse=True)),
            features=features.values,
            counts=features.counts,
            caveats=tuple(caveats),
            decision_trace={
                "primary_reason": primary_reason,
                "primary_channel": (
                    "declaration"
                    if direct
                    else "public_revision_registry"
                    if registry_exact_positive
                    else "cheap_fusion_v1"
                    if fusion_used
                    else "revision_forensics_v3"
                    if forensic_promoted_suspicious
                    else "content_v2"
                ),
                "content_model": content,
                "public_revision_registry": {
                    "available": registry.get("available"),
                    "status": registry.get("status"),
                    "known_exact_positive": registry_exact_positive,
                    "used_for_final_verdict": registry.get("used_for_final_verdict"),
                    "verdict_basis": registry.get("verdict_basis"),
                    "interpretation": (
                        "Exact historical revision identity lookup; not a statistical AI-detection result."
                    ),
                },
                "cheap_fusion_v1": {
                    "available": fusion.get("available"),
                    "status": fusion.get("status"),
                    "calibrated": fusion.get("calibrated"),
                    "decision_usable": fusion.get("decision_usable"),
                    "used_for_final_verdict": fusion_used,
                    "score": fusion.get("score"),
                    "human_null_p_value": fusion.get("human_null_p_value"),
                    "thresholds": fusion.get("thresholds"),
                    "threshold_guarantees": fusion.get("threshold_guarantees"),
                    "threshold_flags": fusion.get("threshold_flags"),
                    "base_model_binding": fusion.get("base_model_binding"),
                    "observed_base_model_sha256": fusion.get("observed_base_model_sha256"),
                    "channels": fusion.get("channels"),
                    "abstention_reasons": fusion.get("abstention_reasons"),
                    "whitebox_included": False,
                },
                "revision_forensics_v3": {
                    "available": forensic.get("available"),
                    "calibrated": forensic.get("calibrated"),
                    "decision_usable": forensic.get("decision_usable"),
                    "threshold_flags": forensic.get("threshold_flags"),
                    "channel_role": forensic.get("channel_role"),
                    "used_for_final_verdict": fusion_used or forensic_promoted_suspicious,
                    "used_via_cheap_fusion": fusion_used,
                    "high_confidence_verdict_allowed": False,
                },
                "direct_disclosure": direct,
                "exact_signature": exact_signature,
                "compound_high": compound_high,
                "rule_evidence_score": round(rule_score, 8),
                "statistical_high": statistical_high,
                "statistical_suspicious": statistical_suspicious,
                "statistical_model": statistical.to_dict() if statistical else {
                    "available": False,
                    "reason": "disabled by statistical_enabled=0",
                },
                "family_contributions": {key: round(value, 8) for key, value in sorted(family_strength.items())},
                "strong_family_count": strong_families,
                "thresholds": {
                    "strong_family_contribution": self.config.strong_family_contribution,
                    "exact_signature_contribution": self.config.exact_signature_contribution,
                    "high_confidence_score": self.config.high_confidence_score,
                    "suspicious_score": self.config.suspicious_score,
                    "min_strong_families": self.config.min_strong_families,
                },
                "deprecated_v0_2_used_for_final_verdict": False,
            },
            map_identity=identity,
            analysis_channels={
                "content_v2": content,
                "public_revision_registry": registry,
                "cheap_fusion_v1": fusion,
                "revision_forensics_v3": forensic,
                "whitebox": {
                    "enabled": False,
                    "status": "not_requested",
                    "discriminator": {
                        "enabled": False,
                        "status": "not_requested",
                        "available": False,
                        "used_for_final_verdict": False,
                    },
                },
                "deprecated_exploratory_v0_2": deprecated_channel,
            },
        )

    @staticmethod
    def _fuse(evidence: list[Evidence]) -> float:
        # Only the strongest item from a family enters the compound score. This
        # avoids double-counting correlated residue statistics.
        family_values: dict[str, float] = {}
        for item in evidence:
            family_values[item.family] = max(family_values.get(item.family, 0.0), item.contribution)
        remaining = 1.0
        for contribution in family_values.values():
            remaining *= 1.0 - _clamp(contribution)
        return 1.0 - remaining

    def _evidence(self, features: FeatureSet) -> list[Evidence]:
        f, c = features.values, features.counts
        cfg = self.config
        evidence: list[Evidence] = []

        if f["explicit_ai_disclosure"] > 0:
            evidence.append(Evidence(
                family="declaration",
                key="explicit_disclosure",
                strength=1.0,
                reliability=1.0,
                description="Metadata explicitly names Mapperatorinator/osuT5 or discloses AI generation.",
                observed={"explicit_ai_disclosure": 1},
            ))

        if f["template_distinctive_ratio"] == 1.0:
            evidence.append(Evidence(
                family="file_template",
                key="exact_template_bundle",
                strength=1.0,
                reliability=cfg.template_reliability,
                description="All five non-default Editor fields exactly match the public Mapperatorinator output template.",
                observed={"template_distinctive_ratio": f["template_distinctive_ratio"]},
            ))

        if c["coordinates"] >= cfg.min_coordinates:
            v29_strength = min(
                _ramp(f["coord_mod32_16_ratio"], cfg.v29_center_start, cfg.v29_center_full),
                _ramp(f["head_coord_mod32_16_ratio"], cfg.v29_center_start, cfg.v29_center_full),
            )
            if v29_strength >= cfg.evidence_activation:
                evidence.append(Evidence(
                    family="spatial_quantization",
                    key="v29_cell_center_signature",
                    strength=v29_strength,
                    reliability=cfg.v29_center_reliability,
                    description="Nearly all object and anchor coordinates occupy the fixed 16 mod 32 cell centers emitted by V28/V29 combined position tokens.",
                    observed={
                        "coordinates": c["coordinates"],
                        "coord_mod32_16_ratio": f["coord_mod32_16_ratio"],
                        "head_coord_mod32_16_ratio": f["head_coord_mod32_16_ratio"],
                    },
                ))
            entropy_distance = abs(f["head_coord_mod32_entropy"] - cfg.v32_entropy_center)
            even_strength = min(
                _ramp(f["head_both_even_ratio"], cfg.v32_even_start, cfg.v32_even_full),
                _ramp(
                    cfg.v32_entropy_max_distance - entropy_distance,
                    0.0,
                    cfg.v32_entropy_max_distance - cfg.v32_entropy_full_distance,
                ),
                _ramp(c["positioned_objects"], cfg.v32_sample_start, cfg.v32_sample_full),
            )
            legacy_strength = min(
                _ramp(f["coord_mod4_ratio"], cfg.legacy_mod4_start, cfg.legacy_mod4_full),
                _ramp(f["coord_grid32_offset_4_8_12_ratio"], cfg.legacy_offset_start, cfg.legacy_offset_full),
                _ramp(
                    cfg.legacy_entropy_ceiling - f["coord_mod32_entropy"],
                    0.0,
                    cfg.legacy_entropy_span,
                ),
            )
            spatial_strength = max(even_strength, legacy_strength)
            if spatial_strength >= cfg.evidence_activation:
                evidence.append(Evidence(
                    family="spatial_quantization",
                    key="coordinate_lattice",
                    strength=spatial_strength,
                    reliability=(
                        cfg.v32_spatial_reliability
                        if even_strength >= legacy_strength
                        else cfg.legacy_spatial_reliability
                    ),
                    description="Coordinates match a calibrated V32 2 px residue distribution or the weaker legacy 4/32 px lattice lead.",
                    observed={
                        "coordinates": c["coordinates"],
                        "head_both_even_ratio": f["head_both_even_ratio"],
                        "coord_even_ratio": f["coord_even_ratio"],
                        "coord_mod4_ratio": f["coord_mod4_ratio"],
                        "coord_grid32_offset_4_8_12_ratio": f["coord_grid32_offset_4_8_12_ratio"],
                        "coord_mod32_entropy": f["coord_mod32_entropy"],
                        "head_coord_mod32_entropy": f["head_coord_mod32_entropy"],
                    },
                ))

        time_strength = 0.0
        observed_time: dict[str, float | int] = {}
        if c["offgrid_object_times"] >= cfg.offgrid_min_objects:
            time_strength = max(
                time_strength,
                _ramp(
                    f["offgrid_time_mod10_5_ratio"],
                    cfg.offgrid_halfstep_start,
                    cfg.offgrid_halfstep_full,
                ),
            )
            observed_time.update({
                "offgrid_object_times": c["offgrid_object_times"],
                "offgrid_time_mod10_5_ratio": f["offgrid_time_mod10_5_ratio"],
            })
        if time_strength >= cfg.evidence_activation:
            evidence.append(Evidence(
                family="temporal_quantization",
                key="time_lattice",
                strength=time_strength,
                reliability=cfg.temporal_reliability,
                description="Unsnapped time stamps concentrate on the decoder's 10 ms half-step.",
                observed=observed_time,
            ))

        if (
            c["sv_epsilon_matches"] >= cfg.epsilon_min_matches
            and f["sv_epsilon_1e10_ratio"] >= cfg.epsilon_min_ratio
        ):
            evidence.append(Evidence(
                family="postprocessing",
                key="epsilon_sv_signature",
                strength=min(
                    _ramp(f["sv_epsilon_1e10_ratio"], cfg.epsilon_ratio_start, cfg.epsilon_ratio_full),
                    _ramp(c["sv_epsilon_matches"], cfg.epsilon_count_start, cfg.epsilon_count_full),
                ),
                reliability=cfg.epsilon_reliability,
                description="Inherited timing values contain the generator's +1e-10 slider-velocity serialization epsilon.",
                observed={
                    "greenlines": c["greenlines"],
                    "sv_epsilon_matches": c["sv_epsilon_matches"],
                    "sv_epsilon_1e10_ratio": f["sv_epsilon_1e10_ratio"],
                },
            ))

        super_timing_strength = min(
            _ramp(
                f["redlines_per_minute"],
                cfg.super_timing_redline_start,
                cfg.super_timing_redline_full,
            ),
            _ramp(
                f["same_offset_red_green_ratio"],
                cfg.super_timing_overlap_start,
                cfg.super_timing_overlap_full,
            ),
        )
        volume_reset_strength = 0.0
        if c["greenlines"] >= cfg.volume_reset_min_greenlines:
            volume_reset_strength = _ramp(
                f["green_plus6ms_after_object_ratio"],
                cfg.volume_reset_start,
                cfg.volume_reset_full,
            )
        post_strength = max(super_timing_strength, volume_reset_strength)
        if post_strength >= cfg.evidence_activation:
            evidence.append(Evidence(
                family="postprocessing",
                key="timing_postprocess_pattern",
                strength=post_strength,
                reliability=cfg.timing_pattern_reliability,
                description="Timing/SV points follow dense super-timing or +6 ms hitsound-reset conventions in the public postprocessor.",
                observed={
                    "redlines_per_minute": f["redlines_per_minute"],
                    "same_offset_red_green_ratio": f["same_offset_red_green_ratio"],
                    "green_plus6ms_after_object_ratio": f["green_plus6ms_after_object_ratio"],
                    "redlines": c["redlines"],
                    "greenlines": c["greenlines"],
                },
            ))

        if c["sliders"] >= cfg.slider_min_count:
            serialization_strength = _ramp(
                f["slider_length_13plus_decimal_ratio"],
                cfg.slider_decimal_start,
                cfg.slider_decimal_full,
            )
            geometry_strength = 0.0
            if c["comparable_slider_pairs"] >= cfg.slider_min_comparable_pairs:
                geometry_strength = _ramp(
                    f["slider_near_not_exact_shape_ratio"],
                    cfg.slider_geometry_start,
                    cfg.slider_geometry_full,
                )
            slider_strength = max(serialization_strength, geometry_strength * cfg.slider_geometry_weight)
            if slider_strength >= cfg.evidence_activation:
                evidence.append(Evidence(
                    family="slider_construction",
                    key="slider_construction_pattern",
                    strength=slider_strength,
                    reliability=cfg.slider_reliability,
                    description="Slider lengths or near-duplicate shapes resemble independent model output rather than exact editor transforms.",
                    observed={
                        "sliders": c["sliders"],
                        "slider_length_13plus_decimal_ratio": f["slider_length_13plus_decimal_ratio"],
                        "comparable_slider_pairs": c["comparable_slider_pairs"],
                        "slider_near_not_exact_shape_ratio": f["slider_near_not_exact_shape_ratio"],
                    },
                ))

        return evidence
