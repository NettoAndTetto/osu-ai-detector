"""CPU-only, auditable discriminator for Mapperatorinator white-box output.

The heavyweight :mod:`osu_ai_detector.whitebox` engine produces descriptive
teacher-forcing statistics.  This module turns those statistics into a fixed
feature vector and applies a regularized logistic *ranking head*.  It has no
NumPy, scikit-learn, PyTorch, or checkpoint dependency at runtime.

The logistic score is deliberately not called an authorship probability.  A
calibrated decision requires an independent, revision-pinned human reference
set.  Artifacts trained by ``scripts/train_whitebox_detector.py`` therefore
ship with empty calibration data; the runtime can consume future human null
scores and reports an upper-tail split-conformal p-value when they are present.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .whitebox_protocol import (
    canonical_search_protocol,
    derive_search_protocol,
    search_protocol_sha256,
)

MODEL_SCHEMA_VERSION = "osu-ai-detector.whitebox-logistic/v5"
RESULT_SCHEMA_VERSION = "osu-ai-detector.whitebox-discriminator-result/v5"
REPRESENTATION_VERSION = "mapperatorinator-family-window/v4"
CHECKPOINT_IDENTITY_SCHEMA_VERSION = "osu-ai-detector.whitebox-checkpoint-identity/v2"
DEFAULT_MODEL = Path(__file__).with_name("models") / "whitebox_logistic_v1.json"

# The published V32 repository contains checkpoints for several osu! modes,
# while the local desktop bundle intentionally carries only the exact
# ``gamemode=0`` files that the standard-mode detector can load.  The original
# calibration artifact bound the entire repository snapshot even though the
# runtime loaded only that subfolder.  Accept this one byte-exact, revision-
# pinned projection as deployment-equivalent; every other identity still fails
# closed.  Keeping this mapping outside the learned artifact preserves the
# artifact used by the final evaluation while making the standard-only local
# distribution usable.
_STANDARD_MODE_IDENTITY_EQUIVALENCES: dict[
    tuple[str, str, str, str], dict[str, Any]
] = {
    (
        "mapperatorinator-whitebox-logistic-v1-501c17853200",
        "v32",
        "a00ccc0a14e8747b762f35712101dc203159ad7437a7fe0800c2e4360e2936f4",
        "9ffaa04b724fda39c2cde41e4e51ab8cfe184b1e2004e0d423d7ebb8c3ab7199",
    ): {
        "repo_id": "OliBomby/Mapperatorinator-v32",
        "resolved_revision": "74f22583400d259bf424819e11027c17933efe54",
        "gamemode": 0,
        "projection": "gamemode=0",
        "observed_snapshot_file_count": 4,
        "observed_model_weight_file_count": 1,
        "observed_tokenizer_file_count": 1,
    }
}

# Minimal calibration identity for the five published standard-mode models.
#
# The older calibration artifact bound whole repository snapshots.  That made
# harmless packaging details (README files, .gitattributes, Hugging Face
# download metadata, and checkpoints for unused game modes) invalidate a
# perfectly valid run.  The discriminator now accepts an identity mismatch only
# when the exact four files actually loaded by the standard-mode runtime match
# this revision-pinned manifest byte-for-byte.  Model weights, tokenizer,
# architecture config, generation config, repository, revision, game mode, and
# the extraction/search protocols remain fail-closed.
_MINIMUM_RUNTIME_PAYLOADS: dict[tuple[str, str, str], dict[str, Any]] = {
    (
        "mapperatorinator-whitebox-logistic-v1-501c17853200",
        "v29",
        "06bdb62e19ecdb8d3ead8a3dd089ad03eaa56dd740d3cd27a958eed7a94e56d0",
    ): {
        "repo_id": "OliBomby/Mapperatorinator-v29.1",
        "resolved_revision": "656db0cd04a8a6a77d94a96e7af89810fb6de5ef",
        "gamemode": 0,
        "files": {
            "config.json": {"bytes": 1756, "sha256": "9c2430265b817aac884ca650cd14c6121e0e4e4b9a7e9a28ba7714a0d88d0963"},
            "generation_config.json": {"bytes": 231, "sha256": "f87c708d474172a24345b50335958e6046fd1322cfaf864f8de3e289749b35bf"},
            "model.safetensors": {"bytes": 877083296, "sha256": "945b49765a3eeef88751474c9aa31d0e4e7670875005d06aebe323bb2a3bc2fe"},
            "tokenizer.json": {"bytes": 3082602, "sha256": "44b7e494652440bbbf964b7fbd25b2cdac0ffed84118121be9a07e01494e06ca"},
        },
    },
    (
        "mapperatorinator-whitebox-logistic-v1-501c17853200",
        "v30",
        "48bc8d1931d87a5a4db9bb086121bf5625da1fae996041928002ff37ba1fd79c",
    ): {
        "repo_id": "OliBomby/Mapperatorinator-v30",
        "resolved_revision": "a4c6e6e69c055711c2293d63161c0e52980e56a1",
        "gamemode": 0,
        "files": {
            "config.json": {"bytes": 2237, "sha256": "3f0bc161f22d838a7a1582ea0231816dbf4e26ee04e030394f1ceb05e3bdb2cb"},
            "generation_config.json": {"bytes": 231, "sha256": "cdd37c2acd98f11d2160f4a6be5766a39495bb148c8526a51611576016e2db97"},
            "model.safetensors": {"bytes": 861280756, "sha256": "28c56449d9bb23afdd2b0a16743ce27dd37aabc5b1a846c78efa63485454fc97"},
            "tokenizer.json": {"bytes": 1876018, "sha256": "d801be48048583822b55cd4c8ef1e382c38934c8e1b9496e0be83cab3048f8e2"},
        },
    },
    (
        "mapperatorinator-whitebox-logistic-v1-501c17853200",
        "v31",
        "c51d89a974e1d125a78823baca1997714a808efb7787a3512c1890c9c436ddcb",
    ): {
        "repo_id": "OliBomby/Mapperatorinator-v31",
        "resolved_revision": "12772791b862b97a11153aa766b2481afa5dda11",
        "gamemode": 0,
        "files": {
            "config.json": {"bytes": 2756, "sha256": "250551fbc9fa300cebb9468c7340b932145c496d4fcaae2f457c9fd82714320b"},
            "generation_config.json": {"bytes": 259, "sha256": "668105b2e2df1f1adbcfa453cd6ec2087b9cce0a8e7cf1edabd5b377ece3f0ac"},
            "model.safetensors": {"bytes": 880954204, "sha256": "8adeaea3dd1be66d1df42fb3fe7e471797be1a89c765f656dd37c489cdda11d6"},
            "tokenizer.json": {"bytes": 3082188, "sha256": "db48ee9de0608264c48cb3746048a5bf401a4e4ac28bf296f57b58c622b33210"},
        },
    },
    (
        "mapperatorinator-whitebox-logistic-v1-501c17853200",
        "v32",
        "a00ccc0a14e8747b762f35712101dc203159ad7437a7fe0800c2e4360e2936f4",
    ): {
        "repo_id": "OliBomby/Mapperatorinator-v32",
        "resolved_revision": "74f22583400d259bf424819e11027c17933efe54",
        "gamemode": 0,
        "files": {
            "config.json": {"bytes": 2972, "sha256": "ff96b8c059179978c93b6f938e39fac945d5682a3495789c00cc159d571a1a22"},
            "generation_config.json": {"bytes": 259, "sha256": "10b901beeb3c982c16c53cd5b37a4f9b3d4674fc70cbc3e64c7b20fdb0433f51"},
            "model.safetensors": {"bytes": 865544236, "sha256": "dc60cf15609a1e93bf2280b360f00ba30b833c8e95d088db21babe44e1c4d6e9"},
            "tokenizer.json": {"bytes": 4974127, "sha256": "6b98be0fc04a95a9e9d4feb8e8b67cc48728a6667e3091dcd5cc528baeca18bd"},
        },
    },
    (
        "mapperatorinator-whitebox-logistic-v1-501c17853200",
        "v32-mini",
        "e1ea1a1543dee28c8e37eb0e96d90765f38ab8cb7b24ccfbf59cd48d9fd00add",
    ): {
        "repo_id": "OliBomby/Mapperatorinator-v32-mini",
        "resolved_revision": "7807f0dc70cab671be012e1f5ddf945b0b8b7278",
        "gamemode": 0,
        "files": {
            "config.json": {"bytes": 2963, "sha256": "df23449d5ddc05e07fb30a576b39aed0703de94c8418086a9ebd6e537a65acc0"},
            "generation_config.json": {"bytes": 259, "sha256": "10b901beeb3c982c16c53cd5b37a4f9b3d4674fc70cbc3e64c7b20fdb0433f51"},
            "model.safetensors": {"bytes": 222891244, "sha256": "a7a795018ea39747be369e571a19319954f999e93925e0c47aa9a807d22c8d7b"},
            "tokenizer.json": {"bytes": 4974127, "sha256": "6b98be0fc04a95a9e9d4feb8e8b67cc48728a6667e3091dcd5cc528baeca18bd"},
        },
    },
}

FAMILIES: tuple[str, ...] = (
    "dist",
    "pos",
    "pos_x",
    "pos_y",
    "hitsound",
    "snap",
    "t",
    "volume",
)
CHECKPOINTS: tuple[str, ...] = ("v29", "v30", "v31", "v32", "v32-mini")
SUMMARY_METRICS: tuple[str, ...] = ("nll", "rank", "log_rank", "entropy", "curvature_z")
RAW_SUMMARY_METRICS: tuple[str, ...] = ("nll", "log_rank", "entropy", "curvature_z")
POLICY_SUMMARY_METRICS: tuple[str, ...] = (
    "nll",
    "log_rank",
    "entropy",
    "curvature_z",
    "sample_nll",
)
SUMMARY_STATISTICS: tuple[str, ...] = ("mean", "p50", "p90")
NUCLEUS_STATISTICS: tuple[str, ...] = ("mean", "p50", "p90")
TEMPTEST_STATISTICS: tuple[str, ...] = ("mean", "p10", "p50", "p90")


def _feature_names() -> tuple[str, ...]:
    names = ["window.log1p_token_count", "window.active_family_fraction"]
    for metric in RAW_SUMMARY_METRICS:
        names.extend(
            f"window.raw_full_vocabulary.{metric}.{statistic}"
            for statistic in SUMMARY_STATISTICS
        )
    names.extend(
        (
            "window.detectllm_lrr.raw_full_vocabulary",
            "window.detectllm_lrr.family_conditioned_heuristic",
            "window.fast_detectgpt_sequence.raw_full_vocabulary",
            "window.fast_detectgpt_sequence.family_conditioned_heuristic",
        )
    )
    names.append("window.generation_policy.coverage_fraction")
    for metric in POLICY_SUMMARY_METRICS:
        names.extend(
            f"window.generation_policy.{metric}.{statistic}"
            for statistic in SUMMARY_STATISTICS
        )
    names.extend(
        (
            "window.generation_policy.detectllm_lrr",
            "window.generation_policy.fast_detectgpt_policy_aware",
            "window.generation_policy.in_support_fraction",
            "window.generation_policy.log1p_longest_support_violation_run",
            "window.generation_policy.removed_mass.mean",
        )
    )
    names.extend(
        f"window.generation_policy.support_size.{statistic}"
        for statistic in NUCLEUS_STATISTICS
    )
    for family in FAMILIES:
        prefix = f"family.{family}"
        names.extend((f"{prefix}.present", f"{prefix}.log1p_token_count", f"{prefix}.token_fraction"))
        for metric in SUMMARY_METRICS:
            names.extend(f"{prefix}.conditioned.{metric}.{statistic}" for statistic in SUMMARY_STATISTICS)
        for metric in RAW_SUMMARY_METRICS:
            names.extend(
                f"{prefix}.raw_full_vocabulary.{metric}.{statistic}"
                for statistic in SUMMARY_STATISTICS
            )
        names.extend(
            (
                f"{prefix}.detectllm_lrr.raw_full_vocabulary",
                f"{prefix}.detectllm_lrr.family_conditioned_heuristic",
                f"{prefix}.fast_detectgpt_sequence.raw_full_vocabulary",
                f"{prefix}.fast_detectgpt_sequence.family_conditioned_heuristic",
            )
        )
        names.append(f"{prefix}.generation_policy.coverage_fraction")
        for metric in POLICY_SUMMARY_METRICS:
            names.extend(
                f"{prefix}.generation_policy.{metric}.{statistic}"
                for statistic in SUMMARY_STATISTICS
            )
        names.extend(
            (
                f"{prefix}.generation_policy.detectllm_lrr",
                f"{prefix}.generation_policy.fast_detectgpt_policy_aware",
                f"{prefix}.generation_policy.in_support_fraction",
                f"{prefix}.generation_policy.log1p_longest_support_violation_run",
                f"{prefix}.generation_policy.removed_mass.mean",
            )
        )
        names.extend(
            f"{prefix}.generation_policy.support_size.{statistic}"
            for statistic in NUCLEUS_STATISTICS
        )
        names.extend(
            f"{prefix}.temp_test.tau_0_9.{statistic}"
            for statistic in TEMPTEST_STATISTICS
        )
        names.append(f"{prefix}.nucleus.in_fraction")
        names.extend(f"{prefix}.nucleus.size.{statistic}" for statistic in NUCLEUS_STATISTICS)
        names.append(f"{prefix}.nucleus.mass.mean")
        names.extend((f"{prefix}.top_p_violation_fraction", f"{prefix}.log1p_longest_top_p_violation_run"))
    names.extend(
        f"context.timing.family.t.temp_test.tau_0_1.{statistic}"
        for statistic in TEMPTEST_STATISTICS
    )
    names.extend(f"checkpoint.{checkpoint}" for checkpoint in CHECKPOINTS)
    names.append("checkpoint.other")
    return tuple(names)


BASE_FEATURE_NAMES = _feature_names()
EXPANDED_FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(f"missing::{name}" for name in BASE_FEATURE_NAMES)


@dataclass(frozen=True)
class WhiteboxFeatureWindow:
    """One fixed-schema feature vector from one checkpoint/audio window."""

    checkpoint: str
    window_index: int
    start_ms: int | None
    end_ms: int | None
    token_count: int
    values: tuple[float, ...]

    def feature_dict(self, *, json_safe: bool = False) -> dict[str, float | None]:
        if json_safe:
            return {
                name: value if math.isfinite(value) else None
                for name, value in zip(BASE_FEATURE_NAMES, self.values, strict=True)
            }
        return dict(zip(BASE_FEATURE_NAMES, self.values, strict=True))


def representation_audit() -> dict[str, Any]:
    """Return the immutable feature recipe embedded in training artifacts."""

    digest = hashlib.sha256("\n".join(BASE_FEATURE_NAMES).encode("utf-8")).hexdigest()
    return {
        "version": REPRESENTATION_VERSION,
        "families": list(FAMILIES),
        "checkpoints": [*CHECKPOINTS, "other"],
        "family_conditioned_metrics": list(SUMMARY_METRICS),
        "raw_full_vocabulary_metrics": list(RAW_SUMMARY_METRICS),
        "summary_statistics": list(SUMMARY_STATISTICS),
        "nucleus_size_statistics": list(NUCLEUS_STATISTICS),
        "temperature_normalization": {
            "method": "TempTest local normalization adapted to the observed tokenizer EventType range",
            "all_families": {"setting": "tau_0_9", "temperature": 0.9},
            "timing_context_family_t": {"setting": "tau_0_1", "temperature": 0.1},
            "statistics": list(TEMPTEST_STATISTICS),
        },
        "detectllm_lrr": {
            "definition": "mean(NLL) / mean(log(one-indexed rank))",
            "primary_scope": "source-model full output vocabulary within one audio window",
            "secondary_scope": "observed EventType-family conditioning, explicitly heuristic",
            "rank_one_policy": "undefined when every token has rank one; represented by NaN plus missing indicator",
        },
        "fast_detectgpt_sequence_discrepancy": {
            "definition": "sum(log p(target)-E_p log p(X)) / sqrt(sum(Var_p log p(X)))",
            "primary_scope": "source-model full output vocabulary within one audio window",
            "secondary_scope": "observed EventType-family conditioning, explicitly heuristic",
            "aggregation": "standardize after summing token numerators and conditional variances",
        },
        "generation_policy_replay": {
            "order": [
                "CFG (already in model_forward)",
                "monotonic time-shift mask",
                "timeshift bias",
                "conditional/global temperature",
                "stateful lookback bias when active",
                "top-k",
                "top-p",
            ],
            "support": "complete output vocabulary; prefix-only processors",
            "metrics": list(POLICY_SUMMARY_METRICS),
            "v29_super_timing": "timing context unavailable because its separate multi-offset beam pipeline is not replayed",
        },
        "coverage": ["present", "log1p_token_count", "token_fraction"],
        "missing_value_policy": (
            "Nonexistent/undefined family statistics are NaN during flattening, then median-imputed with "
            "one missing indicator per base feature. Structural coverage fields remain explicit zeroes."
        ),
        "feature_count": len(BASE_FEATURE_NAMES),
        "feature_names_sha256": digest,
    }


def candidate_map_key(chart_sha256: str, start_ms: int | None, end_ms: int | None) -> str:
    """Stable identity used to merge the same chart interval across checkpoints."""

    digest = str(chart_sha256).strip().casefold()
    if not digest:
        raise ValueError("chart_sha256 is required")
    if (start_ms is None) != (end_ms is None):
        raise ValueError("start_ms and end_ms must either both be set or both be None")
    if start_ms is None:
        interval = "full"
    else:
        start, end = int(start_ms), int(end_ms)
        if start < 0 or start >= end:
            raise ValueError("candidate interval must satisfy 0 <= start_ms < end_ms")
        interval = f"{start}:{end}"
    return f"{digest}@{interval}"


def _finite_number(value: Any, *, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def canonical_checkpoint_settings(settings: Any, schema_version: Any) -> dict[str, Any]:
    """Canonicalize the extraction settings that change white-box features.

    Audit temperature, top-p, the label-symmetric condition view and the white-box result schema are checkpoint
    settings, so calibrated artifacts bind exactly to those values.  Precision,
    attention implementation, and candidate-window multiplicity are bound by
    the separate canonical search protocol.
    """

    if not isinstance(settings, Mapping):
        raise ValueError("white-box checkpoint settings are missing")
    temperature = _finite_number(settings.get("temperature"))
    top_p = _finite_number(settings.get("top_p"))
    schema = str(schema_version or "").strip()
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("white-box checkpoint temperature must be finite and positive")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("white-box checkpoint top_p must be in (0, 1]")
    if not schema:
        raise ValueError("white-box result schema_version is missing")
    canonical: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "schema_version": schema,
    }
    condition_value = settings.get("condition_view")
    if condition_value is not None:
        condition_view = str(condition_value).strip()
        if condition_view != "symmetric_stripped_v1":
            raise ValueError("unsupported white-box condition_view")
        canonical["condition_view"] = condition_view
    runtime_digest_value = settings.get("runtime_source_identity_sha256")
    if runtime_digest_value is not None:
        runtime_digest = str(runtime_digest_value).strip().casefold()
        if len(runtime_digest) != 64 or any(
            character not in "0123456789abcdef" for character in runtime_digest
        ):
            raise ValueError("runtime_source_identity_sha256 must be a SHA-256 digest")
        canonical["runtime_source_identity_sha256"] = runtime_digest
    profile_value = settings.get("generation_policy_profile")
    if profile_value is not None:
        if not isinstance(profile_value, Mapping):
            raise ValueError("generation_policy_profile must be an object")
        if profile_value.get("schema_version") != "mapperatorinator-generation-policy/v1":
            raise ValueError("unsupported generation_policy_profile schema_version")
        if profile_value.get("replay_mode") != "sequential_batch_1_prefix":
            raise ValueError("generation_policy_profile replay_mode must be sequential_batch_1_prefix")
        profile: dict[str, Any] = {
            "schema_version": "mapperatorinator-generation-policy/v1",
            "replay_mode": "sequential_batch_1_prefix",
        }
        for field in (
            "temperature",
            "timing_temperature",
            "mania_column_temperature",
            "taiko_hit_temperature",
        ):
            value = _finite_number(profile_value.get(field))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"generation_policy_profile.{field} must be finite and positive")
            profile[field] = value
        policy_top_p = _finite_number(profile_value.get("top_p"))
        if not math.isfinite(policy_top_p) or not 0 < policy_top_p <= 1:
            raise ValueError("generation_policy_profile.top_p must be in (0, 1]")
        profile["top_p"] = policy_top_p
        raw_top_k = profile_value.get("top_k")
        if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int) or raw_top_k < 0:
            raise ValueError("generation_policy_profile.top_k must be a non-negative integer")
        profile["top_k"] = raw_top_k
        for field in ("cfg_scale", "timeshift_bias"):
            value = _finite_number(profile_value.get(field))
            if not math.isfinite(value):
                raise ValueError(f"generation_policy_profile.{field} must be finite")
            profile[field] = value
        for field in ("lookback_time_ms", "lookahead_time_ms"):
            value = _finite_number(profile_value.get(field))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"generation_policy_profile.{field} must be finite and non-negative")
            profile[field] = value
        for field in ("types_first", "super_timing"):
            value = profile_value.get(field)
            if not isinstance(value, bool):
                raise ValueError(f"generation_policy_profile.{field} must be boolean")
            profile[field] = value
        super_policy = str(
            profile_value.get("v29_super_timing_policy") or ""
        ).strip()
        if super_policy not in {"timing_context_unavailable", "not_applicable"}:
            raise ValueError("generation_policy_profile has invalid v29_super_timing_policy")
        profile["v29_super_timing_policy"] = super_policy
        canonical["generation_policy_profile"] = profile
    return canonical


def checkpoint_settings_sha256(settings_by_checkpoint: Mapping[str, Any]) -> str:
    """Hash a canonical per-checkpoint extraction protocol mapping."""

    canonical: dict[str, dict[str, Any]] = {}
    for raw_name, value in settings_by_checkpoint.items():
        name = str(raw_name).strip().casefold()
        if not name or not isinstance(value, Mapping):
            raise ValueError("checkpoint settings mapping contains an invalid checkpoint")
        if name in canonical:
            raise ValueError(f"duplicate checkpoint settings entry: {name}")
        canonical[name] = canonical_checkpoint_settings(value, value.get("schema_version"))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Hash logical provenance and exact bytes, independent of load transport."""

    payload = {
        key: identity.get(key)
        for key in (
            "schema_version",
            "status",
            "checkpoint",
            "config_name",
            "repo_id",
            "resolved_revision",
            "resolved_snapshot_sha256",
            "resolved_model_weights_sha256",
            "resolved_tokenizer_sha256",
        )
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_checkpoint_identity(identity: Any, checkpoint: str) -> dict[str, Any]:
    """Validate one immutable checkpoint identity emitted by WhiteboxEngine."""

    if not isinstance(identity, Mapping) or identity.get("status") != "ok":
        raise ValueError(f"checkpoint identity for {checkpoint} is missing or unavailable")
    if identity.get("schema_version") != CHECKPOINT_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint identity for {checkpoint} uses an unsupported schema_version"
        )
    digest = str(identity.get("identity_sha256") or "").strip().casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"checkpoint identity for {checkpoint} has an invalid identity_sha256")
    declared = str(identity.get("checkpoint") or checkpoint).strip().casefold()
    if declared != checkpoint:
        raise ValueError(f"checkpoint identity name mismatch: expected {checkpoint}, observed {declared}")
    source_kind = str(identity.get("source_kind") or "").strip()
    if source_kind not in {"local_snapshot", "huggingface_cache_snapshot"}:
        raise ValueError(f"checkpoint identity for {checkpoint} lacks a supported immutable source kind")
    repo_id = str(identity.get("repo_id") or "").strip()
    revision = str(identity.get("resolved_revision") or "").strip()
    if source_kind == "huggingface_cache_snapshot" and (not repo_id or not revision):
        raise ValueError(f"checkpoint identity for {checkpoint} lacks repo_id/resolved_revision")
    if repo_id and not revision:
        raise ValueError(f"checkpoint identity for {checkpoint} has repo_id without resolved_revision")
    for field in (
        "resolved_snapshot_sha256",
        "resolved_model_weights_sha256",
        "resolved_tokenizer_sha256",
    ):
        value = str(identity.get(field) or "").strip().casefold()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"checkpoint identity for {checkpoint} has invalid {field}")
    if checkpoint_identity_sha256(identity) != digest:
        raise ValueError(f"checkpoint identity for {checkpoint} does not reconcile to its fields")
    return copy.deepcopy(dict(identity)) | {"identity_sha256": digest}


def _minimum_runtime_identity_equivalence(
    *,
    model_id: str,
    checkpoint: str,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    checkpoint_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Accept exact loaded bytes while ignoring unrelated snapshot files."""

    expected_digest = str(expected.get("identity_sha256") or "").casefold()
    observed_digest = str(observed.get("identity_sha256") or "").casefold()
    rule = _MINIMUM_RUNTIME_PAYLOADS.get(
        (model_id, checkpoint, expected_digest)
    )
    if rule is None:
        return None

    for field in ("repo_id", "resolved_revision"):
        required = str(rule[field]).casefold()
        if str(expected.get(field) or "").casefold() != required:
            return None
        if str(observed.get(field) or "").casefold() != required:
            return None
    if str(observed.get("checkpoint") or "").strip().casefold() != checkpoint:
        return None
    if str(observed.get("config_name") or "").strip().casefold() != checkpoint:
        return None

    condition = checkpoint_result.get("condition_view")
    condition = condition if isinstance(condition, Mapping) else {}
    preserved = condition.get("preserved_fields")
    preserved = preserved if isinstance(preserved, Mapping) else {}
    if condition.get("label_symmetric") is not True:
        return None
    if condition.get("deployment_allowed") is not True:
        return None
    if preserved.get("gamemode") != rule["gamemode"]:
        return None
    if observed.get("active_runtime_mode") != rule["gamemode"]:
        return None

    raw_files = observed.get("active_runtime_files")
    if not isinstance(raw_files, Mapping):
        return None
    active_files: dict[str, dict[str, Any]] = {}
    for raw_name, raw_identity in raw_files.items():
        name = str(raw_name).replace("\\", "/")
        if not name or "/" in name or not isinstance(raw_identity, Mapping):
            return None
        digest = str(raw_identity.get("sha256") or "").strip().casefold()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            return None
        size = raw_identity.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            return None
        active_files[name] = {"sha256": digest, "bytes": size}
    if active_files != rule["files"]:
        return None

    payload_digest = hashlib.sha256(
        json.dumps(
            active_files,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if observed.get("active_runtime_file_count") != len(active_files):
        return None
    if str(observed.get("active_runtime_payload_sha256") or "").casefold() != payload_digest:
        return None

    runtime_identity_payload = {
        key: observed.get(key)
        for key in (
            "schema_version",
            "checkpoint",
            "config_name",
            "repo_id",
            "resolved_revision",
            "active_runtime_mode",
            "active_runtime_subfolder",
            "active_runtime_payload_sha256",
        )
    }
    runtime_identity_digest = hashlib.sha256(
        json.dumps(
            runtime_identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        str(observed.get("active_runtime_identity_sha256") or "").casefold()
        != runtime_identity_digest
    ):
        return None

    return {
        "checkpoint": checkpoint,
        "expected_identity_sha256": expected_digest,
        "observed_identity_sha256": observed_digest,
        "repo_id": rule["repo_id"],
        "resolved_revision": rule["resolved_revision"],
        "gamemode": rule["gamemode"],
        "projection": "active standard-mode runtime payload",
        "active_runtime_subfolder": observed.get("active_runtime_subfolder"),
        "active_runtime_payload_sha256": payload_digest,
        "reason": (
            "the exact revision-pinned model weights, tokenizer, architecture "
            "config, and generation config loaded for this run match calibration; "
            "unread snapshot files are ignored"
        ),
    }


def _standard_mode_identity_equivalence(
    *,
    model_id: str,
    checkpoint: str,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    checkpoint_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an audit record for one exact standard-only snapshot projection.

    Repository/revision equality alone is deliberately insufficient: the
    expected and observed identity digests, file counts, and preserved gamemode
    must all match the reviewed compatibility record above.
    """

    expected_digest = str(expected.get("identity_sha256") or "").casefold()
    observed_digest = str(observed.get("identity_sha256") or "").casefold()
    rule = _STANDARD_MODE_IDENTITY_EQUIVALENCES.get(
        (model_id, checkpoint, expected_digest, observed_digest)
    )
    if rule is None:
        return None
    for field in ("repo_id", "resolved_revision"):
        required = str(rule[field]).casefold()
        if str(expected.get(field) or "").casefold() != required:
            return None
        if str(observed.get(field) or "").casefold() != required:
            return None
    if str(expected.get("checkpoint") or checkpoint).casefold() != checkpoint:
        return None
    if str(observed.get("checkpoint") or checkpoint).casefold() != checkpoint:
        return None
    if str(expected.get("config_name") or checkpoint).casefold() != checkpoint:
        return None
    if str(observed.get("config_name") or checkpoint).casefold() != checkpoint:
        return None
    for identity_field, rule_field in (
        ("snapshot_file_count", "observed_snapshot_file_count"),
        ("model_weight_file_count", "observed_model_weight_file_count"),
        ("tokenizer_file_count", "observed_tokenizer_file_count"),
    ):
        if observed.get(identity_field) != rule[rule_field]:
            return None
    condition = checkpoint_result.get("condition_view")
    condition = condition if isinstance(condition, Mapping) else {}
    preserved = condition.get("preserved_fields")
    preserved = preserved if isinstance(preserved, Mapping) else {}
    if condition.get("label_symmetric") is not True or condition.get("deployment_allowed") is not True:
        return None
    if preserved.get("gamemode") != rule["gamemode"]:
        return None
    return {
        "checkpoint": checkpoint,
        "expected_identity_sha256": expected_digest,
        "observed_identity_sha256": observed_digest,
        "repo_id": rule["repo_id"],
        "resolved_revision": rule["resolved_revision"],
        "gamemode": rule["gamemode"],
        "projection": rule["projection"],
        "reason": (
            "byte-exact standard-mode checkpoint projection is deployment-equivalent "
            "to the calibrated multi-mode repository snapshot"
        ),
    }


def checkpoint_identities_sha256(identities_by_checkpoint: Mapping[str, Any]) -> str:
    canonical: dict[str, str] = {}
    for raw_name, identity in identities_by_checkpoint.items():
        name = str(raw_name).strip().casefold()
        if not name or name in canonical:
            raise ValueError("checkpoint identities contain an invalid/duplicate name")
        canonical[name] = canonical_checkpoint_identity(identity, name)["identity_sha256"]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _integer(value: Any, *, default: int = 0) -> int:
    number = _finite_number(value, default=float(default))
    return int(number) if math.isfinite(number) else default


def _checkpoint_name(checkpoint: Mapping[str, Any]) -> str:
    value = checkpoint.get("checkpoint") or checkpoint.get("config_name") or "other"
    return str(value).strip().casefold() or "other"


def _summary_value(
    bundle: Mapping[str, Any],
    metric: str,
    statistic: str,
    *,
    section: str = "family_conditioned",
) -> float:
    conditioned = bundle.get(section)
    if not isinstance(conditioned, Mapping):
        return math.nan
    summary = conditioned.get(metric)
    if not isinstance(summary, Mapping) or _integer(summary.get("count")) <= 0:
        return math.nan
    return _finite_number(summary.get(statistic))


def _nested_summary_value(container: Any, name: str, statistic: str) -> float:
    if not isinstance(container, Mapping):
        return math.nan
    summary = container.get(name)
    if not isinstance(summary, Mapping) or _integer(summary.get("count")) <= 0:
        return math.nan
    return _finite_number(summary.get(statistic))


def _temp_test_summary_value(
    bundle: Mapping[str, Any], setting: str, statistic: str
) -> float:
    local_value = bundle.get("local_normalization")
    local = local_value if isinstance(local_value, Mapping) else {}
    setting_value = local.get(setting)
    setting_bundle = setting_value if isinstance(setting_value, Mapping) else {}
    summary_value = setting_bundle.get("temp_test")
    summary = summary_value if isinstance(summary_value, Mapping) else {}
    if _integer(summary.get("count")) <= 0:
        return math.nan
    return _finite_number(summary.get(statistic))


def _detectllm_lrr_value(bundle: Mapping[str, Any], scope: str) -> float:
    container_value = bundle.get("detectllm_lrr")
    container = container_value if isinstance(container_value, Mapping) else {}
    conditioned_value = container.get(scope)
    conditioned = conditioned_value if isinstance(conditioned_value, Mapping) else {}
    if conditioned.get("defined") is not True:
        return math.nan
    return _finite_number(conditioned.get("value"))


def _sequence_discrepancy_value(bundle: Mapping[str, Any], scope: str) -> float:
    container_value = bundle.get("fast_detectgpt_sequence_discrepancy")
    container = container_value if isinstance(container_value, Mapping) else {}
    scoped_value = container.get(scope)
    scoped = scoped_value if isinstance(scoped_value, Mapping) else {}
    if scoped.get("defined") is not True:
        return math.nan
    return _finite_number(scoped.get("value"))


def _populate_generation_policy_features(
    values: dict[str, float], prefix: str, bundle: Mapping[str, Any]
) -> None:
    policy_value = bundle.get("generation_policy")
    policy = policy_value if isinstance(policy_value, Mapping) else {}
    available = max(_integer(policy.get("available_token_count")), 0)
    values[f"{prefix}.generation_policy.coverage_fraction"] = _finite_number(
        policy.get("coverage_fraction"), default=0.0
    )
    pre_value = policy.get("pre_truncation_full_vocabulary")
    pre = pre_value if isinstance(pre_value, Mapping) else {}
    for metric in POLICY_SUMMARY_METRICS:
        for statistic in SUMMARY_STATISTICS:
            values[f"{prefix}.generation_policy.{metric}.{statistic}"] = (
                _nested_summary_value(pre, metric, statistic)
            )
    values[f"{prefix}.generation_policy.detectllm_lrr"] = (
        _detectllm_lrr_value(bundle, "generation_policy_pre_truncation")
    )
    policy_fd_value = bundle.get(
        "fast_detectgpt_policy_aware_sequence_discrepancy"
    )
    policy_fd = policy_fd_value if isinstance(policy_fd_value, Mapping) else {}
    values[f"{prefix}.generation_policy.fast_detectgpt_policy_aware"] = (
        _finite_number(policy_fd.get("value"))
        if policy_fd.get("defined") is True
        else math.nan
    )
    sampling_value = policy.get("sampling_support")
    sampling = sampling_value if isinstance(sampling_value, Mapping) else {}
    values[f"{prefix}.generation_policy.in_support_fraction"] = (
        _finite_number(sampling.get("in_support_fraction"))
        if available
        else math.nan
    )
    for statistic in NUCLEUS_STATISTICS:
        values[f"{prefix}.generation_policy.support_size.{statistic}"] = (
            _nested_summary_value(sampling, "support_size", statistic)
        )
    values[f"{prefix}.generation_policy.removed_mass.mean"] = (
        _nested_summary_value(sampling, "removed_mass", "mean")
    )
    runs_value = policy.get("runs")
    runs = runs_value if isinstance(runs_value, Mapping) else {}
    violations_value = runs.get("sampling_support_violations")
    violations = violations_value if isinstance(violations_value, Mapping) else {}
    longest = _finite_number(violations.get("longest_run")) if available else math.nan
    values[f"{prefix}.generation_policy.log1p_longest_support_violation_run"] = (
        math.log1p(max(longest, 0.0)) if math.isfinite(longest) else math.nan
    )


def _flatten_window(checkpoint_name: str, window: Mapping[str, Any]) -> WhiteboxFeatureWindow:
    total_tokens = max(_integer(window.get("token_count")), 0)
    families_value = window.get("families")
    families = families_value if isinstance(families_value, Mapping) else {}
    counts: dict[str, int] = {}
    for family in FAMILIES:
        bundle = families.get(family)
        counts[family] = max(_integer(bundle.get("token_count")), 0) if isinstance(bundle, Mapping) else 0

    values: dict[str, float] = {
        "window.log1p_token_count": math.log1p(total_tokens),
        "window.active_family_fraction": sum(count > 0 for count in counts.values()) / len(FAMILIES),
    }
    window_summary_value = window.get("summary")
    window_summary = (
        window_summary_value if isinstance(window_summary_value, Mapping) else {}
    )
    for metric in RAW_SUMMARY_METRICS:
        for statistic in SUMMARY_STATISTICS:
            values[f"window.raw_full_vocabulary.{metric}.{statistic}"] = (
                _summary_value(
                    window_summary,
                    metric,
                    statistic,
                    section="raw_full_vocabulary",
                )
            )
    values["window.detectllm_lrr.raw_full_vocabulary"] = _detectllm_lrr_value(
        window_summary, "raw_full_vocabulary"
    )
    values["window.detectllm_lrr.family_conditioned_heuristic"] = (
        _detectllm_lrr_value(window_summary, "family_conditioned")
    )
    values["window.fast_detectgpt_sequence.raw_full_vocabulary"] = (
        _sequence_discrepancy_value(window_summary, "raw_full_vocabulary")
    )
    values["window.fast_detectgpt_sequence.family_conditioned_heuristic"] = (
        _sequence_discrepancy_value(window_summary, "family_conditioned")
    )
    _populate_generation_policy_features(values, "window", window_summary)
    for family in FAMILIES:
        prefix = f"family.{family}"
        bundle_value = families.get(family)
        bundle = bundle_value if isinstance(bundle_value, Mapping) else {}
        count = counts[family]
        values[f"{prefix}.present"] = float(count > 0)
        values[f"{prefix}.log1p_token_count"] = math.log1p(count)
        values[f"{prefix}.token_fraction"] = count / total_tokens if total_tokens else 0.0
        for metric in SUMMARY_METRICS:
            for statistic in SUMMARY_STATISTICS:
                values[f"{prefix}.conditioned.{metric}.{statistic}"] = _summary_value(
                    bundle, metric, statistic
                )
        for metric in RAW_SUMMARY_METRICS:
            for statistic in SUMMARY_STATISTICS:
                values[f"{prefix}.raw_full_vocabulary.{metric}.{statistic}"] = (
                    _summary_value(
                        bundle,
                        metric,
                        statistic,
                        section="raw_full_vocabulary",
                    )
                )
        values[f"{prefix}.detectllm_lrr.raw_full_vocabulary"] = (
            _detectllm_lrr_value(bundle, "raw_full_vocabulary")
        )
        values[f"{prefix}.detectllm_lrr.family_conditioned_heuristic"] = (
            _detectllm_lrr_value(bundle, "family_conditioned")
        )
        values[f"{prefix}.fast_detectgpt_sequence.raw_full_vocabulary"] = (
            _sequence_discrepancy_value(bundle, "raw_full_vocabulary")
        )
        values[f"{prefix}.fast_detectgpt_sequence.family_conditioned_heuristic"] = (
            _sequence_discrepancy_value(bundle, "family_conditioned")
        )
        _populate_generation_policy_features(values, prefix, bundle)
        for statistic in TEMPTEST_STATISTICS:
            values[f"{prefix}.temp_test.tau_0_9.{statistic}"] = (
                _temp_test_summary_value(bundle, "tau_0_9", statistic)
            )

        sampling_value = bundle.get("sampling")
        sampling = sampling_value if isinstance(sampling_value, Mapping) else {}
        values[f"{prefix}.nucleus.in_fraction"] = (
            _finite_number(sampling.get("in_nucleus_fraction")) if count else math.nan
        )
        for statistic in NUCLEUS_STATISTICS:
            values[f"{prefix}.nucleus.size.{statistic}"] = _nested_summary_value(
                sampling, "nucleus_size", statistic
            )
        values[f"{prefix}.nucleus.mass.mean"] = _nested_summary_value(sampling, "nucleus_mass", "mean")

        runs_value = bundle.get("runs")
        runs = runs_value if isinstance(runs_value, Mapping) else {}
        violations_value = runs.get("top_p_violations")
        violations = violations_value if isinstance(violations_value, Mapping) else {}
        values[f"{prefix}.top_p_violation_fraction"] = (
            _finite_number(violations.get("fraction")) if count else math.nan
        )
        longest = _finite_number(violations.get("longest_run")) if count else math.nan
        values[f"{prefix}.log1p_longest_top_p_violation_run"] = (
            math.log1p(max(longest, 0.0)) if math.isfinite(longest) else math.nan
        )

    contexts_value = window.get("contexts")
    contexts = contexts_value if isinstance(contexts_value, Mapping) else {}
    timing_value = contexts.get("timing")
    timing = timing_value if isinstance(timing_value, Mapping) else {}
    timing_families_value = timing.get("families")
    timing_families = (
        timing_families_value
        if isinstance(timing_families_value, Mapping)
        else {}
    )
    timing_t_value = timing_families.get("t")
    timing_t = timing_t_value if isinstance(timing_t_value, Mapping) else {}
    for statistic in TEMPTEST_STATISTICS:
        values[f"context.timing.family.t.temp_test.tau_0_1.{statistic}"] = (
            _temp_test_summary_value(timing_t, "tau_0_1", statistic)
        )

    for known in CHECKPOINTS:
        values[f"checkpoint.{known}"] = float(checkpoint_name == known)
    values["checkpoint.other"] = float(checkpoint_name not in CHECKPOINTS)

    start_raw = window.get("scored_interval_start_ms", window.get("audio_window_start_ms"))
    end_raw = window.get("scored_interval_end_ms", window.get("audio_window_end_ms"))
    start = _integer(start_raw) if start_raw is not None else None
    end = _integer(end_raw) if end_raw is not None else None
    return WhiteboxFeatureWindow(
        checkpoint=checkpoint_name,
        window_index=_integer(window.get("window_index")),
        start_ms=start,
        end_ms=end,
        token_count=total_tokens,
        values=tuple(values[name] for name in BASE_FEATURE_NAMES),
    )


def flatten_whitebox_result(result: Mapping[str, Any]) -> list[WhiteboxFeatureWindow]:
    """Flatten successful checkpoint windows from a white-box engine result.

    Exact duplicate windows within one checkpoint are collapsed deterministically
    (the entry with more scored tokens wins).  Matching time intervals from
    different checkpoints remain separate observations by design.
    """

    payload: Mapping[str, Any] = result
    nested = result.get("result")
    if "checkpoints" not in result and isinstance(nested, Mapping):
        payload = nested
    checkpoint_values = payload.get("checkpoints")
    if not isinstance(checkpoint_values, Sequence) or isinstance(checkpoint_values, (str, bytes)):
        return []

    deduplicated: dict[tuple[str, int | None, int | None, int], WhiteboxFeatureWindow] = {}
    for checkpoint_value in checkpoint_values:
        if not isinstance(checkpoint_value, Mapping) or checkpoint_value.get("status") != "ok":
            continue
        checkpoint_name = _checkpoint_name(checkpoint_value)
        windows_value = checkpoint_value.get("windows")
        if not isinstance(windows_value, Sequence) or isinstance(windows_value, (str, bytes)):
            continue
        for window_value in windows_value:
            if not isinstance(window_value, Mapping):
                continue
            flattened = _flatten_window(checkpoint_name, window_value)
            # The index disambiguates pathological results lacking interval metadata.
            fallback_index = flattened.window_index if flattened.start_ms is None or flattened.end_ms is None else -1
            key = (checkpoint_name, flattened.start_ms, flattened.end_ms, fallback_index)
            existing = deduplicated.get(key)
            if existing is None or flattened.token_count > existing.token_count:
                deduplicated[key] = flattened
    return sorted(
        deduplicated.values(),
        key=lambda item: (
            item.checkpoint,
            item.start_ms if item.start_ms is not None else -1,
            item.end_ms if item.end_ms is not None else -1,
            item.window_index,
        ),
    )


def _finite_sequence(value: Any, name: str, expected: int, *, positive: bool = False) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != expected:
        raise ValueError(f"{name} must contain exactly {expected} numbers")
    result = [_finite_number(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite numbers")
    if positive and not all(item > 0 for item in result):
        raise ValueError(f"{name} must contain only positive numbers")
    return result


def _stable_sigmoid(logit: float) -> float:
    if logit >= 0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _protocol_differences(expected: Any, observed: Any, path: str = "search_protocol") -> list[dict[str, Any]]:
    """Return deterministic, field-level protocol differences for audit UI."""

    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}.{key}"
            if key not in expected:
                differences.append({"path": child, "expected": None, "observed": copy.deepcopy(observed[key])})
            elif key not in observed:
                differences.append({"path": child, "expected": copy.deepcopy(expected[key]), "observed": None})
            else:
                differences.extend(_protocol_differences(expected[key], observed[key], child))
        return differences
    if expected != observed:
        return [{"path": path, "expected": copy.deepcopy(expected), "observed": copy.deepcopy(observed)}]
    return []


class WhiteboxDiscriminator:
    """Apply a frozen JSON logistic head to a white-box engine result."""

    def __init__(self, artifact: Mapping[str, Any]):
        if artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(f"unsupported white-box model schema: {artifact.get('schema_version')!r}")
        feature_names = artifact.get("feature_names")
        normalized_names = (
            list(feature_names)
            if isinstance(feature_names, Sequence) and not isinstance(feature_names, (str, bytes))
            else None
        )
        # Never silently apply weights to a reordered representation.
        if normalized_names != list(BASE_FEATURE_NAMES):
            raise ValueError("artifact feature_names do not match the fixed runtime representation")
        expanded = artifact.get("expanded_feature_names")
        if expanded != list(EXPANDED_FEATURE_NAMES):
            raise ValueError("artifact expanded_feature_names do not match median+missing representation")

        imputation = artifact.get("imputation")
        if not isinstance(imputation, Mapping) or imputation.get("strategy") != "median":
            raise ValueError("artifact must use median imputation")
        scaler = artifact.get("scaler")
        if not isinstance(scaler, Mapping):
            raise ValueError("artifact scaler is missing")
        self.medians = _finite_sequence(imputation.get("values"), "imputation.values", len(BASE_FEATURE_NAMES))
        self.center = _finite_sequence(scaler.get("center"), "scaler.center", len(EXPANDED_FEATURE_NAMES))
        self.scale = _finite_sequence(
            scaler.get("scale"), "scaler.scale", len(EXPANDED_FEATURE_NAMES), positive=True
        )
        self.weights = _finite_sequence(artifact.get("weights"), "weights", len(EXPANDED_FEATURE_NAMES))
        self.intercept = _finite_number(artifact.get("intercept"))
        if not math.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        aggregation = artifact.get("aggregation")
        if not isinstance(aggregation, Mapping) or aggregation.get("method") != "mean_top_k_windows":
            raise ValueError("artifact aggregation must be mean_top_k_windows")
        self.top_k = _integer(aggregation.get("top_k"))
        if self.top_k <= 0:
            raise ValueError("aggregation.top_k must be positive")
        self.model_id = str(artifact.get("model_id") or "unnamed-whitebox-model")
        self.calibration = artifact.get("calibration") if isinstance(artifact.get("calibration"), Mapping) else {}
        required_value = self.calibration.get("required_checkpoints")
        if required_value is None:
            self.required_checkpoints: tuple[str, ...] = ()
        elif isinstance(required_value, Sequence) and not isinstance(required_value, (str, bytes)):
            normalized = tuple(str(value).strip().casefold() for value in required_value)
            if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
                raise ValueError("calibration.required_checkpoints must be unique non-empty names")
            self.required_checkpoints = tuple(sorted(normalized))
        else:
            raise ValueError("calibration.required_checkpoints must be a sequence")

        calibration_settings = self.calibration.get("required_checkpoint_settings")
        self.calibration_settings_declared = isinstance(calibration_settings, Mapping)
        training_value = artifact.get("training")
        training = training_value if isinstance(training_value, Mapping) else {}
        corpus_value = training.get("corpus_audit")
        corpus_audit = corpus_value if isinstance(corpus_value, Mapping) else {}
        training_settings = corpus_audit.get("checkpoint_settings")
        raw_expected = calibration_settings if isinstance(calibration_settings, Mapping) else training_settings
        self.required_checkpoint_settings: dict[str, dict[str, Any]] = {}
        if isinstance(raw_expected, Mapping):
            for raw_name, raw_settings in raw_expected.items():
                name = str(raw_name).strip().casefold()
                if not name or name in self.required_checkpoint_settings:
                    raise ValueError("required checkpoint settings contain an invalid/duplicate name")
                if not isinstance(raw_settings, Mapping):
                    raise ValueError(f"required settings for {name} must be an object")
                self.required_checkpoint_settings[name] = canonical_checkpoint_settings(
                    raw_settings, raw_settings.get("schema_version")
                )
        if self.required_checkpoints and self.required_checkpoint_settings:
            if set(self.required_checkpoint_settings) != set(self.required_checkpoints):
                raise ValueError(
                    "required checkpoint settings must exactly cover calibration.required_checkpoints"
                )
        if isinstance(calibration_settings, Mapping) and isinstance(training_settings, Mapping):
            normalized_training = {
                str(name).strip().casefold(): canonical_checkpoint_settings(
                    settings, settings.get("schema_version") if isinstance(settings, Mapping) else None
                )
                for name, settings in training_settings.items()
            }
            if normalized_training != self.required_checkpoint_settings:
                raise ValueError("calibration checkpoint settings differ from frozen training settings")
        expected_hash = str(self.calibration.get("required_checkpoint_settings_sha256") or "").casefold()
        if expected_hash:
            if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
                raise ValueError("calibration.required_checkpoint_settings_sha256 is invalid")
            if not self.required_checkpoint_settings:
                raise ValueError("checkpoint settings hash is present without checkpoint settings")
            observed_hash = checkpoint_settings_sha256(self.required_checkpoint_settings)
            if observed_hash != expected_hash:
                raise ValueError("calibration checkpoint settings hash does not reconcile")
        calibration_identities_value = self.calibration.get("required_checkpoint_identities")
        training_identities_value = corpus_audit.get("checkpoint_identities")
        self.calibration_identities_declared = isinstance(calibration_identities_value, Mapping)
        identities_value = (
            calibration_identities_value
            if isinstance(calibration_identities_value, Mapping)
            else training_identities_value
        )
        self.required_checkpoint_identities: dict[str, dict[str, Any]] = {}
        if isinstance(identities_value, Mapping):
            for raw_name, raw_identity in identities_value.items():
                name = str(raw_name).strip().casefold()
                if not name or name in self.required_checkpoint_identities:
                    raise ValueError("required checkpoint identities contain an invalid/duplicate name")
                self.required_checkpoint_identities[name] = canonical_checkpoint_identity(raw_identity, name)
        if self.required_checkpoints and self.required_checkpoint_identities:
            if set(self.required_checkpoint_identities) != set(self.required_checkpoints):
                raise ValueError(
                    "required checkpoint identities must exactly cover calibration.required_checkpoints"
                )
        if isinstance(calibration_identities_value, Mapping) and isinstance(training_identities_value, Mapping):
            normalized_training_identities = {
                str(name).strip().casefold(): canonical_checkpoint_identity(
                    identity, str(name).strip().casefold()
                )
                for name, identity in training_identities_value.items()
            }
            if {
                name: identity["identity_sha256"]
                for name, identity in normalized_training_identities.items()
            } != {
                name: identity["identity_sha256"]
                for name, identity in self.required_checkpoint_identities.items()
            }:
                raise ValueError("calibration checkpoint identities differ from frozen training identities")
        expected_identity_hash = str(
            self.calibration.get("required_checkpoint_identities_sha256") or ""
        ).casefold()
        if expected_identity_hash:
            if len(expected_identity_hash) != 64 or any(
                character not in "0123456789abcdef" for character in expected_identity_hash
            ):
                raise ValueError("calibration.required_checkpoint_identities_sha256 is invalid")
            if not self.required_checkpoint_identities:
                raise ValueError("checkpoint identities hash is present without checkpoint identities")
            if checkpoint_identities_sha256(self.required_checkpoint_identities) != expected_identity_hash:
                raise ValueError("calibration checkpoint identities hash does not reconcile")

        calibration_search_value = self.calibration.get("required_search_protocol")
        self.calibration_search_protocol_declared = isinstance(calibration_search_value, Mapping)
        self.required_search_protocol: dict[str, Any] = {}
        if isinstance(calibration_search_value, Mapping):
            self.required_search_protocol = canonical_search_protocol(calibration_search_value)
        expected_search_hash = str(
            self.calibration.get("required_search_protocol_sha256") or ""
        ).strip().casefold()
        self.calibration_search_protocol_hash_declared = bool(expected_search_hash)
        if expected_search_hash:
            if len(expected_search_hash) != 64 or any(
                character not in "0123456789abcdef" for character in expected_search_hash
            ):
                raise ValueError("calibration.required_search_protocol_sha256 is invalid")
            if not self.required_search_protocol:
                raise ValueError("search protocol hash is present without required_search_protocol")
            if search_protocol_sha256(self.required_search_protocol) != expected_search_hash:
                raise ValueError("calibration search protocol hash does not reconcile")
        self._artifact = copy.deepcopy(dict(artifact))

    @classmethod
    def from_path(cls, path: str | Path) -> "WhiteboxDiscriminator":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise ValueError("white-box model artifact must be a JSON object")
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._artifact)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self._artifact, ensure_ascii=False, indent=indent, allow_nan=False)

    def _transform(self, values: Sequence[float]) -> tuple[list[float], list[float], list[bool]]:
        if len(values) != len(BASE_FEATURE_NAMES):
            raise ValueError("window feature vector has an unexpected length")
        missing = [not math.isfinite(float(value)) for value in values]
        imputed = [self.medians[index] if missing[index] else float(value) for index, value in enumerate(values)]
        expanded = imputed + [float(flag) for flag in missing]
        scaled = [
            (value - self.center[index]) / self.scale[index]
            for index, value in enumerate(expanded)
        ]
        return imputed, scaled, missing

    def _score_window(self, window: WhiteboxFeatureWindow, contribution_limit: int) -> dict[str, Any]:
        imputed, scaled, missing = self._transform(window.values)
        contributions = [weight * value for weight, value in zip(self.weights, scaled, strict=True)]
        logit = self.intercept + math.fsum(contributions)
        score = _stable_sigmoid(logit)

        details = []
        for index, (name, contribution) in enumerate(zip(EXPANDED_FEATURE_NAMES, contributions, strict=True)):
            base_index = index if index < len(BASE_FEATURE_NAMES) else index - len(BASE_FEATURE_NAMES)
            details.append(
                {
                    "feature": name,
                    "base_feature": BASE_FEATURE_NAMES[base_index],
                    "raw_value": (
                        window.values[base_index] if math.isfinite(window.values[base_index]) else None
                    ),
                    "imputed_value": imputed[base_index],
                    "was_missing": missing[base_index],
                    "scaled_value": scaled[index],
                    "weight": self.weights[index],
                    "contribution": contribution,
                }
            )
        by_absolute = sorted(details, key=lambda row: (-abs(row["contribution"]), row["feature"]))
        positives = sorted(
            (row for row in details if row["contribution"] > 0),
            key=lambda row: (-row["contribution"], row["feature"]),
        )
        negatives = sorted(
            (row for row in details if row["contribution"] < 0),
            key=lambda row: (row["contribution"], row["feature"]),
        )
        return {
            "checkpoint": window.checkpoint,
            "window_index": window.window_index,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "token_count": window.token_count,
            "ranking_score": score,
            "logit": logit,
            "raw_features": window.feature_dict(json_safe=True),
            "missing_features": [name for name, flag in zip(BASE_FEATURE_NAMES, missing, strict=True) if flag],
            "top_contributions": by_absolute[:contribution_limit],
            "top_positive_contributions": positives[:contribution_limit],
            "top_negative_contributions": negatives[:contribution_limit],
            "contribution_reconciliation": {
                "intercept": self.intercept,
                "feature_contribution_sum": math.fsum(contributions),
                "reconstructed_logit": self.intercept + math.fsum(contributions),
            },
        }

    def _protocol_audit(self, whitebox_result: Mapping[str, Any]) -> dict[str, Any]:
        payload: Mapping[str, Any] = whitebox_result
        nested = whitebox_result.get("result")
        if "checkpoints" not in whitebox_result and isinstance(nested, Mapping):
            payload = nested
        raw_entries = payload.get("checkpoints")
        entries = (
            list(raw_entries)
            if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (str, bytes))
            else []
        )
        names: list[str] = []
        successful: set[str] = set()
        invalid_entries: list[dict[str, Any]] = []
        by_name: dict[str, list[Mapping[str, Any]]] = {}
        for index, raw in enumerate(entries):
            if not isinstance(raw, Mapping):
                invalid_entries.append({"index": index, "reason": "checkpoint entry is not an object"})
                continue
            raw_name = raw.get("checkpoint") or raw.get("config_name")
            name = str(raw_name or "").strip().casefold()
            if not name:
                invalid_entries.append({"index": index, "reason": "checkpoint name is missing"})
                continue
            names.append(name)
            by_name.setdefault(name, []).append(raw)
            if raw.get("status") == "ok":
                successful.add(name)

        observed = sorted(set(names))
        duplicates = sorted(name for name, rows in by_name.items() if len(rows) > 1)
        expected = list(self.required_checkpoints or tuple(sorted(self.required_checkpoint_settings)))
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected)) if expected else []
        unsuccessful = sorted(set(expected) - successful)
        top_settings = payload.get("settings")
        schema_version = payload.get("schema_version") or whitebox_result.get("schema_version")
        observed_settings: dict[str, dict[str, Any]] = {}
        settings_errors: list[dict[str, Any]] = []
        for name, rows in sorted(by_name.items()):
            if len(rows) != 1:
                continue
            row = rows[0]
            checkpoint_settings = row.get("audit_distribution_settings")
            settings_value = checkpoint_settings if isinstance(checkpoint_settings, Mapping) else top_settings
            try:
                canonical = canonical_checkpoint_settings(settings_value, schema_version)
            except ValueError as exc:
                settings_errors.append({"checkpoint": name, "reason": str(exc)})
                continue
            if isinstance(checkpoint_settings, Mapping) and isinstance(top_settings, Mapping):
                try:
                    top_canonical = canonical_checkpoint_settings(top_settings, schema_version)
                except ValueError as exc:
                    settings_errors.append({"checkpoint": name, "reason": f"top-level settings: {exc}"})
                else:
                    conflicting = {
                        key: {"checkpoint": canonical.get(key), "top_level": value}
                        for key, value in top_canonical.items()
                        if key in canonical and canonical.get(key) != value
                    }
                    if conflicting:
                        settings_errors.append({
                            "checkpoint": name,
                            "reason": "checkpoint audit settings conflict with top-level settings",
                            "conflicting_fields": conflicting,
                        })
                    # Extractor rows intentionally keep the shared condition
                    # view in result.settings while audit_distribution_settings
                    # records checkpoint-specific generation parameters.  Bind
                    # the shared value only when the checkpoint's independent
                    # condition audit proves that the same deployment-safe view
                    # was actually applied.  Treating an absent per-checkpoint
                    # copy as a conflict made the production calibration format
                    # fail against itself and forced every white-box result to
                    # abstain despite successful checkpoint coverage.
                    top_condition = top_canonical.get("condition_view")
                    if top_condition is not None and "condition_view" not in canonical:
                        condition_audit = row.get("condition_view")
                        if (
                            not isinstance(condition_audit, Mapping)
                            or condition_audit.get("name") != top_condition
                            or condition_audit.get("label_symmetric") is not True
                            or condition_audit.get("deployment_allowed") is not True
                        ):
                            settings_errors.append({
                                "checkpoint": name,
                                "reason": (
                                    "checkpoint condition audit does not prove the "
                                    "top-level condition_view"
                                ),
                            })
                        else:
                            canonical["condition_view"] = top_condition
            observed_settings[name] = canonical

        settings_mismatches: list[dict[str, Any]] = []
        for name, expected_settings in sorted(self.required_checkpoint_settings.items()):
            actual = observed_settings.get(name)
            if actual is None:
                settings_mismatches.append({
                    "checkpoint": name,
                    "field": "all",
                    "expected": expected_settings,
                    "observed": None,
                })
                continue
            for field in ("temperature", "top_p", "schema_version", "condition_view"):
                if actual.get(field) != expected_settings.get(field):
                    settings_mismatches.append({
                        "checkpoint": name,
                        "field": field,
                        "expected": expected_settings.get(field),
                        "observed": actual.get(field),
                    })

        expected_settings_hash = (
            checkpoint_settings_sha256(self.required_checkpoint_settings)
            if self.required_checkpoint_settings
            else None
        )
        observed_settings_hash = None
        if observed_settings and not settings_errors:
            try:
                observed_settings_hash = checkpoint_settings_sha256(observed_settings)
            except ValueError:
                observed_settings_hash = None

        observed_identities: dict[str, dict[str, Any]] = {}
        identity_errors: list[dict[str, Any]] = []
        for name, rows in sorted(by_name.items()):
            if len(rows) != 1:
                continue
            try:
                observed_identities[name] = canonical_checkpoint_identity(
                    rows[0].get("checkpoint_identity"), name
                )
            except ValueError as exc:
                identity_errors.append({"checkpoint": name, "reason": str(exc)})
        identity_mismatches: list[dict[str, Any]] = []
        identity_equivalences: list[dict[str, Any]] = []
        for name, expected_identity in sorted(self.required_checkpoint_identities.items()):
            actual = observed_identities.get(name)
            expected_digest = expected_identity["identity_sha256"]
            actual_digest = actual.get("identity_sha256") if actual else None
            if actual_digest != expected_digest:
                equivalence = (
                    (
                        _minimum_runtime_identity_equivalence(
                            model_id=self.model_id,
                            checkpoint=name,
                            expected=expected_identity,
                            observed=actual,
                            checkpoint_result=by_name[name][0],
                        )
                        or _standard_mode_identity_equivalence(
                            model_id=self.model_id,
                            checkpoint=name,
                            expected=expected_identity,
                            observed=actual,
                            checkpoint_result=by_name[name][0],
                        )
                    )
                    if actual is not None and len(by_name.get(name, ())) == 1
                    else None
                )
                if equivalence is not None:
                    identity_equivalences.append(equivalence)
                    continue
                identity_mismatches.append({
                    "checkpoint": name,
                    "expected_identity_sha256": expected_digest,
                    "observed_identity_sha256": actual_digest,
                    "expected": expected_identity,
                    "observed": actual,
                })
        expected_identities_hash = (
            checkpoint_identities_sha256(self.required_checkpoint_identities)
            if self.required_checkpoint_identities
            else None
        )
        observed_identities_hash = None
        if observed_identities and not identity_errors:
            try:
                observed_identities_hash = checkpoint_identities_sha256(observed_identities)
            except ValueError:
                observed_identities_hash = None

        observed_search_protocol: dict[str, Any] = {}
        search_protocol_errors: list[dict[str, Any]] = []
        try:
            observed_search_protocol = derive_search_protocol(payload)
        except ValueError as exc:
            search_protocol_errors.append({"reason": str(exc)})
        required_search_protocol_hash = (
            search_protocol_sha256(self.required_search_protocol)
            if self.required_search_protocol
            else None
        )
        observed_search_protocol_hash = (
            search_protocol_sha256(observed_search_protocol)
            if observed_search_protocol
            else None
        )
        search_protocol_mismatches = (
            _protocol_differences(self.required_search_protocol, observed_search_protocol)
            if self.required_search_protocol and observed_search_protocol
            else []
        )
        search_protocol_exact: bool | None = None
        if self.required_search_protocol:
            search_protocol_exact = bool(
                observed_search_protocol
                and not search_protocol_errors
                and not search_protocol_mismatches
                and observed_search_protocol_hash == required_search_protocol_hash
            )

        calibration_values = self.calibration.get("human_scores")
        calibrated_artifact = bool(
            isinstance(calibration_values, Sequence)
            and not isinstance(calibration_values, (str, bytes))
            and any(
                math.isfinite(number := _finite_number(value)) and 0 <= number <= 1
                for value in calibration_values
            )
        )
        reasons: list[str] = []
        if calibrated_artifact and not self.required_checkpoints:
            reasons.append("calibrated artifact lacks calibration.required_checkpoints")
        if calibrated_artifact and not self.calibration_settings_declared:
            reasons.append("calibrated artifact lacks immutable required_checkpoint_settings")
        if calibrated_artifact and not self.calibration_identities_declared:
            reasons.append("calibrated artifact lacks immutable required_checkpoint_identities")
        if calibrated_artifact and not self.calibration_search_protocol_declared:
            reasons.append("calibrated artifact lacks immutable required_search_protocol")
        if calibrated_artifact and not self.calibration_search_protocol_hash_declared:
            reasons.append("calibrated artifact lacks immutable required_search_protocol_sha256")
        if missing:
            reasons.append("missing required checkpoints: " + ", ".join(missing))
        if extra:
            reasons.append("unexpected extra checkpoints: " + ", ".join(extra))
        if duplicates:
            reasons.append("duplicate checkpoint entries: " + ", ".join(duplicates))
        if unsuccessful:
            reasons.append("required checkpoints without successful windows: " + ", ".join(unsuccessful))
        if invalid_entries:
            reasons.append("malformed checkpoint entries are present")
        if settings_errors and self.required_checkpoint_settings:
            reasons.append("checkpoint extraction settings are missing or internally inconsistent")
        if settings_mismatches:
            reasons.append("checkpoint extraction settings do not match the calibrated protocol")
        if identity_errors and self.required_checkpoint_identities:
            reasons.append("checkpoint model/tokenizer identities are missing or invalid")
        if identity_mismatches:
            reasons.append("checkpoint model/tokenizer identities do not match the calibrated artifacts")
        if self.required_search_protocol and search_protocol_errors:
            reasons.append("white-box candidate search protocol is missing or invalid")
        if search_protocol_mismatches:
            reasons.append("white-box candidate search protocol does not match the calibrated protocol")

        checkpoint_exact = bool(expected) and not (
            missing or extra or duplicates or unsuccessful or invalid_entries
        )
        settings_exact: bool | None = None
        if self.required_checkpoint_settings:
            settings_exact = not settings_errors and not settings_mismatches and (
                set(observed_settings) == set(self.required_checkpoint_settings)
            )
            if set(observed_settings) != set(self.required_checkpoint_settings):
                if "checkpoint extraction settings do not exactly cover the required checkpoints" not in reasons:
                    reasons.append("checkpoint extraction settings do not exactly cover the required checkpoints")
        identities_exact: bool | None = None
        if self.required_checkpoint_identities:
            identities_exact = not identity_errors and not identity_mismatches and (
                set(observed_identities) == set(self.required_checkpoint_identities)
            )
            if set(observed_identities) != set(self.required_checkpoint_identities):
                reasons.append("checkpoint identities do not exactly cover the required checkpoints")
        exact_match = bool(
            expected
            and checkpoint_exact
            and (settings_exact is not False)
            and (identities_exact is not False)
            and (search_protocol_exact is not False)
            and not reasons
        )
        return {
            "enforced_for_calibration": calibrated_artifact,
            "required_checkpoints": expected,
            "observed_checkpoint_entries": names,
            "observed_unique_checkpoints": observed,
            "observed_successful_checkpoints": sorted(successful),
            "missing_checkpoints": missing,
            "extra_checkpoints": extra,
            "duplicate_checkpoints": duplicates,
            "unsuccessful_required_checkpoints": unsuccessful,
            "invalid_checkpoint_entries": invalid_entries,
            "required_checkpoint_settings": copy.deepcopy(self.required_checkpoint_settings),
            "checkpoint_settings_binding_source": (
                "calibration" if self.calibration_settings_declared else "training_audit_fallback"
                if self.required_checkpoint_settings else None
            ),
            "observed_checkpoint_settings": observed_settings,
            "required_checkpoint_settings_sha256": expected_settings_hash,
            "observed_checkpoint_settings_sha256": observed_settings_hash,
            "settings_mismatches": settings_mismatches,
            "settings_errors": settings_errors,
            "required_checkpoint_identities": copy.deepcopy(self.required_checkpoint_identities),
            "checkpoint_identity_binding_source": (
                "calibration" if self.calibration_identities_declared else "training_audit_fallback"
                if self.required_checkpoint_identities else None
            ),
            "observed_checkpoint_identities": observed_identities,
            "required_checkpoint_identities_sha256": expected_identities_hash,
            "observed_checkpoint_identities_sha256": observed_identities_hash,
            "identity_mismatches": identity_mismatches,
            "identity_deployment_equivalences": identity_equivalences,
            "identity_errors": identity_errors,
            "required_search_protocol": copy.deepcopy(self.required_search_protocol),
            "observed_search_protocol": copy.deepcopy(observed_search_protocol),
            "required_search_protocol_sha256": required_search_protocol_hash,
            "observed_search_protocol_sha256": observed_search_protocol_hash,
            "search_protocol_mismatches": search_protocol_mismatches,
            "search_protocol_errors": search_protocol_errors,
            "checkpoint_coverage_exact": checkpoint_exact,
            "checkpoint_settings_exact": settings_exact,
            "checkpoint_identities_exact": identities_exact,
            "checkpoint_identities_byte_exact": bool(identities_exact and not identity_equivalences),
            "search_protocol_exact": search_protocol_exact,
            "exact_match": exact_match,
            "reasons": reasons,
        }

    def _calibrate(self, score: float, protocol_audit: Mapping[str, Any]) -> dict[str, Any]:
        values = self.calibration.get("human_scores") if isinstance(self.calibration, Mapping) else None
        human_scores = []
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            human_scores = [
                number for value in values if math.isfinite(number := _finite_number(value)) and 0 <= number <= 1
            ]
        reason = self.calibration.get("reason") if isinstance(self.calibration, Mapping) else None
        if not human_scores:
            return {
                "status": "unavailable",
                "calibrated": False,
                "human_null_size": 0,
                "human_null_p_value": None,
                "thresholds": {},
                "threshold_guarantees": copy.deepcopy(
                    self.calibration.get("thresholds", {}) if isinstance(self.calibration, Mapping) else {}
                ),
                "threshold_flags": {},
                "reason": reason or (
                    "No independent, revision-pinned human null scores are embedded; the logistic output is only a ranking score."
                ),
                "protocol_compatible": protocol_audit.get("exact_match"),
                "protocol_audit": copy.deepcopy(dict(protocol_audit)),
            }
        if not protocol_audit.get("exact_match"):
            detail = "; ".join(str(item) for item in protocol_audit.get("reasons", []))
            return {
                "status": "abstain",
                "calibrated": False,
                "human_null_size": len(human_scores),
                "human_null_p_value": None,
                "thresholds": {},
                "threshold_guarantees": copy.deepcopy(
                    self.calibration.get("thresholds", {}) if isinstance(self.calibration, Mapping) else {}
                ),
                "threshold_flags": {},
                "reason": (
                    "The observed white-box checkpoint/extraction protocol does not exactly match the "
                    "independent human-null calibration protocol"
                    + (f": {detail}" if detail else ".")
                ),
                "protocol_compatible": False,
                "protocol_audit": copy.deepcopy(dict(protocol_audit)),
            }
        p_value = (1 + sum(reference >= score for reference in human_scores)) / (len(human_scores) + 1)
        thresholds_value = self.calibration.get("thresholds")
        thresholds = thresholds_value if isinstance(thresholds_value, Mapping) else {}
        supported_thresholds: dict[str, float] = {}
        flags: dict[str, dict[str, Any]] = {}
        for name, raw in thresholds.items():
            operator = ">"
            supported = True
            value = raw
            if isinstance(raw, Mapping):
                supported = bool(raw.get("supported"))
                operator = str(raw.get("operator") or ">")
                value = raw.get("threshold")
            threshold = _finite_number(value)
            # NP artifacts in this project use a strict order-statistic rule.
            # Unknown/non-strict operators fail closed rather than silently
            # changing the advertised false-positive guarantee.
            if supported and operator == ">" and math.isfinite(threshold):
                supported_thresholds[str(name)] = threshold
                flags[str(name)] = {
                    "supported": True,
                    "threshold": threshold,
                    "operator": ">",
                    "exceeded": score > threshold,
                }
        return {
            "status": "available",
            "calibrated": True,
            "method": "upper-tail split-conformal rank against held-out human scores",
            "human_null_size": len(human_scores),
            "human_null_p_value": p_value,
            "thresholds": supported_thresholds,
            "threshold_guarantees": copy.deepcopy(dict(thresholds)),
            "threshold_flags": flags,
            "reason": reason,
            "protocol_compatible": True,
            "protocol_audit": copy.deepcopy(dict(protocol_audit)),
        }

    def score(self, whitebox_result: Mapping[str, Any], *, contribution_limit: int = 12) -> dict[str, Any]:
        if contribution_limit <= 0:
            raise ValueError("contribution_limit must be positive")
        protocol_audit = self._protocol_audit(whitebox_result)
        flattened = flatten_whitebox_result(whitebox_result)
        base = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "available": bool(flattened),
            "score_semantics": (
                "Regularized logistic ranking score for Mapperatorinator-like white-box statistics; "
                "not an AI-authorship probability."
            ),
            "representation": representation_audit(),
            "calibration_protocol_audit": protocol_audit,
        }
        if not flattened:
            return base | {
                "status": "unavailable",
                "uncalibrated": True,
                "decision_usable": False,
                "reason": "The white-box result contains no successful checkpoint windows.",
                "aggregate": None,
                "calibration": {
                    "status": "unavailable",
                    "calibrated": False,
                    "human_null_size": 0,
                    "human_null_p_value": None,
                    "thresholds": {},
                    "threshold_guarantees": {},
                    "threshold_flags": {},
                    "reason": "Calibration cannot be evaluated because no aggregate score is available.",
                    "protocol_compatible": protocol_audit.get("exact_match"),
                    "protocol_audit": copy.deepcopy(protocol_audit),
                },
                "checkpoint_coverage": [],
                "windows": [],
            }

        windows = [self._score_window(window, contribution_limit) for window in flattened]
        ordered = sorted(
            windows,
            key=lambda row: (
                -row["ranking_score"],
                row["checkpoint"],
                row["start_ms"] if row["start_ms"] is not None else -1,
                row["window_index"],
            ),
        )
        selected = ordered[: min(self.top_k, len(ordered))]
        aggregate_score = math.fsum(row["ranking_score"] for row in selected) / len(selected)
        calibration = self._calibrate(aggregate_score, protocol_audit)
        checkpoints: dict[str, dict[str, Any]] = {}
        for row in windows:
            entry = checkpoints.setdefault(
                row["checkpoint"], {"checkpoint": row["checkpoint"], "window_count": 0, "token_count": 0}
            )
            entry["window_count"] += 1
            entry["token_count"] += row["token_count"]
        return base | {
            "status": "abstain" if protocol_audit.get("enforced_for_calibration") and not calibration["calibrated"] else "ok",
            "uncalibrated": not calibration["calibrated"],
            "decision_usable": bool(calibration["calibrated"]),
            "abstention_reasons": (
                [str(calibration.get("reason"))]
                if protocol_audit.get("enforced_for_calibration") and not calibration["calibrated"]
                else []
            ),
            "aggregate": {
                "ranking_score": aggregate_score,
                "method": "mean_top_k_windows_across_all_checkpoints",
                "configured_top_k": self.top_k,
                "selected_window_count": len(selected),
                "total_window_count": len(windows),
                "selected_windows": [
                    {
                        "checkpoint": row["checkpoint"],
                        "window_index": row["window_index"],
                        "start_ms": row["start_ms"],
                        "end_ms": row["end_ms"],
                        "ranking_score": row["ranking_score"],
                    }
                    for row in selected
                ],
            },
            "calibration": calibration,
            "checkpoint_coverage": sorted(checkpoints.values(), key=lambda row: row["checkpoint"]),
            "windows": windows,
        }


def score_whitebox_discriminator(
    whitebox_result: Mapping[str, Any], artifact: Mapping[str, Any] | str | Path, *, contribution_limit: int = 12
) -> dict[str, Any]:
    """Convenience wrapper for one-shot CPU-only scoring."""

    model = WhiteboxDiscriminator.from_path(artifact) if isinstance(artifact, (str, Path)) else WhiteboxDiscriminator(artifact)
    return model.score(whitebox_result, contribution_limit=contribution_limit)


def unavailable(reason: str, *, model_path: str | Path | None = None) -> dict[str, Any]:
    """Explicit abstention payload for an absent/unloadable discriminator."""

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "unavailable",
        "available": False,
        "uncalibrated": True,
        "decision_usable": False,
        "reason": reason,
        "model_path": str(model_path) if model_path is not None else None,
        "model_id": None,
        "score_semantics": (
            "No ranking score is available; unavailable models/results are never coerced to numeric zero."
        ),
        "aggregate": None,
        "calibration": {
            "status": "unavailable",
            "calibrated": False,
            "human_null_size": 0,
            "human_null_p_value": None,
            "thresholds": {},
            "threshold_guarantees": {},
            "threshold_flags": {},
            "reason": reason,
        },
        "checkpoint_coverage": [],
        "calibration_protocol_audit": {
            "enforced_for_calibration": False,
            "exact_match": False,
            "reasons": [reason],
        },
        "windows": [],
    }


__all__ = [
    "BASE_FEATURE_NAMES",
    "CHECKPOINT_IDENTITY_SCHEMA_VERSION",
    "CHECKPOINTS",
    "DEFAULT_MODEL",
    "EXPANDED_FEATURE_NAMES",
    "FAMILIES",
    "MODEL_SCHEMA_VERSION",
    "REPRESENTATION_VERSION",
    "RESULT_SCHEMA_VERSION",
    "TEMPTEST_STATISTICS",
    "WhiteboxDiscriminator",
    "WhiteboxFeatureWindow",
    "candidate_map_key",
    "canonical_checkpoint_identity",
    "canonical_checkpoint_settings",
    "checkpoint_identities_sha256",
    "checkpoint_identity_sha256",
    "checkpoint_settings_sha256",
    "flatten_whitebox_result",
    "representation_audit",
    "score_whitebox_discriminator",
    "unavailable",
]
