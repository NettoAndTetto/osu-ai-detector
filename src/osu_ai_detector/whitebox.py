"""Optional Mapperatorinator white-box feature extraction.

This module intentionally has *no* import-time dependency on PyTorch, Hydra,
``slider`` or the vendored Mapperatorinator checkout.  The small mathematical
helpers are therefore usable from the lightweight detector/web environment,
while :class:`WhiteboxEngine` loads the heavyweight runtime only when scoring
is requested.

The engine teacher-forces an observed chart against one or more original
Mapperatorinator checkpoints.  Window/context alignment follows the
``Processor.ai_mod`` implementation from Mapperatorinator: overlapping audio
windows are trimmed so an event is counted once.  The primary raw channel
describes the checkpoint's complete output vocabulary.  A separately labelled
family-conditioned channel renormalizes over the observed tokenizer
``EventType`` range and is retained only as a target-family heuristic.

When the vendored revision supports ordinary sequential generation, the
extractor replays its source processors in generation order (time shifts,
biases, temperatures and stateful lookback), followed by the same top-k and
Hugging Face top-p masking rules.  This exposes exact sampling-support and
removed-mass evidence.  Revisions or contexts whose source path cannot be
faithfully reconstructed are marked unavailable instead of approximated.

Raw and policy-aware Fast-DetectGPT-style discrepancies aggregate token
numerators and variances over a scope before standardization.  Per-token and
family-conditioned z summaries remain explicitly heuristic.  DetectLLM LRR is
computed as the ratio of sequence means (mean NLL / mean one-indexed log-rank),
never as the mean of token-wise ratios; an all-rank-one scope is undefined.
These statistics are detector features, not independently calibrated
probabilities that a chart is AI-generated.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .whitebox_protocol import canonical_selection_policy


SCHEMA_VERSION = "osu-ai-detector.whitebox/v4"
CHECKPOINT_IDENTITY_SCHEMA_VERSION = "osu-ai-detector.whitebox-checkpoint-identity/v2"
SYMMETRIC_CONDITION_VIEW = "symmetric_stripped_v1"
DEFAULT_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "Mapperatorinator"
TEMPERATURE_NORMALIZATION_SETTINGS: tuple[tuple[str, float], ...] = (
    ("tau_0_9", 0.9),
    ("tau_0_1", 0.1),
)


@dataclass(frozen=True)
class WhiteboxCheckpoint:
    """A Hydra inference config and optional checkpoint/config overrides."""

    config_name: str
    label: str | None = None
    model_path: str | None = None
    logical_repo_id: str | None = None
    immutable_revision: str | None = None
    hydra_overrides: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return self.label or self.config_name


@dataclass(frozen=True)
class WhiteboxOptions:
    """Configuration for :class:`WhiteboxEngine`.

    ``max_cached_checkpoints`` is intentionally one by default because the
    supported checkpoints are large and consumer GPUs frequently have 6 GB of
    VRAM.  Set it to zero to unload after every checkpoint or increase it for a
    batch service with sufficient memory.
    """

    checkpoints: tuple[WhiteboxCheckpoint, ...] = field(
        default_factory=lambda: (
            WhiteboxCheckpoint("v29"),
            WhiteboxCheckpoint("v30"),
            WhiteboxCheckpoint("v31"),
            WhiteboxCheckpoint("v32"),
            WhiteboxCheckpoint("v32-mini"),
        )
    )
    vendor_root: Path = DEFAULT_VENDOR_ROOT
    device: str = "auto"
    precision: str = "bf16"
    attention_implementation: str = "sdpa"
    seed: int = 1337
    temperature: float = 0.9
    top_p: float = 0.9
    condition_view: str = SYMMETRIC_CONDITION_VIEW
    start_ms: int | None = None
    end_ms: int | None = None
    intervals_ms: tuple[tuple[int, int], ...] = ()
    windows_per_interval: int | None = None
    forward_batch_size: int = 1
    max_windows: int | None = None
    include_token_details: bool = False
    max_token_details_per_window: int = 256
    max_cached_checkpoints: int = 1

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("at least one white-box checkpoint is required")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and greater than zero")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.condition_view not in {SYMMETRIC_CONDITION_VIEW, "native"}:
            raise ValueError("condition_view must be symmetric_stripped_v1 or native")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError("max_windows must be positive")
        if self.start_ms is not None and self.end_ms is not None and self.start_ms >= self.end_ms:
            raise ValueError("start_ms must be less than end_ms")
        for start, end in self.intervals_ms:
            if start < 0 or start >= end:
                raise ValueError("each intervals_ms entry must satisfy 0 <= start < end")
        if self.windows_per_interval is not None and self.windows_per_interval <= 0:
            raise ValueError("windows_per_interval must be positive")
        if self.windows_per_interval is not None and not self.intervals_ms:
            raise ValueError("windows_per_interval requires intervals_ms")
        if (
            isinstance(self.forward_batch_size, bool)
            or not isinstance(self.forward_batch_size, int)
            or self.forward_batch_size <= 0
        ):
            raise ValueError("forward_batch_size must be a positive integer")
        if self.max_token_details_per_window < 0:
            raise ValueError("max_token_details_per_window cannot be negative")
        if self.max_cached_checkpoints < 0:
            raise ValueError("max_cached_checkpoints cannot be negative")


def quantile_summary(values: Iterable[float | int | None]) -> dict[str, float | int]:
    """Return a JSON-safe descriptive summary using linear quantiles."""

    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return {"count": 0}
    ordered = sorted(cleaned)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    mean = math.fsum(ordered) / len(ordered)
    variance = math.fsum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean": mean,
        "std_population": math.sqrt(max(variance, 0.0)),
        "min": ordered[0],
        "p05": quantile(0.05),
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


def select_interval_window_indices(
    frame_times_ms: Sequence[float | int],
    window_duration_ms: float | int,
    intervals_ms: Sequence[tuple[int, int]],
    windows_per_interval: int,
) -> tuple[int, ...]:
    """Select deterministic, non-duplicated audio windows near interval centres.

    Mapperatorinator's audio contexts overlap heavily (V29 advances by roughly
    one tenth of a context).  Scoring every overlapping context adds enormous
    compute and highly correlated observations.  For each label-free proposal
    interval, choose the contexts whose centres are closest to the proposal
    centre, while requiring actual interval overlap.  The complete selection
    rule is later repeated on every human-null chart and covered by the final
    map-level calibration.
    """

    duration = float(window_duration_ms)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("window_duration_ms must be finite and positive")
    if windows_per_interval <= 0:
        raise ValueError("windows_per_interval must be positive")
    times = [float(value) for value in frame_times_ms]
    if any(not math.isfinite(value) for value in times):
        raise ValueError("frame times must be finite")
    selected: set[int] = set()
    for raw_start, raw_end in intervals_ms:
        start, end = int(raw_start), int(raw_end)
        if start < 0 or start >= end:
            raise ValueError("each interval must satisfy 0 <= start < end")
        target_start = (start + end - duration) / 2
        candidates = [
            index
            for index, frame_start in enumerate(times)
            if frame_start < end and frame_start + duration > start
        ]
        ordered = sorted(candidates, key=lambda index: (abs(times[index] - target_start), index))
        selected.update(ordered[:windows_per_interval])
    return tuple(sorted(selected))


def apply_condition_view(generation_config: Any, view: str) -> tuple[Any, dict[str, Any]]:
    """Apply an explicit label-symmetric conditioning contract.

    Ranked human charts often carry BeatmapID/style/mapper identities while a
    generated ``.osu`` does not.  Feeding those native fields to only one class
    creates a trivial white-box shortcut.  The production view therefore keeps
    only gamemode and fixes or removes every chart-identity/difficulty field for
    *both* labels before prompts and model kwargs are built.
    """

    if view == "native":
        return generation_config, {
            "schema_version": "osu-ai-detector.whitebox-condition-view/v1",
            "name": "native",
            "label_symmetric": False,
            "deployment_allowed": False,
            "warning": "native metadata is retained and may confound human/AI comparisons",
        }
    if view != SYMMETRIC_CONDITION_VIEW:
        raise ValueError(f"unsupported white-box condition view: {view}")

    config = copy.copy(generation_config)
    fixed = {
        "beatmap_id": None,
        "difficulty": None,
        "mapper_id": None,
        "year": None,
        "hitsounded": False,
        "hp_drain_rate": None,
        "circle_size": None,
        "overall_difficulty": None,
        "approach_rate": None,
        "slider_multiplier": 1.4,
        "slider_tick_rate": None,
        "keycount": 4,
        "hold_note_ratio": None,
        "scroll_speed_ratio": None,
        "descriptors": None,
        "negative_descriptors": None,
    }
    missing = [name for name in fixed if not hasattr(config, name)]
    if missing:
        raise ValueError(f"generation config is missing condition fields: {missing}")
    for name, value in fixed.items():
        setattr(config, name, value)
    return config, {
        "schema_version": "osu-ai-detector.whitebox-condition-view/v1",
        "name": SYMMETRIC_CONDITION_VIEW,
        "label_symmetric": True,
        "deployment_allowed": True,
        "preserved_fields": {"gamemode": int(config.gamemode)},
        "fixed_or_unknown_fields": fixed,
        "identity_fields_removed": [
            "beatmap_id", "mapper_id", "year", "descriptors", "negative_descriptors"
        ],
        "content_metadata_shortcuts_removed": [
            "difficulty", "hitsounded", "hp_drain_rate", "circle_size",
            "overall_difficulty", "approach_rate", "slider_tick_rate"
        ],
    }


def run_summary(flags: Iterable[bool], *, max_reported_runs: int = 10) -> dict[str, Any]:
    """Summarize consecutive true runs in observation order."""

    observations = [bool(flag) for flag in flags]
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index, flag in enumerate(observations + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append({"start_index": start, "end_index": index - 1, "length": index - start})
            start = None
    positives = sum(observations)
    lengths = [run["length"] for run in runs]
    longest = sorted(runs, key=lambda run: (-run["length"], run["start_index"]))[:max_reported_runs]
    return {
        "observations": len(observations),
        "positives": positives,
        "fraction": positives / len(observations) if observations else 0.0,
        "run_count": len(runs),
        "longest_run": max(lengths, default=0),
        "mean_run_length": math.fsum(lengths) / len(lengths) if lengths else 0.0,
        "longest_runs": longest,
    }


def nucleus_statistics(
    probabilities: Sequence[float],
    target_index: int,
    *,
    top_p: float = 0.9,
    min_tokens_to_keep: int = 1,
) -> dict[str, Any]:
    """Apply Mapperatorinator/Transformers' exact lower-tail top-p rule.

    A deterministic token-index tie break is used by this pure-Python helper.
    Model scoring uses ``torch.sort`` exactly as the installed Transformers
    implementation does; ties at the cutoff should not be interpreted as a
    stable model fingerprint.
    """

    probs = [float(probability) for probability in probabilities]
    if not probs:
        raise ValueError("probabilities cannot be empty")
    if not 0 <= target_index < len(probs):
        raise IndexError("target_index is outside the distribution")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not 1 <= min_tokens_to_keep <= len(probs):
        raise ValueError("min_tokens_to_keep must be within the vocabulary")
    if any(probability < 0 or not math.isfinite(probability) for probability in probs):
        raise ValueError("probabilities must be finite and non-negative")
    total = math.fsum(probs)
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")
    probs = [probability / total for probability in probs]

    ascending = sorted(range(len(probs)), key=lambda index: (probs[index], index))
    lower_tail = 0.0
    removed: set[int] = set()
    protected = set(ascending[-min_tokens_to_keep:])
    for index in ascending:
        lower_tail += probs[index]
        if index not in protected and lower_tail <= 1 - top_p:
            removed.add(index)

    kept = [index for index in range(len(probs)) if index not in removed]
    descending = sorted(range(len(probs)), key=lambda index: (-probs[index], index))
    target_position = descending.index(target_index)
    mass_before = math.fsum(probs[index] for index in descending[:target_position])
    target_probability = probs[target_index]
    return {
        "in_nucleus": target_index in kept,
        "nucleus_size": len(kept),
        "nucleus_mass": math.fsum(probs[index] for index in kept),
        "target_cumulative_mass_before": mass_before,
        "target_cumulative_mass_through": mass_before + target_probability,
        "lower_tail_removed_mass": math.fsum(probs[index] for index in removed),
    }


def distribution_metrics(
    logits: Sequence[float],
    target_index: int,
    *,
    temperature: float = 0.9,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Exact scalar metrics for a finite categorical distribution.

    This is the reference implementation used by unit tests.  The runtime
    engine computes the same equations in vectorized PyTorch operations.
    """

    raw = [float(logit) for logit in logits]
    if not raw:
        raise ValueError("logits cannot be empty")
    if not 0 <= target_index < len(raw):
        raise IndexError("target_index is outside the distribution")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and greater than zero")
    if any(math.isnan(logit) or logit == math.inf for logit in raw):
        raise ValueError("logits cannot contain NaN or positive infinity")
    scaled = [logit / temperature for logit in raw]
    finite = [logit for logit in scaled if math.isfinite(logit)]
    if not finite:
        raise ValueError("at least one logit must be finite")
    maximum = max(finite)
    exponentials = [math.exp(logit - maximum) if math.isfinite(logit) else 0.0 for logit in scaled]
    denominator = math.fsum(exponentials)
    log_normalizer = maximum + math.log(denominator)
    log_probabilities = [logit - log_normalizer for logit in scaled]
    probabilities = [math.exp(log_probability) for log_probability in log_probabilities]
    target_log_probability = log_probabilities[target_index]
    target_probability = probabilities[target_index]
    entropy = -math.fsum(
        probability * log_probability
        for probability, log_probability in zip(probabilities, log_probabilities)
        if probability > 0
    )
    expected_log_probability = math.fsum(
        probability * log_probability
        for probability, log_probability in zip(probabilities, log_probabilities)
        if probability > 0
    )
    variance = math.fsum(
        probability * (log_probability - expected_log_probability) ** 2
        for probability, log_probability in zip(probabilities, log_probabilities)
        if probability > 0
    )
    standard_deviation = math.sqrt(max(variance, 0.0))
    curvature_z = (
        (target_log_probability - expected_log_probability) / standard_deviation
        if standard_deviation > 1e-12 and math.isfinite(target_log_probability)
        else None
    )
    competitors = [logit for index, logit in enumerate(scaled) if index != target_index]
    margin = scaled[target_index] - max(competitors) if competitors else None
    result = {
        "probability": target_probability,
        "log_probability": target_log_probability,
        "nll": -target_log_probability,
        "rank": 1 + sum(logit > scaled[target_index] for logit in scaled),
        "entropy": entropy,
        "margin": margin,
        "expected_log_probability": expected_log_probability,
        "log_probability_variance": max(variance, 0.0),
        "curvature_z": curvature_z,
        "curvature_defined": curvature_z is not None,
    }
    result["log_rank"] = math.log(result["rank"])
    result.update(nucleus_statistics(probabilities, target_index, top_p=top_p))
    return result


def temperature_normalization_statistics(
    logits: Sequence[float],
    target_index: int,
    *,
    temperature: float,
) -> dict[str, float]:
    """Return the exact finite-vocabulary TempTest statistic for one token.

    Let ``p = softmax(logits)`` and ``tau`` be the temperature under test.
    Following Kempton et al. (AISTATS 2025), the local normalization term and
    per-token statistic are

    ``log sum_v p_v**(1/tau)`` and
    ``log_temp_norm - (1/tau - 1) * log p_target``.

    The helper intentionally consumes pre-temperature logits.  The runtime
    applies it to the observed token's tokenizer ``EventType`` range because
    Mapperatorinator's grammar and stateful logits processors make the full
    output vocabulary a misleading comparison set.  This adaptation is an
    auditable feature, not a standalone hypothesis-test p-value.
    """

    raw = [float(logit) for logit in logits]
    if not raw:
        raise ValueError("logits cannot be empty")
    if not 0 <= target_index < len(raw):
        raise IndexError("target_index is outside the distribution")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and greater than zero")
    if any(math.isnan(logit) or logit == math.inf for logit in raw):
        raise ValueError("logits cannot contain NaN or positive infinity")
    if not math.isfinite(raw[target_index]):
        raise ValueError("target logit must be finite")

    def logsumexp(values: Sequence[float]) -> float:
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            raise ValueError("at least one logit must be finite")
        maximum = max(finite)
        return maximum + math.log(
            math.fsum(math.exp(value - maximum) for value in finite)
        )

    log_normalizer = logsumexp(raw)
    inverse_temperature = 1.0 / temperature
    tempered_log_normalizer = logsumexp(
        [logit * inverse_temperature for logit in raw]
    )
    target_log_probability = raw[target_index] - log_normalizer
    log_temperature_normalizer = (
        tempered_log_normalizer - inverse_temperature * log_normalizer
    )
    temp_test = log_temperature_normalizer - (
        inverse_temperature - 1.0
    ) * target_log_probability
    return {
        "temperature": temperature,
        "target_base_log_probability": target_log_probability,
        "log_temperature_normalizer": log_temperature_normalizer,
        "temp_test": temp_test,
    }


def _metric_bundle(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw_names = (
        "probability",
        "nll",
        "rank",
        "log_rank",
        "entropy",
        "margin",
        "expected_log_probability",
        "log_probability_variance",
        "curvature_z",
    )
    family_names = (
        "probability",
        "nll",
        "rank",
        "log_rank",
        "entropy",
        "margin",
        "expected_log_probability",
        "log_probability_variance",
        "curvature_z",
    )
    policy_names = family_names + ("sample_probability", "sample_nll")

    def collect(section: str, name: str) -> list[Any]:
        values: list[Any] = []
        for record in records:
            bundle_value = record.get(section, {})
            bundle = bundle_value if isinstance(bundle_value, Mapping) else {}
            value = bundle.get(name)
            # v3 records persist this field.  Deriving it here also keeps the
            # pure aggregation helper compatible with older fixtures/results.
            if name == "log_rank" and value is None:
                rank = bundle.get("rank")
                if isinstance(rank, (int, float)) and rank >= 1:
                    value = math.log(float(rank))
            values.append(value)
        return values

    def detectllm_lrr(section: str) -> dict[str, Any]:
        """Exact non-perturbative LRR aggregate from the DetectLLM code.

        The official implementation scores ``-mean(log p) / mean(log rank)``.
        Here NLL is already ``-log p``, so the equivalent positive expression
        is ``mean(nll) / mean(log(rank))``.  A sequence containing only rank-1
        observations has a zero denominator and is explicitly undefined.
        """

        pairs: list[tuple[float, float]] = []
        for record in records:
            bundle_value = record.get(section, {})
            bundle = bundle_value if isinstance(bundle_value, Mapping) else {}
            nll = bundle.get("nll")
            rank = bundle.get("rank")
            if (
                isinstance(nll, (int, float))
                and math.isfinite(float(nll))
                and isinstance(rank, (int, float))
                and math.isfinite(float(rank))
                and float(rank) >= 1
            ):
                pairs.append((float(nll), math.log(float(rank))))
        if not pairs:
            return {
                "value": None,
                "defined": False,
                "token_count": 0,
                "mean_nll": None,
                "mean_log_rank": None,
                "undefined_reason": "no finite paired NLL/rank observations",
            }
        mean_nll = math.fsum(pair[0] for pair in pairs) / len(pairs)
        mean_log_rank = math.fsum(pair[1] for pair in pairs) / len(pairs)
        defined = mean_log_rank > 1e-12
        return {
            "value": mean_nll / mean_log_rank if defined else None,
            "defined": defined,
            "token_count": len(pairs),
            "mean_nll": mean_nll,
            "mean_log_rank": mean_log_rank,
            "undefined_reason": None if defined else "all observed tokens have rank 1",
        }

    def sequence_discrepancy(section: str) -> dict[str, Any]:
        """Aggregate conditional discrepancy before standardization.

        Fast-DetectGPT standardizes the sum over a sequence.  Averaging or
        quantiling independently standardized token z-scores is not the same
        statistic because token variances differ.  Conditional independence
        yields a sum of conditional variances in the denominator.
        """

        terms: list[tuple[float, float]] = []
        for record in records:
            bundle_value = record.get(section, {})
            bundle = bundle_value if isinstance(bundle_value, Mapping) else {}
            target = bundle.get("log_probability")
            expected = bundle.get("expected_log_probability")
            variance = bundle.get("log_probability_variance")
            if (
                isinstance(target, (int, float))
                and math.isfinite(float(target))
                and isinstance(expected, (int, float))
                and math.isfinite(float(expected))
                and isinstance(variance, (int, float))
                and math.isfinite(float(variance))
                and float(variance) >= 0
            ):
                terms.append((float(target) - float(expected), float(variance)))
        if not terms:
            return {
                "value": None,
                "defined": False,
                "token_count": 0,
                "numerator": None,
                "variance_sum": None,
                "undefined_reason": "no finite target/expectation/variance observations",
            }
        numerator = math.fsum(term[0] for term in terms)
        variance_sum = math.fsum(term[1] for term in terms)
        defined = variance_sum > 1e-12
        return {
            "value": numerator / math.sqrt(variance_sum) if defined else None,
            "defined": defined,
            "token_count": len(terms),
            "numerator": numerator,
            "variance_sum": variance_sum,
            "undefined_reason": None if defined else "summed conditional variance is zero",
        }

    def policy_aware_sequence_discrepancy() -> dict[str, Any]:
        terms: list[tuple[float, float]] = []
        for record in records:
            policy_value = record.get("generation_policy")
            policy = policy_value if isinstance(policy_value, Mapping) else {}
            target = policy.get("raw_scoring_log_probability")
            expected = policy.get("expected_raw_scoring_log_probability")
            variance = policy.get("raw_scoring_log_probability_variance")
            if (
                isinstance(target, (int, float))
                and math.isfinite(float(target))
                and isinstance(expected, (int, float))
                and math.isfinite(float(expected))
                and isinstance(variance, (int, float))
                and math.isfinite(float(variance))
                and float(variance) >= 0
            ):
                terms.append((float(target) - float(expected), float(variance)))
        if not terms:
            return {
                "value": None,
                "defined": False,
                "token_count": 0,
                "numerator": None,
                "variance_sum": None,
                "undefined_reason": "generation-policy replay is unavailable",
            }
        numerator = math.fsum(term[0] for term in terms)
        variance_sum = math.fsum(term[1] for term in terms)
        defined = variance_sum > 1e-12
        return {
            "value": numerator / math.sqrt(variance_sum) if defined else None,
            "defined": defined,
            "token_count": len(terms),
            "numerator": numerator,
            "variance_sum": variance_sum,
            "undefined_reason": None if defined else "summed policy variance is zero",
        }

    def collect_local(setting: str, name: str) -> list[Any]:
        return [
            record.get("local_normalization", {}).get(setting, {}).get(name)
            for record in records
        ]

    in_nucleus = [bool(record.get("family_conditioned", {}).get("in_nucleus")) for record in records]
    family_ranks = collect("family_conditioned", "rank")
    curvature = collect("family_conditioned", "curvature_z")
    policy_records = [
        record.get("generation_policy")
        for record in records
        if isinstance(record.get("generation_policy"), Mapping)
    ]
    policy_support = [
        bool(record.get("in_sampling_support")) for record in policy_records
    ]
    return {
        "token_count": len(records),
        "raw_full_vocabulary": {name: quantile_summary(collect("raw", name)) for name in raw_names},
        "family_conditioned": {name: quantile_summary(collect("family_conditioned", name)) for name in family_names},
        "detectllm_lrr": {
            "definition": "mean(nll) / mean(log(rank)); rank is one-indexed",
            "raw_full_vocabulary": detectllm_lrr("raw"),
            "family_conditioned": detectllm_lrr("family_conditioned"),
            "generation_policy_pre_truncation": detectllm_lrr(
                "generation_policy"
            ),
            "interpretation": "uncalibrated feature; score direction is learned from revision-pinned data",
        },
        "fast_detectgpt_sequence_discrepancy": {
            "definition": "sum(log p(target) - E_p[log p(X)]) / sqrt(sum(Var_p[log p(X)]))",
            "raw_full_vocabulary": sequence_discrepancy("raw"),
            "family_conditioned": sequence_discrepancy("family_conditioned"),
            "interpretation": (
                "single-source-model conditional discrepancy; family-conditioned scope is an osu! heuristic"
            ),
        },
        "fast_detectgpt_policy_aware_sequence_discrepancy": {
            "definition": (
                "p=raw CFG source; q=renormalized full generation policy after processors/top-k/top-p"
            ),
            **policy_aware_sequence_discrepancy(),
        },
        "generation_policy": {
            "available_token_count": len(policy_records),
            "requested_token_count": len(records),
            "coverage_fraction": (
                len(policy_records) / len(records) if records else 0.0
            ),
            "pre_truncation_full_vocabulary": {
                name: quantile_summary(collect("generation_policy", name))
                for name in policy_names
            },
            "sampling_support": {
                "in_support_count": sum(policy_support),
                "in_support_fraction": (
                    sum(policy_support) / len(policy_support)
                    if policy_support
                    else 0.0
                ),
                "support_size": quantile_summary(
                    collect("generation_policy", "support_size")
                ),
                "support_mass_before_renormalization": quantile_summary(
                    collect(
                        "generation_policy",
                        "support_mass_before_renormalization",
                    )
                ),
                "removed_mass": quantile_summary(
                    collect("generation_policy", "removed_mass")
                ),
                "target_cumulative_mass_before": quantile_summary(
                    collect(
                        "generation_policy", "target_cumulative_mass_before"
                    )
                ),
                "target_cumulative_mass_through": quantile_summary(
                    collect(
                        "generation_policy", "target_cumulative_mass_through"
                    )
                ),
            },
            "runs": {
                "axis": "successive tokens within this scope",
                "sampling_support_violations": run_summary(
                    not member for member in policy_support
                ),
            },
        },
        "local_normalization": {
            setting: {
                "temperature": temperature,
                "scope": "observed tokenizer EventType range before temperature",
                "target_base_log_probability": quantile_summary(
                    collect_local(setting, "target_base_log_probability")
                ),
                "log_temperature_normalizer": quantile_summary(
                    collect_local(setting, "log_temperature_normalizer")
                ),
                "temp_test": quantile_summary(collect_local(setting, "temp_test")),
            }
            for setting, temperature in TEMPERATURE_NORMALIZATION_SETTINGS
        },
        "sampling": {
            "in_nucleus_count": sum(in_nucleus),
            "in_nucleus_fraction": sum(in_nucleus) / len(in_nucleus) if in_nucleus else 0.0,
            "nucleus_size": quantile_summary(collect("family_conditioned", "nucleus_size")),
            "nucleus_mass": quantile_summary(collect("family_conditioned", "nucleus_mass")),
            "target_cumulative_mass_before": quantile_summary(
                collect("family_conditioned", "target_cumulative_mass_before")
            ),
            "target_cumulative_mass_through": quantile_summary(
                collect("family_conditioned", "target_cumulative_mass_through")
            ),
            "lower_tail_removed_mass": quantile_summary(
                collect("family_conditioned", "lower_tail_removed_mass")
            ),
        },
        "runs": {
            "axis": "successive tokens within this scope",
            "top_p_violations": run_summary(not member for member in in_nucleus),
            "family_rank_1": run_summary(rank == 1 for rank in family_ranks),
            "positive_curvature": run_summary(value is not None and value > 0 for value in curvature),
            "curvature_z_ge_1": run_summary(value is not None and value >= 1 for value in curvature),
        },
    }


def aggregate_token_records(
    records: Sequence[Mapping[str, Any]],
    *,
    total_windows: int,
) -> dict[str, Any]:
    """Aggregate JSON-like token records by family and output context."""

    records = list(records)
    family_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    context_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        family_groups[str(record.get("family", "unknown"))].append(record)
        context_groups[str(record.get("context", "unknown"))].append(record)

    def coverage(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        windows = {record.get("window_index") for record in group}
        return {
            "token_count": len(group),
            "token_fraction": len(group) / len(records) if records else 0.0,
            "windows_with_tokens": len(windows),
            "total_windows": total_windows,
            "window_fraction": len(windows) / total_windows if total_windows else 0.0,
        }

    return {
        "coverage": {
            "token_count": len(records),
            "total_windows": total_windows,
            "windows_with_tokens": len({record.get("window_index") for record in records}),
        },
        "overall": _metric_bundle(records),
        "families": {
            family: {"coverage": coverage(group), **_metric_bundle(group)}
            for family, group in sorted(family_groups.items())
        },
        "contexts": {
            context: {"coverage": coverage(group), **_metric_bundle(group)}
            for context, group in sorted(context_groups.items())
        },
    }


def json_safe(value: Any) -> Any:
    """Recursively convert a result to strict JSON-compatible primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


@dataclass
class _Runtime:
    inference_args: Any
    model: Any
    tokenizer: Any
    preprocessor: Any
    processor: Any
    beatmap_class: Any
    generation_config_from_beatmap: Any
    torch: Any
    checkpoint_identity: dict[str, Any]
    runtime_source_identity: dict[str, Any]


_VENDOR_RUNTIME_LOCK = threading.RLock()
_CHECKPOINT_IDENTITY_CACHE: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_RUNTIME_SOURCE_IDENTITY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_source_identity(vendor_root: Path, config_name: str) -> dict[str, Any]:
    """Hash the inference source/config bytes that define policy replay."""

    root = vendor_root.resolve()
    cache_key = (str(root), config_name)
    cached = _RUNTIME_SOURCE_IDENTITY_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    relative_paths = (
        "inference.py",
        "config.py",
        "configs/inference/default.yaml",
        f"configs/inference/{config_name}.yaml",
        "configs/train/default.yaml",
        f"configs/train/{config_name}.yaml",
        "osuT5/osuT5/tokenizer.py",
        "osuT5/osuT5/inference/preprocessor.py",
        "osuT5/osuT5/inference/processor.py",
        "osuT5/osuT5/inference/server.py",
        "osuT5/osuT5/inference/logit_processors.py",
        "osuT5/osuT5/model/modeling_mapperatorinator.py",
    )
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "runtime source identity is missing required files: " + ", ".join(missing)
        )
    files = {
        relative: {
            "sha256": _sha256_file(root / relative),
            "bytes": (root / relative).stat().st_size,
        }
        for relative in relative_paths
    }
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().casefold()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    canonical = {
        "schema_version": "osu-ai-detector.whitebox-runtime-source/v1",
        "config_name": config_name,
        "git_commit": commit,
        "files": files,
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    result = canonical | {"identity_sha256": digest}
    _RUNTIME_SOURCE_IDENTITY_CACHE[cache_key] = result
    return copy.deepcopy(result)


def _snapshot_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    blob_name = resolved.name.casefold()
    # Hugging Face LFS cache blob names are the immutable content SHA-256.
    # Reuse that identity instead of rereading multi-gigabyte weights.  Local
    # snapshots and non-LFS files are hashed byte-for-byte.
    if path.is_symlink() and len(blob_name) == 64 and all(char in "0123456789abcdef" for char in blob_name):
        digest = blob_name
        source = "huggingface_lfs_blob_name"
    else:
        digest = _sha256_file(path)
        source = "file_sha256"
    return {"sha256": digest, "bytes": path.stat().st_size, "identity_source": source}


def _checkpoint_identity(checkpoint: WhiteboxCheckpoint, args: Any, model: Any) -> dict[str, Any]:
    source = str(args.model_path)
    config = getattr(model, "config", None)
    model_revision = str(getattr(config, "_commit_hash", None) or "").strip().casefold()
    declared_revision = str(checkpoint.immutable_revision or "").strip().casefold()
    configured = Path(source).expanduser()
    local_source = configured.exists()
    repo_id = str(checkpoint.logical_repo_id or "").strip() or (None if local_source else source)
    resolved_revision = declared_revision or model_revision
    cache_key = (checkpoint.display_name, source, repo_id or "", resolved_revision)
    cached = _CHECKPOINT_IDENTITY_CACHE.get(cache_key) if not local_source else None
    if cached is not None:
        return dict(cached)

    snapshot: Path | None = configured.resolve() if local_source else None
    resolution_error = None
    if snapshot is None:
        try:
            from huggingface_hub import snapshot_download  # type: ignore  # noqa: PLC0415

            snapshot = Path(
                snapshot_download(
                    repo_id=source,
                    revision=declared_revision or model_revision or "main",
                    local_files_only=True,
                )
            ).resolve()
        except (ImportError, OSError, ValueError) as exc:
            resolution_error = f"{type(exc).__name__}: {exc}"
    if snapshot is None or not snapshot.exists():
        return {
            "schema_version": CHECKPOINT_IDENTITY_SCHEMA_VERSION,
            "status": "unavailable",
            "checkpoint": checkpoint.display_name,
            "config_name": checkpoint.config_name,
            "configured_model_source": source,
            "repo_id": repo_id,
            "resolved_revision": resolved_revision or None,
            "reason": resolution_error or "resolved checkpoint snapshot is unavailable",
            "identity_sha256": None,
        }

    files = [snapshot] if snapshot.is_file() else sorted(
        (item for item in snapshot.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(snapshot).as_posix(),
    )
    manifest: dict[str, dict[str, Any]] = {}
    tokenizer_files: dict[str, str] = {}
    weight_files: dict[str, str] = {}
    for item in files:
        relative = item.name if snapshot.is_file() else item.relative_to(snapshot).as_posix()
        identity = _snapshot_file_identity(item)
        # Transport metadata (HF LFS symlink versus copied local file) must not
        # change the identity of byte-identical, revision-pinned snapshots.
        manifest[relative] = {"sha256": identity["sha256"], "bytes": identity["bytes"]}
        lower = relative.casefold()
        if lower.endswith("tokenizer.json") or lower.endswith("custom_checkpoint_0.pkl"):
            tokenizer_files[relative] = identity["sha256"]
        if lower.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pkl")):
            weight_files[relative] = identity["sha256"]
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    tokenizer_digest = hashlib.sha256(
        json.dumps(tokenizer_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() if tokenizer_files else None
    weights_digest = hashlib.sha256(
        json.dumps(weight_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() if weight_files else None
    identity_errors = []
    if declared_revision and model_revision and declared_revision != model_revision:
        identity_errors.append(
            "loaded model revision does not match WhiteboxCheckpoint.immutable_revision"
        )
    if not local_source and checkpoint.logical_repo_id and repo_id != source:
        identity_errors.append("remote model source does not match WhiteboxCheckpoint.logical_repo_id")
    if repo_id is not None and not resolved_revision:
        identity_errors.append("checkpoint with logical repo provenance lacks an immutable revision")
    if not tokenizer_digest:
        identity_errors.append("resolved snapshot contains no identifiable tokenizer payload")
    if not weights_digest:
        identity_errors.append("resolved snapshot contains no identifiable model weight payload")
    if identity_errors:
        return {
            "schema_version": CHECKPOINT_IDENTITY_SCHEMA_VERSION,
            "status": "unavailable",
            "checkpoint": checkpoint.display_name,
            "config_name": checkpoint.config_name,
            "configured_model_source": source,
            "repo_id": repo_id,
            "resolved_revision": resolved_revision or None,
            "resolved_snapshot_sha256": manifest_digest,
            "resolved_model_weights_sha256": weights_digest,
            "resolved_tokenizer_sha256": tokenizer_digest,
            "reason": "; ".join(identity_errors),
            "identity_sha256": None,
        }
    payload = {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA_VERSION,
        "status": "ok",
        "checkpoint": checkpoint.display_name,
        "config_name": checkpoint.config_name,
        "configured_model_source": source,
        "source_kind": "local_snapshot" if local_source else "huggingface_cache_snapshot",
        "repo_id": repo_id,
        "requested_revision": declared_revision or (None if local_source else "main"),
        "resolved_revision": resolved_revision or None,
        "resolved_snapshot_sha256": manifest_digest,
        "resolved_model_weights_sha256": weights_digest,
        "resolved_tokenizer_sha256": tokenizer_digest,
        "snapshot_file_count": len(manifest),
        "model_weight_file_count": len(weight_files),
        "tokenizer_file_count": len(tokenizer_files),
    }
    # Bind logical provenance and exact bytes, not their transport path.  This
    # lets a calibrated HF snapshot be copied to a frozen local directory
    # without changing identity while still detecting any repo, revision,
    # model-weight, tokenizer or snapshot-content change.
    digest_payload = {
        key: payload.get(key)
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
    identity_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    result = payload | {"identity_sha256": identity_digest}
    if repo_id is not None:
        _CHECKPOINT_IDENTITY_CACHE[cache_key] = result
    return dict(result)


@contextlib.contextmanager
def _working_directory(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _install_local_checkpoint_subfolder_compat(model_utils: Any) -> bool:
    """Normalize the local gamemode-checkpoint sentinel for Transformers.

    Mapperatorinator resolves a local ``snapshot/gamemode=0`` directory to a
    concrete :class:`Path` and the redundant subfolder sentinel ``None``.
    Transformers 4.57 still forwards an explicitly supplied ``None`` to
    ``os.path.join`` and raises ``TypeError``.  An empty subfolder preserves
    the exact resolved directory while remaining a valid path component.

    Keep the shim process-local and refuse to rewrite a remote repository
    resolution: ``None`` is safe only after Mapperatorinator has already
    selected a concrete local path.
    """

    original = model_utils.resolve_model_checkpoint_path
    if getattr(original, "_osu_ai_empty_subfolder_compat", False):
        return False

    @functools.wraps(original)
    def compatible_resolver(*args: Any, **kwargs: Any) -> tuple[Any, str | None]:
        resolved_path, subfolder = original(*args, **kwargs)
        if subfolder is None:
            if not isinstance(resolved_path, Path):
                raise RuntimeError(
                    "a None checkpoint subfolder is valid only after resolving "
                    "a concrete local Path"
                )
            subfolder = ""
        return resolved_path, subfolder

    compatible_resolver._osu_ai_empty_subfolder_compat = True
    model_utils.resolve_model_checkpoint_path = compatible_resolver
    return True


def _read_osu_mode(path: Path) -> int:
    in_general = False
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_general = line.casefold() == "[general]"
            elif in_general and line.casefold().startswith("mode:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return 0
    return 0


def _batched_teacher_forcing_logits(
    torch: Any,
    processor: Any,
    frames: Any,
    prompts: Sequence[Any],
    unconditioned_prompts: Sequence[Any],
    model_kwargses: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> Iterable[tuple[int, Any]]:
    """Yield left-padding-corrected logits in original window order.

    This mirrors Mapperatorinator's ``Processor._batched_inference`` tensor
    assembly but uses an independently frozen batch size for reproducibility.
    It deliberately has no adaptive OOM retry: changing the batch protocol is
    visible, auditable, and calibration-bound instead of silently depending on
    the local GPU.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    count = len(prompts)
    if not count:
        return
    if not (
        len(frames) == count
        and len(unconditioned_prompts) == count
        and len(model_kwargses) == count
    ):
        raise ValueError("teacher-forcing inputs have inconsistent window counts")

    cond_prompt, uncond_prompt, max_len = processor.stack_prompts(
        list(prompts), list(unconditioned_prompts)
    )
    original_lengths = [int(prompt.size(1)) for prompt in prompts]
    first_keys = tuple(model_kwargses[0].keys())
    first_key_set = set(first_keys)
    for index, kwargs in enumerate(model_kwargses):
        if set(kwargs.keys()) != first_key_set:
            raise ValueError(
                f"teacher-forcing model kwargs differ at window {index}"
            )

    for batch_start in range(0, count, batch_size):
        batch_end = min(batch_start + batch_size, count)
        kwargs_batch = {
            key: torch.cat(
                [model_kwargses[index][key] for index in range(batch_start, batch_end)],
                dim=0,
            )
            for key in first_keys
        }
        cond_batch = cond_prompt[batch_start:batch_end]
        uncond_batch = (
            uncond_prompt[batch_start:batch_end]
            if uncond_prompt is not None
            else None
        )
        forward_inputs = kwargs_batch | {
            "inputs": frames[batch_start:batch_end],
            "decoder_input_ids": cond_batch,
            "decoder_attention_mask": cond_batch.ne(processor.tokenizer.pad_id),
            "negative_prompt": uncond_batch,
            "negative_prompt_attention_mask": (
                uncond_batch.ne(processor.tokenizer.pad_id)
                if uncond_batch is not None
                else None
            ),
        }
        batch_logits = processor.model_forward(forward_inputs)
        if not hasattr(batch_logits, "size") or len(batch_logits.size()) != 3:
            raise RuntimeError(
                "teacher-forcing forward pass returned non-[batch, sequence, vocabulary] logits"
            )
        observed_batch = int(batch_logits.size(0))
        expected_batch = batch_end - batch_start
        if observed_batch != expected_batch:
            raise RuntimeError(
                "teacher-forcing forward pass returned an unexpected batch size: "
                f"{observed_batch} versus {expected_batch}"
            )
        for local_index in range(expected_batch):
            window_index = batch_start + local_index
            left_padding = int(max_len) - original_lengths[window_index]
            if left_padding < 0 or left_padding >= int(batch_logits.size(1)):
                raise RuntimeError(
                    f"invalid decoder left padding for window {window_index}: {left_padding}"
                )
            yield window_index, batch_logits[local_index, left_padding:]


class WhiteboxEngine:
    """Lazy, configurable white-box scorer for Mapperatorinator checkpoints."""

    def __init__(self, options: WhiteboxOptions | None = None):
        self.options = options or WhiteboxOptions()
        self._cache: OrderedDict[tuple[Any, ...], _Runtime] = OrderedDict()

    def availability(self) -> dict[str, Any]:
        """Perform a read-only, import-free environment preflight."""

        vendor_root = Path(self.options.vendor_root).resolve()
        missing: list[str] = []
        for relative in (
            "inference.py",
            "config.py",
            "osuT5/osuT5/tokenizer.py",
            "osuT5/osuT5/inference/processor.py",
            "osuT5/osuT5/inference/server.py",
        ):
            if not (vendor_root / relative).is_file():
                missing.append(str(vendor_root / relative))
        packages = {
            package: importlib.util.find_spec(package) is not None
            for package in ("torch", "hydra", "omegaconf", "slider")
        }
        configs = {
            checkpoint.display_name: (
                vendor_root / "configs" / "inference" / f"{checkpoint.config_name}.yaml"
            ).is_file()
            for checkpoint in self.options.checkpoints
        }
        available = not missing and all(packages.values()) and all(configs.values())
        return {
            "available": available,
            "vendor_root": str(vendor_root),
            "missing_vendor_files": missing,
            "optional_packages": packages,
            "checkpoint_configs": configs,
            "remediation": None if available else (
                "Run the white-box service in the generation Conda environment and ensure "
                "vendor/Mapperatorinator plus the selected inference configs are present."
            ),
        }

    def close(self) -> None:
        """Release cached models and, when available, CUDA allocator cache."""

        with _VENDOR_RUNTIME_LOCK:
            self._cache.clear()
            gc.collect()
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

    def __enter__(self) -> "WhiteboxEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def score(
        self,
        beatmap_path: str | Path,
        audio_path: str | Path,
        *,
        candidate_interval_selection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score one pair, optionally binding per-map label-free candidates.

        The override is applied for this call only while the engine-wide
        runtime lock is held.  Cached checkpoints therefore remain reusable
        across a batch without exposing mutable per-map options to concurrent
        requests.  Custom ``start_ms``, ``end_ms``, and ``max_windows`` remain
        those of the base options and are later rejected by calibrated protocol
        auditing when non-null.
        """

        bound_options = self.options
        attached_selection: dict[str, Any] | None = None
        if candidate_interval_selection is not None:
            if not isinstance(candidate_interval_selection, Mapping):
                raise ValueError("candidate_interval_selection must be an object")
            raw_intervals = candidate_interval_selection.get("intervals_ms")
            if not isinstance(raw_intervals, Sequence) or isinstance(raw_intervals, (str, bytes)):
                raise ValueError("candidate_interval_selection.intervals_ms must be a sequence")
            intervals: list[tuple[int, int]] = []
            for index, raw in enumerate(raw_intervals):
                if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
                    raise ValueError(
                        f"candidate_interval_selection.intervals_ms[{index}] must be a pair"
                    )
                if (
                    isinstance(raw[0], bool)
                    or isinstance(raw[1], bool)
                    or not isinstance(raw[0], int)
                    or not isinstance(raw[1], int)
                ):
                    raise ValueError("candidate interval coordinates must be integers")
                if raw[0] < 0 or raw[0] >= raw[1]:
                    raise ValueError("candidate intervals must satisfy 0 <= start_ms < end_ms")
                if intervals and raw[0] < intervals[-1][1]:
                    raise ValueError("candidate intervals must be ordered and non-overlapping")
                intervals.append((raw[0], raw[1]))
            if not intervals:
                raise ValueError("candidate_interval_selection must contain at least one interval")
            policy = canonical_selection_policy(candidate_interval_selection.get("selection_policy"))
            bound_options = dataclasses.replace(
                self.options,
                intervals_ms=tuple(intervals),
                windows_per_interval=policy["windows_per_interval"],
            )
            attached_selection = copy.deepcopy(dict(candidate_interval_selection))

        with _VENDOR_RUNTIME_LOCK:
            previous_options = self.options
            self.options = bound_options
            try:
                result = self._score_with_current_options(beatmap_path, audio_path)
            finally:
                self.options = previous_options
        if attached_selection is not None:
            result["candidate_interval_selection"] = attached_selection
        return result

    def _score_with_current_options(
        self,
        beatmap_path: str | Path,
        audio_path: str | Path,
    ) -> dict[str, Any]:
        """Score using options already protected by the runtime lock."""

        beatmap = Path(beatmap_path).resolve()
        audio = Path(audio_path).resolve()
        base = {
            "schema_version": SCHEMA_VERSION,
            "method": {
                "name": "Mapperatorinator checkpoint teacher-forcing",
                "window_alignment": (
                    "Processor.ai_mod-compatible; lookback/lookahead overlap is trimmed so each event is counted once"
                ),
                "raw_distribution": "CFG-adjusted model logits over the complete output vocabulary",
                "family_distribution": (
                    "target-family-conditioned heuristic; never presented as the source model support"
                ),
                "generation_policy_distribution": (
                    "full-vocabulary replay of monotonic mask, bias, prefix-conditioned/global temperature, "
                    "stateful lookback when active, top-k, then Transformers top-p"
                ),
                "sequence_discrepancy_formula": (
                    "sum(log p(observed)-E_q[log p(X)])/sqrt(sum(Var_q[log p(X)]))"
                ),
                "interpretation": "features only; no output field is a calibrated AI probability",
            },
            "input": {"beatmap": str(beatmap), "audio": str(audio)},
            "settings": {
                "temperature": self.options.temperature,
                "top_p": self.options.top_p,
                "condition_view": self.options.condition_view,
                "device": self.options.device,
                "precision": self.options.precision,
                "attention_implementation": self.options.attention_implementation,
                "start_ms": self.options.start_ms,
                "end_ms": self.options.end_ms,
                "intervals_ms": self.options.intervals_ms,
                "windows_per_interval": self.options.windows_per_interval,
                "forward_batch_size": self.options.forward_batch_size,
                "max_windows": self.options.max_windows,
                "include_token_details": self.options.include_token_details,
            },
            "availability": self.availability(),
        }
        input_errors = []
        if not beatmap.is_file():
            input_errors.append(f"beatmap file does not exist: {beatmap}")
        if not audio.is_file():
            input_errors.append(f"audio file does not exist: {audio}")
        if input_errors:
            return json_safe(base | {"status": "error", "errors": input_errors, "checkpoints": []})
        if not base["availability"]["available"]:
            return json_safe(base | {"status": "unavailable", "checkpoints": []})

        mode = _read_osu_mode(beatmap)
        checkpoint_results = []
        for checkpoint in self.options.checkpoints:
            try:
                checkpoint_results.append(self._score_checkpoint(checkpoint, beatmap, audio, mode))
            except (ImportError, ModuleNotFoundError) as exc:
                checkpoint_results.append(self._failure(checkpoint, "unavailable", "runtime_import", exc))
            except Exception as exc:  # A failed checkpoint must not take down the web/CLI process.
                checkpoint_results.append(self._failure(checkpoint, "error", "scoring", exc))
                self._cleanup_cuda()

        successful = sum(result.get("status") == "ok" for result in checkpoint_results)
        unavailable = sum(result.get("status") == "unavailable" for result in checkpoint_results)
        if successful == len(checkpoint_results):
            status = "ok"
        elif successful:
            status = "partial"
        elif unavailable == len(checkpoint_results):
            status = "unavailable"
        else:
            status = "error"
        return json_safe(
            base
            | {
                "status": status,
                "summary": {
                    "requested_checkpoints": len(checkpoint_results),
                    "successful_checkpoints": successful,
                    "unavailable_checkpoints": unavailable,
                    "error_checkpoints": len(checkpoint_results) - successful - unavailable,
                    "total_scored_tokens": sum(
                        int(result.get("coverage", {}).get("token_count", 0))
                        for result in checkpoint_results
                    ),
                },
                "checkpoints": checkpoint_results,
            }
        )

    @staticmethod
    def dumps(result: Mapping[str, Any], *, indent: int | None = 2) -> str:
        """Serialize a result with strict JSON (NaN/Infinity are forbidden)."""

        return json.dumps(json_safe(result), ensure_ascii=False, indent=indent, allow_nan=False)

    def _failure(
        self,
        checkpoint: WhiteboxCheckpoint,
        status: str,
        stage: str,
        exc: BaseException,
    ) -> dict[str, Any]:
        return {
            "checkpoint": checkpoint.display_name,
            "config_name": checkpoint.config_name,
            "status": status,
            "error": {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
                "remediation": (
                    "Verify the generation Conda environment, vendored source/config, checkpoint cache, "
                    "audio decoder and available CPU/GPU memory."
                ),
            },
        }

    def _runtime_key(self, checkpoint: WhiteboxCheckpoint, mode: int) -> tuple[Any, ...]:
        return (
            str(Path(self.options.vendor_root).resolve()),
            checkpoint.config_name,
            checkpoint.model_path,
            checkpoint.hydra_overrides,
            mode,
            self.options.device,
            self.options.precision,
            self.options.attention_implementation,
        )

    def _get_runtime(
        self,
        checkpoint: WhiteboxCheckpoint,
        beatmap: Path,
        audio: Path,
        mode: int,
    ) -> tuple[_Runtime, bool]:
        key = self._runtime_key(checkpoint, mode)
        if key in self._cache:
            runtime = self._cache.pop(key)
            self._cache[key] = runtime
            return runtime, False
        runtime = self._load_runtime(checkpoint, beatmap, audio, mode)
        if self.options.max_cached_checkpoints == 0:
            return runtime, True
        self._cache[key] = runtime
        while len(self._cache) > self.options.max_cached_checkpoints:
            self._cache.popitem(last=False)
            self._cleanup_cuda()
        return runtime, False

    def _load_runtime(
        self,
        checkpoint: WhiteboxCheckpoint,
        beatmap: Path,
        audio: Path,
        mode: int,
    ) -> _Runtime:
        vendor_root = Path(self.options.vendor_root).resolve()
        config_dir = vendor_root / "configs" / "inference"
        if not (config_dir / f"{checkpoint.config_name}.yaml").is_file():
            raise FileNotFoundError(f"Mapperatorinator inference config not found: {checkpoint.config_name}")
        vendor_text = str(vendor_root)
        if vendor_text not in sys.path:
            sys.path.insert(0, vendor_text)

        with _working_directory(vendor_root):
            # These imports are intentionally local.  Importing the structured
            # configs registers Hydra's /inference/base and /diffusion/base.
            import config  # type: ignore  # noqa: F401, PLC0415
            import osu_diffusion.config  # type: ignore  # noqa: F401, PLC0415
            import torch  # type: ignore  # noqa: PLC0415
            from hydra import compose, initialize_config_dir  # type: ignore  # noqa: PLC0415
            from inference import (  # type: ignore  # noqa: PLC0415
                compile_args,
                load_model_with_server,
                setup_inference_environment,
            )
            from omegaconf import OmegaConf  # type: ignore  # noqa: PLC0415
            from slider import Beatmap  # type: ignore  # noqa: PLC0415
            from osuT5.osuT5.inference.preprocessor import Preprocessor  # type: ignore  # noqa: PLC0415
            from osuT5.osuT5.inference.processor import (  # type: ignore  # noqa: PLC0415
                Processor,
                generation_config_from_beatmap,
            )
            from osuT5.osuT5.utils import model_utils  # type: ignore  # noqa: PLC0415

            _install_local_checkpoint_subfolder_compat(model_utils)

            with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
                cfg = compose(config_name=checkpoint.config_name, overrides=list(checkpoint.hydra_overrides))
            args = OmegaConf.to_object(cfg)
            args.beatmap_path = str(beatmap)
            args.audio_path = str(audio)
            args.output_path = str(vendor_root / ".whitebox-output")
            args.device = self.options.device
            args.precision = self.options.precision
            args.attn_implementation = self.options.attention_implementation
            args.compile = False
            args.generate_positions = False
            args.use_server = False
            args.gamemode = mode
            if checkpoint.model_path is not None:
                args.model_path = checkpoint.model_path
            compile_args(args, verbose=False)
            setup_inference_environment(self.options.seed)
            model, tokenizer = load_model_with_server(
                args.model_path,
                args.train,
                args.device,
                use_server=False,
                precision=args.precision,
                attn_implementation=args.attn_implementation,
                gamemode=mode,
                auto_select_gamemode_model=args.auto_select_gamemode_model,
            )
            preprocessor = Preprocessor(args, parallel=False)
            processor = Processor(args, model, tokenizer)
        runtime_source_identity = _runtime_source_identity(
            vendor_root, checkpoint.config_name
        )
        return _Runtime(
            inference_args=args,
            model=model,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            processor=processor,
            beatmap_class=Beatmap,
            generation_config_from_beatmap=generation_config_from_beatmap,
            torch=torch,
            checkpoint_identity=_checkpoint_identity(checkpoint, args, model),
            runtime_source_identity=runtime_source_identity,
        )

    def _score_checkpoint(
        self,
        checkpoint: WhiteboxCheckpoint,
        beatmap_path: Path,
        audio_path: Path,
        mode: int,
    ) -> dict[str, Any]:
        with _VENDOR_RUNTIME_LOCK:
            runtime, ephemeral = self._get_runtime(checkpoint, beatmap_path, audio_path, mode)
            try:
                with _working_directory(Path(self.options.vendor_root).resolve()):
                    return self._run_teacher_forcing(checkpoint, runtime, beatmap_path, audio_path)
            finally:
                if ephemeral:
                    del runtime
                    self._cleanup_cuda()

    @staticmethod
    def _replay_generation_policy_logits(
        torch: Any,
        processor: Any,
        tokenizer: Any,
        logits: Any,
        sequence_prompt: Any,
        start_token: int,
        args: Any,
        *,
        lookback_active: bool,
    ) -> Any:
        """Replay Mapperatorinator's custom pre-top-k/top-p processor chain.

        ``processor.model_forward`` has already applied CFG.  The remaining
        processors are instantiated from the vendored upstream module and run
        one sample/token at a time, preserving the stateful lookback behavior
        and upstream batch-1 conditional-temperature semantics.  The four
        pinned source snapshots currently ship byte-identical
        ``logit_processors.py`` files; the vendor tree is separately frozen.
        """

        from transformers import LogitsProcessorList, TemperatureLogitsWarper  # noqa: PLC0415
        from osuT5.osuT5.inference.logit_processors import (  # noqa: PLC0415
            ConditionalTemperatureLogitsWarper,
            LookbackBiasLogitsWarper,
            MonotonicTimeShiftLogitsProcessor,
            TimeshiftBias,
            get_beat_type_tokens,
            get_mania_type_tokens,
            get_scroll_speed_tokens,
        )
        from osuT5.osuT5.event import EventType  # noqa: PLC0415

        if len(logits) == 0:
            return logits.clone()
        types_first = bool(args.train.data.types_first)
        chain = LogitsProcessorList()
        chain.append(MonotonicTimeShiftLogitsProcessor(tokenizer))
        timeshift_bias = float(args.timeshift_bias)
        if timeshift_bias != 0:
            chain.append(
                TimeshiftBias(
                    timeshift_bias,
                    tokenizer.event_start[EventType.TIME_SHIFT],
                    tokenizer.event_end[EventType.TIME_SHIFT],
                )
            )
        if types_first:
            chain.append(
                ConditionalTemperatureLogitsWarper(
                    float(args.temperature),
                    float(args.timing_temperature),
                    float(args.mania_column_temperature),
                    float(args.taiko_hit_temperature),
                    types_first,
                    get_beat_type_tokens(tokenizer),
                    get_mania_type_tokens(tokenizer),
                    get_scroll_speed_tokens(tokenizer),
                )
            )
        else:
            chain.append(TemperatureLogitsWarper(float(args.temperature)))
        if lookback_active:
            chain.append(
                LookbackBiasLogitsWarper(
                    float(processor.lookback_time),
                    tokenizer,
                    types_first,
                    logits.device,
                )
            )

        prompt = sequence_prompt.to(device=logits.device, dtype=torch.long)
        processed: list[Any] = []
        for index in range(len(logits)):
            prefix_end = int(start_token) + index
            if prefix_end <= 0 or prefix_end > len(prompt):
                raise RuntimeError(
                    f"invalid source-policy prefix boundary {prefix_end} for prompt length {len(prompt)}"
                )
            prefix = prompt[:prefix_end].unsqueeze(0)
            scores = chain(prefix, logits[index].unsqueeze(0).clone())
            processed.append(scores.squeeze(0))
        return torch.stack(processed, dim=0)

    def _run_teacher_forcing(
        self,
        checkpoint: WhiteboxCheckpoint,
        runtime: _Runtime,
        beatmap_path: Path,
        audio_path: Path,
    ) -> dict[str, Any]:
        torch = runtime.torch
        processor = runtime.processor
        tokenizer = runtime.tokenizer
        args = runtime.inference_args
        beatmap = runtime.beatmap_class.from_path(beatmap_path)
        native_generation_config = runtime.generation_config_from_beatmap(
            beatmap, beatmap_path, tokenizer
        )
        generation_config, condition_view_audit = apply_condition_view(
            native_generation_config, self.options.condition_view
        )

        audio = runtime.preprocessor.load(str(audio_path))
        frames, frame_times, song_length = runtime.preprocessor.segment(audio)
        original_window_count = len(frame_times)
        original_indices = torch.arange(original_window_count)
        if self.options.start_ms is not None:
            keep = frame_times >= self.options.start_ms
            frames, frame_times, original_indices = frames[keep], frame_times[keep], original_indices[keep]
        if self.options.end_ms is not None:
            keep = frame_times < self.options.end_ms
            frames, frame_times, original_indices = frames[keep], frame_times[keep], original_indices[keep]
        if self.options.intervals_ms:
            if self.options.windows_per_interval is not None:
                selected = select_interval_window_indices(
                    [float(value) for value in frame_times.detach().cpu().tolist()],
                    float(processor.miliseconds_per_sequence),
                    self.options.intervals_ms,
                    self.options.windows_per_interval,
                )
                if selected:
                    indices = list(selected)
                    frames = frames[indices]
                    frame_times = frame_times[indices]
                    original_indices = original_indices[indices]
                else:
                    frames = frames[:0]
                    frame_times = frame_times[:0]
                    original_indices = original_indices[:0]
            else:
                keep = torch.zeros_like(frame_times, dtype=torch.bool)
                for interval_start, interval_end in self.options.intervals_ms:
                    keep |= (frame_times >= interval_start) & (frame_times < interval_end)
                frames, frame_times, original_indices = frames[keep], frame_times[keep], original_indices[keep]
        if self.options.max_windows is not None:
            frames = frames[: self.options.max_windows]
            frame_times = frame_times[: self.options.max_windows]
            original_indices = original_indices[: self.options.max_windows]
        if len(frame_times) == 0:
            raise ValueError("no audio windows remain after applying the requested range")

        gen_in, gen_out, required = processor._get_viable_template(
            in_context=runtime.inference_args.in_context,
            out_context=runtime.inference_args.output_type,
            extra_in_context=None,
            gamemode=int(beatmap.mode),
        )
        in_context = processor.get_in_context(
            in_context=gen_in,
            beatmap_path=str(beatmap_path),
            extra_in_context=None,
            song_length=song_length,
        )
        out_context = processor.get_out_context(
            out_context=gen_out,
            generation_config=generation_config,
            given_context=gen_out,
            beatmap_path=str(beatmap_path),
            extra_in_context=None,
            song_length=song_length,
            verbose=False,
        )
        model_kwargs = processor._get_model_cond_kwargs(generation_config)
        prompts, unconditioned_prompts, kwargses = processor._prepare_parallel_inputs(
            frame_times,
            song_length,
            in_context,
            out_context,
            model_kwargs,
            required,
        )

        all_records: list[dict[str, Any]] = []
        window_results: list[dict[str, Any]] = []
        with torch.inference_mode():
            for window_index, logits_all in _batched_teacher_forcing_logits(
                torch,
                processor,
                frames,
                prompts,
                unconditioned_prompts,
                kwargses,
                batch_size=self.options.forward_batch_size,
            ):
                frame_time_tensor = frame_times[window_index]
                original_index_tensor = original_indices[window_index]
                prompt = prompts[window_index]
                frame_time = float(frame_time_tensor.item())
                original_window_index = int(original_index_tensor.item())
                window_records: list[dict[str, Any]] = []
                context_results: dict[str, Any] = {}
                # Selection may contain disjoint intervals.  Overlap trimming
                # must follow the window's position in the *full song*, not
                # its ordinal in the selected subset.
                trim_lookback = original_window_index != 0
                trim_lookahead = original_window_index != original_window_count - 1
                scored_start = frame_time + processor.lookback_time if trim_lookback else frame_time
                scored_end = (
                    frame_time + processor.lookahead_max_time
                    if trim_lookahead
                    else frame_time + processor.miliseconds_per_sequence
                )

                for context_order, context in enumerate(out_context):
                    context_name = str(context["context_type"].value)
                    start_event, end_event = processor._get_events_time_range(
                        context["event_times"],
                        frame_time,
                        frame_time + processor.miliseconds_per_sequence,
                    )
                    events = context["events"][start_event:end_event]
                    event_times = context["event_times"][start_event:end_event]
                    tokens = processor._encode(events, frame_time).squeeze(0)
                    sequence_prompt = prompt.squeeze(0)
                    if processor.add_out_context_types:
                        # get_prompt() intentionally omits the context EOS from
                        # the final output context because generation is meant
                        # to continue there.  Non-strict lookup still respects
                        # an EOS for preceding contexts and uses prompt end for
                        # that final context.
                        start_token, end_token = processor._get_token_context(
                            sequence_prompt,
                            tokenizer.context_sos[context["context_type"]],
                            tokenizer.context_eos[context["context_type"]],
                            strict=False,
                        )
                    else:
                        start_token, end_token = processor._get_token_context(
                            sequence_prompt,
                            tokenizer.sos_id,
                            tokenizer.eos_id,
                        )
                    context_logits = logits_all[start_token - 1 : end_token - 1]
                    exact_policy_available = not (
                        bool(getattr(args, "super_timing", False))
                        and checkpoint.config_name.casefold().startswith("v29")
                        and context_name == "timing"
                    )
                    generation_policy_logits = None
                    if exact_policy_available:
                        generation_policy_logits = self._replay_generation_policy_logits(
                            torch,
                            processor,
                            tokenizer,
                            context_logits,
                            sequence_prompt,
                            start_token,
                            args,
                            lookback_active=(
                                original_window_index != 0
                                and bool(args.train.data.types_first)
                                and float(processor.lookback_time) > 0
                            ),
                        )
                    prompt_truncated_events = 0
                    if len(context_logits) < len(events):
                        # Processor.get_prompts() repeatedly halves
                        # max_token_length and get_context_tokens() keeps the
                        # *suffix* when a dense context would exceed the model
                        # decoder length.  Mirror that exact behavior so very
                        # dense but valid charts remain scoreable and disclose
                        # the lost coverage instead of failing alignment.
                        prompt_truncated_events = len(events) - len(context_logits)
                        events = events[prompt_truncated_events:]
                        event_times = event_times[prompt_truncated_events:]
                        tokens = tokens[prompt_truncated_events:]
                    if len(context_logits) != len(events):
                        raise RuntimeError(
                            f"teacher-forcing alignment mismatch for {context_name} in window {window_index}: "
                            f"{len(context_logits)} logits versus {len(events)} events"
                        )
                    keep_start, keep_end = processor._get_events_time_range(
                        event_times,
                        scored_start,
                        scored_end,
                    )
                    kept_tokens = tokens[keep_start:keep_end]
                    kept_logits = context_logits[keep_start:keep_end]
                    kept_policy_logits = (
                        generation_policy_logits[keep_start:keep_end]
                        if generation_policy_logits is not None
                        else None
                    )
                    kept_times = event_times[keep_start:keep_end]
                    records = self._tensor_records(
                        torch,
                        kept_logits,
                        kept_tokens,
                        tokenizer,
                        generation_policy_logits=kept_policy_logits,
                        generation_top_p=float(args.top_p),
                        generation_top_k=int(args.top_k),
                        window_index=window_index,
                        context=context_name,
                        context_order=context_order,
                        prompt_offset=start_token + keep_start,
                        event_times=kept_times,
                    )
                    window_records.extend(records)
                    context_results[context_name] = {
                        "encoded_events_in_audio_window": len(events) + prompt_truncated_events,
                        "events_dropped_by_decoder_prompt_limit": prompt_truncated_events,
                        "scored_events_after_overlap_trim": len(records),
                        "generation_policy_replay": {
                            "available": exact_policy_available,
                            "reason": (
                                None
                                if exact_policy_available
                                else "V29 super_timing uses a separate multi-offset beam/top-k pipeline"
                            ),
                        },
                        "summary": _metric_bundle(records),
                        "families": {
                            family: _metric_bundle([record for record in records if record["family"] == family])
                            for family in sorted({record["family"] for record in records})
                        },
                    }

                window_records.sort(key=lambda record: (record["context_order"], record["prompt_position"]))
                all_records.extend(window_records)
                window_result: dict[str, Any] = {
                    "window_index": window_index,
                    "source_audio_window_index": original_window_index,
                    "audio_window_start_ms": int(round(frame_time)),
                    "audio_window_end_ms": int(round(frame_time + processor.miliseconds_per_sequence)),
                    "scored_interval_start_ms": int(round(scored_start)),
                    "scored_interval_end_ms": int(round(scored_end)),
                    "token_count": len(window_records),
                    "summary": _metric_bundle(window_records),
                    "families": {
                        family: _metric_bundle(
                            [record for record in window_records if record["family"] == family]
                        )
                        for family in sorted({record["family"] for record in window_records})
                    },
                    "contexts": context_results,
                }
                if self.options.include_token_details:
                    limit = self.options.max_token_details_per_window
                    window_result["token_details"] = window_records[:limit]
                    window_result["token_details_truncated"] = max(0, len(window_records) - limit)
                window_results.append(window_result)

        aggregate = aggregate_token_records(all_records, total_windows=len(window_results))
        return {
            "checkpoint": checkpoint.display_name,
            "config_name": checkpoint.config_name,
            "status": "ok",
            "model_path": str(args.model_path),
            "checkpoint_identity": runtime.checkpoint_identity,
            "runtime_source_identity": runtime.runtime_source_identity,
            "model_generation_settings": {
                "temperature": float(args.temperature),
                "timing_temperature": float(args.timing_temperature),
                "mania_column_temperature": float(args.mania_column_temperature),
                "taiko_hit_temperature": float(args.taiko_hit_temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "cfg_scale": float(args.cfg_scale),
                "timeshift_bias": float(args.timeshift_bias),
                "types_first": bool(args.train.data.types_first),
                "lookback_time_ms": float(processor.lookback_time),
                "lookahead_time_ms": float(processor.lookahead_time),
                "super_timing": bool(getattr(args, "super_timing", False)),
                "upstream_parallel": bool(getattr(args, "parallel", False)),
                "note": (
                    "Raw teacher-forced logits include model_forward's CFG processor. Family-conditioned "
                    "metrics deliberately use the separately reported audit temperature/top_p and do not "
                    "claim to replay the stateful lookback processor."
                ),
            },
            "condition_view": condition_view_audit,
            "audit_distribution_settings": {
                "temperature": self.options.temperature,
                "top_p": self.options.top_p,
                "conditioning": "observed tokenizer EventType range",
                "runtime_source_identity_sha256": runtime.runtime_source_identity[
                    "identity_sha256"
                ],
                "generation_policy_profile": {
                    "schema_version": "mapperatorinator-generation-policy/v1",
                    "replay_mode": "sequential_batch_1_prefix",
                    "temperature": float(args.temperature),
                    "timing_temperature": float(args.timing_temperature),
                    "mania_column_temperature": float(
                        args.mania_column_temperature
                    ),
                    "taiko_hit_temperature": float(args.taiko_hit_temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                    "cfg_scale": float(args.cfg_scale),
                    "timeshift_bias": float(args.timeshift_bias),
                    "types_first": bool(args.train.data.types_first),
                    "lookback_time_ms": float(processor.lookback_time),
                    "lookahead_time_ms": float(processor.lookahead_time),
                    "super_timing": bool(getattr(args, "super_timing", False)),
                    "v29_super_timing_policy": (
                        "timing_context_unavailable"
                        if bool(getattr(args, "super_timing", False))
                        and checkpoint.config_name.casefold().startswith("v29")
                        else "not_applicable"
                    ),
                },
            },
            "song_length_ms": float(song_length),
            "window_count": len(window_results),
            "source_audio_window_count": original_window_count,
            **aggregate,
            "windows": window_results,
        }

    def _tensor_records(
        self,
        torch: Any,
        logits: Any,
        targets: Any,
        tokenizer: Any,
        *,
        generation_policy_logits: Any | None,
        generation_top_p: float,
        generation_top_k: int,
        window_index: int,
        context: str,
        context_order: int,
        prompt_offset: int,
        event_times: Sequence[float],
    ) -> list[dict[str, Any]]:
        if len(targets) == 0:
            return []
        if not 0 < generation_top_p <= 1:
            raise ValueError("generation_top_p must be in (0, 1]")
        if generation_top_k < 0:
            raise ValueError("generation_top_k cannot be negative")
        logits = logits.to(torch.float32)
        targets = targets.to(torch.long)
        row_indices = torch.arange(len(targets))
        target_column = targets.unsqueeze(1)
        raw_log_probabilities = torch.log_softmax(logits, dim=-1)
        raw_probabilities = raw_log_probabilities.exp()
        raw_target_log_probability = raw_log_probabilities.gather(1, target_column).squeeze(1)
        raw_target_logits = logits.gather(1, target_column).squeeze(1)
        raw_rank = (logits > raw_target_logits.unsqueeze(1)).sum(dim=1) + 1
        raw_entropy = torch.where(
            raw_probabilities > 0,
            -raw_probabilities * raw_log_probabilities,
            torch.zeros_like(raw_probabilities),
        ).sum(dim=1)
        raw_expected_log_probability = torch.where(
            raw_probabilities > 0,
            raw_probabilities * raw_log_probabilities,
            torch.zeros_like(raw_probabilities),
        ).sum(dim=1)
        raw_log_probability_variance = torch.where(
            raw_probabilities > 0,
            raw_probabilities
            * (raw_log_probabilities - raw_expected_log_probability.unsqueeze(1)).square(),
            torch.zeros_like(raw_probabilities),
        ).sum(dim=1).clamp_min(0)
        raw_standard_deviation = raw_log_probability_variance.sqrt()
        competitor_logits = logits.clone()
        competitor_logits.scatter_(1, target_column, -torch.inf)
        raw_margin = raw_target_logits - competitor_logits.max(dim=1).values

        policy: dict[str, Any] | None = None
        if generation_policy_logits is not None:
            if tuple(generation_policy_logits.shape) != tuple(logits.shape):
                raise RuntimeError(
                    "generation-policy logits do not align with raw teacher-forcing logits"
                )
            policy_logits = generation_policy_logits.to(torch.float32)
            policy_log_probabilities = torch.log_softmax(policy_logits, dim=-1)
            policy_probabilities = policy_log_probabilities.exp()
            policy_target_log_probability = policy_log_probabilities.gather(
                1, target_column
            ).squeeze(1)
            policy_target_logits = policy_logits.gather(1, target_column).squeeze(1)
            policy_rank = (
                (policy_logits > policy_target_logits.unsqueeze(1)).sum(dim=1) + 1
            )
            policy_entropy = torch.where(
                policy_probabilities > 0,
                -policy_probabilities * policy_log_probabilities,
                torch.zeros_like(policy_probabilities),
            ).sum(dim=1)
            policy_expected_log_probability = torch.where(
                policy_probabilities > 0,
                policy_probabilities * policy_log_probabilities,
                torch.zeros_like(policy_probabilities),
            ).sum(dim=1)
            policy_log_probability_variance = torch.where(
                policy_probabilities > 0,
                policy_probabilities
                * (
                    policy_log_probabilities
                    - policy_expected_log_probability.unsqueeze(1)
                ).square(),
                torch.zeros_like(policy_probabilities),
            ).sum(dim=1).clamp_min(0)
            policy_standard_deviation = policy_log_probability_variance.sqrt()
            policy_competitors = policy_logits.clone()
            policy_competitors.scatter_(1, target_column, -torch.inf)
            policy_margin = policy_target_logits - policy_competitors.max(dim=1).values
            policy = {
                "logits": policy_logits,
                "log_probabilities": policy_log_probabilities,
                "probabilities": policy_probabilities,
                "target_log_probability": policy_target_log_probability,
                "rank": policy_rank,
                "entropy": policy_entropy,
                "expected_log_probability": policy_expected_log_probability,
                "variance": policy_log_probability_variance,
                "standard_deviation": policy_standard_deviation,
                "margin": policy_margin,
            }

        descriptors: list[tuple[str, int, int, Any]] = []
        target_list = targets.tolist()
        for token in target_list:
            event = tokenizer.decode(int(token))
            descriptors.append(
                (
                    str(event.type.value),
                    int(tokenizer.event_start[event.type]),
                    int(tokenizer.event_end[event.type]),
                    event.value,
                )
            )

        records: list[dict[str, Any]] = []
        for index, (token, descriptor) in enumerate(zip(target_list, descriptors)):
            family, family_start, family_end, event_value = descriptor
            raw = {
                "probability": float(raw_target_log_probability[index].exp().item()),
                "log_probability": float(raw_target_log_probability[index].item()),
                "nll": float(-raw_target_log_probability[index].item()),
                "rank": int(raw_rank[index].item()),
                "entropy": float(raw_entropy[index].item()),
                "margin": float(raw_margin[index].item()),
                "expected_log_probability": float(
                    raw_expected_log_probability[index].item()
                ),
                "log_probability_variance": float(
                    raw_log_probability_variance[index].item()
                ),
                "curvature_z": (
                    float(
                        (
                            (
                                raw_target_log_probability[index]
                                - raw_expected_log_probability[index]
                            )
                            / raw_standard_deviation[index]
                        ).item()
                    )
                    if float(raw_standard_deviation[index].item()) > 1e-12
                    else None
                ),
                "curvature_defined": bool(
                    float(raw_standard_deviation[index].item()) > 1e-12
                ),
            }
            raw["log_rank"] = math.log(raw["rank"])
            generation_policy: dict[str, Any] | None = None
            if policy is not None:
                policy_logits_row = policy["logits"][index]
                top_k_logits = policy_logits_row.clone()
                vocabulary_size = int(top_k_logits.numel())
                effective_top_k = min(int(generation_top_k), vocabulary_size)
                if effective_top_k > 0 and effective_top_k < vocabulary_size:
                    threshold = torch.topk(top_k_logits, effective_top_k).values[-1]
                    top_k_logits[top_k_logits < threshold] = -torch.inf

                # Exact Transformers TopPLogitsWarper lower-tail semantics,
                # applied after top-k over the complete processed vocabulary.
                sorted_logits, sorted_indices = torch.sort(
                    top_k_logits, descending=False
                )
                cumulative_ascending = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove_sorted = cumulative_ascending <= (1 - generation_top_p)
                remove_sorted[-1:] = False  # min_tokens_to_keep=1
                remove = torch.zeros_like(remove_sorted).scatter(
                    0, sorted_indices, remove_sorted
                )
                final_logits = top_k_logits.masked_fill(remove, -torch.inf)
                support = torch.isfinite(final_logits)
                final_probabilities = final_logits.softmax(dim=-1)
                top_k_probabilities = top_k_logits.softmax(dim=-1)
                local_target = int(token)
                target_in_support = bool(support[local_target].item())
                support_mass = float(top_k_probabilities[support].sum().item())
                sample_probability = (
                    float(final_probabilities[local_target].item())
                    if target_in_support
                    else 0.0
                )
                descending_probs, descending_indices = torch.sort(
                    top_k_probabilities, descending=True
                )
                target_positions = (
                    descending_indices == local_target
                ).nonzero(as_tuple=True)[0]
                target_position = int(target_positions[0].item())
                mass_before = float(descending_probs[:target_position].sum().item())

                # Fast-DetectGPT's general p/q form: p is the raw CFG source
                # model and q is the final, renormalized generation policy.
                raw_scoring_log_probs = raw_log_probabilities[index]
                expected_raw_scoring_log_probability = torch.where(
                    final_probabilities > 0,
                    final_probabilities * raw_scoring_log_probs,
                    torch.zeros_like(final_probabilities),
                ).sum()
                raw_scoring_log_probability_variance = torch.where(
                    final_probabilities > 0,
                    final_probabilities
                    * (
                        raw_scoring_log_probs
                        - expected_raw_scoring_log_probability
                    ).square(),
                    torch.zeros_like(final_probabilities),
                ).sum().clamp_min(0)

                policy_std = float(policy["standard_deviation"][index].item())
                policy_log_probability = float(
                    policy["target_log_probability"][index].item()
                )
                policy_rank = int(policy["rank"][index].item())
                generation_policy = {
                    "probability": (
                        math.exp(policy_log_probability)
                        if math.isfinite(policy_log_probability)
                        else 0.0
                    ),
                    "log_probability": policy_log_probability,
                    "nll": (
                        -policy_log_probability
                        if math.isfinite(policy_log_probability)
                        else None
                    ),
                    "rank": policy_rank,
                    "log_rank": math.log(policy_rank),
                    "entropy": float(policy["entropy"][index].item()),
                    "margin": float(policy["margin"][index].item()),
                    "expected_log_probability": float(
                        policy["expected_log_probability"][index].item()
                    ),
                    "log_probability_variance": float(
                        policy["variance"][index].item()
                    ),
                    "curvature_z": (
                        float(
                            (
                                (
                                    policy["target_log_probability"][index]
                                    - policy["expected_log_probability"][index]
                                )
                                / policy["standard_deviation"][index]
                            ).item()
                        )
                        if policy_std > 1e-12
                        and math.isfinite(policy_log_probability)
                        else None
                    ),
                    "curvature_defined": bool(
                        policy_std > 1e-12 and math.isfinite(policy_log_probability)
                    ),
                    "in_sampling_support": target_in_support,
                    "support_size": int(support.sum().item()),
                    "support_mass_before_renormalization": support_mass,
                    "sample_probability": sample_probability,
                    "sample_nll": (
                        -math.log(sample_probability)
                        if sample_probability > 0
                        else None
                    ),
                    "target_cumulative_mass_before": mass_before,
                    "target_cumulative_mass_through": (
                        mass_before
                        + float(top_k_probabilities[local_target].item())
                    ),
                    "removed_mass": float(top_k_probabilities[~support].sum().item()),
                    "top_k": int(generation_top_k),
                    "top_p": float(generation_top_p),
                    "raw_scoring_log_probability": float(
                        raw_target_log_probability[index].item()
                    ),
                    "expected_raw_scoring_log_probability": float(
                        expected_raw_scoring_log_probability.item()
                    ),
                    "raw_scoring_log_probability_variance": float(
                        raw_scoring_log_probability_variance.item()
                    ),
                }
            family_base_logits = logits[index, family_start:family_end]
            family_logits = family_base_logits / self.options.temperature
            local_target = int(token) - family_start
            family_log_probs = torch.log_softmax(family_logits, dim=-1)
            family_probs = family_log_probs.exp()
            target_log_probability = family_log_probs[local_target]
            target_probability = family_probs[local_target]
            family_entropy = torch.where(
                family_probs > 0,
                -family_probs * family_log_probs,
                torch.zeros_like(family_probs),
            ).sum()
            expected_log_probability = torch.where(
                family_probs > 0,
                family_probs * family_log_probs,
                torch.zeros_like(family_probs),
            ).sum()
            log_probability_variance = torch.where(
                family_probs > 0,
                family_probs * (family_log_probs - expected_log_probability).square(),
                torch.zeros_like(family_probs),
            ).sum().clamp_min(0)
            standard_deviation = log_probability_variance.sqrt()
            curvature_z = (
                float(((target_log_probability - expected_log_probability) / standard_deviation).item())
                if float(standard_deviation.item()) > 1e-12
                else None
            )
            family_rank = int((family_logits > family_logits[local_target]).sum().item()) + 1
            if len(family_logits) > 1:
                competitors = family_logits.clone()
                competitors[local_target] = -torch.inf
                family_margin = float((family_logits[local_target] - competitors.max()).item())
            else:
                family_margin = None

            # Exact Transformers TopPLogitsWarper semantics used by Mapperatorinator.
            sorted_logits, sorted_indices = torch.sort(family_logits, descending=False)
            cumulative_ascending = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove_sorted = cumulative_ascending <= (1 - self.options.top_p)
            remove_sorted[-1:] = False  # min_tokens_to_keep=1
            remove = torch.zeros_like(remove_sorted).scatter(0, sorted_indices, remove_sorted)
            keep = ~remove
            descending_probs, descending_indices = torch.sort(family_probs, descending=True)
            target_position = int((descending_indices == local_target).nonzero(as_tuple=True)[0][0].item())
            mass_before = float(descending_probs[:target_position].sum().item())
            family_conditioned = {
                "probability": float(target_probability.item()),
                "log_probability": float(target_log_probability.item()),
                "nll": float(-target_log_probability.item()),
                "rank": family_rank,
                "log_rank": math.log(family_rank),
                "entropy": float(family_entropy.item()),
                "margin": family_margin,
                "expected_log_probability": float(expected_log_probability.item()),
                "log_probability_variance": float(log_probability_variance.item()),
                "curvature_z": curvature_z,
                "curvature_defined": curvature_z is not None,
                "in_nucleus": bool(keep[local_target].item()),
                "nucleus_size": int(keep.sum().item()),
                "nucleus_mass": float(family_probs[keep].sum().item()),
                "target_cumulative_mass_before": mass_before,
                "target_cumulative_mass_through": mass_before + float(target_probability.item()),
                "lower_tail_removed_mass": float(family_probs[remove].sum().item()),
            }
            base_log_normalizer = torch.logsumexp(family_base_logits, dim=-1)
            base_target_log_probability = (
                family_base_logits[local_target] - base_log_normalizer
            )
            local_normalization: dict[str, dict[str, float]] = {}
            for setting, temperature in TEMPERATURE_NORMALIZATION_SETTINGS:
                inverse_temperature = 1.0 / temperature
                log_temperature_normalizer = (
                    torch.logsumexp(
                        family_base_logits * inverse_temperature, dim=-1
                    )
                    - inverse_temperature * base_log_normalizer
                )
                temp_test = log_temperature_normalizer - (
                    inverse_temperature - 1.0
                ) * base_target_log_probability
                local_normalization[setting] = {
                    "temperature": temperature,
                    "target_base_log_probability": float(
                        base_target_log_probability.item()
                    ),
                    "log_temperature_normalizer": float(
                        log_temperature_normalizer.item()
                    ),
                    "temp_test": float(temp_test.item()),
                }
            records.append(
                {
                    "window_index": window_index,
                    "context": context,
                    "context_order": context_order,
                    "prompt_position": prompt_offset + index,
                    "event_time_ms": float(event_times[index]),
                    "token_id": int(token),
                    "family": family,
                    "event_value": json_safe(event_value),
                    "family_token_start_inclusive": family_start,
                    "family_token_end_exclusive": family_end,
                    "raw": raw,
                    "family_conditioned": family_conditioned,
                    "generation_policy": generation_policy,
                    "local_normalization": local_normalization,
                }
            )
        return records

    @staticmethod
    def _cleanup_cuda() -> None:
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def score_whitebox(
    beatmap_path: str | Path,
    audio_path: str | Path,
    *,
    options: WhiteboxOptions | None = None,
) -> dict[str, Any]:
    """One-shot convenience wrapper that always releases cached checkpoints."""

    with WhiteboxEngine(options) as engine:
        return engine.score(beatmap_path, audio_path)


__all__ = [
    "SCHEMA_VERSION",
    "SYMMETRIC_CONDITION_VIEW",
    "TEMPERATURE_NORMALIZATION_SETTINGS",
    "WhiteboxCheckpoint",
    "WhiteboxOptions",
    "WhiteboxEngine",
    "aggregate_token_records",
    "apply_condition_view",
    "distribution_metrics",
    "json_safe",
    "nucleus_statistics",
    "quantile_summary",
    "run_summary",
    "score_whitebox",
    "select_interval_window_indices",
    "temperature_normalization_statistics",
]
