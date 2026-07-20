"""Canonical, fail-closed white-box candidate-search protocol bindings.

The expensive white-box statistic is a *searched* statistic: a cheap content
model proposes several intervals and the discriminator keeps high-scoring
windows across those proposals.  Human-null calibration is valid only when
production performs exactly the same label-free proposal and window search.

This module intentionally uses only the Python standard library.  It can be
used by the lightweight scoring runtime, corpus scripts, and artifact auditors
without importing PyTorch or the Mapperatorinator checkout.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


SEARCH_PROTOCOL_SCHEMA_VERSION = "osu-ai-detector.whitebox-search-protocol/v1"
CANDIDATE_SELECTION_SCHEMA_VERSION = 2
CANDIDATE_SELECTION_METHOD = "first + last + densest + non-redundant top content windows"
SELECTION_POLICY_SCHEMA_VERSION = 1
SELECTION_POLICY_METHOD = "nearest_audio_contexts_to_each_merged_interval_centre"
INTERVAL_BINDING_POLICY = "exact_ordered_candidate_intervals_equal_result_settings_intervals"

SELECTION_PARAMETER_KEYS = frozenset(
    {"content_windows", "edge_duration_ms", "dense_duration_ms", "join_gap_ms"}
)
SELECTION_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "windows_per_interval",
        "overlap_required",
        "deduplicate_overlapping_proposals",
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], name: str) -> None:
    observed = {str(key) for key in value}
    missing = sorted(set(expected) - observed)
    extra = sorted(observed - set(expected))
    if missing or extra:
        raise ValueError(f"{name} keys are not canonical: missing={missing}, extra={extra}")


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty_text(value: Any, name: str, *, casefold: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    return result.casefold() if casefold else result


def _sha256(value: Any, name: str) -> str:
    digest = _nonempty_text(value, name, casefold=True)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return digest


def _intervals(value: Any, name: str) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must contain at least one interval")
    result: list[list[int]] = []
    previous: tuple[int, int] | None = None
    for index, raw in enumerate(value):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
            raise ValueError(f"{name}[{index}] must be a [start_ms, end_ms] pair")
        start = _integer(raw[0], f"{name}[{index}][0]", minimum=0)
        end = _integer(raw[1], f"{name}[{index}][1]", minimum=1)
        if start >= end:
            raise ValueError(f"{name}[{index}] must satisfy 0 <= start_ms < end_ms")
        current = (start, end)
        if previous is not None and current[0] < previous[1]:
            raise ValueError(f"{name} must be ordered and non-overlapping")
        result.append([start, end])
        previous = current
    return result


def canonical_selection_policy(value: Any) -> dict[str, Any]:
    """Validate and canonicalize the audio-window selection policy."""

    policy = _mapping(value, "candidate_interval_selection.selection_policy")
    _exact_keys(policy, SELECTION_POLICY_KEYS, "candidate_interval_selection.selection_policy")
    if policy.get("schema_version") != SELECTION_POLICY_SCHEMA_VERSION:
        raise ValueError("candidate selection policy uses an unsupported schema_version")
    if policy.get("method") != SELECTION_POLICY_METHOD:
        raise ValueError("candidate selection policy uses an unsupported method")
    if policy.get("overlap_required") is not True:
        raise ValueError("candidate selection policy must require interval overlap")
    if policy.get("deduplicate_overlapping_proposals") is not True:
        raise ValueError("candidate selection policy must deduplicate overlapping proposals")
    return {
        "schema_version": SELECTION_POLICY_SCHEMA_VERSION,
        "method": SELECTION_POLICY_METHOD,
        "windows_per_interval": _integer(
            policy.get("windows_per_interval"),
            "candidate_interval_selection.selection_policy.windows_per_interval",
            minimum=1,
        ),
        "overlap_required": True,
        "deduplicate_overlapping_proposals": True,
    }


def canonical_search_protocol(value: Any) -> dict[str, Any]:
    """Return the unique JSON representation of a white-box search recipe.

    Per-map interval coordinates are deliberately excluded: they are data
    produced by the bound content model.  :func:`derive_search_protocol`
    separately requires those coordinates to exactly match the intervals the
    engine actually received in ``result.settings``.
    """

    protocol = _mapping(value, "search protocol")
    _exact_keys(
        protocol,
        {"schema_version", "candidate_interval_selection", "execution"},
        "search protocol",
    )
    if protocol.get("schema_version") != SEARCH_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("search protocol uses an unsupported schema_version")

    selection = _mapping(
        protocol.get("candidate_interval_selection"),
        "search protocol candidate_interval_selection",
    )
    _exact_keys(
        selection,
        {
            "schema_version",
            "method",
            "label_free",
            "parameters",
            "selection_policy",
            "content_model_id",
            "content_model_sha256",
        },
        "search protocol candidate_interval_selection",
    )
    if selection.get("schema_version") != CANDIDATE_SELECTION_SCHEMA_VERSION:
        raise ValueError("candidate interval selection uses an unsupported schema_version")
    if selection.get("method") != CANDIDATE_SELECTION_METHOD:
        raise ValueError("candidate interval selection uses an unsupported method")
    if selection.get("label_free") is not True:
        raise ValueError("candidate interval selection must be explicitly label-free")
    parameters = _mapping(selection.get("parameters"), "candidate interval selection parameters")
    _exact_keys(parameters, SELECTION_PARAMETER_KEYS, "candidate interval selection parameters")
    canonical_parameters = {
        "content_windows": _integer(parameters.get("content_windows"), "content_windows", minimum=0),
        "edge_duration_ms": _integer(parameters.get("edge_duration_ms"), "edge_duration_ms", minimum=1),
        "dense_duration_ms": _integer(parameters.get("dense_duration_ms"), "dense_duration_ms", minimum=1),
        "join_gap_ms": _integer(parameters.get("join_gap_ms"), "join_gap_ms", minimum=0),
    }
    policy = canonical_selection_policy(selection.get("selection_policy"))

    execution = _mapping(protocol.get("execution"), "search protocol execution")
    _exact_keys(
        execution,
        {
            "result_schema_version",
            "precision",
            "attention_implementation",
            "start_ms",
            "end_ms",
            "max_windows",
            "windows_per_interval",
            "forward_batch_size",
            "interval_binding_policy",
        },
        "search protocol execution",
    )
    for field in ("start_ms", "end_ms", "max_windows"):
        if execution.get(field) is not None:
            raise ValueError(f"search protocol execution.{field} must be null")
    windows_per_interval = _integer(
        execution.get("windows_per_interval"),
        "search protocol execution.windows_per_interval",
        minimum=1,
    )
    if windows_per_interval != policy["windows_per_interval"]:
        raise ValueError("selection policy and execution windows_per_interval do not match")
    forward_batch_size = _integer(
        execution.get("forward_batch_size"),
        "search protocol execution.forward_batch_size",
        minimum=1,
    )
    if execution.get("interval_binding_policy") != INTERVAL_BINDING_POLICY:
        raise ValueError("search protocol uses an unsupported interval binding policy")

    return {
        "schema_version": SEARCH_PROTOCOL_SCHEMA_VERSION,
        "candidate_interval_selection": {
            "schema_version": CANDIDATE_SELECTION_SCHEMA_VERSION,
            "method": CANDIDATE_SELECTION_METHOD,
            "label_free": True,
            "parameters": canonical_parameters,
            "selection_policy": policy,
            "content_model_id": _nonempty_text(
                selection.get("content_model_id"), "candidate interval selection content_model_id"
            ),
            "content_model_sha256": _sha256(
                selection.get("content_model_sha256"),
                "candidate interval selection content_model_sha256",
            ),
        },
        "execution": {
            "result_schema_version": _nonempty_text(
                execution.get("result_schema_version"), "white-box result schema_version"
            ),
            "precision": _nonempty_text(execution.get("precision"), "white-box precision", casefold=True),
            "attention_implementation": _nonempty_text(
                execution.get("attention_implementation"),
                "white-box attention implementation",
                casefold=True,
            ),
            "start_ms": None,
            "end_ms": None,
            "max_windows": None,
            "windows_per_interval": windows_per_interval,
            "forward_batch_size": forward_batch_size,
            "interval_binding_policy": INTERVAL_BINDING_POLICY,
        },
    }


def derive_search_protocol(whitebox_result: Any) -> dict[str, Any]:
    """Derive a protocol from one result and validate its actual search.

    This is intentionally fail-closed.  Missing settings, a custom range, a
    global window cap, mismatched candidate/actual intervals, or incomplete
    model/runtime provenance all raise :class:`ValueError`.
    """

    outer = _mapping(whitebox_result, "white-box result")
    payload = outer
    nested = outer.get("result")
    if "settings" not in outer and isinstance(nested, Mapping):
        payload = nested
    selection = _mapping(
        payload.get("candidate_interval_selection"),
        "white-box result candidate_interval_selection",
    )
    settings = _mapping(payload.get("settings"), "white-box result settings")

    selected_intervals = _intervals(
        selection.get("intervals_ms"), "candidate_interval_selection.intervals_ms"
    )
    if "intervals_ms" not in settings:
        raise ValueError("white-box result settings.intervals_ms is missing")
    actual_intervals = _intervals(settings.get("intervals_ms"), "result.settings.intervals_ms")
    if actual_intervals != selected_intervals:
        raise ValueError(
            "candidate_interval_selection.intervals_ms does not exactly match "
            "result.settings.intervals_ms"
        )
    for field in ("start_ms", "end_ms", "max_windows"):
        if field not in settings:
            raise ValueError(f"white-box result settings.{field} is missing")
        if settings.get(field) is not None:
            raise ValueError(f"white-box result settings.{field} must be null for calibrated search")
    if "windows_per_interval" not in settings:
        raise ValueError("white-box result settings.windows_per_interval is missing")
    actual_windows_per_interval = _integer(
        settings.get("windows_per_interval"), "result.settings.windows_per_interval", minimum=1
    )

    protocol = {
        "schema_version": SEARCH_PROTOCOL_SCHEMA_VERSION,
        "candidate_interval_selection": {
            "schema_version": selection.get("schema_version"),
            "method": selection.get("method"),
            "label_free": selection.get("label_free"),
            "parameters": selection.get("parameters"),
            "selection_policy": selection.get("selection_policy"),
            "content_model_id": selection.get("content_model_id"),
            "content_model_sha256": selection.get("content_model_sha256"),
        },
        "execution": {
            "result_schema_version": payload.get("schema_version"),
            "precision": settings.get("precision"),
            "attention_implementation": settings.get("attention_implementation"),
            "start_ms": settings.get("start_ms"),
            "end_ms": settings.get("end_ms"),
            "max_windows": settings.get("max_windows"),
            "windows_per_interval": actual_windows_per_interval,
            "forward_batch_size": settings.get("forward_batch_size"),
            "interval_binding_policy": INTERVAL_BINDING_POLICY,
        },
    }
    canonical = canonical_search_protocol(protocol)
    if canonical["candidate_interval_selection"]["selection_policy"]["windows_per_interval"] != actual_windows_per_interval:
        raise ValueError(
            "candidate selection policy windows_per_interval does not match result.settings"
        )
    return canonical


def search_protocol_sha256(value: Any) -> str:
    """Hash a canonical white-box search protocol."""

    canonical = canonical_search_protocol(value)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def copy_canonical_search_protocol(value: Any) -> dict[str, Any]:
    """Explicit deep-copy helper for callers embedding the protocol in reports."""

    return copy.deepcopy(canonical_search_protocol(value))


__all__ = [
    "CANDIDATE_SELECTION_METHOD",
    "CANDIDATE_SELECTION_SCHEMA_VERSION",
    "INTERVAL_BINDING_POLICY",
    "SEARCH_PROTOCOL_SCHEMA_VERSION",
    "SELECTION_POLICY_METHOD",
    "SELECTION_POLICY_SCHEMA_VERSION",
    "canonical_search_protocol",
    "canonical_selection_policy",
    "copy_canonical_search_protocol",
    "derive_search_protocol",
    "search_protocol_sha256",
]
