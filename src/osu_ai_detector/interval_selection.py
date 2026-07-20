"""Label-free candidate interval selection for expensive white-box scoring."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from .parser import Beatmap
from .whitebox_protocol import (
    CANDIDATE_SELECTION_METHOD,
    CANDIDATE_SELECTION_SCHEMA_VERSION,
    SELECTION_POLICY_METHOD,
    SELECTION_POLICY_SCHEMA_VERSION,
    canonical_selection_policy,
)


DEFAULT_CONTENT_WINDOWS = 3
DEFAULT_EDGE_DURATION_MS = 30_000
DEFAULT_DENSE_DURATION_MS = 45_000
DEFAULT_JOIN_GAP_MS = 4_000


@dataclasses.dataclass(frozen=True)
class SelectedInterval:
    start_ms: int
    end_ms: int
    reasons: tuple[str, ...]


def densest_interval(beatmap: Beatmap, duration_ms: int = 45_000) -> tuple[int, int] | None:
    times = sorted(obj.time for obj in beatmap.hit_objects)
    if not times:
        return None
    end_index = 0
    best_start, best_count = times[0], 0
    for start_index, time_value in enumerate(times):
        while end_index < len(times) and times[end_index] < time_value + duration_ms:
            end_index += 1
        if end_index - start_index > best_count:
            best_count = end_index - start_index
            best_start = time_value
    start = max(best_start - 1_500, 0)
    return start, start + duration_ms


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def merge_intervals(
    intervals: Iterable[tuple[int, int, str]],
    *,
    join_gap_ms: int = 4_000,
) -> tuple[SelectedInterval, ...]:
    ordered = sorted((int(start), int(end), str(reason)) for start, end, reason in intervals if start < end)
    if not ordered:
        return ()
    result: list[SelectedInterval] = []
    start, end, reason = ordered[0]
    reasons = {reason}
    for next_start, next_end, next_reason in ordered[1:]:
        if next_start <= end + join_gap_ms:
            end = max(end, next_end)
            reasons.add(next_reason)
        else:
            result.append(SelectedInterval(start, end, tuple(sorted(reasons))))
            start, end, reasons = next_start, next_end, {next_reason}
    result.append(SelectedInterval(start, end, tuple(sorted(reasons))))
    return tuple(result)


def select_whitebox_intervals(
    beatmap: Beatmap,
    content_analysis: dict[str, Any] | None,
    *,
    content_windows: int = DEFAULT_CONTENT_WINDOWS,
    edge_duration_ms: int = DEFAULT_EDGE_DURATION_MS,
    dense_duration_ms: int = DEFAULT_DENSE_DURATION_MS,
    join_gap_ms: int = DEFAULT_JOIN_GAP_MS,
) -> dict[str, Any]:
    """Select deterministic intervals without looking at an AI label.

    The first and last intervals protect against a content model that is weak
    on a new generator.  A density interval catches high-object sections, and
    up to three non-redundant content-score windows provide an inexpensive
    learned proposal.  This exact search is later applied to every human-null
    calibration chart, so selection multiplicity is included in its p-value.
    """

    base = {
        "schema_version": CANDIDATE_SELECTION_SCHEMA_VERSION,
        "method": CANDIDATE_SELECTION_METHOD,
        "label_free": True,
        "content_model_id": (content_analysis or {}).get("model_id"),
        "content_model_calibrated": bool((content_analysis or {}).get("calibrated")),
        "parameters": {
            "content_windows": content_windows,
            "edge_duration_ms": edge_duration_ms,
            "dense_duration_ms": dense_duration_ms,
            "join_gap_ms": join_gap_ms,
        },
    }
    if not beatmap.hit_objects:
        return base | {
            "intervals_ms": [],
            "interval_details": [],
            "total_selected_ms": 0,
            "chart_object_range_ms": None,
        }
    first = min(obj.time for obj in beatmap.hit_objects)
    last = max((obj.end_time or obj.time) for obj in beatmap.hit_objects)
    candidates: list[tuple[int, int, str]] = [
        (max(first - 1_500, 0), max(first - 1_500, 0) + edge_duration_ms, "first_objects"),
        (max(last - edge_duration_ms, 0), last + 1_500, "last_objects"),
    ]
    dense = densest_interval(beatmap, dense_duration_ms)
    if dense is not None:
        candidates.append((*dense, "densest_objects"))

    ranked = []
    for window in (content_analysis or {}).get("windows", []):
        try:
            ranked.append((float(window["score"]), int(window["start_ms"]), int(window["end_ms"])))
        except (KeyError, TypeError, ValueError):
            continue
    chosen: list[tuple[int, int]] = []
    for score, start, end in sorted(ranked, reverse=True):
        interval = (start, end)
        # Adjacent 8-second-stride windows mostly contain the same events.
        # Greedy de-duplication makes the proposal cover distinct sections.
        if any(_overlap(interval, previous) / max(min(end - start, previous[1] - previous[0]), 1) > 0.5 for previous in chosen):
            continue
        chosen.append(interval)
        candidates.append((start, end, f"content_top_{len(chosen)}"))
        if len(chosen) >= content_windows:
            break

    merged = merge_intervals(candidates, join_gap_ms=join_gap_ms)
    return {
        **base,
        "intervals_ms": [[item.start_ms, item.end_ms] for item in merged],
        "interval_details": [dataclasses.asdict(item) for item in merged],
        "total_selected_ms": sum(item.end_ms - item.start_ms for item in merged),
        "chart_object_range_ms": [first, last],
    }


def build_candidate_selection_policy(windows_per_interval: int = 1) -> dict[str, Any]:
    """Return the one supported deterministic audio-window search policy."""

    return canonical_selection_policy(
        {
            "schema_version": SELECTION_POLICY_SCHEMA_VERSION,
            "method": SELECTION_POLICY_METHOD,
            "windows_per_interval": windows_per_interval,
            "overlap_required": True,
            "deduplicate_overlapping_proposals": True,
        }
    )


def build_bound_candidate_selection(
    beatmap: Beatmap,
    content_analysis: dict[str, Any],
    *,
    content_model_sha256: str,
    windows_per_interval: int = 1,
) -> dict[str, Any]:
    """Build the single calibrated, label-free candidate-selection record.

    The returned object carries both the logical content-model identifier and
    the exact artifact bytes.  It is suitable for attaching directly as
    ``result["candidate_interval_selection"]``.
    """

    digest = str(content_model_sha256).strip().casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("content_model_sha256 must be a lowercase 64-character SHA-256 digest")
    model_id = content_analysis.get("model_id") if isinstance(content_analysis, dict) else None
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("content_analysis must contain a non-empty model_id")
    if isinstance(windows_per_interval, bool) or not isinstance(windows_per_interval, int) or windows_per_interval <= 0:
        raise ValueError("windows_per_interval must be a positive integer")
    selection = select_whitebox_intervals(beatmap, content_analysis)
    if not selection.get("intervals_ms"):
        raise ValueError("label-free candidate selection produced no intervals")
    policy = build_candidate_selection_policy(windows_per_interval)
    return selection | {
        "content_model_id": model_id.strip(),
        "content_model_sha256": digest,
        "selection_policy": policy,
    }


__all__ = [
    "DEFAULT_CONTENT_WINDOWS",
    "DEFAULT_DENSE_DURATION_MS",
    "DEFAULT_EDGE_DURATION_MS",
    "DEFAULT_JOIN_GAP_MS",
    "SelectedInterval",
    "build_bound_candidate_selection",
    "build_candidate_selection_policy",
    "densest_interval",
    "merge_intervals",
    "select_whitebox_intervals",
]
