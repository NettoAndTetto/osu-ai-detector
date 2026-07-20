"""Serialization-independent beatmap content representation.

The first detector prototype mixed authorship evidence with details such as a
UTF-8 BOM, an editor template and default hitsample fields.  Those fields are
useful for revision forensics, but they are neither stable under an editor save
nor evidence about the mapping itself.  This module creates an explicit
allow-listed representation of semantic content and a separately labelled set
of mechanical fingerprints.  File text and metadata never enter the returned
model vector.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import statistics
from collections.abc import Iterable, Sequence

from .advanced_features import extract_windows
from .parser import Beatmap, HitObject, active_redline, inferred_snap


SCHEMA_VERSION = 2

# Exact allow-lists are deliberately expressed as predicates over known feature
# families.  A newly added raw-file feature therefore cannot silently enter a
# trained model.
_BASE_SEMANTIC = {
    "window_seconds",
    "object_count",
    "object_density",
    "circle_ratio",
    "spinner_ratio",
    "new_combo_ratio",
    "hitsound_nonzero",
    "hitsound_bit_2",
    "hitsound_bit_4",
    "hitsound_bit_8",
}
_RHYTHM_MECHANICAL_PREFIXES = ("rhythm_time_mod10",)
_SPACE_MECHANICAL_PARTS = (
    "both_even",
    "both_mod4",
    "coord_even",
    "coord_mod4",
    "coord_mod8",
    "coord_mod16",
    "entropy_mod",
    "top_mod32",
    "_mod4_",
    "_mod32_",
)
_SLIDER_EXCLUDED_PREFIXES = (
    "slider_decimal_",
)
_TIMING_EXCLUDED = {
    "timing_sample_index_minus1",
    "timing_default_red_fields",
    "timing_beat_length_decimals",
    "timing_beat_length_12plus",
}
_TIMING_MECHANICAL = {
    "timing_green_plus6",
    "timing_green_at_object",
    "timing_red_green_same_offset",
    "timing_sv_quantized_001",
    "timing_sv_quantized_005",
    "timing_sv_epsilon_count",
    "timing_sv_epsilon_ratio",
}


@dataclasses.dataclass(frozen=True)
class ContentWindow:
    start_ms: int
    end_ms: int
    object_count: int
    semantic: dict[str, float]
    mechanical: dict[str, float]
    sequence_tokens: tuple[str, ...]


def feature_channel(name: str) -> str:
    """Return ``semantic``, ``mechanical`` or ``excluded`` for a feature.

    ``mechanical`` still describes parsed hit objects/timing behavior, but may
    be easy to erase (grid residues, integer-ms residues, known postprocessor
    offsets).  It must be shown separately in reports and cannot by itself
    produce a high-confidence result.
    """

    if name in _BASE_SEMANTIC:
        return "semantic"
    if name.startswith("file_") or name in _TIMING_EXCLUDED:
        return "excluded"
    if name.startswith(_SLIDER_EXCLUDED_PREFIXES):
        return "excluded"
    if name.startswith(_RHYTHM_MECHANICAL_PREFIXES):
        return "mechanical"
    if name.startswith("space_"):
        return "mechanical" if any(part in name for part in _SPACE_MECHANICAL_PARTS) else "semantic"
    if name.startswith("rhythm_") or name.startswith("slider_"):
        return "semantic"
    if name.startswith("timing_"):
        return "mechanical" if name in _TIMING_MECHANICAL else "semantic"
    # Fail closed: unknown feature families are not model inputs.
    return "excluded"


def _bucket(value: float, boundaries: Sequence[float]) -> int:
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return index
    return len(boundaries)


def _shape_token(obj: HitObject) -> str:
    if obj.kind != "slider" or not obj.anchors:
        return "_"
    points = [(obj.x, obj.y), *obj.anchors]
    vectors = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    lengths = [math.hypot(dx, dy) for dx, dy in vectors]
    total = sum(lengths)
    if total <= 1e-9:
        return "z"
    length_bins = "".join(str(min(int(length / total * 8), 7)) for length in lengths[:4])
    turn_bins: list[str] = []
    for first, second in zip(vectors, vectors[1:]):
        # Absolute angle is invariant to global rotation and reflection.
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        angle = abs(math.atan2(cross, dot)) / math.pi
        turn_bins.append(str(min(int(angle * 8), 7)))
    return f"l{length_bins}t{''.join(turn_bins[:3])}"


def canonical_sequence_tokens(beatmap: Beatmap, objects: Sequence[HitObject]) -> tuple[str, ...]:
    """Create translation/rotation/reflection-invariant local event tokens."""

    result: list[str] = []
    previous: HitObject | None = None
    previous_angle: float | None = None
    for obj in objects:
        kind = {"circle": "C", "slider": "S", "spinner": "P", "hold": "H"}.get(obj.kind, "U")
        snap_result = inferred_snap(beatmap, obj.time)
        snap = min(snap_result[0], 16) if snap_result else 0
        interval_bucket = distance_bucket = turn_bucket = 0
        if previous is not None:
            redline = active_redline(beatmap, obj.time)
            beat_interval = (
                (obj.time - previous.time) / redline.beat_length
                if redline is not None and redline.beat_length > 0
                else 0.0
            )
            interval_bucket = _bucket(
                beat_interval,
                (0.07, 0.10, 0.14, 0.19, 0.27, 0.38, 0.55, 0.80, 1.10, 1.60, 2.20),
            )
            dx, dy = obj.x - previous.x, obj.y - previous.y
            distance_bucket = _bucket(math.hypot(dx, dy), (24, 48, 72, 96, 128, 160, 200, 256, 340, 450))
            angle = math.atan2(dy, dx)
            if previous_angle is not None:
                turn = abs(((angle - previous_angle + math.pi) % (2 * math.pi)) - math.pi)
                turn_bucket = min(int(turn / math.pi * 8), 7)
            previous_angle = angle
        anchor_bucket = min(len(obj.anchors), 5) if obj.kind == "slider" else 0
        repeat_bucket = min(obj.repeats, 4) if obj.kind == "slider" else 0
        length_bucket = _bucket(obj.pixel_length or 0.0, (40, 80, 120, 180, 260, 380)) if obj.kind == "slider" else 0
        curve = obj.curve_type if obj.curve_type in {"B", "C", "L", "P"} else "_"
        result.append(
            f"{kind}|s{snap}|i{interval_bucket}|d{distance_bucket}|t{turn_bucket}"
            f"|c{curve}|a{anchor_bucket}|r{repeat_bucket}|l{length_bucket}"
            f"|h{min(obj.hit_sound, 15)}|n{int(obj.new_combo)}|g{_shape_token(obj)}"
        )
        previous = obj
    return tuple(result)


def _normalized_entropy(values: Sequence[str], alphabet_size: int | None = None) -> float:
    if not values:
        return 0.0
    counts = collections.Counter(values)
    entropy = -sum((count / len(values)) * math.log(count / len(values)) for count in counts.values())
    denominator = math.log(alphabet_size or max(len(counts), 2))
    return entropy / denominator if denominator > 0 else 0.0


def _conditional_entropy(values: Sequence[str]) -> float:
    if len(values) < 2:
        return 0.0
    transitions: dict[str, list[str]] = collections.defaultdict(list)
    for first, second in zip(values, values[1:]):
        transitions[first].append(second)
    total = len(values) - 1
    return sum(
        len(next_values) / total * _normalized_entropy(next_values)
        for next_values in transitions.values()
    )


def _unique_ratio(values: Sequence[str], size: int) -> float:
    total = len(values) - size + 1
    if total <= 0:
        return 0.0
    grams = ["\x1f".join(values[index:index + size]) for index in range(total)]
    return len(set(grams)) / total


def _maximum_ngram_fraction(values: Sequence[str], size: int) -> float:
    total = len(values) - size + 1
    if total <= 0:
        return 0.0
    counts = collections.Counter(
        "\x1f".join(values[index:index + size]) for index in range(total)
    )
    return max(counts.values(), default=0) / total


def _longest_run_ratio(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    longest = current = 1
    for previous, value in zip(values, values[1:]):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest / len(values)


def _lz78_phrase_ratio(values: Sequence[str]) -> float:
    """Small-sample LZ78 complexity normalized by event count."""

    if not values:
        return 0.0
    dictionary: set[tuple[str, ...]] = set()
    phrase: tuple[str, ...] = ()
    phrases = 0
    for value in values:
        candidate = phrase + (value,)
        if candidate in dictionary:
            phrase = candidate
        else:
            dictionary.add(candidate)
            phrases += 1
            phrase = ()
    if phrase:
        phrases += 1
    return phrases / len(values)


def canonical_sequence_features(tokens: Sequence[str]) -> dict[str, float]:
    """Content-only repetition/complexity summaries for one local window."""

    values = list(tokens)
    kinds = [value.split("|", 1)[0] for value in values]
    structural = ["|".join(value.split("|")[:8]) for value in values]
    result = {
        "sequence::events": float(len(values)),
        "sequence::token_entropy": _normalized_entropy(values),
        "sequence::structural_entropy": _normalized_entropy(structural),
        "sequence::kind_entropy": _normalized_entropy(kinds, 5),
        "sequence::kind_conditional_entropy": _conditional_entropy(kinds),
        "sequence::structural_conditional_entropy": _conditional_entropy(structural),
        "sequence::unigram_unique_ratio": _unique_ratio(structural, 1),
        "sequence::bigram_unique_ratio": _unique_ratio(structural, 2),
        "sequence::trigram_unique_ratio": _unique_ratio(structural, 3),
        "sequence::fourgram_unique_ratio": _unique_ratio(structural, 4),
        "sequence::bigram_max_fraction": _maximum_ngram_fraction(structural, 2),
        "sequence::trigram_max_fraction": _maximum_ngram_fraction(structural, 3),
        "sequence::longest_token_run_ratio": _longest_run_ratio(structural),
        "sequence::longest_kind_run_ratio": _longest_run_ratio(kinds),
        "sequence::lz78_phrase_ratio": _lz78_phrase_ratio(structural),
    }
    for lag in (1, 2, 4, 8, 16):
        comparisons = len(structural) - lag
        result[f"sequence::lag_{lag}_match_ratio"] = (
            sum(first == second for first, second in zip(structural, structural[lag:])) / comparisons
            if comparisons > 0
            else 0.0
        )
    return result


def extract_content_windows(
    beatmap: Beatmap,
    window_ms: int = 24_000,
    stride_ms: int = 8_000,
    min_objects: int = 12,
) -> tuple[ContentWindow, ...]:
    result: list[ContentWindow] = []
    for window in extract_windows(beatmap, window_ms=window_ms, stride_ms=stride_ms, min_objects=min_objects):
        semantic: dict[str, float] = {}
        mechanical: dict[str, float] = {}
        for name, value in window.values.items():
            channel = feature_channel(name)
            if channel == "semantic":
                semantic[name] = float(value)
            elif channel == "mechanical":
                mechanical[name] = float(value)
        objects = tuple(obj for obj in beatmap.hit_objects if window.start_ms <= obj.time < window.end_ms)
        sequence_tokens = canonical_sequence_tokens(beatmap, objects)
        semantic.update(canonical_sequence_features(sequence_tokens))
        result.append(
            ContentWindow(
                start_ms=window.start_ms,
                end_ms=window.end_ms,
                object_count=window.object_count,
                semantic=semantic,
                mechanical=mechanical,
                sequence_tokens=sequence_tokens,
            )
        )
    return tuple(result)


def iter_ngrams(tokens: Sequence[str], min_n: int = 1, max_n: int = 4) -> Iterable[str]:
    for size in range(min_n, max_n + 1):
        for index in range(len(tokens) - size + 1):
            yield " ".join(tokens[index:index + size])


def hash_canonical_ngrams(tokens: Sequence[str], dimensions: int = 512) -> dict[int, float]:
    counts: collections.Counter[int] = collections.Counter()
    for ngram in iter_ngrams(tokens):
        digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8, person=b"osu-cont-v2").digest()
        raw = int.from_bytes(digest, "little")
        counts[raw % dimensions] += 1 if raw >> 63 else -1
    transformed = {index: math.copysign(math.log1p(abs(count)), count) for index, count in counts.items() if count}
    norm = math.sqrt(sum(value * value for value in transformed.values()))
    return {index: value / norm for index, value in transformed.items()} if norm else {}


def aggregate_map_features(
    windows: Sequence[ContentWindow],
    *,
    include_mechanical: bool = False,
    ngram_dimensions: int = 512,
) -> dict[str, float]:
    """Aggregate local windows to a fixed-size, JSON-friendly map vector."""

    feature_values: dict[str, list[float]] = collections.defaultdict(list)
    all_tokens: list[str] = []
    for window in windows:
        values = dict(window.semantic)
        if include_mechanical:
            values.update({f"mechanical::{key}": value for key, value in window.mechanical.items()})
        for name, value in values.items():
            if math.isfinite(value):
                feature_values[name].append(float(value))
        all_tokens.extend(window.sequence_tokens)

    result: dict[str, float] = {"map_window_count": float(len(windows))}
    for name, values in sorted(feature_values.items()):
        ordered = sorted(values)
        q90_index = min(round((len(ordered) - 1) * 0.9), len(ordered) - 1)
        result[f"{name}::mean"] = statistics.fmean(values)
        result[f"{name}::std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        result[f"{name}::max"] = max(values)
        result[f"{name}::q90"] = ordered[q90_index]
    for index, value in hash_canonical_ngrams(all_tokens, ngram_dimensions).items():
        result[f"ngram::{index}"] = value
    return result


def representation_audit() -> dict[str, object]:
    """Machine-readable guarantees embedded in models and reports."""

    return {
        "schema_version": SCHEMA_VERSION,
        "raw_text_used": False,
        "metadata_used": False,
        "serialization_features_used": False,
        "sequence_invariances": ["translation", "global_rotation", "reflection"],
        "mechanical_channel_is_separate": True,
        "sequence_complexity_features": True,
        "excluded_examples": sorted(_TIMING_EXCLUDED) + ["file_*", "slider_decimal_*"],
    }
