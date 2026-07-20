from __future__ import annotations

import collections
import dataclasses
import math
import re
import statistics
from typing import Iterable

from .parser import Beatmap, HitObject, inferred_snap


@dataclasses.dataclass(frozen=True)
class FeatureSet:
    values: dict[str, float]
    counts: dict[str, int]
    notes: tuple[str, ...] = ()

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _normalized_entropy(values: Iterable[int], modulus: int) -> float:
    data = list(values)
    if not data:
        return 1.0
    counts = collections.Counter(v % modulus for v in data)
    entropy = -sum((n / len(data)) * math.log(n / len(data)) for n in counts.values())
    return entropy / math.log(modulus)


def _decimal_places(raw: str | None) -> int:
    if not raw or "." not in raw:
        return 0
    token = raw.lower().split("e", 1)[0]
    return len(token.rsplit(".", 1)[1].rstrip("0"))


def _declaration_feature(beatmap: Beatmap) -> tuple[float, list[str]]:
    searchable = " ".join(
        beatmap.metadata.get(key, "")
        for key in ("Title", "TitleUnicode", "Creator", "Version", "Source", "Tags")
    ).lower()
    patterns = {
        "Mapperatorinator": r"\bmapperatorinator\b",
        "osuT5": r"\bosu\s*t5\b",
        "AI-generated disclosure": r"\b(ai[- ]generated|generated (?:with|by) ai)\b",
    }
    # A negated phrase ("not AI-generated", "without Mapperatorinator") is
    # not a self-disclosure.  Metadata is user-controlled, so the declaration
    # shortcut must be narrow enough that a denial cannot become a high verdict.
    negation = re.compile(
        r"(?:\b(?:not|no|without|never)\b|\bnon[- ]|不是|并非|非|未使用|未用|不用|没有使用).{0,24}$",
        re.IGNORECASE,
    )
    matches: list[str] = []
    for label, pattern in patterns.items():
        affirmative = False
        for occurrence in re.finditer(pattern, searchable):
            prefix = searchable[max(0, occurrence.start() - 40):occurrence.start()]
            if not negation.search(prefix):
                affirmative = True
                break
        if affirmative:
            matches.append(label)
    return (1.0 if matches else 0.0), matches


def _template_features(beatmap: Beatmap) -> tuple[dict[str, float], list[str]]:
    editor = beatmap.properties.get("Editor", {})
    general = beatmap.properties.get("General", {})
    checks = {
        "Bookmarks:-330001": editor.get("Bookmarks") == "-330001",
        "TimelineZoom:2.20004": editor.get("TimelineZoom") == "2.20004",
        "GridSize:8": editor.get("GridSize") == "8",
        "BeatDivisor:4": editor.get("BeatDivisor") == "4",
        "DistanceSpacing:1.0": editor.get("DistanceSpacing") == "1.0",
        "OverlayPosition:Above": general.get("OverlayPosition") == "Above",
        "SampleSet:All": general.get("SampleSet") == "All",
        "Countdown:0": general.get("Countdown") == "0",
    }
    exact = [label for label, matched in checks.items() if matched]
    # The first five fields form the unusually specific current template. The
    # general fields are weak defaults and only contribute to the softer score.
    distinctive = sum(checks[label] for label in list(checks)[:5])
    return {
        "template_distinctive_ratio": distinctive / 5.0,
        "template_all_field_ratio": sum(checks.values()) / len(checks),
        # U+034F before Combo3 is also emitted by many editor-authored maps;
        # record it for auditability but never treat it as a watermark.
        "template_combo3_cgj": float("\u034fCombo3" in beatmap.raw_text),
    }, exact


def _spatial_features(beatmap: Beatmap) -> tuple[dict[str, float], dict[str, int]]:
    objects = [obj for obj in beatmap.hit_objects if obj.kind not in {"spinner", "hold"}]
    head_pairs = [(obj.x, obj.y) for obj in objects]
    all_pairs = [point for obj in objects for point in obj.points]
    heads = [value for pair in head_pairs for value in pair]
    coords = [value for pair in all_pairs for value in pair]

    residue_counts = collections.Counter(value % 32 for value in coords)
    top4 = sum(n for _, n in residue_counts.most_common(4))
    distances32 = [min(value % 32, (-value) % 32) for value in coords]

    values = {
        "head_both_even_ratio": _ratio(sum(x % 2 == 0 and y % 2 == 0 for x, y in head_pairs), len(head_pairs)),
        "head_both_mod4_ratio": _ratio(sum(x % 4 == 0 and y % 4 == 0 for x, y in head_pairs), len(head_pairs)),
        "coord_even_ratio": _ratio(sum(v % 2 == 0 for v in coords), len(coords)),
        "coord_mod4_ratio": _ratio(sum(v % 4 == 0 for v in coords), len(coords)),
        "coord_mod8_ratio": _ratio(sum(v % 8 == 0 for v in coords), len(coords)),
        "coord_mod32_entropy": _normalized_entropy(coords, 32),
        "coord_mod32_top4_ratio": _ratio(top4, len(coords)),
        "coord_mod32_16_ratio": _ratio(sum(v % 32 == 16 for v in coords), len(coords)),
        "coord_grid32_offset_4_8_12_ratio": _ratio(sum(d in {4, 8, 12} for d in distances32), len(distances32)),
        "head_coord_mod32_entropy": _normalized_entropy(heads, 32),
        "head_coord_mod32_16_ratio": _ratio(sum(v % 32 == 16 for v in heads), len(heads)),
    }
    counts = {
        "positioned_objects": len(head_pairs),
        "coordinates": len(coords),
        "slider_anchor_coordinates": max(len(coords) - len(heads), 0),
    }
    return values, counts


def _temporal_features(beatmap: Beatmap) -> tuple[dict[str, float], dict[str, int]]:
    times = [obj.time for obj in beatmap.hit_objects]
    offgrid_times: list[int] = []
    snapped = 0
    floor_exclusive = 0
    round_exclusive = 0
    floor_or_round_cases = 0
    exact_floor = 0
    exact_round = 0

    for time in times:
        inferred = inferred_snap(beatmap, time)
        if inferred is None:
            continue
        divisor, ideal = inferred
        if divisor == 0:
            offgrid_times.append(time)
            continue
        snapped += 1
        floor_value = math.floor(ideal + 1e-9)
        round_value = math.floor(ideal + 0.5)
        if time == floor_value:
            exact_floor += 1
        if time == round_value:
            exact_round += 1
        if floor_value != round_value:
            floor_or_round_cases += 1
            if time == floor_value:
                floor_exclusive += 1
            elif time == round_value:
                round_exclusive += 1

    all_residue_entropy = _normalized_entropy(times, 10)
    values = {
        "snapped_object_ratio": _ratio(snapped, len(times)),
        "offgrid_time_mod10_5_ratio": _ratio(sum(t % 10 == 5 for t in offgrid_times), len(offgrid_times)),
        "all_time_mod10_5_ratio": _ratio(sum(t % 10 == 5 for t in times), len(times)),
        "all_time_mod10_entropy": all_residue_entropy,
        "snap_floor_exclusive_ratio": _ratio(floor_exclusive, floor_or_round_cases),
        "snap_round_exclusive_ratio": _ratio(round_exclusive, floor_or_round_cases),
        "snap_exact_floor_ratio": _ratio(exact_floor, snapped),
        "snap_exact_round_ratio": _ratio(exact_round, snapped),
    }
    counts = {
        "object_times": len(times),
        "offgrid_object_times": len(offgrid_times),
        "snapped_object_times": snapped,
        "floor_round_discriminating_times": floor_or_round_cases,
    }
    return values, counts


def _timing_features(beatmap: Beatmap) -> tuple[dict[str, float], dict[str, int]]:
    red = [tp for tp in beatmap.timing_points if tp.uninherited and tp.beat_length > 0]
    green = [tp for tp in beatmap.timing_points if not tp.uninherited and tp.beat_length < 0]
    object_times = {obj.time for obj in beatmap.hit_objects}
    duration_ms = 0
    if beatmap.hit_objects:
        duration_ms = max(obj.time for obj in beatmap.hit_objects) - min(obj.time for obj in beatmap.hit_objects)
    duration_minutes = max(duration_ms / 60000.0, 0.25)

    red_offsets = collections.Counter(round(tp.offset) for tp in red)
    green_offsets = collections.Counter(round(tp.offset) for tp in green)
    same_red_green = sum(min(n, green_offsets.get(offset, 0)) for offset, n in red_offsets.items())
    svs = [tp.slider_velocity for tp in green if tp.slider_velocity is not None and math.isfinite(tp.slider_velocity)]

    def quantized(value: float, step: float, tolerance: float = 2e-7) -> bool:
        return abs(value / step - round(value / step)) <= tolerance

    epsilon_matches = 0
    for point in green:
        sv = point.slider_velocity
        if sv is None or sv <= 0:
            continue
        qsv = round(sv * 100) / 100
        if qsv <= 0:
            continue
        exact = -100.0 / qsv
        observed = point.beat_length
        delta = observed - exact
        if 2e-11 <= delta <= 2e-10:
            epsilon_matches += 1

    green_at_objects = sum(round(tp.offset) in object_times for tp in green)
    green_plus6 = sum(round(tp.offset) - 6 in object_times for tp in green)
    default_red_fields = sum(
        tp.meter == 4 and tp.sample_set == 2 and tp.sample_index == -1 for tp in red
    )

    values = {
        "redlines_per_minute": len(red) / duration_minutes,
        "redline_object_ratio": _ratio(len(red), len(beatmap.hit_objects)),
        "greenline_object_ratio": _ratio(len(green), len(beatmap.hit_objects)),
        "same_offset_red_green_ratio": _ratio(same_red_green, len(red)),
        "green_at_object_ratio": _ratio(green_at_objects, len(green)),
        "green_plus6ms_after_object_ratio": _ratio(green_plus6, len(green)),
        "sv_quantized_0_01_ratio": _ratio(sum(quantized(sv, 0.01) for sv in svs), len(svs)),
        "sv_quantized_0_05_ratio": _ratio(sum(quantized(sv, 0.05) for sv in svs), len(svs)),
        "sv_epsilon_1e10_ratio": _ratio(epsilon_matches, len(green)),
        "redline_default_fields_ratio": _ratio(default_red_fields, len(red)),
    }
    counts = {
        "timing_points": len(beatmap.timing_points),
        "redlines": len(red),
        "greenlines": len(green),
        "slider_velocities": len(svs),
        "sv_epsilon_matches": epsilon_matches,
    }
    return values, counts


def _shape_invariant(slider: HitObject) -> tuple[float, ...] | None:
    points = [(slider.x, slider.y), *slider.anchors]
    if len(points) < 3:
        return None
    segments = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    lengths = [math.hypot(x, y) for x, y in segments]
    total = sum(lengths)
    if total <= 1e-9:
        return None
    result: list[float] = [length / total for length in lengths]
    for first, second in zip(segments, segments[1:]):
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        result.append(abs(math.atan2(cross, dot)) / math.pi)
    return tuple(result)


def _slider_features(beatmap: Beatmap) -> tuple[dict[str, float], dict[str, int]]:
    sliders = [obj for obj in beatmap.hit_objects if obj.kind == "slider"]
    decimals = [_decimal_places(obj.raw_pixel_length) for obj in sliders]
    comparable = 0
    exact = 0
    near_not_exact = 0

    for first, second in zip(sliders, sliders[1:]):
        if second.time - first.time > 3000:
            continue
        a = _shape_invariant(first)
        b = _shape_invariant(second)
        if a is None or b is None or len(a) != len(b) or first.curve_type != second.curve_type:
            continue
        comparable += 1
        rms = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))
        if rms < 1e-9:
            exact += 1
        elif rms < 0.08:
            near_not_exact += 1

    anchor_counts = [len(obj.anchors) for obj in sliders]
    values = {
        "slider_ratio": _ratio(len(sliders), len(beatmap.hit_objects)),
        "slider_length_10plus_decimal_ratio": _ratio(sum(n >= 10 for n in decimals), len(decimals)),
        "slider_length_13plus_decimal_ratio": _ratio(sum(n >= 13 for n in decimals), len(decimals)),
        "slider_length_decimal_mean": statistics.fmean(decimals) if decimals else 0.0,
        "slider_near_not_exact_shape_ratio": _ratio(near_not_exact, comparable),
        "slider_exact_shape_ratio": _ratio(exact, comparable),
        "slider_multi_anchor_ratio": _ratio(sum(n >= 2 for n in anchor_counts), len(anchor_counts)),
    }
    counts = {
        "sliders": len(sliders),
        "comparable_slider_pairs": comparable,
    }
    return values, counts


def extract_features(beatmap: Beatmap) -> FeatureSet:
    values: dict[str, float] = {
        "mode": float(beatmap.mode),
        "object_count": float(beatmap.object_count),
        "utf8_bom": float(beatmap.had_utf8_bom),
    }
    counts: dict[str, int] = {"objects": beatmap.object_count}
    notes: list[str] = []

    declared, declarations = _declaration_feature(beatmap)
    values["explicit_ai_disclosure"] = declared
    if declarations:
        notes.append("Metadata disclosure: " + ", ".join(declarations))

    template_values, template_matches = _template_features(beatmap)
    values.update(template_values)
    if len(template_matches) >= 5:
        notes.append("Matches several fields from the Mapperatorinator output template")

    for extractor in (_spatial_features, _temporal_features, _timing_features, _slider_features):
        feature_values, feature_counts = extractor(beatmap)
        values.update(feature_values)
        counts.update(feature_counts)

    return FeatureSet(values=values, counts=counts, notes=tuple(notes))
