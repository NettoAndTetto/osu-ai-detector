from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import statistics
from typing import Iterable, Sequence

from .parser import Beatmap, HitObject, active_redline, inferred_snap


EPSILON = 1e-12


@dataclasses.dataclass(frozen=True)
class WindowFeatures:
    start_ms: int
    end_ms: int
    object_count: int
    values: dict[str, float]
    sequence_tokens: tuple[str, ...]


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _entropy(items: Iterable[object], alphabet_size: int | None = None) -> float:
    values = list(items)
    if not values:
        return 0.0
    counts = collections.Counter(values)
    entropy = -sum((n / len(values)) * math.log(n / len(values)) for n in counts.values())
    denominator = math.log(alphabet_size or max(len(counts), 2))
    return entropy / denominator if denominator > 0 else 0.0


def _close(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= tolerance


def _decimal_places(raw: str | None) -> int:
    if not raw or "." not in raw:
        return 0
    mantissa = raw.lower().split("e", 1)[0]
    return len(mantissa.rsplit(".", 1)[1].rstrip("0"))


def _put_distribution(values: dict[str, float], prefix: str, data: Sequence[float], scale: float = 1.0) -> None:
    normalized = [float(item) / scale for item in data]
    values[f"{prefix}_mean"] = _mean(normalized)
    values[f"{prefix}_std"] = _std(normalized)
    for label, q in (("q10", 0.10), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q90", 0.90)):
        values[f"{prefix}_{label}"] = _quantile(normalized, q)


def _coordinate_features(values: dict[str, float], prefix: str, pairs: Sequence[tuple[int, int]]) -> None:
    flat = [coordinate for pair in pairs for coordinate in pair]
    values[f"{prefix}_pairs"] = float(len(pairs))
    values[f"{prefix}_coordinates"] = float(len(flat))
    if not pairs:
        for key in (
            "both_even", "both_mod4_0", "both_mod4_2", "coord_even", "coord_mod4_0",
            "coord_mod4_2", "coord_mod8_0", "coord_mod16_0", "entropy_mod4",
            "entropy_mod8", "entropy_mod16", "entropy_mod32", "top_mod32",
            "folded_mod32_entropy", "folded_mod32_target_4_8_12",
            "folded_mod32_target_4_8_12_nonzero", "folded_mod32_nonzero_coordinates",
            "folded_mod32_nonzero_ratio",
        ):
            values[f"{prefix}_{key}"] = 0.0
        return

    values[f"{prefix}_both_even"] = _ratio(sum(x % 2 == 0 and y % 2 == 0 for x, y in pairs), len(pairs))
    values[f"{prefix}_both_mod4_0"] = _ratio(sum(x % 4 == 0 and y % 4 == 0 for x, y in pairs), len(pairs))
    values[f"{prefix}_both_mod4_2"] = _ratio(sum(x % 4 == 2 and y % 4 == 2 for x, y in pairs), len(pairs))
    values[f"{prefix}_coord_even"] = _ratio(sum(item % 2 == 0 for item in flat), len(flat))
    values[f"{prefix}_coord_mod4_0"] = _ratio(sum(item % 4 == 0 for item in flat), len(flat))
    values[f"{prefix}_coord_mod4_2"] = _ratio(sum(item % 4 == 2 for item in flat), len(flat))
    values[f"{prefix}_coord_mod8_0"] = _ratio(sum(item % 8 == 0 for item in flat), len(flat))
    values[f"{prefix}_coord_mod16_0"] = _ratio(sum(item % 16 == 0 for item in flat), len(flat))
    for modulus in (4, 8, 16, 32):
        values[f"{prefix}_entropy_mod{modulus}"] = _entropy((item % modulus for item in flat), modulus)
    for residue in range(4):
        values[f"{prefix}_mod4_{residue}"] = _ratio(sum(item % 4 == residue for item in flat), len(flat))
    # V28/V29 use a fixed half-cell offset (normally residue 16), while V32
    # refines a 32 px cell in 2 px steps. Keeping the actual residue histogram
    # distinguishes those mechanisms from a generic low-entropy human grid.
    for residue in range(32):
        values[f"{prefix}_mod32_{residue}"] = _ratio(sum(item % 32 == residue for item in flat), len(flat))
    residue = collections.Counter(item % 32 for item in flat)
    values[f"{prefix}_top_mod32"] = _ratio(residue.most_common(1)[0][1], len(flat))

    # A public community prototype (the “Fraudinator” Colab linked from the
    # September 2025 Jayblue false-positive discussion) folded absolute jump
    # deltas around a 32 px cell and inspected peaks at 4/8/12 px.  Keep the
    # exact, auditable statistic as a weak mechanical feature.  Applying it to
    # deltas below makes it invariant to translating the whole map; calibration
    # and hard grid-snap negatives are still required because editor grids can
    # create the same peaks.
    folded = [min(abs(item) % 32, 32 - (abs(item) % 32)) for item in flat]
    nonzero_folded = [item for item in folded if item]
    values[f"{prefix}_folded_mod32_entropy"] = _entropy(folded, 17)
    values[f"{prefix}_folded_mod32_nonzero_coordinates"] = float(len(nonzero_folded))
    values[f"{prefix}_folded_mod32_nonzero_ratio"] = _ratio(len(nonzero_folded), len(folded))
    values[f"{prefix}_folded_mod32_target_4_8_12"] = _ratio(
        sum(item in {4, 8, 12} for item in folded), len(folded)
    )
    values[f"{prefix}_folded_mod32_target_4_8_12_nonzero"] = _ratio(
        sum(item in {4, 8, 12} for item in nonzero_folded), len(nonzero_folded)
    )


def _shape(slider: HitObject) -> tuple[float, ...] | None:
    points = [(slider.x, slider.y), *slider.anchors]
    if len(points) < 3:
        return None
    segments = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    lengths = [math.hypot(x, y) for x, y in segments]
    total = sum(lengths)
    if total <= EPSILON:
        return None
    descriptor = [length / total for length in lengths]
    for first, second in zip(segments, segments[1:]):
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        descriptor.append(math.atan2(cross, dot) / math.pi)
    return tuple(descriptor)


def _rhythm_features(beatmap: Beatmap, objects: Sequence[HitObject], values: dict[str, float]) -> list[float]:
    times = [obj.time for obj in objects]
    intervals = [b - a for a, b in zip(times, times[1:]) if b >= a]
    beat_intervals: list[float] = []
    for previous, current in zip(objects, objects[1:]):
        redline = active_redline(beatmap, current.time)
        if redline and redline.beat_length > 0:
            beat_intervals.append((current.time - previous.time) / redline.beat_length)

    values["rhythm_interval_count"] = float(len(intervals))
    _put_distribution(values, "rhythm_interval_ms", intervals, 1000.0)
    _put_distribution(values, "rhythm_interval_beats", beat_intervals)
    values["rhythm_interval_cv"] = _ratio(_std(intervals), _mean(intervals))
    values["rhythm_interval_repeat"] = _ratio(
        sum(abs(a - b) <= 1 for a, b in zip(intervals, intervals[1:])), max(len(intervals) - 1, 0)
    )

    targets = (1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 3 / 4, 1.0, 1.5, 2.0)
    quantized: list[int] = []
    for interval in beat_intervals:
        nearest = min(range(len(targets)), key=lambda index: abs(interval - targets[index]))
        quantized.append(nearest if abs(interval - targets[nearest]) <= 0.025 else len(targets))
    values["rhythm_known_interval_ratio"] = _ratio(sum(item < len(targets) for item in quantized), len(quantized))
    values["rhythm_interval_token_entropy"] = _entropy(quantized, len(targets) + 1)
    values["rhythm_interval_bigram_entropy"] = _entropy(zip(quantized, quantized[1:]))
    values["rhythm_interval_trigram_entropy"] = _entropy(zip(quantized, quantized[1:], quantized[2:]))

    snaps: list[int] = []
    snap_errors: list[float] = []
    for time in times:
        inferred = inferred_snap(beatmap, time)
        if inferred is None:
            snaps.append(0)
        else:
            divisor, ideal = inferred
            snaps.append(divisor)
            snap_errors.append(abs(time - ideal))
    for divisor in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16):
        values[f"rhythm_snap_{divisor}"] = _ratio(sum(item == divisor for item in snaps), len(snaps))
    values["rhythm_snap_entropy"] = _entropy(snaps, 17)
    values["rhythm_snap_error_mean"] = _mean(snap_errors)
    values["rhythm_snap_error_q90"] = _quantile(snap_errors, 0.9)
    values["rhythm_time_mod10_entropy"] = _entropy((time % 10 for time in times), 10)
    for residue in range(10):
        values[f"rhythm_time_mod10_{residue}"] = _ratio(sum(time % 10 == residue for time in times), len(times))
    return beat_intervals


def _spatial_features(objects: Sequence[HitObject], values: dict[str, float]) -> None:
    positioned = [obj for obj in objects if obj.kind not in {"spinner", "hold"}]
    circles = [(obj.x, obj.y) for obj in positioned if obj.kind == "circle"]
    slider_heads = [(obj.x, obj.y) for obj in positioned if obj.kind == "slider"]
    heads = [(obj.x, obj.y) for obj in positioned]
    anchors = [anchor for obj in positioned if obj.kind == "slider" for anchor in obj.anchors]
    all_points = [point for obj in positioned for point in obj.points]
    for prefix, pairs in (
        ("space_head", heads), ("space_circle", circles), ("space_slider_head", slider_heads),
        ("space_anchor", anchors), ("space_point", all_points),
    ):
        _coordinate_features(values, prefix, pairs)

    anchor_deltas = [
        (anchor[0] - obj.x, anchor[1] - obj.y)
        for obj in positioned if obj.kind == "slider" for anchor in obj.anchors
    ]
    _coordinate_features(values, "space_anchor_delta", anchor_deltas)

    jumps = [(b.x - a.x, b.y - a.y) for a, b in zip(positioned, positioned[1:])]
    _coordinate_features(values, "space_jump_delta", jumps)
    distances = [math.hypot(x, y) for x, y in jumps]
    _put_distribution(values, "space_jump", distances, 640.0)
    values["space_jump_cv"] = _ratio(_std(distances), _mean(distances))
    rounded_distances = [round(item) for item in distances]
    values["space_jump_repeat"] = _ratio(
        sum(a == b for a, b in zip(rounded_distances, rounded_distances[1:])), max(len(rounded_distances) - 1, 0)
    )

    angles = [math.atan2(y, x) for x, y in jumps if x or y]
    angle_bins = [int(((angle + math.pi) / (2 * math.pi)) * 16) % 16 for angle in angles]
    values["space_direction_entropy"] = _entropy(angle_bins, 16)
    values["space_direction_dominant"] = _ratio(
        collections.Counter(angle_bins).most_common(1)[0][1] if angle_bins else 0, len(angle_bins)
    )
    turns = [((b - a + math.pi) % (2 * math.pi)) - math.pi for a, b in zip(angles, angles[1:])]
    values["space_turn_straight"] = _ratio(sum(abs(item) < math.pi / 18 for item in turns), len(turns))
    values["space_turn_reverse"] = _ratio(sum(abs(abs(item) - math.pi) < math.pi / 18 for item in turns), len(turns))
    values["space_turn_right_angle"] = _ratio(sum(abs(abs(item) - math.pi / 2) < math.pi / 18 for item in turns), len(turns))
    values["space_turn_abs_mean"] = _mean([abs(item) / math.pi for item in turns])
    values["space_turn_abs_std"] = _std([abs(item) / math.pi for item in turns])

    grid_cells = [
        (min(max(x // 64, 0), 7), min(max(y // 64, 0), 5)) for x, y in heads
    ]
    values["space_grid_entropy"] = _entropy(grid_cells, 48)
    values["space_grid_occupancy"] = _ratio(len(set(grid_cells)), 48)
    values["space_outside_playfield"] = _ratio(
        sum(x < 0 or x > 512 or y < 0 or y > 384 for x, y in all_points), len(all_points)
    )


def _slider_features(objects: Sequence[HitObject], values: dict[str, float]) -> None:
    sliders = [obj for obj in objects if obj.kind == "slider"]
    values["slider_count"] = float(len(sliders))
    values["slider_ratio"] = _ratio(len(sliders), len(objects))
    anchor_counts = [len(obj.anchors) for obj in sliders]
    values["slider_anchor_mean"] = _mean(anchor_counts)
    values["slider_anchor_q90"] = _quantile(anchor_counts, 0.9)
    values["slider_multi_anchor"] = _ratio(sum(item >= 2 for item in anchor_counts), len(anchor_counts))
    for curve_type in ("B", "C", "L", "P"):
        values[f"slider_curve_{curve_type}"] = _ratio(sum(obj.curve_type == curve_type for obj in sliders), len(sliders))
    values["slider_repeat"] = _ratio(sum(obj.repeats > 1 for obj in sliders), len(sliders))

    decimals = [_decimal_places(obj.raw_pixel_length) for obj in sliders]
    values["slider_decimal_mean"] = _mean(decimals)
    values["slider_decimal_10plus"] = _ratio(sum(item >= 10 for item in decimals), len(decimals))
    values["slider_decimal_13plus"] = _ratio(sum(item >= 13 for item in decimals), len(decimals))

    length_ratios: list[float] = []
    for slider in sliders:
        points = [(slider.x, slider.y), *slider.anchors]
        polyline = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        if slider.pixel_length and slider.pixel_length > EPSILON:
            length_ratios.append(polyline / slider.pixel_length)
    _put_distribution(values, "slider_polyline_pixel_ratio", length_ratios)

    comparable = exact = near = 0
    for index, first in enumerate(sliders):
        shape_a = _shape(first)
        if shape_a is None:
            continue
        for second in sliders[index + 1:]:
            if second.time - first.time > 8000:
                break
            shape_b = _shape(second)
            if shape_b is None or len(shape_a) != len(shape_b) or first.curve_type != second.curve_type:
                continue
            comparable += 1
            # Mirror/rotation invariant: compare both signed and reflected turn descriptors.
            rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(shape_a, shape_b)) / len(shape_a))
            reflected = tuple(shape_b[: len(second.anchors)] + tuple(-x for x in shape_b[len(second.anchors):]))
            rms_reflected = math.sqrt(sum((a - b) ** 2 for a, b in zip(shape_a, reflected)) / len(shape_a))
            distance = min(rms, rms_reflected)
            if distance < 1e-9:
                exact += 1
            elif distance < 0.06:
                near += 1
    values["slider_shape_pairs"] = float(comparable)
    values["slider_shape_exact"] = _ratio(exact, comparable)
    values["slider_shape_near_not_exact"] = _ratio(near, comparable)
    values["slider_shape_near_exact_ratio"] = _ratio(near, exact + near)


def _timing_features(beatmap: Beatmap, start_ms: int, end_ms: int, objects: Sequence[HitObject], values: dict[str, float]) -> None:
    points = [tp for tp in beatmap.timing_points if start_ms <= tp.offset < end_ms]
    red = [tp for tp in points if tp.uninherited and tp.beat_length > 0]
    green = [tp for tp in points if not tp.uninherited and tp.beat_length < 0]
    object_times = {obj.time for obj in objects}
    values["timing_points"] = float(len(points))
    values["timing_red"] = float(len(red))
    values["timing_green"] = float(len(green))
    values["timing_red_per_second"] = _ratio(len(red) * 1000, end_ms - start_ms)
    values["timing_green_per_object"] = _ratio(len(green), len(objects))
    values["timing_sample_index_minus1"] = _ratio(sum(tp.sample_index == -1 for tp in points), len(points))
    values["timing_default_red_fields"] = _ratio(
        sum(tp.meter == 4 and tp.sample_set == 2 and tp.sample_index == -1 for tp in red), len(red)
    )
    red_offsets = collections.Counter(round(tp.offset) for tp in red)
    green_offsets = collections.Counter(round(tp.offset) for tp in green)
    overlap = sum(min(count, green_offsets.get(offset, 0)) for offset, count in red_offsets.items())
    values["timing_red_green_same_offset"] = _ratio(overlap, len(red))
    values["timing_green_at_object"] = _ratio(sum(round(tp.offset) in object_times for tp in green), len(green))
    values["timing_green_plus6"] = _ratio(sum(round(tp.offset) - 6 in object_times for tp in green), len(green))

    svs = [tp.slider_velocity for tp in green if tp.slider_velocity and math.isfinite(tp.slider_velocity)]
    values["timing_sv_quantized_001"] = _ratio(sum(_close(sv * 100, round(sv * 100), 2e-7) for sv in svs), len(svs))
    values["timing_sv_quantized_005"] = _ratio(sum(_close(sv * 20, round(sv * 20), 2e-7) for sv in svs), len(svs))
    epsilon_matches = 0
    for point in green:
        sv = point.slider_velocity
        if not sv or sv <= 0:
            continue
        qsv = round(sv * 100) / 100
        if qsv <= 0:
            continue
        delta = point.beat_length - (-100.0 / qsv)
        if 2e-11 <= delta <= 2e-10:
            epsilon_matches += 1
    values["timing_sv_epsilon_count"] = float(epsilon_matches)
    values["timing_sv_epsilon_ratio"] = _ratio(epsilon_matches, len(green))
    raw_decimals = [_decimal_places(tp.raw_beat_length) for tp in points]
    values["timing_beat_length_decimals"] = _mean(raw_decimals)
    values["timing_beat_length_12plus"] = _ratio(sum(item >= 12 for item in raw_decimals), len(raw_decimals))


def _serialization_features(beatmap: Beatmap, objects: Sequence[HitObject], values: dict[str, float]) -> None:
    editor = beatmap.properties.get("Editor", {})
    distinctive = (
        editor.get("Bookmarks") == "-330001",
        editor.get("TimelineZoom") == "2.20004",
        editor.get("GridSize") == "8",
        editor.get("BeatDivisor") == "4",
        editor.get("DistanceSpacing") == "1.0",
    )
    values["file_template_ratio"] = _ratio(sum(distinctive), len(distinctive))
    values["file_utf8_bom"] = float(beatmap.had_utf8_bom)
    values["file_combo3_cgj"] = float("\u034fCombo3" in beatmap.raw_text)
    values["file_trailing_colon"] = _ratio(sum(obj.raw.endswith(":") for obj in objects), len(objects))
    values["file_sample_index_minus1"] = _ratio(sum(":-1:" in obj.raw for obj in objects), len(objects))
    values["file_zero_sample_tail"] = _ratio(sum(obj.raw.endswith("0:0:0:0:") for obj in objects), len(objects))


def _bucket(value: float, boundaries: Sequence[float]) -> int:
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return index
    return len(boundaries)


def sequence_tokens(beatmap: Beatmap, objects: Sequence[HitObject]) -> tuple[str, ...]:
    tokens: list[str] = []
    previous: HitObject | None = None
    previous_angle: float | None = None
    for obj in objects:
        kind = {"circle": "C", "slider": "S", "spinner": "P", "hold": "H"}.get(obj.kind, "U")
        inferred = inferred_snap(beatmap, obj.time)
        snap = inferred[0] if inferred else 0
        interval_bucket = 0
        distance_bucket = 0
        turn_bucket = 0
        if previous is not None:
            redline = active_redline(beatmap, obj.time)
            beat_interval = (obj.time - previous.time) / redline.beat_length if redline and redline.beat_length > 0 else 0
            interval_bucket = _bucket(beat_interval, (0.07, 0.10, 0.14, 0.19, 0.27, 0.38, 0.55, 0.80, 1.10, 1.60, 2.20))
            dx, dy = obj.x - previous.x, obj.y - previous.y
            distance_bucket = _bucket(math.hypot(dx, dy), (24, 48, 72, 96, 128, 160, 200, 256, 340, 450))
            angle = math.atan2(dy, dx)
            if previous_angle is not None:
                turn = abs(((angle - previous_angle + math.pi) % (2 * math.pi)) - math.pi)
                turn_bucket = min(int(turn / math.pi * 8), 7)
            previous_angle = angle
        anchor_bucket = min(len(obj.anchors), 4) if obj.kind == "slider" else 0
        repeat_bucket = min(obj.repeats, 3) if obj.kind == "slider" else 0
        curve = obj.curve_type if obj.curve_type in {"B", "C", "L", "P"} else "_"
        position = f"{min(max(obj.x // 64, 0), 7)}{min(max(obj.y // 64, 0), 5)}"
        residue = f"{obj.x % 32:02d}{obj.y % 32:02d}"
        mod4 = f"{obj.x % 4}{obj.y % 4}"
        anchor_even = _ratio(
            sum(x % 2 == 0 and y % 2 == 0 for x, y in obj.anchors), len(obj.anchors)
        ) if obj.anchors else 0.0
        anchor_even_bucket = min(int(anchor_even * 4 + 1e-9), 4)
        token = (
            f"{kind}|s{min(snap,16)}|i{interval_bucket}|d{distance_bucket}|t{turn_bucket}"
            f"|p{position}|q{residue}|m{mod4}|e{anchor_even_bucket}|c{curve}|a{anchor_bucket}|r{repeat_bucket}"
            f"|h{min(obj.hit_sound,15)}|n{int(obj.new_combo)}"
        )
        tokens.append(token)
        previous = obj
    return tuple(tokens)


def extract_window_features(beatmap: Beatmap, start_ms: int, end_ms: int) -> WindowFeatures:
    objects = [obj for obj in beatmap.hit_objects if start_ms <= obj.time < end_ms]
    values: dict[str, float] = {
        "window_seconds": max((end_ms - start_ms) / 1000.0, 0.001),
        "object_count": float(len(objects)),
        "object_density": _ratio(len(objects) * 1000, end_ms - start_ms),
        "circle_ratio": _ratio(sum(obj.kind == "circle" for obj in objects), len(objects)),
        "spinner_ratio": _ratio(sum(obj.kind == "spinner" for obj in objects), len(objects)),
        "new_combo_ratio": _ratio(sum(obj.new_combo for obj in objects), len(objects)),
        "hitsound_nonzero": _ratio(sum(obj.hit_sound != 0 for obj in objects), len(objects)),
    }
    for bit in (2, 4, 8):
        values[f"hitsound_bit_{bit}"] = _ratio(sum(bool(obj.hit_sound & bit) for obj in objects), len(objects))
    _rhythm_features(beatmap, objects, values)
    _spatial_features(objects, values)
    _slider_features(objects, values)
    _timing_features(beatmap, start_ms, end_ms, objects, values)
    _serialization_features(beatmap, objects, values)
    return WindowFeatures(
        start_ms=start_ms,
        end_ms=end_ms,
        object_count=len(objects),
        values=values,
        sequence_tokens=sequence_tokens(beatmap, objects),
    )


def extract_windows(
    beatmap: Beatmap,
    window_ms: int = 16_000,
    stride_ms: int = 8_000,
    min_objects: int = 12,
) -> tuple[WindowFeatures, ...]:
    if not beatmap.hit_objects:
        return ()
    first = min(obj.time for obj in beatmap.hit_objects)
    last = max(obj.time for obj in beatmap.hit_objects)
    start = (first // stride_ms) * stride_ms
    windows: list[WindowFeatures] = []
    while start <= last:
        item = extract_window_features(beatmap, start, start + window_ms)
        if item.object_count >= min_objects:
            windows.append(item)
        start += stride_ms
    if not windows and len(beatmap.hit_objects) >= min_objects:
        windows.append(extract_window_features(beatmap, first, last + 1))
    return tuple(windows)


def iter_ngrams(tokens: Sequence[str], min_n: int = 1, max_n: int = 5) -> Iterable[str]:
    for size in range(min_n, max_n + 1):
        for index in range(0, len(tokens) - size + 1):
            yield " ".join(tokens[index:index + size])


def hash_ngrams(tokens: Sequence[str], dimensions: int = 4096) -> dict[int, float]:
    counts: collections.Counter[int] = collections.Counter()
    for ngram in iter_ngrams(tokens):
        digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8, person=b"osu-ai-v1").digest()
        raw = int.from_bytes(digest, "little")
        index = raw % dimensions
        sign = 1 if raw & (1 << 63) else -1
        counts[index] += sign
    transformed = {index: math.copysign(math.log1p(abs(count)), count) for index, count in counts.items() if count}
    norm = math.sqrt(sum(value * value for value in transformed.values()))
    return {index: value / norm for index, value in transformed.items()} if norm else {}
