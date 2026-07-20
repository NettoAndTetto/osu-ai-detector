from __future__ import annotations

import codecs
import dataclasses
import math
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class TimingPoint:
    offset: float
    beat_length: float
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool
    effects: int
    raw_beat_length: str

    @property
    def kiai(self) -> bool:
        return bool(self.effects & 1)

    @property
    def slider_velocity(self) -> float | None:
        if self.uninherited or self.beat_length >= 0:
            return None
        return -100.0 / self.beat_length


@dataclasses.dataclass(frozen=True)
class HitObject:
    x: int
    y: int
    time: int
    type_flags: int
    hit_sound: int
    kind: str
    anchors: tuple[tuple[int, int], ...] = ()
    curve_type: str | None = None
    repeats: int = 1
    pixel_length: float | None = None
    raw_pixel_length: str | None = None
    end_time: int | None = None
    raw: str = ""

    @property
    def new_combo(self) -> bool:
        return bool(self.type_flags & 4)

    @property
    def points(self) -> tuple[tuple[int, int], ...]:
        if self.kind in {"spinner", "hold"}:
            return ()
        return ((self.x, self.y),) + self.anchors


@dataclasses.dataclass(frozen=True)
class Beatmap:
    path: Path
    format_version: int
    sections: dict[str, tuple[str, ...]]
    properties: dict[str, dict[str, str]]
    timing_points: tuple[TimingPoint, ...]
    hit_objects: tuple[HitObject, ...]
    raw_text: str
    had_utf8_bom: bool

    @property
    def mode(self) -> int:
        try:
            return int(float(self.properties.get("General", {}).get("Mode", "0")))
        except ValueError:
            return 0

    @property
    def metadata(self) -> dict[str, str]:
        return self.properties.get("Metadata", {})

    @property
    def slider_multiplier(self) -> float:
        try:
            return float(self.properties.get("Difficulty", {}).get("SliderMultiplier", "1.4"))
        except ValueError:
            return 1.4

    @property
    def object_count(self) -> int:
        return len(self.hit_objects)


class BeatmapParseError(ValueError):
    pass


def _decode(path: Path) -> tuple[str, bool]:
    data = path.read_bytes()
    had_bom = data.startswith(codecs.BOM_UTF8)
    if had_bom:
        data = data[len(codecs.BOM_UTF8) :]
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding), had_bom
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), had_bom


def _as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return default


def _parse_timing(lines: tuple[str, ...]) -> tuple[TimingPoint, ...]:
    result: list[TimingPoint] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        parts = stripped.split(",")
        if len(parts) < 2:
            continue
        try:
            result.append(
                TimingPoint(
                    offset=float(parts[0]),
                    beat_length=float(parts[1]),
                    meter=_as_int(parts[2], 4) if len(parts) > 2 else 4,
                    sample_set=_as_int(parts[3], 0) if len(parts) > 3 else 0,
                    sample_index=_as_int(parts[4], 0) if len(parts) > 4 else 0,
                    volume=_as_int(parts[5], 100) if len(parts) > 5 else 100,
                    uninherited=(_as_int(parts[6], 1) == 1) if len(parts) > 6 else True,
                    effects=_as_int(parts[7], 0) if len(parts) > 7 else 0,
                    raw_beat_length=parts[1].strip(),
                )
            )
        except ValueError:
            continue
    return tuple(sorted(result, key=lambda tp: tp.offset))


def _parse_anchor(token: str) -> tuple[int, int] | None:
    pair = token.split(":", 1)
    if len(pair) != 2:
        return None
    try:
        return int(float(pair[0])), int(float(pair[1]))
    except ValueError:
        return None


def _parse_hit_objects(lines: tuple[str, ...]) -> tuple[HitObject, ...]:
    result: list[HitObject] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        parts = stripped.split(",")
        if len(parts) < 5:
            continue
        try:
            x, y, time, type_flags, hit_sound = (
                int(float(parts[0])),
                int(float(parts[1])),
                int(float(parts[2])),
                int(float(parts[3])),
                int(float(parts[4])),
            )
        except ValueError:
            continue

        if type_flags & 128:
            kind = "hold"
        elif type_flags & 8:
            kind = "spinner"
        elif type_flags & 2:
            kind = "slider"
        else:
            kind = "circle"

        anchors: list[tuple[int, int]] = []
        curve_type: str | None = None
        repeats = 1
        pixel_length: float | None = None
        raw_pixel_length: str | None = None
        end_time: int | None = None

        if kind == "slider" and len(parts) > 5:
            curve = parts[5].split("|")
            curve_type = curve[0] if curve else None
            anchors = [a for a in (_parse_anchor(token) for token in curve[1:]) if a is not None]
            repeats = max(_as_int(parts[6], 1), 1) if len(parts) > 6 else 1
            if len(parts) > 7:
                raw_pixel_length = parts[7].strip()
                try:
                    pixel_length = float(raw_pixel_length)
                except ValueError:
                    pixel_length = None
        elif kind in {"spinner", "hold"} and len(parts) > 5:
            end_token = parts[5].split(":", 1)[0]
            end_time = _as_int(end_token, time)

        result.append(
            HitObject(
                x=x,
                y=y,
                time=time,
                type_flags=type_flags,
                hit_sound=hit_sound,
                kind=kind,
                anchors=tuple(anchors),
                curve_type=curve_type,
                repeats=repeats,
                pixel_length=pixel_length,
                raw_pixel_length=raw_pixel_length,
                end_time=end_time,
                raw=stripped,
            )
        )
    return tuple(sorted(result, key=lambda obj: obj.time))


def parse_beatmap(path: str | Path) -> Beatmap:
    path = Path(path)
    text, had_bom = _decode(path)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if header_index is None:
        raise BeatmapParseError(f"Not an osu! beatmap: {path}")
    header = lines[header_index].strip()
    if not header.lower().startswith("osu file format v"):
        raise BeatmapParseError(f"Not an osu! beatmap: {path}")
    try:
        version = int(header.rsplit("v", 1)[1].strip())
    except (IndexError, ValueError) as exc:
        raise BeatmapParseError(f"Invalid osu! format header: {path}") from exc

    mutable_sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            mutable_sections.setdefault(current, [])
        elif current is not None:
            mutable_sections[current].append(line)

    sections = {name: tuple(content) for name, content in mutable_sections.items()}
    properties: dict[str, dict[str, str]] = {}
    for name, content in sections.items():
        values: dict[str, str] = {}
        for line in content:
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            values[key.strip()] = value.strip()
        properties[name] = values

    return Beatmap(
        path=path,
        format_version=version,
        sections=sections,
        properties=properties,
        timing_points=_parse_timing(sections.get("TimingPoints", ())),
        hit_objects=_parse_hit_objects(sections.get("HitObjects", ())),
        raw_text=text,
        had_utf8_bom=had_bom,
    )


def active_redline(beatmap: Beatmap, time: float) -> TimingPoint | None:
    redlines = [tp for tp in beatmap.timing_points if tp.uninherited and tp.beat_length > 0]
    if not redlines:
        return None
    active = redlines[0]
    for point in redlines:
        if point.offset > time:
            break
        active = point
    return active


def inferred_snap(beatmap: Beatmap, time: int, leniency_ms: float = 2.0) -> tuple[int, float] | None:
    point = active_redline(beatmap, time)
    if point is None or not math.isfinite(point.beat_length) or point.beat_length <= 0:
        return None
    beats = (time - point.offset) / point.beat_length
    for divisor in range(1, 17):
        snapped_beats = round(beats * divisor) / divisor
        ideal = point.offset + snapped_beats * point.beat_length
        if abs(time - ideal) < leniency_ms:
            return divisor, ideal
    return 0, time
