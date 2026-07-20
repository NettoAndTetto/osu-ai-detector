"""FastAPI service for the multi-channel detector and its single-page UI.

The HTTP API keeps the original upload/text endpoints and response fields,
while adding explicit content-v2 and optional white-box channels.  White-box
analysis is disabled by default and all uploaded material is held in a scoped
temporary directory for the duration of one request.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import copy
import dataclasses
import gzip
import hashlib
import html as html_lib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError('Install web dependencies with: python -m pip install -e ".[web]"') from exc

from . import __version__
from .detector import (
    DEFAULT_CONTENT_MODEL,
    DEFAULT_DEEP_FUSION_MODEL,
    DEFAULT_FORENSIC_MODEL,
    DEFAULT_FUSION_MODEL,
    DEFAULT_REVISION_REGISTRY,
    Detector,
    DetectorConfig,
)
from .interval_selection import (
    build_bound_candidate_selection,
    build_candidate_selection_policy,
    select_whitebox_intervals,
)
from .parser import BeatmapParseError, parse_beatmap
from .whitebox import (
    DEFAULT_VENDOR_ROOT,
    WhiteboxCheckpoint,
    WhiteboxEngine,
    WhiteboxOptions,
)
from .whitebox_model import (
    DEFAULT_MODEL as DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL,
    WhiteboxDiscriminator,
    unavailable as whitebox_discriminator_unavailable,
)
from .web_ui import HTML_PAGE


FEATURE_CATALOG: dict[str, str] = {
    "explicit_ai_disclosure": "Whether metadata explicitly discloses Mapperatorinator, osuT5, or AI generation.",
    "template_distinctive_ratio": "Legacy Mapperatorinator file-template match ratio (exploratory only).",
    "head_both_even_ratio": "Ratio of hit-object head coordinates that are both even (exploratory only).",
    "coord_mod32_16_ratio": "Ratio of coordinates on the 16 mod 32 lattice (exploratory only).",
    "offgrid_time_mod10_5_ratio": "Ratio of off-beat timestamps concentrated on 10 ms half-steps (exploratory only).",
    "sv_epsilon_1e10_ratio": "Legacy postprocessor +1e-10 SV serialization residue ratio (exploratory only).",
    "slider_near_not_exact_shape_ratio": "Ratio of similar but non-identical slider geometry (exploratory only).",
    "object_count": "Number of hit objects in the beatmap.",
}

COUNT_CATALOG: dict[str, str] = {
    "objects": "Total hit objects",
    "positioned_objects": "Objects included in positional statistics",
    "coordinates": "Hit-object head and slider-anchor coordinates",
    "timing_points": "Timing points",
    "redlines": "Uninherited timing points",
    "greenlines": "Inherited timing points",
    "sliders": "Sliders",
}

EVIDENCE_CATALOG: dict[str, str] = {
    "explicit_disclosure": "Beatmap metadata explicitly discloses Mapperatorinator, osuT5, or AI use.",
    "exact_template_bundle": "Exact legacy file-template match; it may have been copied and is not decisive alone.",
    "v29_cell_center_signature": "V28/V29 16 mod 32 lattice clue; known to produce human false positives.",
    "coordinate_lattice": "Coordinate-quantization clue; exploratory only.",
    "time_lattice": "Timing-quantization clue; exploratory only.",
    "epsilon_sv_signature": "SV floating-point residue clue; exploratory only.",
    "timing_postprocess_pattern": "Legacy postprocessor timing clue; exploratory only.",
    "slider_construction_pattern": "Slider-construction statistical clue; exploratory only.",
}

VERDICT_CATALOG: dict[str, str] = {
    "high_confidence_ai": "Explicit disclosure, or a calibrated in-distribution primary score above the high threshold; human review is still required.",
    "suspicious": "The primary score exceeds the elevated threshold and warrants review; it is not sufficient for a public accusation.",
    "inconclusive": "No sufficiently decisive evidence was found; this does not establish human authorship.",
    "insufficient_data": "Too few objects for stable analysis.",
    "unsupported_mode": "The primary model currently supports osu!standard only.",
}

CONFIG_CATALOG: dict[str, str] = {
    field.name: "deprecated v0.2 exploratory-channel setting; does not control the primary verdict"
    for field in dataclasses.fields(DetectorConfig)
}

ALLOWED_AUDIO_SUFFIXES = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac"}
ALLOWED_CHECKPOINTS = ("v29", "v30", "v31", "v32", "v32-mini")
ANALYSIS_METHODS = ("source", "whitebox", "forensic", "content")
MAX_FORWARD_BATCH_SIZE = 64


@dataclasses.dataclass(frozen=True)
class WebSettings:
    runtime_profile: str = "map-only"
    max_file_bytes: int = 20 * 1024 * 1024
    max_audio_bytes: int = 256 * 1024 * 1024
    max_total_upload_bytes: int = 300 * 1024 * 1024
    max_archive_uncompressed_bytes: int = 300 * 1024 * 1024
    max_archive_entries: int = 1000
    max_maps_per_request: int = 250
    allowed_audio_roots: tuple[Path, ...] = ()
    content_model_path: Path = DEFAULT_CONTENT_MODEL
    forensic_model_path: Path = DEFAULT_FORENSIC_MODEL
    revision_registry_path: Path = DEFAULT_REVISION_REGISTRY
    fusion_model_path: Path = DEFAULT_FUSION_MODEL
    whitebox_vendor_root: Path = DEFAULT_VENDOR_ROOT
    whitebox_discriminator_model_path: Path = DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL
    deep_fusion_model_path: Path = DEFAULT_DEEP_FUSION_MODEL
    cache_root: Path = Path(os.getenv("LOCALAPPDATA", tempfile.gettempdir())) / "osu-ai-detector" / "cache"
    cache_max_bytes: int = 20 * 1024 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "WebSettings":
        def integer(name: str, default: int) -> int:
            value = int(os.getenv(name, str(default)))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            return value

        roots = tuple(
            Path(item).expanduser().resolve()
            for item in os.getenv("OSU_AI_AUDIO_ROOTS", "").split(os.pathsep)
            if item.strip()
        )
        runtime_profile = os.getenv("OSU_AI_WEB_PROFILE", "map-only").strip().casefold()
        if runtime_profile not in {"map-only", "audio-review"}:
            raise ValueError("OSU_AI_WEB_PROFILE must be map-only or audio-review")
        return cls(
            runtime_profile=runtime_profile,
            max_file_bytes=integer("OSU_AI_MAX_FILE_BYTES", cls.max_file_bytes),
            max_audio_bytes=integer("OSU_AI_MAX_AUDIO_BYTES", cls.max_audio_bytes),
            max_total_upload_bytes=integer("OSU_AI_MAX_TOTAL_UPLOAD_BYTES", cls.max_total_upload_bytes),
            max_archive_uncompressed_bytes=integer(
                "OSU_AI_MAX_ARCHIVE_UNCOMPRESSED_BYTES", cls.max_archive_uncompressed_bytes
            ),
            max_archive_entries=integer("OSU_AI_MAX_ARCHIVE_ENTRIES", cls.max_archive_entries),
            max_maps_per_request=integer("OSU_AI_MAX_MAPS", cls.max_maps_per_request),
            allowed_audio_roots=roots,
            content_model_path=Path(os.getenv("OSU_AI_CONTENT_MODEL", str(DEFAULT_CONTENT_MODEL))).expanduser().resolve(),
            forensic_model_path=Path(
                os.getenv("OSU_AI_FORENSIC_MODEL", str(DEFAULT_FORENSIC_MODEL))
            ).expanduser().resolve(),
            revision_registry_path=Path(
                os.getenv("OSU_AI_REVISION_REGISTRY", str(DEFAULT_REVISION_REGISTRY))
            ).expanduser().resolve(),
            fusion_model_path=Path(
                os.getenv("OSU_AI_FUSION_MODEL", str(DEFAULT_FUSION_MODEL))
            ).expanduser().resolve(),
            whitebox_vendor_root=Path(
                os.getenv("OSU_AI_MAPPERATORINATOR_ROOT", str(DEFAULT_VENDOR_ROOT))
            ).expanduser().resolve(),
            whitebox_discriminator_model_path=Path(
                os.getenv(
                    "OSU_AI_WHITEBOX_DISCRIMINATOR_MODEL",
                    str(DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL),
                )
            ).expanduser().resolve(),
            deep_fusion_model_path=Path(
                os.getenv("OSU_AI_DEEP_FUSION_MODEL", str(DEFAULT_DEEP_FUSION_MODEL))
            ).expanduser().resolve(),
            cache_root=Path(
                os.getenv("OSU_AI_CACHE_ROOT", str(cls.cache_root))
            ).expanduser().resolve(),
            cache_max_bytes=integer("OSU_AI_CACHE_MAX_BYTES", cls.cache_max_bytes),
        )


@dataclasses.dataclass(frozen=True)
class MapInput:
    display_name: str
    data: bytes
    requested_audio_name: str | None = None
    archive_audio_name: str | None = None
    archive_audio_data: bytes | None = None


@dataclasses.dataclass(frozen=True)
class AudioInput:
    name: str
    data: bytes


@dataclasses.dataclass(frozen=True)
class WhiteboxRequest:
    enabled: bool = False
    checkpoints: tuple[str, ...] = ("v29", "v30", "v31", "v32", "v32-mini")
    audio_path: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    max_windows: int | None = None
    include_token_details: bool = False
    max_token_details_per_window: int = 256
    device: str = "auto"
    precision: str = "bf16"
    temperature: float = 0.9
    top_p: float = 0.9
    forward_batch_size: int = 1
    discriminator_enabled: bool = False
    deep_fusion_enabled: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "WhiteboxRequest":
        values = dict(values or {})
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("unknown whitebox fields: " + ", ".join(unknown))
        checkpoints = values.get("checkpoints", cls.checkpoints)
        if isinstance(checkpoints, str):
            stripped = checkpoints.strip()
            if stripped.startswith("["):
                checkpoints = json.loads(stripped)
            else:
                checkpoints = [item.strip() for item in stripped.split(",") if item.strip()]
        checkpoints = tuple(str(item) for item in checkpoints)
        if not checkpoints:
            raise ValueError("whitebox checkpoints cannot be empty")
        invalid = sorted(set(checkpoints) - set(ALLOWED_CHECKPOINTS))
        if invalid:
            raise ValueError("unsupported whitebox checkpoints: " + ", ".join(invalid))
        values["checkpoints"] = checkpoints
        result = cls(**values)
        if result.start_ms is not None and result.start_ms < 0:
            raise ValueError("whitebox start_ms cannot be negative")
        if result.end_ms is not None and result.end_ms <= 0:
            raise ValueError("whitebox end_ms must be positive")
        if result.start_ms is not None and result.end_ms is not None and result.start_ms >= result.end_ms:
            raise ValueError("whitebox start_ms must be less than end_ms")
        if result.max_windows is not None and not 1 <= result.max_windows <= 10000:
            raise ValueError("whitebox max_windows must be in [1, 10000]")
        if not 0 <= result.max_token_details_per_window <= 10000:
            raise ValueError("whitebox max_token_details_per_window must be in [0, 10000]")
        if result.precision not in {"fp32", "bf16", "amp"}:
            raise ValueError("whitebox precision must be fp32, bf16 or amp")
        if result.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("whitebox device must be auto, cpu, cuda or mps")
        if not 0 < float(result.top_p) <= 1:
            raise ValueError("whitebox top_p must be in (0, 1]")
        if float(result.temperature) <= 0:
            raise ValueError("whitebox temperature must be positive")
        if (
            isinstance(result.forward_batch_size, bool)
            or not isinstance(result.forward_batch_size, int)
            or not 1 <= result.forward_batch_size <= MAX_FORWARD_BATCH_SIZE
        ):
            raise ValueError(
                f"whitebox forward_batch_size must be an integer in [1, {MAX_FORWARD_BATCH_SIZE}]"
            )
        if result.discriminator_enabled and not result.enabled:
            raise ValueError("whitebox discriminator requires raw whitebox enabled=true")
        if result.deep_fusion_enabled and not result.enabled:
            raise ValueError("audio-backed deep fusion requires raw whitebox enabled=true")
        if result.deep_fusion_enabled and not result.discriminator_enabled:
            raise ValueError("audio-backed deep fusion requires whitebox discriminator enabled=true")
        return result

    def engine_options(self, settings: WebSettings) -> WhiteboxOptions:
        return WhiteboxOptions(
            checkpoints=tuple(WhiteboxCheckpoint(name) for name in self.checkpoints),
            vendor_root=settings.whitebox_vendor_root,
            device=self.device,
            precision=self.precision,
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            forward_batch_size=self.forward_batch_size,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            max_windows=self.max_windows,
            include_token_details=self.include_token_details,
            max_token_details_per_window=self.max_token_details_per_window,
            max_cached_checkpoints=1,
        )


class TextAnalysisRequest(BaseModel):
    filename: str = "map.osu"
    content: str
    detector_config: dict[str, Any] = Field(default_factory=dict)
    include_features: bool = True
    content_model_enabled: bool = True
    forensic_model_enabled: bool = True
    revision_registry_enabled: bool = True
    fusion_model_enabled: bool = True
    whitebox: dict[str, Any] = Field(default_factory=dict)


@dataclasses.dataclass
class AnalysisJob:
    """Thread-safe state for one streamed analysis request.

    Full reports are written as independent JSON shards under ``root/results``.
    The in-memory state only retains compact cards and progress events, so a
    large archive never becomes one giant HTTP response.
    """

    job_id: str
    root: Path
    total_maps: int
    checkpoints: tuple[str, ...]
    selected_methods: tuple[str, ...] = ANALYSIS_METHODS
    created_unix: float = dataclasses.field(default_factory=time.time)
    status: str = "queued"
    phase: str = "queued"
    message: str = "Waiting for analysis resources"
    completed_units: int = 0
    total_units: int = 0
    completed_maps: int = 0
    current_checkpoint: str | None = None
    current_map: int | None = None
    compact_results: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)
    method_results: dict[int, dict[str, dict[str, Any]]] = dataclasses.field(default_factory=dict)
    errors: list[dict[str, str]] = dataclasses.field(default_factory=list)
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    cancel_requested: threading.Event = dataclasses.field(default_factory=threading.Event, repr=False)
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock, repr=False)

    def emit(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        with self.lock:
            self.events.append(
                {
                    "id": len(self.events) + 1,
                    "event": event,
                    "data": copy.deepcopy(dict(payload or {})),
                }
            )

    def progress(self) -> dict[str, Any]:
        with self.lock:
            percent = (
                round(100.0 * self.completed_units / self.total_units, 2)
                if self.total_units
                else 0.0
            )
            return {
                "status": self.status,
                "phase": self.phase,
                "message": self.message,
                "percent": percent,
                "completed_units": self.completed_units,
                "total_units": self.total_units,
                "completed_maps": self.completed_maps,
                "total_maps": self.total_maps,
                "current_checkpoint": self.current_checkpoint,
                "current_map": self.current_map,
            }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "schema_version": 1,
                "job_id": self.job_id,
                "created_unix": self.created_unix,
                "progress": self.progress(),
                "result_count": len(self.compact_results),
                "method_result_count": sum(len(value) for value in self.method_results.values()),
                "selected_methods": list(self.selected_methods),
                "error_count": len(self.errors),
                "errors": copy.deepcopy(self.errors),
            }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _whitebox_cache_key(
    item: MapInput,
    audio_path: Path,
    selection: Mapping[str, Any] | None,
    request: WhiteboxRequest,
    settings: WebSettings,
    discriminator: WhiteboxDiscriminator | None,
    audio_sha256: str,
) -> str | None:
    # Only cache results bound to a published discriminator protocol.  Raw,
    # exploratory or unavailable discriminator runs are intentionally excluded.
    if discriminator is None:
        return None
    model_path = settings.whitebox_discriminator_model_path
    if not model_path.is_file():
        return None
    identities = getattr(discriminator, "required_checkpoint_identities", None)
    if not isinstance(identities, Mapping) or not identities:
        return None
    payload = {
        # v2 binds the relaxed active-runtime identity policy.  Do not reuse
        # v1 entries whose discriminator may have abstained only because of
        # unread README/cache/other-gamemode files.
        "schema": "osu-ai-web-whitebox-cache/v2",
        "detector_version": __version__,
        "map_sha256": hashlib.sha256(item.data).hexdigest(),
        "audio_sha256": audio_sha256,
        "checkpoints": list(request.checkpoints),
        "device_independent_protocol": {
            "precision": request.precision,
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "forward_batch_size": request.forward_batch_size,
            "start_ms": request.start_ms,
            "end_ms": request.end_ms,
            "max_windows": request.max_windows,
            "include_token_details": request.include_token_details,
            "max_token_details_per_window": request.max_token_details_per_window,
        },
        "candidate_selection": selection,
        "required_checkpoint_identities": identities,
        "discriminator_sha256": _sha256_path(model_path),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_path(settings: WebSettings, key: str) -> Path:
    return settings.cache_root / "whitebox-complete" / key[:2] / f"{key}.json.gz"


def _read_whitebox_cache(settings: WebSettings, key: str | None) -> dict[str, Any] | None:
    if not key:
        return None
    path = _cache_path(settings, key)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping) or value.get("cache_key") != key:
            return None
        channel = value.get("whitebox")
        if not isinstance(channel, Mapping):
            return None
        os.utime(path, None)
        return copy.deepcopy(dict(channel))
    except (gzip.BadGzipFile, json.JSONDecodeError, OSError, ValueError):
        return None


def _trim_cache(settings: WebSettings) -> None:
    root = settings.cache_root / "whitebox-complete"
    if not root.is_dir():
        return
    files = [path for path in root.glob("*/*.json.gz") if path.is_file()]
    rows = sorted(((path.stat().st_mtime, path.stat().st_size, path) for path in files), key=lambda row: row[0])
    total = sum(size for _, size, _ in rows)
    for _, size, path in rows:
        if total <= settings.cache_max_bytes:
            break
        try:
            path.unlink()
            total -= size
        except OSError:
            continue


def _write_whitebox_cache(settings: WebSettings, key: str | None, channel: Mapping[str, Any]) -> None:
    if not key:
        return
    discriminator = channel.get("discriminator")
    if not isinstance(discriminator, Mapping) or discriminator.get("decision_usable") is not True:
        return
    if channel.get("status") != "ok":
        return
    path = _cache_path(settings, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump({"cache_key": key, "created_unix": time.time(), "whitebox": channel}, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    os.replace(temporary, path)
    _trim_cache(settings)


def _detector_default_from_env() -> DetectorConfig:
    inline = os.getenv("OSU_AI_DETECTOR_CONFIG")
    file_path = os.getenv("OSU_AI_DETECTOR_CONFIG_FILE")
    if inline and file_path:
        raise ValueError("Set only one of OSU_AI_DETECTOR_CONFIG and OSU_AI_DETECTOR_CONFIG_FILE")
    if file_path:
        inline = Path(file_path).read_text(encoding="utf-8")
    if not inline:
        return DetectorConfig()
    values = json.loads(inline)
    if not isinstance(values, dict):
        raise ValueError("default detector configuration must be a JSON object")
    return DetectorConfig.from_dict(values)


def _safe_name(value: str | None, fallback: str = "upload.osu") -> str:
    name = PurePosixPath((value or fallback).replace("\\", "/")).name
    cleaned = "".join(char for char in name if char.isprintable() and char not in "\x00\r\n")
    return cleaned[:240] or fallback


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _audio_filename(data: bytes) -> str | None:
    in_general = False
    for raw_line in _decode_text(data).splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_general = line.casefold() == "[general]"
        elif in_general and line.casefold().startswith("audiofilename:"):
            value = line.split(":", 1)[1].strip()
            return value or None
    return None


def _zip_audio_info(
    map_info: zipfile.ZipInfo,
    requested_name: str,
    infos_by_name: Mapping[str, zipfile.ZipInfo],
) -> zipfile.ZipInfo | None:
    requested = PurePosixPath(requested_name.replace("\\", "/"))
    if ".." in requested.parts or requested.is_absolute():
        return None
    relative = (PurePosixPath(map_info.filename).parent / requested).as_posix().casefold()
    return infos_by_name.get(relative)


def _decode_upload(filename: str, data: bytes, settings: WebSettings) -> list[MapInput]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".osu":
        if len(data) > settings.max_file_bytes:
            raise ValueError(f"{filename}: file exceeds {settings.max_file_bytes} bytes")
        return [MapInput(_safe_name(filename), data, requested_audio_name=_audio_filename(data))]
    if suffix not in {".osz", ".zip"}:
        raise ValueError(f"{filename}: only .osu, .osz and .zip are accepted as map inputs")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{filename}: invalid ZIP/OSZ archive") from exc
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > settings.max_archive_entries:
            raise ValueError(f"{filename}: archive has too many entries")
        maps = [info for info in infos if info.filename.lower().endswith(".osu")]
        if not maps:
            raise ValueError(f"{filename}: archive contains no .osu files")
        if any(info.file_size > settings.max_file_bytes for info in maps):
            raise ValueError(f"{filename}: a map exceeds the configured file limit")
        infos_by_name = {info.filename.replace("\\", "/").casefold(): info for info in infos}
        audio_cache: dict[str, bytes] = {}
        expanded = 0
        result: list[MapInput] = []
        for info in maps:
            map_data = archive.read(info)
            expanded += len(map_data)
            requested_audio = _audio_filename(map_data)
            audio_info = _zip_audio_info(info, requested_audio, infos_by_name) if requested_audio else None
            audio_data = None
            audio_name = None
            if audio_info is not None and Path(audio_info.filename).suffix.lower() in ALLOWED_AUDIO_SUFFIXES:
                key = audio_info.filename.casefold()
                if audio_info.file_size > settings.max_audio_bytes:
                    raise ValueError(f"{filename}/{audio_info.filename}: audio exceeds the configured limit")
                if key not in audio_cache:
                    audio_cache[key] = archive.read(audio_info)
                    expanded += len(audio_cache[key])
                audio_data = audio_cache[key]
                audio_name = _safe_name(audio_info.filename, "audio.mp3")
            if expanded > settings.max_archive_uncompressed_bytes:
                raise ValueError(f"{filename}: expanded map/audio content exceeds the configured limit")
            result.append(
                MapInput(
                    f"{_safe_name(filename)}::{_safe_name(info.filename)}",
                    map_data,
                    requested_audio_name=requested_audio,
                    archive_audio_name=audio_name,
                    archive_audio_data=audio_data,
                )
            )
        return result


def _resolve_local_audio(value: str, settings: WebSettings) -> Path:
    if not settings.allowed_audio_roots:
        raise ValueError("local audio paths are disabled; configure OSU_AI_AUDIO_ROOTS or upload audio")
    candidate = Path(value).expanduser().resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("local audio path is not a file")
    if not any(candidate.is_relative_to(root) for root in settings.allowed_audio_roots):
        raise ValueError("local audio path is outside OSU_AI_AUDIO_ROOTS")
    if candidate.suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
        raise ValueError("local audio file type is not supported")
    if candidate.stat().st_size > settings.max_audio_bytes:
        raise ValueError("local audio exceeds the configured limit")
    return candidate


def _write_audio(root: Path, name: str, data: bytes, cache: dict[str, Path]) -> Path:
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        suffix = ".bin"
    key = hashlib.sha256(data).hexdigest() + suffix
    if key not in cache:
        path = root / f"audio-{len(cache):03d}{suffix}"
        path.write_bytes(data)
        cache[key] = path
    return cache[key]


def _filter_candidate_intervals_for_custom_range(
    selection: dict[str, Any],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> dict[str, Any]:
    """Intersect automatic proposals with a user range without scanning a song.

    The engine also receives the explicit start/end settings.  Keeping that
    second gate is intentional: even if no automatic proposal intersects the
    requested range, the request returns zero raw windows rather than silently
    falling back to a whole-song search.
    """

    if start_ms is None and end_ms is None:
        return selection
    filtered: list[list[int]] = []
    for raw in selection.get("intervals_ms", []):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        start, end = int(raw[0]), int(raw[1])
        bounded_start = max(start, start_ms or 0)
        bounded_end = min(end, end_ms) if end_ms is not None else end
        if bounded_start < bounded_end:
            filtered.append([bounded_start, bounded_end])
    result = dict(selection)
    result["custom_range_filter"] = {
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "automatic_candidate_count": len(selection.get("intervals_ms", [])),
        "intersecting_candidate_count": len(filtered),
        "behavior_when_no_overlap": (
            "retain automatic proposals behind the engine range gate; score zero windows, never the whole song"
        ),
    }
    if filtered:
        result["intervals_ms"] = filtered
        result["interval_details"] = [
            {
                "start_ms": start,
                "end_ms": end,
                "reasons": ["automatic_candidate_intersected_with_custom_range"],
            }
            for start, end in filtered
        ]
        result["total_selected_ms"] = sum(end - start for start, end in filtered)
    return result


def _whitebox_candidate_selection(
    path: Path,
    report: Mapping[str, Any],
    request: WhiteboxRequest,
    *,
    content_model_sha256: str | None,
) -> dict[str, Any]:
    """Build the per-map bounded search record used by every Web white-box run."""

    beatmap = parse_beatmap(path)
    content = report.get("analysis_channels", {}).get("content_v2", {})
    canonical_error: str | None = None
    try:
        if not isinstance(content, dict) or content.get("available") is not True:
            raise ValueError("content analysis is unavailable")
        if not isinstance(content_model_sha256, str):
            raise ValueError("content artifact SHA-256 is unavailable")
        selection = build_bound_candidate_selection(
            beatmap,
            content,
            content_model_sha256=content_model_sha256,
            windows_per_interval=1,
        )
        selection["search_mode"] = "calibration_bound_label_free_auto_candidates"
        selection["calibration_eligible"] = True
        selection["protocol_notice"] = (
            "Candidate intervals were proposed automatically by the active content model and bound to its "
            "model ID and file SHA-256. Calibrated scoring remains conditional on the full protocol audit."
        )
    except (TypeError, ValueError) as exc:
        canonical_error = str(exc)
        selection = select_whitebox_intervals(beatmap, None)
        if not selection.get("intervals_ms"):
            raise ValueError("cannot build a bounded label-free white-box search: beatmap has no scorable objects")
        selection.update(
            {
                "content_model_sha256": content_model_sha256,
                "selection_policy": build_candidate_selection_policy(1),
                "search_mode": "safe_label_free_fallback",
                "calibration_eligible": False,
                "protocol_notice": (
                    "The content model or its immutable identity is unavailable. Raw diagnostics are limited "
                    "to the opening, closing, and densest object intervals; the full song is not scanned, and "
                    "calibrated secondary scoring must abstain."
                ),
                "protocol_incomplete_reason": canonical_error,
            }
        )

    manual_overrides = {
        "start_ms": request.start_ms,
        "end_ms": request.end_ms,
        "max_windows": request.max_windows,
    }
    if any(value is not None for value in manual_overrides.values()):
        selection = _filter_candidate_intervals_for_custom_range(
            selection,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
        )
        selection.update(
            {
                "search_mode": "bounded_custom_raw_exploration",
                "calibration_eligible": False,
                "manual_search_overrides": manual_overrides,
                "protocol_notice": (
                    "Manual range or window limits are retained for raw diagnostics, restricted to their "
                    "intersection with automatic candidate intervals. These non-default settings invalidate "
                    "the calibrated protocol, so secondary scoring must remain diagnostic and abstain."
                ),
            }
        )
    return selection


def _load_whitebox_discriminator(
    request: WhiteboxRequest, settings: WebSettings
) -> tuple[WhiteboxDiscriminator | None, dict[str, Any] | None]:
    if not request.discriminator_enabled:
        return None, None
    path = settings.whitebox_discriminator_model_path
    if not path.is_file():
        return None, whitebox_discriminator_unavailable(
            f"white-box discriminator artifact is missing: {path}; the channel abstains",
            model_path=path,
        )
    try:
        return WhiteboxDiscriminator.from_path(path), None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, whitebox_discriminator_unavailable(
            f"cannot load white-box discriminator artifact: {type(exc).__name__}: {exc}",
            model_path=path,
        )


def _analyze_maps(
    maps: list[MapInput],
    config: DetectorConfig,
    include_features: bool,
    *,
    settings: WebSettings,
    content_model_enabled: bool,
    forensic_model_enabled: bool,
    revision_registry_enabled: bool,
    fusion_model_enabled: bool,
    whitebox: WhiteboxRequest,
    uploaded_audio: AudioInput | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    detector = Detector(
        config,
        content_model_path=settings.content_model_path,
        content_model_enabled=content_model_enabled,
        forensic_model_path=settings.forensic_model_path,
        forensic_model_enabled=forensic_model_enabled,
        revision_registry_path=settings.revision_registry_path,
        revision_registry_enabled=revision_registry_enabled,
        fusion_model_path=settings.fusion_model_path,
        fusion_model_enabled=fusion_model_enabled,
        deep_fusion_model_path=settings.deep_fusion_model_path,
        deep_fusion_model_enabled=whitebox.deep_fusion_enabled,
    )
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    engine = WhiteboxEngine(whitebox.engine_options(settings)) if whitebox.enabled else None
    discriminator, discriminator_error = _load_whitebox_discriminator(whitebox, settings)
    content_model_sha256 = detector.observed_base_model_sha256.get("content_v2")

    def attach_discriminator(
        report: dict[str, Any], raw_result: dict[str, Any], *, has_audio: bool
    ) -> None:
        if not whitebox.discriminator_enabled:
            raw_result["discriminator"] = {
                "enabled": False,
                "status": "not_requested",
                "available": False,
                "used_for_final_verdict": False,
            }
            return
        if discriminator is None:
            detail = dict(discriminator_error or whitebox_discriminator_unavailable("model unavailable"))
        else:
            try:
                detail = discriminator.score(raw_result)
            except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
                detail = whitebox_discriminator_unavailable(
                    f"white-box discriminator scoring failed: {type(exc).__name__}: {exc}",
                    model_path=settings.whitebox_discriminator_model_path,
                )
        detail["enabled"] = True
        detail["model_path"] = str(settings.whitebox_discriminator_model_path)
        detail["used_for_final_verdict"] = False
        raw_result["discriminator"] = detail
        # Deep fusion is an explicitly requested, audio-backed post-analysis
        # channel.  It is deliberately absent from ordinary map-only reports
        # and from requests where no audio was available.  When requested with
        # audio, artifact/binding/calibration failures are returned as a
        # detailed abstention instead of being converted to a numeric zero.
        if whitebox.deep_fusion_enabled and has_audio:
            cheap_fusion = report.get("analysis_channels", {}).get("cheap_fusion_v1", {})
            deep_detail = detector.analyze_deep_fusion(
                cheap_fusion,
                detail,
                whitebox_model_path=settings.whitebox_discriminator_model_path,
            )
            deep_detail["audio_required_and_present"] = True
            deep_detail["independent_result_only"] = True
            deep_detail["map_only_verdict_unchanged"] = True
            report.setdefault("analysis_channels", {})["deep_fusion_v1"] = deep_detail
            raw_result["deep_fusion"] = deep_detail
    local_audio: Path | None = None
    if whitebox.enabled and whitebox.audio_path:
        local_audio = _resolve_local_audio(whitebox.audio_path, settings)
    try:
        with tempfile.TemporaryDirectory(prefix="osu-ai-web-") as directory:
            root = Path(directory)
            audio_cache: dict[str, Path] = {}
            uploaded_audio_path = (
                _write_audio(root, uploaded_audio.name, uploaded_audio.data, audio_cache)
                if uploaded_audio is not None
                else None
            )
            for index, item in enumerate(maps):
                path = root / f"map-{index:05d}.osu"
                path.write_bytes(item.data)
                try:
                    report = detector.analyze(path).to_dict(include_features=include_features)
                    report["path"] = item.display_name
                    if engine is None:
                        whitebox_result: dict[str, Any] = {
                            "enabled": False,
                            "status": "not_requested",
                            "used_for_final_verdict": False,
                            "discriminator": {
                                "enabled": False,
                                "status": "not_requested",
                                "available": False,
                                "used_for_final_verdict": False,
                            },
                        }
                    else:
                        archive_audio = (
                            _write_audio(root, item.archive_audio_name or "audio.mp3", item.archive_audio_data, audio_cache)
                            if item.archive_audio_data is not None
                            else None
                        )
                        selected_audio = uploaded_audio_path or local_audio or archive_audio
                        if selected_audio is None:
                            whitebox_result = {
                                "enabled": True,
                                "status": "error",
                                "used_for_final_verdict": False,
                                "error": {
                                    "type": "AudioRequired",
                                    "message": (
                                        "white-box analysis requires an uploaded audio file, an allowed local "
                                        "audio_path, or AudioFilename content present in the OSZ"
                                    ),
                                },
                            }
                        else:
                            candidate_selection = _whitebox_candidate_selection(
                                path,
                                report,
                                whitebox,
                                content_model_sha256=content_model_sha256,
                            )
                            whitebox_result = engine.score(
                                path,
                                selected_audio,
                                candidate_interval_selection=candidate_selection,
                            )
                            # Real engines attach this themselves.  setdefault
                            # also keeps test/dry-run engines and future remote
                            # adapters lossless at the HTTP boundary.
                            whitebox_result.setdefault(
                                "candidate_interval_selection", candidate_selection
                            )
                            whitebox_result["enabled"] = True
                            whitebox_result["used_for_final_verdict"] = False
                            whitebox_result["audio_source"] = (
                                "uploaded" if uploaded_audio_path else "allowed_local_path" if local_audio else "osz"
                            )
                        attach_discriminator(
                            report,
                            whitebox_result,
                            has_audio=selected_audio is not None,
                        )
                    report.setdefault("analysis_channels", {})["whitebox"] = whitebox_result
                    reports.append(report)
                except (OSError, ValueError, BeatmapParseError) as exc:
                    errors.append({
                        "path": item.display_name,
                        "error": str(exc).replace(str(path), item.display_name),
                    })
    finally:
        if engine is not None:
            engine.close()
    return reports, errors


def _build_response(
    maps: list[MapInput],
    config: DetectorConfig,
    include_features: bool,
    started: float,
    *,
    settings: WebSettings,
    content_model_enabled: bool,
    forensic_model_enabled: bool,
    revision_registry_enabled: bool,
    fusion_model_enabled: bool,
    whitebox: WhiteboxRequest,
    uploaded_audio: AudioInput | None = None,
) -> dict[str, Any]:
    reports, errors = _analyze_maps(
        maps,
        config,
        include_features,
        settings=settings,
        content_model_enabled=content_model_enabled,
        forensic_model_enabled=forensic_model_enabled,
        revision_registry_enabled=revision_registry_enabled,
        fusion_model_enabled=fusion_model_enabled,
        whitebox=whitebox,
        uploaded_audio=uploaded_audio,
    )
    verdicts = collections.Counter(report["verdict"] for report in reports)
    content_available = sum(
        bool(report.get("analysis_channels", {}).get("content_v2", {}).get("available"))
        for report in reports
    )
    content_abstentions = sum(
        bool(report.get("analysis_channels", {}).get("content_v2", {}).get("ood", {}).get("abstain"))
        for report in reports
    )
    forensic_status = collections.Counter(
        report.get("analysis_channels", {}).get("revision_forensics_v3", {}).get("status", "missing")
        for report in reports
    )
    registry_status = collections.Counter(
        report.get("analysis_channels", {}).get("public_revision_registry", {}).get("status", "missing")
        for report in reports
    )
    fusion_status = collections.Counter(
        report.get("analysis_channels", {}).get("cheap_fusion_v1", {}).get("status", "missing")
        for report in reports
    )
    whitebox_status = collections.Counter(
        report.get("analysis_channels", {}).get("whitebox", {}).get("status", "missing")
        for report in reports
    )
    discriminator_status = collections.Counter(
        report.get("analysis_channels", {})
        .get("whitebox", {})
        .get("discriminator", {})
        .get("status", "missing")
        for report in reports
    )
    deep_fusion_status = collections.Counter(
        report.get("analysis_channels", {}).get("deep_fusion_v1", {}).get("status", "missing")
        for report in reports
        if "deep_fusion_v1" in report.get("analysis_channels", {})
    )
    default_config = DetectorConfig()
    return {
        "schema_version": 2,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "summary": {
            "submitted_maps": len(maps),
            "analyzed_maps": len(reports),
            "error_count": len(errors),
            "verdict_counts": dict(sorted(verdicts.items())),
            "content_v2_available_maps": content_available,
            "content_v2_ood_or_unavailable_abstentions": content_abstentions,
            "revision_forensics_v3_status_counts": dict(sorted(forensic_status.items())),
            "public_revision_registry_status_counts": dict(sorted(registry_status.items())),
            "cheap_fusion_v1_status_counts": dict(sorted(fusion_status.items())),
            "whitebox_status_counts": dict(sorted(whitebox_status.items())),
            "whitebox_discriminator_status_counts": dict(sorted(discriminator_status.items())),
            "deep_fusion_v1_status_counts": dict(sorted(deep_fusion_status.items())),
        },
        "detector": {
            "scope": Detector.scope,
            "effective_config": config.to_dict(),
            "is_frozen_calibrated_profile": config == default_config,
            "content_model_path": str(settings.content_model_path),
            "content_model_enabled": content_model_enabled,
            "forensic_model_path": str(settings.forensic_model_path),
            "forensic_model_enabled": forensic_model_enabled,
            "revision_registry_path": str(settings.revision_registry_path),
            "revision_registry_enabled": revision_registry_enabled,
            "fusion_model_path": str(settings.fusion_model_path),
            "fusion_model_enabled": fusion_model_enabled,
            "whitebox_request": dataclasses.asdict(whitebox),
            "whitebox_discriminator_model_path": str(settings.whitebox_discriminator_model_path),
            "deep_fusion_model_path": str(settings.deep_fusion_model_path),
            "deep_fusion_model_enabled": whitebox.deep_fusion_enabled,
            "score_notice": (
                "scores are not authorship probabilities; uncalibrated channels abstain; v0.2 and white-box "
                "discriminator diagnostics do not control the final verdict; optional audio-backed deep fusion "
                "is reported independently and never silently rewrites the map-only verdict"
            ),
        },
        "feature_catalog": FEATURE_CATALOG if include_features else {},
        "count_catalog": COUNT_CATALOG,
        "evidence_catalog": EVIDENCE_CATALOG,
        "verdict_catalog": VERDICT_CATALOG,
        "config_catalog": CONFIG_CATALOG,
        "reports": reports,
        "errors": errors,
    }


def _first_numeric(value: Any, key: str) -> float | None:
    """Find the first finite numeric field without retaining a large traversal."""

    if isinstance(value, Mapping):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            number = float(candidate)
            if number == number and number not in {float("inf"), float("-inf")}:
                return number
        for child in value.values():
            found = _first_numeric(child, key)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _first_numeric(child, key)
            if found is not None:
                return found
    return None


def _channel_percentile(channel: Mapping[str, Any] | None) -> float | None:
    if not isinstance(channel, Mapping):
        return None
    if channel.get("decision_usable") is False or channel.get("abstain") is True:
        return None
    p_value = _first_numeric(channel, "human_null_p_value")
    if p_value is None:
        p_value = _first_numeric(channel, "development_upper_tail_p")
    if p_value is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - p_value)) * 100.0, 2)


METHOD_COPY = {
    "source": ("Source Provenance Check", "Checks explicit AI-use disclosures and exact matches to registered files."),
    "whitebox": (
        "Mapperatorinator Model Agreement",
        "Runs the original Mapperatorinator models and measures agreement with their generation process. Requires audio and takes longer.",
    ),
    "forensic": ("Generator Trace Check", "Checks coordinates, timing, SV, and file formatting for mechanical generation traces."),
    "content": ("Mapping Structure Check", "Compares rhythm, movement, and object arrangement with the human reference set."),
}


def _threshold_values(channel: Mapping[str, Any]) -> dict[str, float]:
    raw = channel.get("thresholds")
    raw = raw if isinstance(raw, Mapping) else {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        candidate = value.get("threshold") if isinstance(value, Mapping) else value
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            result[str(key)] = float(candidate)
    return result


def _threshold_flags(channel: Mapping[str, Any]) -> dict[str, bool]:
    raw = channel.get("threshold_flags")
    raw = raw if isinstance(raw, Mapping) else {}
    return {
        str(key): bool(value.get("exceeded")) if isinstance(value, Mapping) else bool(value)
        for key, value in raw.items()
    }


def _metric_status(channel: Mapping[str, Any]) -> tuple[str, str, str]:
    if channel.get("decision_usable") is not True:
        return "unable", "Unavailable", "bluegrey"
    calibration = channel.get("calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    flags = _threshold_flags(channel if channel.get("threshold_flags") else calibration)
    if flags.get("high") or any(key.startswith("high") and value for key, value in flags.items()):
        return "high", "Highly anomalous", "red"
    if flags.get("elevated") or any(key.startswith("elevated") and value for key, value in flags.items()):
        return "elevated", "Anomalous", "orange"
    percentile = _channel_percentile(channel)
    if percentile is not None and percentile >= 95.0:
        return "near", "Near threshold", "yellow"
    return "below", "Below alert threshold", "grey"


def _window_text(row: Mapping[str, Any]) -> str | None:
    start = row.get("window_start_ms", row.get("start_ms"))
    end = row.get("window_end_ms", row.get("end_ms"))
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return f"{float(start) / 1000:.1f}–{float(end) / 1000:.1f} seconds"


def _reason_family(method: str, row: Mapping[str, Any]) -> tuple[str, str]:
    feature = str(row.get("base_feature") or row.get("feature") or "").casefold()
    family = str(row.get("family") or "").casefold()
    token = f"{family} {feature}"
    mappings = {
        "forensic": (
            (("coordinate", "position", "grid"), "Coordinate-grid trace"),
            (("timing", "quant"), "Timing-quantization trace"),
            (("sv", "float", "decimal"), "SV or decimal residue"),
            (("template", "format", "serialization"), "File-template trace"),
        ),
        "content": (
            (("rhythm", "timing", "interval"), "Rhythm-structure anomaly"),
            (("movement", "distance", "angle", "velocity"), "Movement-pattern anomaly"),
            (("slider",), "Slider-structure anomaly"),
            (("object", "ngram", "ordering", "type"), "Object-ordering anomaly"),
        ),
        "whitebox": (
            (("token", "vocabulary", "likelihood", "log_prob"), "Token-likelihood anomaly"),
            (("rank", "top_p", "generation_policy"), "Generation-candidate rank anomaly"),
            (("checkpoint", "coverage"), "Multi-model agreement anomaly"),
            (("sequence", "detectgpt", "lrr"), "Generated-sequence feature anomaly"),
        ),
    }
    for needles, label in mappings.get(method, ()):
        if any(needle in token for needle in needles):
            return label, label
    defaults = {
        "content": "Local mapping-structure anomaly",
        "forensic": "Mechanical generation trace",
        "whitebox": "Original-model prediction anomaly",
    }
    return defaults.get(method, "Anomalous metric"), feature or family or "Unnamed metric"


def _reason_observation(method: str, row: Mapping[str, Any], value: Any) -> tuple[str, str]:
    """Return a compact observation and a plain-language explanation.

    The raw detector features do not share one monotonic direction.  In
    particular, a zero-valued entropy feature can be strong positive evidence
    because zero means that coordinates are concentrated on one grid residue.
    Keep that raw observation separate from the learned model contribution so
    the UI never implies that every feature is "larger = more suspicious".
    """

    feature = str(row.get("base_feature") or row.get("feature") or "").casefold()
    numeric = float(value) if isinstance(value, (int, float)) else None
    raw_text = f"{numeric:.4f}" if numeric is not None else str(value or "—")

    if method == "forensic" and "entropy_mod" in feature:
        return (
            f"Grid-residue entropy {raw_text}",
            "Dispersion of coordinates across fixed grid residues: values near 0 are concentrated, while values near 1 are dispersed",
        )
    if method == "forensic" and feature == "timing_sv_quantized_001" and numeric is not None:
        return (
            f"0.01-step matches {numeric * 100:.1f}%",
            "Share of SV values in this interval that fall exactly on a 0.01 step",
        )
    if method == "forensic" and feature == "file_trailing_colon" and numeric is not None:
        return (
            f"Line-ending template matches {numeric * 100:.1f}%",
            "Share of hit-object lines in this interval that end with the template-style colon",
        )
    if numeric is not None and 0.0 <= numeric <= 1.0:
        return f"Raw value {raw_text}", "Raw ratio or normalized value for this metric"
    return f"Raw value {raw_text}", "Raw value supplied to the model for this metric"


def _positive_reasons(method: str, channel: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    if method == "whitebox":
        windows = channel.get("windows")
        if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
            for window in windows:
                if not isinstance(window, Mapping):
                    continue
                contributions = window.get("top_positive_contributions")
                if not isinstance(contributions, Sequence) or isinstance(contributions, (str, bytes)):
                    continue
                for raw in contributions:
                    if isinstance(raw, Mapping):
                        rows.append({**raw, "checkpoint": window.get("checkpoint"), "start_ms": window.get("start_ms"), "end_ms": window.get("end_ms")})
    else:
        raw_rows = channel.get("source_anchored_positive_evidence") if method == "forensic" else channel.get("top_evidence")
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
            rows.extend(row for row in raw_rows if isinstance(row, Mapping))
    positive = [row for row in rows if float(row.get("contribution") or 0.0) > 0]
    positive.sort(key=lambda row: -float(row.get("contribution") or 0.0))
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in positive:
        family, metric = _reason_family(method, row)
        if family in seen:
            continue
        seen.add(family)
        value = row.get("raw_value", row.get("value"))
        contribution = float(row.get("contribution") or 0.0)
        observation, explanation = _reason_observation(method, row, value)
        feature = str(row.get("base_feature") or row.get("feature") or metric)
        detail_parts = [
            explanation,
            f"Raw metric: {feature}",
            f"{observation}",
            f"Effect on the internal anomaly score: {contribution:+.4f}",
            "Details are ranked by effect size, not raw value; suspicious directions differ between metrics",
        ]
        window = _window_text(row)
        if window:
            detail_parts.append(f"Interval: {window}")
        if row.get("checkpoint"):
            detail_parts.append(f"Model: {row['checkpoint']}")
        result.append({
            "label": family,
            "value": value,
            "observation": observation,
            "contribution": contribution,
            "impact": "increase",
            "detail": "; ".join(detail_parts),
            "window": window,
            "checkpoint": row.get("checkpoint"),
        })
        if len(result) == 5:
            break
    return result


def _compact_method(report: Mapping[str, Any], index: int, method: str) -> dict[str, Any]:
    channels_value = report.get("analysis_channels")
    channels = channels_value if isinstance(channels_value, Mapping) else {}
    title, description = METHOD_COPY[method]
    identity_value = report.get("map_identity")
    identity = identity_value if isinstance(identity_value, Mapping) else {}
    base: dict[str, Any] = {
        "index": index,
        "path": str(report.get("path") or f"Beatmap {index + 1}"),
        "title": str(identity.get("title") or ""),
        "version": str(identity.get("version") or ""),
        "creator": str(identity.get("creator") or ""),
        "method": method,
        "name": title,
        "description": description,
        "percentile_notice": "Anomaly percentile relative to the human reference set; this is not the probability of AI authorship.",
    }
    if method == "source":
        registry_value = channels.get("public_revision_registry")
        registry = registry_value if isinstance(registry_value, Mapping) else {}
        trace_value = report.get("decision_trace")
        trace = trace_value if isinstance(trace_value, Mapping) else {}
        exact = bool(registry.get("known_exact_positive"))
        disclosure = bool(trace.get("direct_disclosure"))
        available = registry.get("available", True) is not False
        reasons: list[dict[str, Any]] = []
        if exact:
            reasons.append({"label": "Registered file match", "value": "Exact SHA-256 match", "direction": "up", "detail": "This file is byte-identical to a registered historical revision."})
        if disclosure:
            reasons.append({"label": "Explicit AI-use disclosure", "value": "Metadata match", "direction": "up", "detail": "Beatmap metadata explicitly mentions Mapperatorinator, osuT5, or AI generation."})
        if exact or disclosure:
            state, label, tone = "match", "Explicit match", "red"
        elif not available:
            state, label, tone = "unable", "Unavailable", "bluegrey"
        else:
            state, label, tone = "clear", "No match found", "grey"
        return base | {
            "kind": "fact",
            "state": state,
            "label": label,
            "tone": tone,
            "exact_registry_match": exact,
            "explicit_disclosure": disclosure,
            "reasons": reasons[:5],
            "technical": {
                "registry_available": available,
                "registry_status": registry.get("status"),
                "match_policy": registry.get("match_policy"),
            },
        }

    if method == "content":
        channel_value = channels.get("content_v2")
    elif method == "forensic":
        channel_value = channels.get("revision_forensics_v3")
    else:
        whitebox_value = channels.get("whitebox")
        whitebox = whitebox_value if isinstance(whitebox_value, Mapping) else {}
        channel_value = whitebox.get("discriminator")
    channel = channel_value if isinstance(channel_value, Mapping) else {}
    state, label, tone = _metric_status(channel)
    score = channel.get("score")
    if score is None and isinstance(channel.get("aggregate"), Mapping):
        score = channel["aggregate"].get("ranking_score")
    calibration = channel.get("calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    protocol_audit = channel.get("calibration_protocol_audit")
    protocol_audit = protocol_audit if isinstance(protocol_audit, Mapping) else {}
    mismatch_checkpoints = sorted(
        {
            str(item.get("checkpoint"))
            for item in protocol_audit.get("identity_mismatches", [])
            if isinstance(item, Mapping) and item.get("checkpoint")
        }
    )
    threshold_source = channel if channel.get("thresholds") else calibration
    return base | {
        "kind": "statistical",
        "state": state,
        "label": label,
        "tone": tone,
        "percentile": _channel_percentile(channel),
        "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        "thresholds": _threshold_values(threshold_source),
        "threshold_flags": _threshold_flags(threshold_source),
        "reasons": _positive_reasons(method, channel),
        "model_id": channel.get("model_id"),
        "technical": {
            "model_id": channel.get("model_id"),
            "decision_usable": channel.get("decision_usable"),
            "calibrated": channel.get("calibrated", calibration.get("calibrated")),
            "human_null_size": channel.get("calibration_maps", calibration.get("human_null_size")),
            "calibration_reason": (
                calibration.get("reason") if channel.get("decision_usable") is not True else None
            ),
            "protocol_mismatch_checkpoints": mismatch_checkpoints or None,
        },
    }


def _compact_report(report: Mapping[str, Any], index: int, methods: Sequence[str] = ANALYSIS_METHODS) -> dict[str, Any]:
    identity = report.get("map_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    return {
        "index": index,
        "path": str(report.get("path") or f"Beatmap {index + 1}"),
        "title": str(identity.get("title") or ""),
        "version": str(identity.get("version") or ""),
        "creator": str(identity.get("creator") or ""),
        "methods": {method: _compact_method(report, index, method) for method in methods},
        "reasons": [],
    }


def _merge_whitebox_parts(
    parts: Sequence[Mapping[str, Any]],
    candidate_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not parts:
        return {
            "enabled": True,
            "status": "error",
            "used_for_final_verdict": False,
            "checkpoints": [],
            "error": {"type": "NoCheckpointResults", "message": "No white-box checkpoint shards were produced"},
        }
    merged = copy.deepcopy(dict(parts[0]))
    checkpoint_results: list[dict[str, Any]] = []
    configs: dict[str, Any] = {}
    for part in parts:
        checkpoint_results.extend(copy.deepcopy(list(part.get("checkpoints") or [])))
        availability = part.get("availability")
        if isinstance(availability, Mapping):
            raw_configs = availability.get("checkpoint_configs")
            if isinstance(raw_configs, Mapping):
                configs.update(raw_configs)
    successful = sum(item.get("status") == "ok" for item in checkpoint_results)
    unavailable = sum(item.get("status") == "unavailable" for item in checkpoint_results)
    if successful == len(checkpoint_results) and checkpoint_results:
        status = "ok"
    elif successful:
        status = "partial"
    elif checkpoint_results and unavailable == len(checkpoint_results):
        status = "unavailable"
    else:
        status = "error"
    merged["status"] = status
    merged["checkpoints"] = checkpoint_results
    merged["summary"] = {
        "requested_checkpoints": len(checkpoint_results),
        "successful_checkpoints": successful,
        "unavailable_checkpoints": unavailable,
        "error_checkpoints": len(checkpoint_results) - successful - unavailable,
        "total_scored_tokens": sum(
            int(item.get("coverage", {}).get("token_count", 0))
            for item in checkpoint_results
        ),
    }
    if isinstance(merged.get("availability"), Mapping):
        merged["availability"] = dict(merged["availability"])
        merged["availability"]["checkpoint_configs"] = configs
        merged["availability"]["available"] = bool(configs) and all(configs.values())
    if candidate_selection is not None:
        merged["candidate_interval_selection"] = copy.deepcopy(dict(candidate_selection))
    merged["enabled"] = True
    merged["used_for_final_verdict"] = False
    return merged


def _attach_streamed_postprocessors(
    report: dict[str, Any],
    whitebox_result: dict[str, Any],
    *,
    detector: Detector,
    request: WhiteboxRequest,
    settings: WebSettings,
    discriminator: WhiteboxDiscriminator | None,
    discriminator_error: Mapping[str, Any] | None,
    has_audio: bool,
) -> None:
    if not request.discriminator_enabled:
        whitebox_result["discriminator"] = {
            "enabled": False,
            "status": "not_requested",
            "available": False,
            "used_for_final_verdict": False,
        }
    else:
        if discriminator is None:
            detail = dict(
                discriminator_error or whitebox_discriminator_unavailable("model unavailable")
            )
        else:
            try:
                detail = discriminator.score(whitebox_result)
            except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
                detail = whitebox_discriminator_unavailable(
                    f"white-box discriminator scoring failed: {type(exc).__name__}: {exc}",
                    model_path=settings.whitebox_discriminator_model_path,
                )
        detail["enabled"] = True
        detail["model_path"] = str(settings.whitebox_discriminator_model_path)
        detail["used_for_final_verdict"] = False
        whitebox_result["discriminator"] = detail
        if request.deep_fusion_enabled and has_audio:
            cheap_fusion = report.get("analysis_channels", {}).get("cheap_fusion_v1", {})
            deep_detail = detector.analyze_deep_fusion(
                cheap_fusion,
                detail,
                whitebox_model_path=settings.whitebox_discriminator_model_path,
            )
            deep_detail["audio_required_and_present"] = True
            deep_detail["independent_result_only"] = True
            deep_detail["map_only_verdict_unchanged"] = True
            report.setdefault("analysis_channels", {})["deep_fusion_v1"] = deep_detail
            whitebox_result["deep_fusion"] = deep_detail


def _job_set_progress(job: AnalysisJob, **values: Any) -> None:
    with job.lock:
        for key, value in values.items():
            setattr(job, key, value)
        payload = job.progress()
    job.emit("progress", payload)


def _publish_job_method(job: AnalysisJob, index: int, method: str, report: Mapping[str, Any]) -> None:
    compact = _compact_method(report, index, method)
    with job.lock:
        methods = job.method_results.setdefault(index, {})
        methods[method] = compact
        bundle = job.compact_results.setdefault(
            index,
            {
                "index": index,
                "path": str(report.get("path") or f"Beatmap {index + 1}"),
                "title": "",
                "version": "",
                "creator": "",
                "methods": {},
                "reasons": [],
            },
        )
        identity = report.get("map_identity")
        identity = identity if isinstance(identity, Mapping) else {}
        bundle.update({
            "path": str(report.get("path") or f"Beatmap {index + 1}"),
            "title": str(identity.get("title") or ""),
            "version": str(identity.get("version") or ""),
            "creator": str(identity.get("creator") or ""),
        })
        bundle["methods"][method] = compact
    job.emit("method_result", compact)


def _publish_job_report(job: AnalysisJob, index: int, report: dict[str, Any]) -> None:
    result_path = job.root / "results" / f"{index:05d}.json"
    _atomic_json(result_path, report)
    compact = _compact_report(report, index, job.selected_methods)
    with job.lock:
        existing = job.compact_results.get(index)
        if existing:
            compact["methods"].update(existing.get("methods") or {})
        job.compact_results[index] = compact
        job.method_results[index] = copy.deepcopy(compact["methods"])
        job.completed_maps += 1
        job.completed_units += 1
    job.emit("result", compact)
    _job_set_progress(job)


def _run_analysis_job(
    job: AnalysisJob,
    maps: list[MapInput],
    config: DetectorConfig,
    include_features: bool,
    *,
    settings: WebSettings,
    content_model_enabled: bool,
    forensic_model_enabled: bool,
    revision_registry_enabled: bool,
    fusion_model_enabled: bool,
    whitebox: WhiteboxRequest,
    uploaded_audio: AudioInput | None,
) -> None:
    """Run one request with checkpoint-major GPU scheduling and report sharding."""

    checkpoints = whitebox.checkpoints if whitebox.enabled else ()
    with job.lock:
        job.status = "running"
        job.phase = "map_only"
        job.message = "Parsing beatmaps and computing non-white-box methods"
        job.total_units = len(maps) * (2 + len(checkpoints))
    job.emit("progress", job.progress())
    try:
        detector = Detector(
            config,
            content_model_path=settings.content_model_path,
            content_model_enabled=content_model_enabled,
            forensic_model_path=settings.forensic_model_path,
            forensic_model_enabled=forensic_model_enabled,
            revision_registry_path=settings.revision_registry_path,
            revision_registry_enabled=revision_registry_enabled,
            fusion_model_path=settings.fusion_model_path,
            fusion_model_enabled=False,
            deep_fusion_model_path=settings.deep_fusion_model_path,
            deep_fusion_model_enabled=False,
        )
        discriminator, discriminator_error = _load_whitebox_discriminator(whitebox, settings)
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.phase = "failed"
            job.message = f"Analyzer initialization failed: {type(exc).__name__}: {exc}"
            job.errors.append({"path": "Job", "error": job.message})
            job.emit("failed", job.snapshot())
        return
    content_model_sha256 = detector.observed_base_model_sha256.get("content_v2")
    work = job.root / "work"
    work.mkdir(parents=True, exist_ok=True)
    audio_cache: dict[str, Path] = {}
    audio_hashes: dict[str, str] = {}
    valid: list[dict[str, Any]] = []
    try:
        local_audio = (
            _resolve_local_audio(whitebox.audio_path, settings)
            if whitebox.enabled and whitebox.audio_path
            else None
        )
        uploaded_audio_path = (
            _write_audio(work, uploaded_audio.name, uploaded_audio.data, audio_cache)
            if uploaded_audio is not None
            else None
        )
        for index, item in enumerate(maps):
            if job.cancel_requested.is_set():
                raise InterruptedError("analysis cancelled")
            path = work / f"map-{index:05d}.osu"
            path.write_bytes(item.data)
            try:
                report = detector.analyze(path).to_dict(include_features=include_features)
                report["path"] = item.display_name
                archive_audio = (
                    _write_audio(
                        work,
                        item.archive_audio_name or "audio.mp3",
                        item.archive_audio_data,
                        audio_cache,
                    )
                    if item.archive_audio_data is not None
                    else None
                )
                selected_audio = uploaded_audio_path or local_audio or archive_audio
                selection = (
                    _whitebox_candidate_selection(
                        path,
                        report,
                        whitebox,
                        content_model_sha256=content_model_sha256,
                    )
                    if whitebox.enabled and selected_audio is not None
                    else None
                )
                cache_key = None
                cached_whitebox = None
                if whitebox.enabled and selected_audio is not None and selection is not None:
                    audio_key = str(selected_audio.resolve())
                    if audio_key not in audio_hashes:
                        audio_hashes[audio_key] = _sha256_path(selected_audio)
                    cache_key = _whitebox_cache_key(
                        item,
                        selected_audio,
                        selection,
                        whitebox,
                        settings,
                        discriminator,
                        audio_hashes[audio_key],
                    )
                    cached_whitebox = _read_whitebox_cache(settings, cache_key)
                    if cached_whitebox is not None:
                        cached_whitebox["cache"] = {"status": "hit", "key": cache_key}
                        cached_whitebox["audio_source"] = (
                            "uploaded" if uploaded_audio_path else "allowed_local_path" if local_audio else "osz"
                        )
                        report.setdefault("analysis_channels", {})["whitebox"] = cached_whitebox
                report_path = job.root / "map_only" / f"{index:05d}.json"
                _atomic_json(report_path, report)
                for method in ("source", "forensic", "content"):
                    if method in job.selected_methods:
                        _publish_job_method(job, index, method, report)
                valid.append(
                    {
                        "index": index,
                        "item": item,
                        "path": path,
                        "audio": selected_audio,
                        "selection": selection,
                        "report_path": report_path,
                        "cache_key": cache_key,
                        "cached": cached_whitebox is not None,
                    }
                )
                with job.lock:
                    job.completed_units += 1
                    job.current_map = index
                _job_set_progress(job)
                if not whitebox.enabled:
                    report.setdefault("analysis_channels", {})["whitebox"] = {
                        "enabled": False,
                        "status": "not_requested",
                        "used_for_final_verdict": False,
                        "discriminator": {
                            "enabled": False,
                            "status": "not_requested",
                            "available": False,
                            "used_for_final_verdict": False,
                        },
                    }
                    _publish_job_report(job, index, report)
                elif cached_whitebox is not None:
                    if "whitebox" in job.selected_methods:
                        _publish_job_method(job, index, "whitebox", report)
                    with job.lock:
                        job.completed_units += len(checkpoints)
                    _publish_job_report(job, index, report)
                elif selected_audio is None:
                    whitebox_result = {
                        "enabled": True,
                        "status": "error",
                        "used_for_final_verdict": False,
                        "error": {
                            "type": "AudioRequired",
                            "message": (
                                "white-box analysis requires uploaded audio, an allowed local path, "
                                "or the exact AudioFilename path embedded in the OSZ; select the audio file manually"
                            ),
                        },
                    }
                    _attach_streamed_postprocessors(
                        report,
                        whitebox_result,
                        detector=detector,
                        request=whitebox,
                        settings=settings,
                        discriminator=discriminator,
                        discriminator_error=discriminator_error,
                        has_audio=False,
                    )
                    report.setdefault("analysis_channels", {})["whitebox"] = whitebox_result
                    if "whitebox" in job.selected_methods:
                        _publish_job_method(job, index, "whitebox", report)
                    with job.lock:
                        job.completed_units += len(checkpoints)
                    _publish_job_report(job, index, report)
            except (OSError, ValueError, BeatmapParseError) as exc:
                error = {"path": item.display_name, "error": str(exc).replace(str(path), item.display_name)}
                with job.lock:
                    job.errors.append(error)
                    job.completed_units += 2 + len(checkpoints)
                job.emit("item_error", error)
                _job_set_progress(job)

        gpu_items = [
            entry
            for entry in valid
            if whitebox.enabled and entry["audio"] is not None and not entry.get("cached")
        ]
        for checkpoint_index, checkpoint_name in enumerate(checkpoints):
            if not gpu_items:
                break
            options = dataclasses.replace(
                whitebox.engine_options(settings),
                checkpoints=(WhiteboxCheckpoint(checkpoint_name),),
                max_cached_checkpoints=1,
            )
            engine = WhiteboxEngine(options)
            _job_set_progress(
                job,
                phase="whitebox",
                message=f"Processing the beatmap batch with {checkpoint_name}",
                current_checkpoint=checkpoint_name,
            )
            try:
                for entry in gpu_items:
                    if job.cancel_requested.is_set():
                        raise InterruptedError("analysis cancelled")
                    part = engine.score(
                        entry["path"],
                        entry["audio"],
                        candidate_interval_selection=entry["selection"],
                    )
                    part_path = (
                        job.root
                        / "whitebox"
                        / f"{entry['index']:05d}"
                        / f"{checkpoint_index:02d}-{checkpoint_name}.json"
                    )
                    _atomic_json(part_path, part)
                    with job.lock:
                        job.completed_units += 1
                        job.current_map = int(entry["index"])
                    _job_set_progress(job)
                    if checkpoint_index == len(checkpoints) - 1:
                        report = json.loads(entry["report_path"].read_text(encoding="utf-8"))
                        parts = [
                            json.loads(
                                (
                                    job.root
                                    / "whitebox"
                                    / f"{entry['index']:05d}"
                                    / f"{part_index:02d}-{name}.json"
                                ).read_text(encoding="utf-8")
                            )
                            for part_index, name in enumerate(checkpoints)
                        ]
                        whitebox_result = _merge_whitebox_parts(parts, entry["selection"])
                        whitebox_result["audio_source"] = (
                            "uploaded"
                            if uploaded_audio_path
                            else "allowed_local_path"
                            if local_audio
                            else "osz"
                        )
                        _attach_streamed_postprocessors(
                            report,
                            whitebox_result,
                            detector=detector,
                            request=whitebox,
                            settings=settings,
                            discriminator=discriminator,
                            discriminator_error=discriminator_error,
                            has_audio=True,
                        )
                        report.setdefault("analysis_channels", {})["whitebox"] = whitebox_result
                        _write_whitebox_cache(settings, entry.get("cache_key"), whitebox_result)
                        if "whitebox" in job.selected_methods:
                            _publish_job_method(job, int(entry["index"]), "whitebox", report)
                        _publish_job_report(job, int(entry["index"]), report)
            finally:
                engine.close()

        resolved_root = job.root.resolve()
        # Keep map-only and per-checkpoint shards for safe local resume/debugging;
        # only uploaded working copies are transient.
        for directory_name in ("work",):
            transient = (job.root / directory_name).resolve()
            if transient.parent == resolved_root:
                shutil.rmtree(transient, ignore_errors=True)
        with job.lock:
            job.status = "complete"
            job.phase = "complete"
            job.message = "All sharded results are available"
            job.current_checkpoint = None
            job.current_map = None
            job.completed_units = job.total_units
            job.emit("complete", job.snapshot())
    except InterruptedError:
        with job.lock:
            job.status = "cancelled"
            job.phase = "cancelled"
            job.message = "Job cancelled; completed results remain available"
            job.emit("cancelled", job.snapshot())
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.phase = "failed"
            job.message = f"Analysis failed: {type(exc).__name__}: {exc}"
            job.errors.append({"path": "Job", "error": job.message})
            job.emit("failed", job.snapshot())


def create_app():
    settings = WebSettings.from_env()
    service_detector_config = _detector_default_from_env()
    app = FastAPI(
        title="osu! AI mapping detector",
        version=__version__,
        description="Detailed content and optional Mapperatorinator white-box analysis.",
        docs_url=None,
        redoc_url=None,
    )
    jobs: dict[str, AnalysisJob] = {}
    jobs_lock = threading.RLock()
    jobs_root = Path(tempfile.mkdtemp(prefix="osu-ai-web-jobs-"))
    # GPU work is deliberately serialized across requests.  Within one job the
    # scheduler is checkpoint-major, so a model is loaded once and scores the
    # whole map batch before the next checkpoint is loaded.
    job_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="osu-ai-gpu-job",
    )
    app.state.analysis_jobs = jobs
    app.state.analysis_job_executor = job_executor
    app.state.analysis_jobs_root = jobs_root

    @app.on_event("shutdown")
    def shutdown_job_executor() -> None:
        with jobs_lock:
            for job in jobs.values():
                job.cancel_requested.set()
        job_executor.shutdown(wait=False, cancel_futures=True)

    def parse_config(raw: str | Mapping[str, Any] | None) -> DetectorConfig:
        try:
            values = json.loads(raw) if isinstance(raw, str) and raw.strip() else dict(raw or {})
            if not isinstance(values, dict):
                raise ValueError("detector_config must be a JSON object")
            merged = service_detector_config.to_dict()
            merged.update(values)
            return DetectorConfig.from_dict(merged)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def parse_whitebox(raw: str | Mapping[str, Any] | None, **overrides: Any) -> WhiteboxRequest:
        try:
            values = json.loads(raw) if isinstance(raw, str) and raw.strip() else dict(raw or {})
            if not isinstance(values, dict):
                raise ValueError("whitebox_config must be a JSON object")
            values.update({key: value for key, value in overrides.items() if value is not None})
            return WhiteboxRequest.from_mapping(values)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def parse_methods(raw: str | Sequence[str] | None, whitebox: WhiteboxRequest) -> tuple[tuple[str, ...], WhiteboxRequest]:
        try:
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                requested = ["source", "forensic", "content"]
                if whitebox.enabled:
                    requested.insert(1, "whitebox")
            elif isinstance(raw, str):
                requested = json.loads(raw) if raw.lstrip().startswith("[") else raw.split(",")
            else:
                requested = list(raw)
            normalized = []
            for item in requested:
                name = str(item).strip().casefold()
                if name and name not in normalized:
                    normalized.append(name)
            invalid = sorted(set(normalized) - set(ANALYSIS_METHODS))
            if invalid:
                raise ValueError("unsupported methods: " + ", ".join(invalid))
            if not normalized:
                raise ValueError("select at least one detection method")
            if "whitebox" in normalized:
                if "content" not in normalized:
                    normalized.append("content")
                whitebox = dataclasses.replace(
                    whitebox,
                    enabled=True,
                    discriminator_enabled=True,
                    deep_fusion_enabled=False,
                )
            else:
                whitebox = dataclasses.replace(whitebox, enabled=False, deep_fusion_enabled=False)
            ordered = tuple(method for method in ANALYSIS_METHODS if method in normalized)
            return ordered, whitebox
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def require_job(job_id: str) -> AnalysisJob:
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="analysis job not found")
        return job

    async def read_job_inputs(
        files: list[UploadFile],
        audio: UploadFile | None,
        whitebox: WhiteboxRequest,
    ) -> tuple[list[MapInput], AudioInput | None, list[dict[str, str]]]:
        maps: list[MapInput] = []
        errors: list[dict[str, str]] = []
        total_bytes = 0
        for upload in files:
            data = await upload.read(settings.max_total_upload_bytes + 1)
            total_bytes += len(data)
            if total_bytes > settings.max_total_upload_bytes:
                raise HTTPException(status_code=413, detail="request exceeds maximum total upload size")
            try:
                maps.extend(_decode_upload(_safe_name(upload.filename), data, settings))
            except ValueError as exc:
                errors.append({"path": _safe_name(upload.filename), "error": str(exc)})
            if len(maps) > settings.max_maps_per_request:
                raise HTTPException(status_code=413, detail="request contains too many beatmaps")
        uploaded_audio = None
        if audio is not None and audio.filename:
            if whitebox.audio_path:
                raise HTTPException(status_code=422, detail="choose uploaded audio or whitebox_audio_path, not both")
            audio_name = _safe_name(audio.filename, "audio.mp3")
            if Path(audio_name).suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
                raise HTTPException(status_code=422, detail="unsupported uploaded audio type")
            audio_data = await audio.read(settings.max_audio_bytes + 1)
            total_bytes += len(audio_data)
            if len(audio_data) > settings.max_audio_bytes or total_bytes > settings.max_total_upload_bytes:
                raise HTTPException(status_code=413, detail="uploaded audio/request exceeds configured size")
            uploaded_audio = AudioInput(audio_name, audio_data)
        if not maps:
            raise HTTPException(status_code=400, detail={"message": "no analyzable .osu files", "errors": errors})
        if whitebox.enabled and uploaded_audio is None and not whitebox.audio_path:
            missing = [item for item in maps if item.archive_audio_data is None]
            if missing:
                requested = sorted({item.requested_audio_name or "(beatmap does not declare AudioFilename)" for item in missing})
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "manual_audio_selection_required",
                        "message": "The declared audio file is missing or cannot be matched exactly. Select the audio file manually.",
                        "requested_audio": requested,
                        "affected_maps": [item.display_name for item in missing[:20]],
                    },
                )
        return maps, uploaded_audio, errors

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        return HTMLResponse(
            HTML_PAGE,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
                    "form-action 'self'; connect-src 'self'; img-src 'self' data:; "
                    "style-src 'unsafe-inline'; script-src 'unsafe-inline'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/health")
    async def health():
        preflight_engine = WhiteboxEngine(
            WhiteboxOptions(
                checkpoints=(WhiteboxCheckpoint("v31"),),
                vendor_root=settings.whitebox_vendor_root,
            )
        )
        try:
            preflight = preflight_engine.availability()
        finally:
            preflight_engine.close()
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        except (ImportError, RuntimeError):
            cuda_available = False
            cuda_device_count = 0
        return {
            "status": "ok",
            "runtime_profile": settings.runtime_profile,
            "detector_version": __version__,
            "detector_scope": Detector.scope,
            "content_v2": {
                "available": settings.content_model_path.is_file(),
                "model_path": str(settings.content_model_path),
            },
            "revision_forensics_v3": {
                "available": settings.forensic_model_path.is_file(),
                "model_path": str(settings.forensic_model_path),
                "missing_behavior": "explicit unavailable + abstention",
            },
            "public_revision_registry": {
                "available": settings.revision_registry_path.is_file(),
                "registry_path": str(settings.revision_registry_path),
                "match_policy": "exact SHA-256 only; identity/set/replacement entries never inherit positivity",
            },
            "cheap_fusion_v1": {
                "available": settings.fusion_model_path.is_file(),
                "model_path": str(settings.fusion_model_path),
                "channels": ["content_v2", "revision_forensics_v3"],
                "missing_behavior": "explicit abstention; content/forensic fallback remains active",
            },
            "whitebox": preflight,
            "compute": {
                "cuda_available": cuda_available,
                "cuda_device_count": cuda_device_count,
                "whitebox_cpu_supported": True,
                "whitebox_cpu_warning": "CPU execution is supported but substantially slower; CUDA is recommended.",
            },
            "whitebox_discriminator": {
                "available": settings.whitebox_discriminator_model_path.is_file(),
                "model_path": str(settings.whitebox_discriminator_model_path),
                "default_enabled": False,
                "missing_behavior": "explicit unavailable + abstention; never numeric zero",
            },
            "deep_fusion_v1": {
                "available": settings.deep_fusion_model_path.is_file(),
                "model_path": str(settings.deep_fusion_model_path),
                "default_enabled": False,
                "channels": ["cheap_fusion_v1", "whitebox_discriminator_v1"],
                "requires": ["audio", "raw whitebox", "whitebox discriminator"],
                "used_for_final_verdict": False,
                "missing_behavior": "explicit unavailable + abstention when requested; absent otherwise",
            },
        }

    @app.get("/api/config")
    async def config():
        return {
            "runtime_profile": settings.runtime_profile,
            "independent_methods": [
                {"id": method, "name": METHOD_COPY[method][0], "description": METHOD_COPY[method][1]}
                for method in ANALYSIS_METHODS
            ],
            "overall_verdict": {"enabled": False, "policy": "four independent methods; no combined verdict"},
            "detector": service_detector_config.to_dict(),
            "calibrated_detector": DetectorConfig().to_dict(),
            "deprecated_exploratory_detector_defaults": DetectorConfig().to_dict(),
            "service_default_is_frozen_calibrated_profile": service_detector_config == DetectorConfig(),
            "detector_field_catalog": CONFIG_CATALOG,
            "content_v2": {
                "available": settings.content_model_path.is_file(),
                "model_path": str(settings.content_model_path),
                "missing_behavior": "explicit unavailable + abstention; never silent fallback",
            },
            "revision_forensics_v3": {
                "default_enabled": True,
                "available": settings.forensic_model_path.is_file(),
                "model_path": str(settings.forensic_model_path),
                "high_verdict_policy": "calibrated high may only cap the verdict at suspicious/corroborate",
                "missing_or_uncalibrated_behavior": "explicit abstention",
            },
            "public_revision_registry": {
                "default_enabled": True,
                "available": settings.revision_registry_path.is_file(),
                "registry_path": str(settings.revision_registry_path),
                "positive_policy": (
                    "only an exact checksum with known_exact_positive=true can set a known historical revision high; "
                    "this is identity lookup, not statistical detection"
                ),
                "mismatch_policy": "BeatmapID/BeatmapSetID/replacement/missing records never inherit positivity",
            },
            "cheap_fusion_v1": {
                "default_enabled": True,
                "available": settings.fusion_model_path.is_file(),
                "model_path": str(settings.fusion_model_path),
                "channels": ["content_v2", "revision_forensics_v3"],
                "whitebox_included": False,
                "missing_or_abstain_behavior": "use existing content fallback and forensic suspicious cap",
            },
            "whitebox": {
                "default_enabled": False,
                "allowed_checkpoints": list(ALLOWED_CHECKPOINTS),
                "config_fields": [field.name for field in dataclasses.fields(WhiteboxRequest)],
                "forward_batch_size_range": [1, MAX_FORWARD_BATCH_SIZE],
                "candidate_search": {
                    "default": "label-free automatic proposals bound to the active content artifact ID and SHA-256",
                    "fallback": "bounded first/last/densest proposals; calibrated discriminator abstains",
                    "manual_override": "raw exploration remains available; calibrated discriminator abstains",
                },
                "local_audio_roots": [str(path) for path in settings.allowed_audio_roots],
                "discriminator": {
                    "default_enabled": False,
                    "available": settings.whitebox_discriminator_model_path.is_file(),
                    "model_path": str(settings.whitebox_discriminator_model_path),
                    "used_for_final_verdict": False,
                    "missing_behavior": "explicit unavailable + abstention; never numeric zero",
                },
                "audio_backed_deep_review": {
                    "default_enabled": False,
                    "available": settings.deep_fusion_model_path.is_file(),
                    "model_path": str(settings.deep_fusion_model_path),
                    "requires": ["audio", "raw whitebox", "whitebox discriminator"],
                    "used_for_final_verdict": False,
                    "display_policy": "independent review; map-only verdict remains unchanged",
                    "missing_behavior": "explicit abstention when requested with audio; absent otherwise",
                },
            },
            "batch_analysis": {
                "endpoint": "/api/jobs",
                "scheduling": "checkpoint-major",
                "max_parallel_gpu_jobs": 1,
                "progress_transport": "server-sent events",
                "result_transport": "one compact event per method and one JSON shard per map",
                "legacy_endpoint": "/api/analyze remains available for compatibility",
            },
            "service_limits": dataclasses.asdict(settings),
            "environment_variables": {
                "OSU_AI_CONTENT_MODEL": "content_v2 JSON artifact path",
                "OSU_AI_FORENSIC_MODEL": "revision_forensics_v3 JSON artifact path",
                "OSU_AI_REVISION_REGISTRY": "exact-checksum public incident revision registry path",
                "OSU_AI_FUSION_MODEL": "independently calibrated cheap content+forensic fusion JSON artifact path",
                "OSU_AI_MAPPERATORINATOR_ROOT": "vendored Mapperatorinator source root",
                "OSU_AI_WHITEBOX_DISCRIMINATOR_MODEL": "optional CPU white-box discriminator JSON artifact path",
                "OSU_AI_DEEP_FUSION_MODEL": "optional calibrated cheap+white-box deep fusion JSON artifact path",
                "OSU_AI_AUDIO_ROOTS": "os.pathsep-separated roots allowed for local audio_path",
                "OSU_AI_MAX_FILE_BYTES": "maximum bytes per uploaded .osu",
                "OSU_AI_MAX_AUDIO_BYTES": "maximum bytes per audio file",
                "OSU_AI_MAX_TOTAL_UPLOAD_BYTES": "maximum request upload bytes",
                "OSU_AI_MAX_ARCHIVE_UNCOMPRESSED_BYTES": "maximum map+audio bytes expanded from archives",
                "OSU_AI_MAX_ARCHIVE_ENTRIES": "maximum archive entries",
                "OSU_AI_MAX_MAPS": "maximum maps per request",
                "OSU_AI_DETECTOR_CONFIG": "deprecated exploratory-channel service defaults",
                "OSU_AI_DETECTOR_CONFIG_FILE": "JSON file for deprecated exploratory-channel defaults",
                "OSU_AI_STATISTICAL_MODEL": "alternate deprecated v0.2 statistical JSON",
                "OSU_AI_WEB_PROFILE": "UI/runtime hint: map-only or audio-review",
                "OSU_AI_CACHE_ROOT": "persistent derived-result cache directory",
                "OSU_AI_CACHE_MAX_BYTES": "maximum derived-result cache bytes; default 20 GiB",
            },
        }

    @app.get("/api/cache")
    async def cache_status():
        root = settings.cache_root / "whitebox-complete"
        files = [path for path in root.glob("*/*.json.gz") if path.is_file()] if root.is_dir() else []
        return {
            "root": str(settings.cache_root),
            "max_bytes": settings.cache_max_bytes,
            "entries": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "contains_original_uploads": False,
        }

    @app.delete("/api/cache")
    async def clear_cache():
        root = settings.cache_root.resolve()
        target = (root / "whitebox-complete").resolve()
        if target.parent != root:
            raise HTTPException(status_code=500, detail="invalid cache path")
        shutil.rmtree(target, ignore_errors=True)
        return {"cleared": True, "root": str(root), "contains_original_uploads": False}

    @app.get("/api/features")
    async def features():
        return {
            "features": FEATURE_CATALOG,
            "counts": COUNT_CATALOG,
            "evidence": EVIDENCE_CATALOG,
            "verdicts": VERDICT_CATALOG,
            "configuration": CONFIG_CATALOG,
        }

    @app.post("/api/jobs", status_code=202)
    async def create_analysis_job(
        files: list[UploadFile] = File(...),
        audio: UploadFile | None = File(None),
        detector_config: str = Form("{}"),
        include_features: bool = Form(False),
        content_model_enabled: bool = Form(True),
        forensic_model_enabled: bool = Form(True),
        revision_registry_enabled: bool = Form(True),
        fusion_model_enabled: bool = Form(True),
        selected_methods: str = Form(""),
        whitebox_config: str = Form("{}"),
        whitebox_enabled: bool | None = Form(None),
        whitebox_checkpoints: str = Form(""),
        whitebox_audio_path: str = Form(""),
        whitebox_start_ms: int | None = Form(None),
        whitebox_end_ms: int | None = Form(None),
        whitebox_max_windows: int | None = Form(None),
        whitebox_include_token_details: bool | None = Form(None),
        whitebox_max_token_details_per_window: int | None = Form(None),
        whitebox_device: str = Form(""),
        whitebox_precision: str = Form(""),
        whitebox_forward_batch_size: int | None = Form(None),
        whitebox_discriminator_enabled: bool | None = Form(None),
        deep_fusion_enabled: bool | None = Form(None),
    ):
        config_value = parse_config(detector_config)
        whitebox_value = parse_whitebox(
            whitebox_config,
            enabled=whitebox_enabled,
            checkpoints=whitebox_checkpoints.strip() or None,
            audio_path=whitebox_audio_path.strip() or None,
            start_ms=whitebox_start_ms,
            end_ms=whitebox_end_ms,
            max_windows=whitebox_max_windows,
            include_token_details=whitebox_include_token_details,
            max_token_details_per_window=whitebox_max_token_details_per_window,
            device=whitebox_device.strip() or None,
            precision=whitebox_precision.strip() or None,
            forward_batch_size=whitebox_forward_batch_size,
            discriminator_enabled=whitebox_discriminator_enabled,
            deep_fusion_enabled=deep_fusion_enabled,
        )
        methods_value, whitebox_value = parse_methods(selected_methods, whitebox_value)
        maps, uploaded_audio, input_errors = await read_job_inputs(files, audio, whitebox_value)
        job_id = uuid.uuid4().hex
        job = AnalysisJob(
            job_id=job_id,
            root=jobs_root / job_id,
            total_maps=len(maps),
            checkpoints=whitebox_value.checkpoints if whitebox_value.enabled else (),
            selected_methods=methods_value,
            errors=input_errors,
        )
        job.root.mkdir(parents=True, exist_ok=False)
        job.emit("queued", job.snapshot())
        with jobs_lock:
            jobs[job_id] = job
        job_executor.submit(
            _run_analysis_job,
            job,
            maps,
            config_value,
            include_features,
            settings=settings,
            content_model_enabled="content" in methods_value,
            forensic_model_enabled="forensic" in methods_value,
            revision_registry_enabled="source" in methods_value,
            fusion_model_enabled=False,
            whitebox=whitebox_value,
            uploaded_audio=uploaded_audio,
        )
        return {
            "schema_version": 1,
            "job_id": job_id,
            "status_url": f"/api/jobs/{job_id}",
            "events_url": f"/api/jobs/{job_id}/events",
            "results_url": f"/api/jobs/{job_id}/results",
            "scheduling": "checkpoint-major",
            "selected_methods": list(methods_value),
        }

    @app.get("/api/jobs/{job_id}")
    async def analysis_job_status(job_id: str):
        return require_job(job_id).snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def analysis_job_events(job_id: str, after: int = 0):
        job = require_job(job_id)

        async def stream():
            cursor = max(0, after)
            while True:
                with job.lock:
                    pending = [event for event in job.events if int(event["id"]) > cursor]
                    terminal = job.status in {"complete", "failed", "cancelled"}
                    last_id = int(job.events[-1]["id"]) if job.events else 0
                for event in pending:
                    cursor = int(event["id"])
                    payload = json.dumps(event["data"], ensure_ascii=False, separators=(",", ":"))
                    yield f"id: {cursor}\nevent: {event['event']}\ndata: {payload}\n\n"
                if terminal and cursor >= last_id:
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/results")
    async def analysis_job_results(job_id: str, after: int = -1, limit: int = 50):
        job = require_job(job_id)
        bounded_limit = max(1, min(limit, 200))
        with job.lock:
            items = [
                copy.deepcopy(job.compact_results[index])
                for index in sorted(job.compact_results)
                if index > after
            ][:bounded_limit]
        return {"job_id": job_id, "results": items, "count": len(items)}

    @app.get("/api/jobs/{job_id}/results/{index}")
    async def analysis_job_result(job_id: str, index: int):
        job = require_job(job_id)
        if index < 0 or index >= job.total_maps:
            raise HTTPException(status_code=404, detail="result shard not found")
        path = job.root / "results" / f"{index:05d}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="result shard not ready")
        return FileResponse(path, media_type="application/json", filename=f"result-{index + 1}.json")

    @app.get("/api/jobs/{job_id}/report.json")
    async def analysis_job_json_export(job_id: str):
        job = require_job(job_id)

        async def stream_json():
            yield '{"schema_version":2,"job_id":' + json.dumps(job_id) + ',"reports":['
            first = True
            for index in range(job.total_maps):
                path = job.root / "results" / f"{index:05d}.json"
                if not path.is_file():
                    continue
                if not first:
                    yield ","
                first = False
                with path.open("r", encoding="utf-8") as handle:
                    while chunk := handle.read(1024 * 1024):
                        yield chunk
                await asyncio.sleep(0)
            yield "]}"

        return StreamingResponse(
            stream_json(),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="osu-ai-report-{job_id[:8]}.json"'},
        )

    @app.get("/api/jobs/{job_id}/report.html", response_class=HTMLResponse)
    async def analysis_job_html_export(job_id: str):
        job = require_job(job_id)
        with job.lock:
            results = [copy.deepcopy(job.compact_results[index]) for index in sorted(job.compact_results)]
        sections: list[str] = []
        for result in results:
            heading = html_lib.escape(result.get("title") or result.get("path") or "Beatmap")
            version = html_lib.escape(result.get("version") or "")
            cards = []
            for method in ANALYSIS_METHODS:
                card = (result.get("methods") or {}).get(method)
                if not card:
                    continue
                metric = (
                    f"{card.get('percentile'):.2f} percentile"
                    if isinstance(card.get("percentile"), (int, float))
                    else html_lib.escape(str(card.get("label") or ""))
                )
                reasons = "".join(
                    f"<li>{html_lib.escape(str(row.get('label') or ''))}</li>"
                    for row in card.get("reasons") or []
                ) or "<li>No sufficiently strong individual anomaly indicators</li>"
                cards.append(
                    f"<section><h3>{html_lib.escape(card['name'])}</h3>"
                    f"<p><strong>{metric}</strong> · {html_lib.escape(str(card.get('label') or ''))}</p>"
                    f"<ul>{reasons}</ul></section>"
                )
            sections.append(f"<article><h2>{heading} {version}</h2>{''.join(cards)}</article>")
        page = (
            "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<title>osu! AI Detection Report</title><style>body{font:15px/1.55 system-ui;margin:32px;max-width:1100px}"
            "article{border-top:2px solid #333;padding:18px 0}section{border-left:3px solid #bbb;padding:1px 16px;margin:14px 0}"
            "h1,h2,h3{line-height:1.25}small{color:#666}</style><h1>osu! AI Detection Report</h1>"
            "<p><small>The four methods are reported independently. This report provides no overall verdict, and statistical percentiles are not AI-authorship probabilities.</small></p>"
            + "".join(sections)
            + "</html>"
        )
        return HTMLResponse(page, headers={"Content-Disposition": f'attachment; filename="osu-ai-report-{job_id[:8]}.html"'})

    @app.delete("/api/jobs/{job_id}")
    async def cancel_analysis_job(job_id: str):
        job = require_job(job_id)
        job.cancel_requested.set()
        return {"job_id": job_id, "cancel_requested": True, "status": job.status}

    @app.post("/api/analyze")
    async def analyze_uploads(
        files: list[UploadFile] = File(...),
        audio: UploadFile | None = File(None),
        detector_config: str = Form("{}"),
        include_features: bool = Form(True),
        content_model_enabled: bool = Form(True),
        forensic_model_enabled: bool = Form(True),
        revision_registry_enabled: bool = Form(True),
        fusion_model_enabled: bool = Form(True),
        whitebox_config: str = Form("{}"),
        whitebox_enabled: bool | None = Form(None),
        whitebox_checkpoints: str = Form(""),
        whitebox_audio_path: str = Form(""),
        whitebox_start_ms: int | None = Form(None),
        whitebox_end_ms: int | None = Form(None),
        whitebox_max_windows: int | None = Form(None),
        whitebox_include_token_details: bool | None = Form(None),
        whitebox_max_token_details_per_window: int | None = Form(None),
        whitebox_device: str = Form(""),
        whitebox_precision: str = Form(""),
        whitebox_forward_batch_size: int | None = Form(None),
        whitebox_discriminator_enabled: bool | None = Form(None),
        deep_fusion_enabled: bool | None = Form(None),
    ):
        started = time.perf_counter()
        config_value = parse_config(detector_config)
        whitebox_value = parse_whitebox(
            whitebox_config,
            enabled=whitebox_enabled,
            checkpoints=whitebox_checkpoints.strip() or None,
            audio_path=whitebox_audio_path.strip() or None,
            start_ms=whitebox_start_ms,
            end_ms=whitebox_end_ms,
            max_windows=whitebox_max_windows,
            include_token_details=whitebox_include_token_details,
            max_token_details_per_window=whitebox_max_token_details_per_window,
            device=whitebox_device.strip() or None,
            precision=whitebox_precision.strip() or None,
            forward_batch_size=whitebox_forward_batch_size,
            discriminator_enabled=whitebox_discriminator_enabled,
            deep_fusion_enabled=deep_fusion_enabled,
        )
        maps: list[MapInput] = []
        errors: list[dict[str, str]] = []
        total_bytes = 0
        for upload in files:
            data = await upload.read(settings.max_total_upload_bytes + 1)
            total_bytes += len(data)
            if total_bytes > settings.max_total_upload_bytes:
                raise HTTPException(status_code=413, detail="request exceeds maximum total upload size")
            try:
                maps.extend(_decode_upload(_safe_name(upload.filename), data, settings))
            except ValueError as exc:
                errors.append({"path": _safe_name(upload.filename), "error": str(exc)})
            if len(maps) > settings.max_maps_per_request:
                raise HTTPException(status_code=413, detail="request contains too many beatmaps")
        uploaded_audio = None
        if audio is not None and audio.filename:
            if whitebox_value.audio_path:
                raise HTTPException(status_code=422, detail="choose uploaded audio or whitebox_audio_path, not both")
            audio_name = _safe_name(audio.filename, "audio.mp3")
            if Path(audio_name).suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
                raise HTTPException(status_code=422, detail="unsupported uploaded audio type")
            audio_data = await audio.read(settings.max_audio_bytes + 1)
            total_bytes += len(audio_data)
            if len(audio_data) > settings.max_audio_bytes or total_bytes > settings.max_total_upload_bytes:
                raise HTTPException(status_code=413, detail="uploaded audio/request exceeds configured size")
            uploaded_audio = AudioInput(audio_name, audio_data)
        if not maps:
            raise HTTPException(status_code=400, detail={"message": "no analyzable .osu files", "errors": errors})
        try:
            result = await asyncio.to_thread(
                _build_response,
                maps,
                config_value,
                include_features,
                started,
                settings=settings,
                content_model_enabled=content_model_enabled,
                forensic_model_enabled=forensic_model_enabled,
                revision_registry_enabled=revision_registry_enabled,
                fusion_model_enabled=fusion_model_enabled,
                whitebox=whitebox_value,
                uploaded_audio=uploaded_audio,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["errors"] = errors + result["errors"]
        result["summary"]["error_count"] = len(result["errors"])
        return result

    @app.post("/api/analyze-text")
    async def analyze_text(request: TextAnalysisRequest):
        data = request.content.encode("utf-8")
        if len(data) > settings.max_file_bytes:
            raise HTTPException(status_code=413, detail="content exceeds maximum file size")
        config_value = parse_config(request.detector_config)
        try:
            whitebox_value = WhiteboxRequest.from_mapping(request.whitebox)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        started = time.perf_counter()
        try:
            return await asyncio.to_thread(
                _build_response,
                [MapInput(_safe_name(request.filename), data)],
                config_value,
                request.include_features,
                started,
                settings=settings,
                content_model_enabled=request.content_model_enabled,
                forensic_model_enabled=request.forensic_model_enabled,
                revision_registry_enabled=request.revision_registry_enabled,
                fusion_model_enabled=request.fusion_model_enabled,
                whitebox=whitebox_value,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
