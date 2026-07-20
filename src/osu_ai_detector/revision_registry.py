"""Exact-checksum public incident revision lookup.

No label is inherited from BeatmapID, BeatmapSetID, filename, mapper or
difficulty name.  This is essential because reported maps may be replaced or
re-ranked after the incident.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .parser import Beatmap


DEFAULT_REGISTRY = Path(__file__).with_name("models") / "revision_registry.json"


@functools.lru_cache(maxsize=4)
def load_revision_registry(path: str = str(DEFAULT_REGISTRY)) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported revision registry schema")
    return value


def _hashes(beatmap: Beatmap) -> tuple[str, str, int]:
    try:
        raw = beatmap.path.read_bytes()
    except OSError:
        raw = beatmap.raw_text.encode("utf-8")
    return (
        hashlib.sha256(raw).hexdigest(),
        hashlib.md5(raw).hexdigest(),  # nosec B324 - osu! revision checksum convention
        len(raw),
    )


def match_revision(beatmap: Beatmap, registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    registry_value = dict(registry) if registry is not None else load_revision_registry()
    sha256, md5, size = _hashes(beatmap)
    metadata = beatmap.metadata
    try:
        beatmap_id = int(metadata.get("BeatmapID", ""))
    except ValueError:
        beatmap_id = None
    try:
        set_id = int(metadata.get("BeatmapSetID", ""))
    except ValueError:
        set_id = None

    exact = None
    identity_history = []
    for entry in registry_value.get("exact_revisions", []):
        if str(entry.get("sha256", "")).casefold() == sha256:
            exact = entry
        if beatmap_id is not None and int(entry.get("beatmap_id", -1)) == beatmap_id:
            identity_history.append(entry)
    missing_history = [
        entry
        for entry in registry_value.get("missing_reported_revisions", [])
        if (beatmap_id is not None and int(entry.get("beatmap_id", -1)) == beatmap_id)
        or (set_id is not None and int(entry.get("beatmapset_id", -1)) == set_id)
    ]
    if exact is not None:
        # SHA-256 is authoritative; MD5/length mismatches indicate a corrupt
        # registry artifact rather than an alternate fuzzy match.
        if str(exact.get("md5", "")).casefold() != md5 or int(exact.get("bytes", -1)) != size:
            raise ValueError("Revision registry SHA match has inconsistent MD5/byte length")
        status = "exact_verified_public_positive" if exact.get("exact_verified_positive") else "exact_context_revision"
        reason = str(exact.get("interpretation"))
    elif identity_history or missing_history:
        status = "identity_seen_but_checksum_not_matched"
        reason = (
            "Beatmap identity appears in the incident registry, but this file checksum is different. "
            "It is an unknown/replaced revision and does not inherit any historical positive label."
        )
    else:
        status = "not_in_public_revision_registry"
        reason = "No exact or identity-level entry in the bundled public incident registry."
    return {
        "available": True,
        "registry_id": registry_value.get("registry_id"),
        "registry_source_audit_sha256": registry_value.get("source_audit_sha256"),
        "status": status,
        "reason": reason,
        "observed": {"sha256": sha256, "md5": md5, "bytes": size, "beatmap_id": beatmap_id, "beatmapset_id": set_id},
        "exact_match": exact,
        "identity_history": identity_history,
        "missing_revision_history": missing_history,
        "known_exact_positive": bool(exact and exact.get("exact_verified_positive")),
        "used_as_statistical_training_label": False,
    }


def unavailable(reason: str, path: str | Path | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "registry_path": str(path) if path is not None else None,
        "status": "unavailable",
        "reason": reason,
        "exact_match": None,
        "identity_history": [],
        "missing_revision_history": [],
        "known_exact_positive": False,
        "used_as_statistical_training_label": False,
    }
