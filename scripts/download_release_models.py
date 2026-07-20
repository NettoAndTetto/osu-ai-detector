#!/usr/bin/env python3
"""Download and verify the model assets required by the public application."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path, manifest_name: str) -> None:
    payload = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    for item in payload.get("files", []):
        relative = Path(item["path"])
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise RuntimeError(f"model artifact is missing or unsafe: {relative.as_posix()}")
        if target.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"model artifact size mismatch: {relative.as_posix()}")
        if sha256(target) != item["sha256"]:
            raise RuntimeError(f"model artifact SHA-256 mismatch: {relative.as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, default=Path("release-spec.json"))
    parser.add_argument("--hf-home", type=Path, default=Path(".runtime-assets/huggingface"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    app_root = args.app_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else app_root / args.spec
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    hf_home = (args.hf_home if args.hf_home.is_absolute() else app_root / args.hf_home).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    own_snapshot = Path(
        snapshot_download(
            repo_id="NettoAndTetto/osu-ai-detector-models",
            revision=spec["model_revision"],
            cache_dir=cache_dir,
            local_files_only=args.offline,
        )
    ).resolve()
    verify_manifest(own_snapshot, "artifact-manifest.json")
    model_target = app_root / "src" / "osu_ai_detector" / "models"
    model_target.mkdir(parents=True, exist_ok=True)
    for item in json.loads((own_snapshot / "artifact-manifest.json").read_text(encoding="utf-8"))["files"]:
        source = own_snapshot / item["path"]
        destination = model_target / Path(item["path"]).name
        shutil.copy2(source, destination)
        if sha256(destination) != item["sha256"]:
            raise RuntimeError(f"copied model artifact failed verification: {destination.name}")

    for checkpoint in spec["upstream_models"]:
        snapshot_download(
            repo_id=checkpoint["repo_id"],
            revision=checkpoint["revision"],
            cache_dir=cache_dir,
            local_files_only=args.offline,
        )

    receipt = {
        "schema_version": 1,
        "release": spec["tag"],
        "model_repo": "NettoAndTetto/osu-ai-detector-models",
        "model_revision": spec["model_revision"],
        "upstream_models": spec["upstream_models"],
    }
    (app_root / ".runtime-assets" / "model-install-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
