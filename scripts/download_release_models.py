#!/usr/bin/env python3
"""Install and verify the model assets required by the public application.

The normal path downloads pinned Hugging Face snapshots.  If that one network
attempt fails, users may download repository files in a browser, put them in
the documented ``manual-models`` folders, and rerun this command with
``--manual-root``.  Manual files are imported into the same revision-pinned
Hugging Face cache used by the application; inference needs no special mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_sha256(path: Path) -> str:
    """Authenticate an HF LFS link without rereading its multi-GB blob."""

    blob_name = path.resolve().name.casefold()
    if path.is_symlink() and len(blob_name) == 64 and all(
        char in "0123456789abcdef" for char in blob_name
    ):
        return blob_name
    return sha256(path)


def verify_manifest(root: Path, manifest_name: str) -> None:
    payload = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    for item in payload.get("files", []):
        relative = Path(item["path"])
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise RuntimeError(f"model artifact is missing or unsafe: {relative.as_posix()}")
        # Hugging Face snapshots normally expose files as links into the same
        # repository cache's blobs directory. Validate the manifest's logical
        # member path before following that link, then authenticate its bytes
        # below instead of rejecting the standard cache layout.
        target = root / relative
        if not target.is_file():
            raise RuntimeError(f"model artifact is missing or unsafe: {relative.as_posix()}")
        if target.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"model artifact size mismatch: {relative.as_posix()}")
        if sha256(target) != item["sha256"]:
            raise RuntimeError(f"model artifact SHA-256 mismatch: {relative.as_posix()}")


def _cached_snapshot(repo_id: str, revision: str, cache_dir: Path) -> Path | None:
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        ).resolve()
    except (LocalEntryNotFoundError, OSError, ValueError):
        return None


def _manual_folder(manual_root: Path, name: str) -> Path:
    folder = (manual_root / name).resolve()
    if not folder.is_dir():
        raise RuntimeError(
            f"required manual model folder is missing: {folder}\n"
            "See the 'Manual model download fallback' section in README.md."
        )
    return folder


def _copy_manual_snapshot(source: Path, repo_id: str, revision: str, cache_dir: Path) -> Path:
    """Import ordinary downloaded files into a pinned Hugging Face snapshot."""

    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.casefold()):
        raise RuntimeError(f"manual upstream revision is not an immutable commit: {repo_id}@{revision}")
    repository_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"
    destination = repository_cache / "snapshots" / revision
    destination.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()):
        relative = item.relative_to(source)
        if item.is_symlink() or any(part in {"", ".", ".."} for part in relative.parts):
            raise RuntimeError(f"unsafe manual model member: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise RuntimeError(f"unsupported manual model member: {item}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(item, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"manual model folder is empty: {source}")

    snapshot = _cached_snapshot(repo_id, revision, cache_dir)
    if snapshot is None:
        raise RuntimeError(f"manual snapshot could not be opened from the local cache: {repo_id}@{revision}")
    return snapshot


def _verify_upstream_snapshot(snapshot: Path, expected: dict) -> None:
    for item in expected["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise RuntimeError(f"unsafe upstream manifest path: {relative.as_posix()}")
        target = snapshot / relative
        if not target.is_file():
            raise RuntimeError(
                f"snapshot for {expected['repo_id']} is incomplete; missing: {relative.as_posix()}"
            )
        if target.stat().st_size != int(item["bytes"]):
            raise RuntimeError(
                f"snapshot file size mismatch for {expected['repo_id']}: {relative.as_posix()}"
            )
        if snapshot_sha256(target) != item["sha256"]:
            raise RuntimeError(
                f"snapshot SHA-256 mismatch for {expected['repo_id']}: {relative.as_posix()}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, default=Path("release-spec.json"))
    parser.add_argument(
        "--upstream-manifest",
        type=Path,
        default=Path("upstream-model-manifest.json"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path(".runtime-assets/huggingface"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--manual-root",
        type=Path,
        help="import missing repositories from the documented manual-models directory",
    )
    args = parser.parse_args()

    app_root = args.app_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else app_root / args.spec
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    upstream_manifest_path = (
        args.upstream_manifest
        if args.upstream_manifest.is_absolute()
        else app_root / args.upstream_manifest
    )
    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
    expected_upstream = {
        item["id"]: item for item in upstream_manifest["repositories"]
    }
    if set(expected_upstream) != {item["id"] for item in spec["upstream_models"]}:
        raise RuntimeError("upstream model manifest does not match release-spec.json")
    hf_home = (args.hf_home if args.hf_home.is_absolute() else app_root / args.hf_home).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manual_root = None
    if args.manual_root is not None:
        manual_root = (
            args.manual_root if args.manual_root.is_absolute() else app_root / args.manual_root
        ).resolve()

    own_repo = "NettoAndTetto/osu-ai-detector-models"
    own_snapshot = _cached_snapshot(own_repo, spec["model_revision"], cache_dir)
    if own_snapshot is not None:
        try:
            verify_manifest(own_snapshot, "artifact-manifest.json")
        except (FileNotFoundError, RuntimeError):
            own_snapshot = None
    if own_snapshot is None and manual_root is not None:
        own_snapshot = _manual_folder(manual_root, "osu-ai-detector-models")
    if own_snapshot is None:
        own_snapshot = Path(
            snapshot_download(
                repo_id=own_repo,
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
        expected = expected_upstream[checkpoint["id"]]
        if (
            expected["repo_id"] != checkpoint["repo_id"]
            or expected["revision"] != checkpoint["revision"]
        ):
            raise RuntimeError(
                f"upstream model identity mismatch in release manifests: {checkpoint['id']}"
            )
        snapshot = _cached_snapshot(checkpoint["repo_id"], checkpoint["revision"], cache_dir)
        if snapshot is not None:
            try:
                _verify_upstream_snapshot(snapshot, expected)
            except RuntimeError:
                snapshot = None
        if snapshot is None and manual_root is not None:
            snapshot = _copy_manual_snapshot(
                _manual_folder(manual_root, checkpoint["id"]),
                checkpoint["repo_id"],
                checkpoint["revision"],
                cache_dir,
            )
        if snapshot is None:
            snapshot = Path(
                snapshot_download(
                    repo_id=checkpoint["repo_id"],
                    revision=checkpoint["revision"],
                    cache_dir=cache_dir,
                    local_files_only=args.offline,
                )
            ).resolve()
        _verify_upstream_snapshot(snapshot, expected)

    receipt = {
        "schema_version": 1,
        "release": spec["tag"],
        "model_repo": "NettoAndTetto/osu-ai-detector-models",
        "model_revision": spec["model_revision"],
        "upstream_models": spec["upstream_models"],
        "upstream_manifest_sha256": sha256(upstream_manifest_path),
        "manual_import": manual_root is not None,
    }
    receipt_path = app_root / ".runtime-assets" / "model-install-receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
