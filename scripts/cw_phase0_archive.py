#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a hashed Phase-0 artifact manifest and verified gzip archive.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    archive = Path(args.archive).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    marker = run_root / ("DONE" if args.exit_status == 0 else "FAILED")
    marker.touch()
    (run_root / "PHASE1_FORBIDDEN").touch()
    files = {}
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[str(path.relative_to(run_root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "schema": "cw_judge_signal_phase0_artifact_manifest_v1",
        "run_name": run_root.name,
        "exit_status": args.exit_status,
        "phase1_forbidden": True,
        "files": files,
    }
    manifest_path = run_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".partial")
    if partial.exists():
        partial.unlink()
    with tarfile.open(partial, "w:gz") as handle:
        handle.add(run_root, arcname=run_root.name)
    with tarfile.open(partial, "r:gz") as handle:
        members = handle.getmembers()
        if not members or f"{run_root.name}/artifact_manifest.json" not in {member.name for member in members}:
            raise RuntimeError("Archive verification failed: artifact manifest missing")
    partial.replace(archive)
    print(json.dumps({
        "event": "phase0_archive_written",
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "exit_status": args.exit_status,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
