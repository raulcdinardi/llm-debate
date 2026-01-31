#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def _default_out_dir(repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    return Path("models") / safe


def main() -> None:
    p = argparse.ArgumentParser(description="Download an HF model snapshot to a local directory (fail-fast).")
    p.add_argument("--repo-id", required=True, help="HuggingFace model repo id, e.g. Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--revision", default=None, help="Optional git revision/tag/commit")
    p.add_argument("--out-dir", default=None, help="Output directory (default: models/<repo-id with / -> __>)")
    args = p.parse_args()

    out_dir = _default_out_dir(args.repo_id) if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    print(local_path)


if __name__ == "__main__":
    main()

