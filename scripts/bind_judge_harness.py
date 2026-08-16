from __future__ import annotations

import argparse
from pathlib import Path

from llm_local_rl.judge_harness import (
    JUDGE_HARNESS_MANIFEST,
    judge_harness_ids,
    validate_judge_harness_manifest,
    write_judge_harness_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind a judge LoRA directory to the exact harness used for its training."
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--harness", required=True, choices=judge_harness_ids())
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(f"Not a PEFT adapter directory: {adapter_dir}")
    existing = adapter_dir / JUDGE_HARNESS_MANIFEST
    if existing.is_file():
        # Never silently rewrite adapter provenance. An identical existing
        # binding is idempotent; a different or stale binding fails closed.
        validate_judge_harness_manifest(
            adapter_dir=adapter_dir,
            harness_id=args.harness,
        )
        print(existing)
        return 0
    path = write_judge_harness_manifest(
        adapter_dir=adapter_dir,
        harness_id=args.harness,
    )
    validate_judge_harness_manifest(
        adapter_dir=adapter_dir,
        harness_id=args.harness,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
