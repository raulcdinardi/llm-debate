#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="data/secret_word_sft/train.jsonl")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    path = Path(args.path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    for r in random.sample(rows, min(args.n, len(rows))):
        print(f"SOURCE: {r['meta']['source']}  SECRET: {r['meta']['secret_word']}")
        print("PROMPT:", r["prompt"][:200].replace("\n", "\\n"))
        print("COMPLETION:", r["completion"][:200].replace("\n", "\\n"))
        print("-" * 80)


if __name__ == "__main__":
    main()
