#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm_local_rl.behavior_policy_microprobe import (
    run_temperature_four_cell_microprobe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a zero-update four-cell temperature/logprob semantic probe."
    )
    parser.add_argument("--behavior-temperature", type=float, default=0.8)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    logits = torch.randn(
        (args.rows, args.vocab_size),
        generator=generator,
        dtype=torch.float32,
    ) * 3.0
    # Draw actions from the actual T-behavior distribution. This makes the
    # p/q importance-ratio mean approach one in the mismatched cell, proving
    # that ratio_mean≈1 is not a parity signal.
    target_ids = torch.multinomial(
        torch.softmax(logits / args.behavior_temperature, dim=-1),
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
    report = run_temperature_four_cell_microprobe(
        logits=logits,
        target_ids=target_ids,
        behavior_temperature=args.behavior_temperature,
        abs_tol=args.abs_tol,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
