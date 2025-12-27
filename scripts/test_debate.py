#!/usr/bin/env python3
"""Run debate(s) and save logs.

Usage:
    python test_debate.py                    # Run one debate
    python test_debate.py -n 5               # Run 5 debates
    python test_debate.py --question "..."   # Custom question
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Allow running as a script without installing the package.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run debate(s) and save logs")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of debates to run")
    parser.add_argument("--question", "-q", default="What is 7 * 8?", help="Question to debate")
    parser.add_argument("--ground-truth", "-g", default="56", help="Ground truth answer")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--real-judge", action="store_true", help="Use LLM judge instead of mock")
    return parser.parse_args()


def save_log(result, log_dir: str) -> Path:
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = Path(log_dir) / f"debate_{timestamp}.json"

    log_data = {
        "timestamp": timestamp,
        "question": result.question,
        "ground_truth": result.ground_truth,
        "verdict": result.verdict,
        "judge_reasoning": result.judge_reasoning,
        "agent_a": {
            "frozen_solution": result.trajectory_a.frozen_solution,
            "r1": result.trajectory_a.metrics.get("r1", ""),
            "r2": result.trajectory_a.metrics.get("r2", ""),
            "r3": result.trajectory_a.metrics.get("r3", ""),
            "transitions": [
                {
                    "round": t.round_num,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "completion_logprobs": t.completion_logprobs,
                }
                for t in result.trajectory_a.transitions
            ],
        },
        "agent_b": {
            "frozen_solution": result.trajectory_b.frozen_solution,
            "r1": result.trajectory_b.metrics.get("r1", ""),
            "r2": result.trajectory_b.metrics.get("r2", ""),
            "r3": result.trajectory_b.metrics.get("r3", ""),
            "transitions": [
                {
                    "round": t.round_num,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "completion_logprobs": t.completion_logprobs,
                }
                for t in result.trajectory_b.transitions
            ],
        },
        "metrics": result.metrics,
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_path


async def main():
    args = parse_args()

    print("Setting up Tinker client...")
    from tinker_debate.tinker_client import TinkerDebateClient

    client = await TinkerDebateClient.create()

    from tinker_debate.debate_env import (
        DebateRolloutClient,
        DebateTokenRolloutClient,
        mock_judge_random,
        run_debate_batch_token_only,
    )
    from tinker_debate.debate_types import DebateConfig

    async def generate_fn(prompts, max_tokens, temp):
        return await client.generate(prompts, max_tokens=max_tokens, temperature=temp)

    rollout_client = DebateRolloutClient(generate_fn=generate_fn)

    async def sample_tokens_fn(prompt_tokens_list, max_tokens, temp):
        return await client.sample_token_prompts(
            prompt_tokens_list=prompt_tokens_list,
            max_tokens=max_tokens,
            temperature=temp,
        )

    token_rollout_client = DebateTokenRolloutClient(
        sample_fn=sample_tokens_fn,
        decode_fn=lambda toks: client.tokenizer.decode(toks, skip_special_tokens=True),
    )

    config = DebateConfig.cheap()
    judge_fn = None if args.real_judge else mock_judge_random

    print(f"Running {args.num} debate(s)...")
    print(f"  Question: {args.question}")
    print(f"  Judge: {'LLM' if args.real_judge else 'mock (random)'}")
    print(f"  Logs: {args.log_dir}/")
    print()

    stats = {"A": 0, "B": 0, "correct_wins": 0, "total": 0}

    batch = [(args.question, args.ground_truth) for _ in range(args.num)]
    results = await run_debate_batch_token_only(
        batch,
        token_rollout_client,
        client.tokenizer,
        config,
        rollout_client,
        judge_fn=judge_fn,
    )

    for i, result in enumerate(results):
        log_path = save_log(result, args.log_dir)

        stats[result.verdict] += 1
        stats["total"] += 1

        winner = result.get_winner_trajectory()
        if winner.frozen_solution == args.ground_truth:
            stats["correct_wins"] += 1

        sol_a = result.trajectory_a.frozen_solution or "?"
        sol_b = result.trajectory_b.frozen_solution or "?"
        correct = "✓" if winner.frozen_solution == args.ground_truth else "✗"

        print(f"  [{i+1}/{args.num}] {result.verdict} wins | A={sol_a} B={sol_b} | {correct} | {log_path.name}")

    print()
    print(
        f"Done. A:{stats['A']} B:{stats['B']} | Correct wins: {stats['correct_wins']}/{stats['total']}"
    )
    print("View logs: python view_logs.py --list")
    print("Watch live: python view_logs.py --watch")


if __name__ == "__main__":
    asyncio.run(main())
