from __future__ import annotations

import argparse
import json
from statistics import mean

from transformers import AutoTokenizer

from llm_local_rl.envs import HTSequenceEnv
from llm_local_rl.episodes import SingleTurnEpisodeBuilder
from llm_local_rl.vllm_sampling import VllmRuntimeConfig, VllmSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-turn HT baseline with the rewrite stack.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--reward-mode", choices=["num_h", "num_transitions"], default="num_h")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    env = HTSequenceEnv(sequence_len=args.sequence_len, reward_mode=args.reward_mode)
    builder = SingleTurnEpisodeBuilder(validate_behavior_policy_contract=False)
    sampler = VllmSampler(runtime=VllmRuntimeConfig(model_path=args.model_path))

    instances = env.sample_instances(n=args.num_samples, seed=args.seed)
    records: list[dict] = []
    for idx, instance in enumerate(instances):
        request_seed = None if args.seed is None else args.seed + idx
        sample = builder.build_and_score(
            env=env,
            tokenizer=tokenizer,
            sampler=sampler,
            instance=instance,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=request_seed,
            min_p=args.min_p,
        )
        turn = sample.turns[0]
        records.append(
            {
                "instance_id": sample.instance_id,
                "reward": sample.reward,
                "reward_metrics": sample.reward_metrics,
                "completion_text": turn.metadata.get("text", ""),
                "completion_token_ids": turn.completion_token_ids,
                "completion_logprobs": turn.completion_logprobs,
            }
        )

    parse_successes = [float(row["reward_metrics"]["parse_success"]) for row in records]
    rewards = [float(row["reward"]) for row in records]
    summary = {
        "num_samples": len(records),
        "sequence_len": args.sequence_len,
        "reward_mode": args.reward_mode,
        "temperature": args.temperature,
        "mean_reward": mean(rewards) if rewards else 0.0,
        "mean_parse_success": mean(parse_successes) if parse_successes else 0.0,
        "records": records,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
