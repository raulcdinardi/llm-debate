from __future__ import annotations

import argparse

from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.driver import TrainingDriver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HT single-turn rollout+train with the rewrite stack.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--reward-mode", choices=["num_h", "num_transitions"], default="num_h")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adapter-name", default="shared")
    parser.add_argument("--thinking-mode", default="default", choices=["default", "no_think", "force_think"])
    parser.add_argument("--no-trace-model-io", action="store_true")
    parser.add_argument("--trace-model-io-dir", default=None)
    parser.add_argument("--trace-top-logprobs", type=int, default=5)
    parser.add_argument("--no-resource-logging", action="store_true")
    parser.add_argument("--resource-log-interval-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_layout = "shared" if args.adapter_name == "shared" else "split"
    driver = TrainingDriver(
        config=TrainRunConfig(
            model_path=args.model_path,
            output_dir=args.output_dir,
            rollout=RolloutConfig(
                env_name="ht_sequence",
                mode="single_turn",
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                min_p=args.min_p,
                seed=args.seed,
            ),
            steps=args.steps,
            learning_rate=args.learning_rate,
            adapter_layout=adapter_layout,
            sequence_len=args.sequence_len,
            reward_mode=args.reward_mode,
            thinking_mode=args.thinking_mode,
            trace_model_io=not args.no_trace_model_io,
            trace_model_io_dir=args.trace_model_io_dir,
            trace_top_logprobs=args.trace_top_logprobs,
            resource_logging=not args.no_resource_logging,
            resource_log_interval_s=args.resource_log_interval_s,
        )
    )
    driver.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
