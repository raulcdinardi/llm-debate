from __future__ import annotations

import argparse

from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.driver import TrainingDriver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic rollout+train driver for the rewrite stack.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env", default="ht_sequence", choices=["ht_sequence", "coin_flip"])
    parser.add_argument("--mode", default="debate", choices=["single_turn", "debate"])
    parser.add_argument("--adapter-layout", default="shared", choices=["shared", "split"])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--reward-mode", default="num_h", choices=["num_h", "num_transitions"])
    parser.add_argument("--thinking-mode", default="default", choices=["default", "no_think", "force_think"])
    parser.add_argument("--advantage-mode", default="zscore", choices=["identity", "centered_mean", "zscore"])
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--debate-rounds", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--debate-r1-reward", default="task", choices=["task", "judge_pointwise", "judge", "none"])
    parser.add_argument("--debate-r23-reward", default="constant", choices=["constant", "none"])
    parser.add_argument("--debate-r23-constant", type=float, default=1.0)
    parser.add_argument("--debate-r23-mode", default="symmetric", choices=["symmetric", "winner_only"])
    parser.add_argument("--debate-judge-adapter", default="policy", choices=["policy", "base", "solution", "debate"])
    parser.add_argument("--debate-external-judge-url", default=None)
    parser.add_argument("--debate-external-judge-timeout-s", type=float, default=600.0)
    parser.add_argument("--debate-round-adapter-names", nargs="*", default=["solution", "debate", "debate"])
    parser.add_argument("--train-minibatch-size", type=int, default=0)
    parser.add_argument("--sampler-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--sampler-max-model-len", type=int, default=512)
    parser.add_argument("--no-trace-model-io", action="store_true")
    parser.add_argument("--trace-model-io-dir", default=None)
    parser.add_argument("--trace-top-logprobs", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume:
        driver = TrainingDriver.resume(output_dir=args.output_dir)
    else:
        driver = TrainingDriver(
            config=TrainRunConfig(
                model_path=args.model_path,
                output_dir=args.output_dir,
                rollout=RolloutConfig(
                    env_name=args.env,
                    mode=args.mode,
                    num_samples=args.num_samples,
                    num_groups=args.num_groups,
                    group_size=args.group_size,
                    rollout_batch_size=args.rollout_batch_size,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    min_p=args.min_p,
                    seed=args.seed,
                ),
                steps=args.steps,
                learning_rate=args.learning_rate,
                adapter_layout=args.adapter_layout,
                sequence_len=args.sequence_len,
                reward_mode=args.reward_mode,
                thinking_mode=args.thinking_mode,
                advantage_mode=args.advantage_mode,
                ppo_clip_epsilon=args.ppo_clip_epsilon,
                debate_rounds=args.debate_rounds,
                debate_r1_reward=args.debate_r1_reward,
                debate_r23_reward=args.debate_r23_reward,
                debate_r23_constant=args.debate_r23_constant,
                debate_r23_mode=args.debate_r23_mode,
                debate_judge_adapter=args.debate_judge_adapter,
                debate_external_judge_url=args.debate_external_judge_url,
                debate_external_judge_timeout_s=args.debate_external_judge_timeout_s,
                debate_round_adapter_names=tuple(args.debate_round_adapter_names),
                train_minibatch_size=args.train_minibatch_size,
                sampler_gpu_memory_utilization=args.sampler_gpu_memory_utilization,
                sampler_max_model_len=args.sampler_max_model_len,
                trace_model_io=not args.no_trace_model_io,
                trace_model_io_dir=args.trace_model_io_dir,
                trace_top_logprobs=args.trace_top_logprobs,
            )
        )
    driver.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
