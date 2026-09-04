from __future__ import annotations

import argparse
import json

from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.judge_harness import (
    judge_harness_ids,
    resolve_judge_harness_id,
)


def _parse_init_adapter_dirs(values: list[str]) -> dict[str, str]:
    adapter_dirs = {}
    for value in values:
        name, sep, path = value.partition("=")
        if sep != "=" or not name or not path:
            raise ValueError("--init-adapter-dir entries must be NAME=PATH.")
        adapter_dirs[name] = path
    return adapter_dirs


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic rollout+train driver for the rewrite stack.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--env",
        default="ht_sequence",
        choices=[
            "ht_sequence",
            "coin_flip",
            "short_story",
            "secret_word",
            "quality_debate",
            "countdown_code",
            "constrained_writing",
            "mmlu_pro_pairwise",
        ],
    )
    parser.add_argument("--mode", default="debate", choices=["single_turn", "debate"])
    parser.add_argument("--adapter-layout", default="shared", choices=["shared", "split"])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-groups", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--request-seed-mode", default="none", choices=["none", "per_request"])
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sequence-len", type=int, default=8)
    parser.add_argument("--reward-mode", default="num_h", choices=["num_h", "num_transitions"])
    parser.add_argument("--quality-data-dir", default=None)
    parser.add_argument("--quality-split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--quality-hard-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-source", default="Gutenberg")
    parser.add_argument("--quality-topic-contains", default="Science fiction")
    parser.add_argument("--quality-download", action="store_true")
    parser.add_argument("--mmlu-pro-data-path", default=None)
    parser.add_argument("--thinking-mode", default="default", choices=["default", "no_think", "force_think"])
    parser.add_argument("--advantage-mode", default="zscore", choices=["identity", "centered_mean", "zscore"])
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--debate-rounds", type=int, default=3, help="Maximum debate depth (>=1).")
    parser.add_argument(
        "--debate-min-rounds",
        type=int,
        default=0,
        help="Minimum per-debate depth; 0 fixes every debate at --debate-rounds.",
    )
    parser.add_argument(
        "--debate-r1-reward",
        default="task",
        choices=[
            "task", "judge_pointwise", "judge", "judge_rejection_task",
            "judge_delta_task", "judge_soft_task_gap", "none",
        ],
    )
    parser.add_argument(
        "--debate-r23-reward", default="constant", choices=["constant", "soft_judge", "none"]
    )
    parser.add_argument("--debate-r23-constant", type=float, default=1.0)
    parser.add_argument("--debate-r1-judge-delta-q", type=float, default=1.0)
    parser.add_argument("--debate-incoherent-r23-reward", type=float, default=-0.5)
    parser.add_argument("--debate-r23-format-failure-penalty", type=float, default=0.0)
    parser.add_argument("--debate-r23-mode", default="symmetric", choices=["symmetric", "winner_only"])
    parser.add_argument("--debate-r23-advantage-scope", default="per_round", choices=["per_round", "merged_r23"])
    parser.add_argument("--debate-judge-adapter", default="policy", choices=["policy", "base", "solution", "debate", "judge"])
    parser.add_argument("--debate-external-judge-url", default=None)
    parser.add_argument("--debate-judge-server-url", default=None)
    parser.add_argument("--debate-judge-server-adapter-path", default=None)
    parser.add_argument("--debate-mock-judge-seed", type=int, default=None)
    parser.add_argument("--debate-external-judge-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--debate-judge-harness",
        default=None,
        choices=judge_harness_ids(),
        help="Versioned judge harness controlling rendering, parsing, prefill, and output contract.",
    )
    parser.add_argument(
        "--debate-judge-prompt-format",
        default=None,
        choices=["chat", "base_model_sft", "single_token_sft"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--debate-judge-max-tokens", type=int, default=0)
    parser.add_argument("--debate-judge-temperature", type=float, default=1.0)
    parser.add_argument("--debate-judge-top-p", type=float, default=1.0)
    parser.add_argument("--debate-judge-top-k", type=int, default=-1)
    parser.add_argument("--debate-judge-min-p", type=float, default=0.0)
    parser.add_argument("--debate-judge-presence-penalty", type=float, default=0.0)
    parser.add_argument("--debate-judge-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--debate-judge-seed", type=int, default=None)
    parser.add_argument(
        "--debate-judge-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Judge both A/B orders and require referent agreement for a coherent verdict.",
    )
    parser.add_argument(
        "--debate-judge-constrain-single-token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict a frozen single-token judge to tokenizer-valid A/B token ids.",
    )
    parser.add_argument(
        "--debate-judge-score-mode",
        default="hard_verdict",
        choices=["hard_verdict", "order_sym_soft_logit"],
        help="Keep legacy hard verdicts or compute an order-symmetrized soft judge score.",
    )
    parser.add_argument(
        "--judge-label-token-contract",
        default="none",
        choices=["none", "lfm25_ab_whitespace_compat_v1", "lfm25_openbookqa_spaced_ab_v1"],
        help=(
            "Temporary tokenizer compatibility contract for soft judge scoring. "
            "Replace and Phase-0 validate when the judge tokenizer or answer stem changes."
        ),
    )
    parser.add_argument(
        "--train-judge",
        "--train-judge-coherence-grpo",
        dest="train_judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train the judge adapter from both transcript orderings. The legacy "
            "--train-judge-coherence-grpo spelling is retained as a CLI alias."
        ),
    )
    parser.add_argument(
        "--judge-training-objective",
        choices=["grpo", "supervised_label_ce_js"],
        default="grpo",
        help=(
            "Optimization objective for the trainable judge. 'grpo' preserves sampled-action "
            "PPO/GRPO; 'supervised_label_ce_js' applies direct two-class label CE plus "
            "differentiable referent-aligned JS coherence. Debate adapters remain GRPO."
        ),
    )
    parser.add_argument(
        "--judge-coherence-js-weight",
        type=float,
        default=1.0,
        help="Coefficient lambda_js in label_ce + lambda_js * JS/ln(2).",
    )
    parser.add_argument(
        "--judge-grpo-reward-mode",
        choices=["coherence", "label", "label_js"],
        default="coherence",
        help=(
            "Judge GRPO raw reward source: 'coherence' scores referent agreement of the "
            "two orderings; 'label' scores each sampled verdict against the ground-truth "
            "trajectory reward (+1 gold referent, -1 otherwise, 0 on reward ties). "
            "JS coherence is a separate direct loss, never a sampled-action reward. "
            "The legacy 'label_js' spelling atomically migrates an enabled judge to "
            "supervised_label_ce_js."
        ),
    )
    parser.add_argument("--debate-round-adapter-names", nargs="*", default=["solution", "debate", "debate"])
    parser.add_argument(
        "--debate-prompt-format",
        default="chat",
        choices=["chat", "qwen35_base_text_prefill"],
    )
    parser.add_argument(
        "--debate-stop-on-concluded",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a real CONCLUDED string stop for R2/R3. This is supported only by "
            "SGLang with qwen35_base_text_prefill."
        ),
    )
    parser.add_argument("--base-r2-prefill", default="The reasons that my solution is better than my opponent's are:\n1)")
    parser.add_argument("--base-r3-prefill", default="Responding to my opponent's criticism:\n1)")
    parser.add_argument("--debate-r1-max-tokens", type=int, default=0, help="0 uses --max-tokens for debate R1.")
    parser.add_argument("--debate-r23-max-tokens", type=int, default=0, help="0 uses --max-tokens for debate rounds after R1.")
    parser.add_argument("--debate-r2-max-tokens", type=int, default=0, help="0 inherits --debate-r23-max-tokens for debate R2.")
    parser.add_argument("--debate-r3-max-tokens", type=int, default=0, help="0 inherits --debate-r23-max-tokens for debate R3.")
    parser.add_argument(
        "--rollout-grad-accum-steps",
        type=int,
        default=1,
        help="Debate: number of rollout micro-batches (num_groups debates each) accumulated before one train step.",
    )
    parser.add_argument(
        "--rollout-assistant-prefill",
        default=None,
        help=(
            "Optional assistant prefix encoded jointly with single-turn or debate-R1 Base-text prompts. "
            "It is prompt context, not part of the trainable completion. Omit to use the "
            "env/prompt-format default; pass an explicit empty string to force no prefill."
        ),
    )
    parser.add_argument("--train-minibatch-size", type=int, default=0)
    parser.add_argument("--train-max-tokens", type=int, default=0)
    parser.add_argument("--train-length-bucket-batches", action="store_true")
    parser.add_argument(
        "--train-logprob-backend",
        default="full_logits",
        choices=["full_logits", "selective_lm_head"],
        help="Training logprob implementation. selective_lm_head projects only trained token positions.",
    )
    parser.add_argument("--compile-train-logprob-helper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--train-adapter-names",
        nargs="*",
        default=[],
        help="If provided, train/save only these adapter names. Other adapters remain available for rollout sampling.",
    )
    parser.add_argument("--stop-parsed-reward-hacking-min", type=float, default=None)
    parser.add_argument("--stop-parsed-reward-hacking-max", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--on-policy-logprob-check",
        action="store_true",
        default=True,
        help=(
            "Required fail-closed check before PPO ratio/backward when rollout and trainer "
            "behavior-policy logprobs differ. This safety gate cannot be disabled."
        ),
    )
    parser.add_argument(
        "--on-policy-logprob-warn-only",
        action="store_true",
        help=(
            "Measure and record strict logprob-parity violations but continue through "
            "PPO/backward. Use only after an explicit warning-band gate override."
        ),
    )
    parser.add_argument("--on-policy-logprob-abs-tol", type=float, default=1e-3)
    parser.add_argument("--on-policy-logprob-warning-path", default=None)
    parser.add_argument("--on-policy-logprob-max-records-per-batch", type=int, default=8)
    parser.add_argument("--sampler-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--sampler-max-model-len", type=int, default=512)
    parser.add_argument("--sampler-max-num-seqs", type=int, default=0)
    parser.add_argument("--sampler-enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sampler-teardown-before-training",
        action="store_true",
        help="Legacy fallback that rebuilds the engine; vLLM now sleeps by default when this is omitted.",
    )
    parser.add_argument(
        "--sampler-sleep-before-training",
        action="store_true",
        help="Enable memory-saver sleep explicitly (automatic for vLLM; required for SGLang).",
    )
    parser.add_argument(
        "--sampler-sleep-level",
        type=int,
        choices=[1, 2],
        default=1,
        help="vLLM sleep level; level 1 preserves CPU-offloaded base weights and is fastest for LoRA-only updates.",
    )
    parser.add_argument(
        "--sampler-backend",
        default="vllm",
        choices=["vllm", "transformers", "sglang"],
        help="Rollout inference backend. transformers reuses the training LoRA via HF generate(); sglang uses an external SGLang HTTP server.",
    )
    parser.add_argument("--sampler-sglang-base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--sampler-sglang-timeout-s", type=float, default=600.0)
    parser.add_argument("--sampler-sglang-pin-loras", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sampler-sglang-unload-stale-adapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-adapter-path", default=None, help="Initialize a shared-adapter run from this LoRA.")
    parser.add_argument(
        "--init-adapter-dir",
        action="append",
        default=[],
        help="Initialize a named adapter for split runs. Repeat as NAME=PATH.",
    )
    parser.add_argument("--init-adapter-dirs-json", default=None)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--target-parameters", default="")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--no-trace-model-io", action="store_true")
    parser.add_argument("--trace-model-io-dir", default=None)
    parser.add_argument("--trace-top-logprobs", type=int, default=5)
    parser.add_argument("--no-resource-logging", action="store_true")
    parser.add_argument("--resource-log-interval-s", type=float, default=5.0)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="llm-local-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-upload-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-checkpoint-every", type=int, default=10)
    parser.add_argument("--optimizer-checkpoint-every", type=int, default=50)
    parser.add_argument("--rollout-shard-every", type=int, default=10)
    parser.add_argument("--wandb-table-samples-per-shard", type=int, default=32)
    parser.add_argument("--reference-kl-every", type=int, default=10, help="0 disables sampled KL to initialization.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.judge_grpo_reward_mode == "label_js":
        args.judge_grpo_reward_mode = "coherence"
        if args.train_judge:
            args.judge_training_objective = "supervised_label_ce_js"
    return args


def main() -> int:
    from llm_local_rl.driver import TrainingDriver

    args = parse_args()
    if args.sampler_teardown_before_training and args.sampler_sleep_before_training:
        raise ValueError("Use either --sampler-teardown-before-training or --sampler-sleep-before-training, not both.")
    if args.resume:
        if args.init_adapter_path is not None:
            raise ValueError("--init-adapter-path is only valid for fresh runs, not --resume.")
        if args.init_adapter_dir:
            raise ValueError("--init-adapter-dir is only valid for fresh runs, not --resume.")
        if args.init_adapter_dirs_json is not None:
            raise ValueError("--init-adapter-dirs-json is only valid for fresh runs, not --resume.")
        driver = TrainingDriver.resume(output_dir=args.output_dir)
    else:
        if args.init_adapter_path is not None and args.adapter_layout != "shared":
            raise ValueError("--init-adapter-path currently supports only --adapter-layout shared.")
        if args.init_adapter_path is not None and (args.init_adapter_dir or args.init_adapter_dirs_json is not None):
            raise ValueError("Use either --init-adapter-path or named --init-adapter-dir inputs, not both.")
        if args.init_adapter_dirs_json is not None and args.init_adapter_dir:
            raise ValueError("Use either --init-adapter-dirs-json or repeated --init-adapter-dir inputs, not both.")
        init_adapter_dirs = None
        if args.init_adapter_path is not None:
            init_adapter_dirs = {"shared": args.init_adapter_path}
        elif args.init_adapter_dirs_json is not None:
            init_adapter_dirs = json.loads(args.init_adapter_dirs_json)
        elif args.init_adapter_dir:
            init_adapter_dirs = _parse_init_adapter_dirs(args.init_adapter_dir)
        driver = TrainingDriver(
            config=TrainRunConfig(
                model_path=args.model_path,
                tokenizer_path=args.tokenizer_path,
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
                    top_p=args.top_p,
                    min_p=args.min_p,
                    seed=args.seed,
                    request_seed_mode=args.request_seed_mode,
                ),
                steps=args.steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                adapter_layout=args.adapter_layout,
                sequence_len=args.sequence_len,
                reward_mode=args.reward_mode,
                quality_data_dir=args.quality_data_dir,
                quality_split=args.quality_split,
                quality_hard_only=args.quality_hard_only,
                quality_source=args.quality_source,
                quality_topic_contains=args.quality_topic_contains,
                quality_download=args.quality_download,
                mmlu_pro_data_path=args.mmlu_pro_data_path,
                thinking_mode=args.thinking_mode,
                advantage_mode=args.advantage_mode,
                ppo_clip_epsilon=args.ppo_clip_epsilon,
                debate_rounds=args.debate_rounds,
                debate_min_rounds=args.debate_min_rounds,
                debate_r1_reward=args.debate_r1_reward,
                debate_r23_reward=args.debate_r23_reward,
                debate_r23_constant=args.debate_r23_constant,
                debate_r1_judge_delta_q=args.debate_r1_judge_delta_q,
                debate_incoherent_r23_reward=args.debate_incoherent_r23_reward,
                debate_r23_format_failure_penalty=args.debate_r23_format_failure_penalty,
                debate_r23_mode=args.debate_r23_mode,
                debate_r23_advantage_scope=args.debate_r23_advantage_scope,
                debate_judge_adapter=args.debate_judge_adapter,
                debate_external_judge_url=args.debate_external_judge_url,
                debate_judge_server_url=args.debate_judge_server_url,
                debate_judge_server_adapter_path=args.debate_judge_server_adapter_path,
                debate_mock_judge_seed=args.debate_mock_judge_seed,
                debate_external_judge_timeout_s=args.debate_external_judge_timeout_s,
                debate_judge_harness=resolve_judge_harness_id(
                    harness_id=args.debate_judge_harness,
                    legacy_prompt_format=args.debate_judge_prompt_format,
                    num_rounds=args.debate_rounds,
                ),
                debate_judge_max_tokens=args.debate_judge_max_tokens,
                debate_judge_temperature=args.debate_judge_temperature,
                debate_judge_top_p=args.debate_judge_top_p,
                debate_judge_top_k=args.debate_judge_top_k,
                debate_judge_min_p=args.debate_judge_min_p,
                debate_judge_presence_penalty=args.debate_judge_presence_penalty,
                debate_judge_repetition_penalty=args.debate_judge_repetition_penalty,
                debate_judge_seed=args.debate_judge_seed,
                debate_judge_bidirectional=args.debate_judge_bidirectional,
                debate_judge_constrain_single_token=args.debate_judge_constrain_single_token,
                debate_judge_score_mode=args.debate_judge_score_mode,
                judge_label_token_contract=args.judge_label_token_contract,
                train_judge=args.train_judge,
                judge_training_objective=args.judge_training_objective,
                judge_coherence_js_weight=args.judge_coherence_js_weight,
                judge_grpo_reward_mode=args.judge_grpo_reward_mode,
                debate_round_adapter_names=tuple(args.debate_round_adapter_names),
                debate_prompt_format=args.debate_prompt_format,
                debate_stop_on_concluded=args.debate_stop_on_concluded,
                base_r2_prefill=args.base_r2_prefill,
                base_r3_prefill=args.base_r3_prefill,
                debate_r1_max_tokens=args.debate_r1_max_tokens,
                debate_r23_max_tokens=args.debate_r23_max_tokens,
                debate_r2_max_tokens=args.debate_r2_max_tokens,
                debate_r3_max_tokens=args.debate_r3_max_tokens,
                rollout_grad_accum_steps=args.rollout_grad_accum_steps,
                rollout_assistant_prefill=args.rollout_assistant_prefill,
                train_minibatch_size=args.train_minibatch_size,
                train_max_tokens=args.train_max_tokens,
                train_length_bucket_batches=args.train_length_bucket_batches,
                train_logprob_backend=args.train_logprob_backend,
                compile_train_logprob_helper=args.compile_train_logprob_helper,
                train_adapter_names=tuple(args.train_adapter_names),
                stop_parsed_reward_hacking_min=args.stop_parsed_reward_hacking_min,
                stop_parsed_reward_hacking_max=args.stop_parsed_reward_hacking_max,
                gradient_checkpointing=args.gradient_checkpointing,
                on_policy_logprob_check=args.on_policy_logprob_check,
                on_policy_logprob_warn_only=args.on_policy_logprob_warn_only,
                on_policy_logprob_abs_tol=args.on_policy_logprob_abs_tol,
                on_policy_logprob_warning_path=args.on_policy_logprob_warning_path,
                on_policy_logprob_max_records_per_batch=args.on_policy_logprob_max_records_per_batch,
                sampler_gpu_memory_utilization=args.sampler_gpu_memory_utilization,
                sampler_max_model_len=args.sampler_max_model_len,
                sampler_max_num_seqs=args.sampler_max_num_seqs,
                sampler_enforce_eager=args.sampler_enforce_eager,
                sampler_teardown_before_training=args.sampler_teardown_before_training,
                sampler_sleep_before_training=args.sampler_sleep_before_training,
                sampler_sleep_level=args.sampler_sleep_level,
                sampler_backend=args.sampler_backend,
                sampler_sglang_base_url=args.sampler_sglang_base_url,
                sampler_sglang_timeout_s=args.sampler_sglang_timeout_s,
                sampler_sglang_pin_loras=args.sampler_sglang_pin_loras,
                sampler_sglang_unload_stale_adapters=args.sampler_sglang_unload_stale_adapters,
                init_adapter_dirs=init_adapter_dirs,
                target_modules=_parse_csv_tuple(args.target_modules),
                target_parameters=_parse_csv_tuple(args.target_parameters),
                lora_rank=args.lora_rank,
                trace_model_io=not args.no_trace_model_io,
                trace_model_io_dir=args.trace_model_io_dir,
                trace_top_logprobs=args.trace_top_logprobs,
                resource_logging=not args.no_resource_logging,
                resource_log_interval_s=args.resource_log_interval_s,
                wandb_enabled=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=args.wandb_group,
                wandb_run_name=args.wandb_run_name,
                wandb_mode=args.wandb_mode,
                wandb_upload_artifacts=args.wandb_upload_artifacts,
                adapter_checkpoint_every=args.adapter_checkpoint_every,
                optimizer_checkpoint_every=args.optimizer_checkpoint_every,
                rollout_shard_every=args.rollout_shard_every,
                wandb_table_samples_per_shard=args.wandb_table_samples_per_shard,
                reference_kl_every=args.reference_kl_every,
            )
        )
    driver.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
