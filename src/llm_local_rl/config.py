from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path

from llm_local_rl.behavior_policy import BehaviorPolicySpec
from llm_local_rl.judge_harness import (
    CHAT_SOLUTION_TAGGED_V1,
    CONSTITUTION_SINGLE_TOKEN_V1,
    JudgeHarnessSpec,
    SOLUTION_R1_RATIONALE_V1,
    get_judge_harness,
    resolve_judge_harness_id,
)
from llm_local_rl.soft_judge import (
    JUDGE_LABEL_TOKEN_CONTRACT_NONE,
    JUDGE_LABEL_TOKEN_CONTRACTS,
    LFM25_AB_WHITESPACE_COMPAT_V1,
    LFM25_OPENBOOKQA_SPACED_AB_V1,
)


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout sizing.

    For single-turn rollouts, num_samples is the total number of sampled
    completions per step. num_rollouts_per_instance controls how many
    completions are sampled for each distinct instance, so the driver draws
    num_samples // num_rollouts_per_instance instances.
    """

    env_name: str = "ht_sequence"
    mode: str = "debate"
    num_samples: int = 16
    num_rollouts_per_instance: int = 1
    num_groups: int = 2
    group_size: int = 8
    rollout_batch_size: int = 0
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int | None = None
    request_seed_mode: str = "none"


@dataclass(frozen=True)
class TrainRunConfig:
    model_path: str
    output_dir: str
    tokenizer_path: str | None = None
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adapter_layout: str = "shared"
    sequence_len: int = 8
    reward_mode: str = "num_h"
    strict_ht_format: bool = False
    constrained_writing_rules_per_speaker: int = 2
    constrained_writing_reward_scope: str = "both"
    constrained_writing_sides: str = "both"
    constrained_writing_rule_family: str = "generic"
    constrained_writing_reward_mode: str = "additive"
    constrained_writing_letter_temperature: float = 1.0
    constrained_writing_anchors: str = "on"
    quality_data_dir: str | None = None
    quality_split: str = "train"
    quality_hard_only: bool = True
    quality_source: str | None = "Gutenberg"
    quality_topic_contains: str | None = "Science fiction"
    quality_download: bool = False
    mmlu_pro_data_path: str | None = None
    thinking_mode: str = "default"
    advantage_mode: str = "zscore"
    ppo_clip_epsilon: float = 0.2
    debate_rounds: int = 3
    debate_rounds_per_group: tuple[int, ...] = ()
    debate_r1_reward: str = "task"
    debate_r23_reward: str = "constant"
    debate_r23_constant: float = 1.0
    debate_r1_judge_delta_q: float = 1.0
    debate_incoherent_r23_reward: float = -0.5
    debate_r23_format_failure_penalty: float = 0.0
    debate_r23_mode: str = "symmetric"
    debate_r23_advantage_scope: str = "per_round"
    debate_judge_adapter: str = "policy"
    debate_external_judge_url: str | None = None
    debate_judge_server_url: str | None = None
    debate_judge_server_adapter_path: str | None = None
    debate_mock_judge_seed: int | None = None
    debate_external_judge_timeout_s: float = 600.0
    debate_judge_harness: str = CHAT_SOLUTION_TAGGED_V1
    debate_judge_max_tokens: int = 0
    debate_judge_temperature: float = 1.0
    debate_judge_top_p: float = 1.0
    debate_judge_top_k: int = -1
    debate_judge_min_p: float = 0.0
    debate_judge_presence_penalty: float = 0.0
    debate_judge_repetition_penalty: float = 1.0
    debate_judge_seed: int | None = None
    debate_judge_bidirectional: bool = False
    debate_judge_constrain_single_token: bool = False
    debate_judge_score_mode: str = "hard_verdict"
    judge_label_token_contract: str = JUDGE_LABEL_TOKEN_CONTRACT_NONE
    train_judge: bool = False
    judge_training_objective: str = "grpo"
    judge_coherence_js_weight: float = 1.0
    judge_grpo_reward_mode: str = "coherence"
    debate_round_adapter_names: tuple[str, ...] = ("solution", "debate", "debate")
    debate_prompt_format: str = "chat"
    debate_stop_on_concluded: bool = False
    base_r2_prefill: str = "The reasons that my solution is better than my opponent's are:\n1)"
    base_r3_prefill: str = "Responding to my opponent's criticism:\n1)"
    debate_r1_max_tokens: int = 0
    debate_r23_max_tokens: int = 0
    debate_r2_max_tokens: int = 0
    debate_r3_max_tokens: int = 0
    rollout_grad_accum_steps: int = 1
    rollout_assistant_prefill: str | None = None
    train_minibatch_size: int = 0
    train_max_tokens: int = 0
    train_length_bucket_batches: bool = False
    train_logprob_backend: str = "full_logits"
    compile_train_logprob_helper: bool = False
    train_adapter_names: tuple[str, ...] = ()
    stop_parsed_reward_hacking_min: float | None = None
    stop_parsed_reward_hacking_max: float | None = None
    gradient_checkpointing: bool = True
    on_policy_logprob_check: bool = True
    on_policy_logprob_warn_only: bool = False
    on_policy_logprob_abs_tol: float = 1e-3
    on_policy_logprob_warning_path: str | None = None
    on_policy_logprob_max_records_per_batch: int = 8
    sampler_gpu_memory_utilization: float = 0.55
    sampler_max_model_len: int = 512
    sampler_max_num_seqs: int = 0
    sampler_enforce_eager: bool = True
    sampler_max_lora_rank: int = 32
    sampler_max_loras: int = 4
    sampler_teardown_before_training: bool = False
    sampler_sleep_before_training: bool = False
    sampler_sleep_level: int = 1
    sampler_backend: str = "vllm"
    sampler_sglang_base_url: str = "http://127.0.0.1:30000"
    sampler_sglang_timeout_s: float = 600.0
    sampler_sglang_pin_loras: bool = False
    sampler_sglang_unload_stale_adapters: bool = True
    init_adapter_dirs: dict[str, str] | None = None
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    target_parameters: tuple[str, ...] = ()
    lora_rank: int = 32
    trace_model_io: bool = True
    trace_model_io_dir: str | None = None
    trace_top_logprobs: int = 5
    resource_logging: bool = True
    resource_log_interval_s: float = 5.0
    wandb_enabled: bool = True
    wandb_project: str = "llm-local-rl"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_upload_artifacts: bool = True
    adapter_checkpoint_every: int = 10
    optimizer_checkpoint_every: int = 50
    rollout_shard_every: int = 10
    wandb_table_samples_per_shard: int = 32
    reference_kl_every: int = 10

    def to_dict(self) -> dict:
        data = asdict(self)
        data["behavior_policy_contract"] = self.behavior_policy().to_dict()
        data["judge_label_token_contract_temporary"] = (
            self.judge_label_token_contract == LFM25_AB_WHITESPACE_COMPAT_V1
        )
        return data

    def behavior_policy(self) -> BehaviorPolicySpec:
        return BehaviorPolicySpec.from_rollout_config(self.rollout)

    def judge_harness(self) -> JudgeHarnessSpec:
        """Return the single resolved judge contract used by every execution path."""
        return get_judge_harness(self.debate_judge_harness)

    def configured_debate_depths(self) -> tuple[int, ...]:
        if self.debate_rounds_per_group:
            return self.debate_rounds_per_group
        return (self.debate_rounds,) * max(1, self.rollout.group_size // 2)

    def effective_debate_min_rounds(self) -> int:
        return min(self.configured_debate_depths())

    def effective_debate_max_rounds(self) -> int:
        return max(self.configured_debate_depths())

    def resolved_debate_round_adapter_names(self) -> tuple[str, ...]:
        if not self.debate_round_adapter_names:
            raise ValueError("debate_round_adapter_names must not be empty")
        return tuple(
            self.debate_round_adapter_names[min(index, len(self.debate_round_adapter_names) - 1)]
            for index in range(self.effective_debate_max_rounds())
        )

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_checkpoint_every", self.adapter_checkpoint_every),
            ("rollout_shard_every", self.rollout_shard_every),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.optimizer_checkpoint_every < 0:
            raise ValueError("optimizer_checkpoint_every must be non-negative")
        if self.wandb_table_samples_per_shard < 0:
            raise ValueError("wandb_table_samples_per_shard must be non-negative")
        if self.reference_kl_every < 0:
            raise ValueError("reference_kl_every must be non-negative")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.sampler_teardown_before_training and self.sampler_sleep_before_training:
            raise ValueError(
                "sampler_teardown_before_training and sampler_sleep_before_training "
                "are mutually exclusive"
            )
        if self.sampler_sleep_level not in (1, 2):
            raise ValueError("sampler_sleep_level must be 1 or 2")
        judge_modes = sum(
            value is not None
            for value in (
                self.debate_external_judge_url,
                self.debate_judge_server_url,
                self.debate_mock_judge_seed,
            )
        )
        if judge_modes > 1:
            raise ValueError(
                "debate_external_judge_url, debate_judge_server_url, and debate_mock_judge_seed "
                "are mutually exclusive."
            )
        if not 0.0 < float(self.rollout.top_p) <= 1.0:
            raise ValueError(f"rollout.top_p must be in (0, 1], got {self.rollout.top_p}")
        judge_harness = self.judge_harness()
        uses_configured_harness = self.debate_mock_judge_seed is None
        if self.debate_rounds < 1:
            raise ValueError("debate_rounds must be at least 1")
        object.__setattr__(
            self,
            "debate_rounds_per_group",
            tuple(self.debate_rounds_per_group),
        )
        if self.debate_rounds_per_group:
            if self.rollout.mode != "debate":
                raise ValueError("debate_rounds_per_group requires rollout.mode='debate'")
            if self.rollout.group_size < 2 or self.rollout.group_size % 2 != 0:
                raise ValueError("Debate requires an even rollout.group_size of at least 2")
            expected_depths = self.rollout.group_size // 2
            if len(self.debate_rounds_per_group) != expected_depths:
                raise ValueError(
                    "debate_rounds_per_group must contain exactly one depth per debate "
                    f"in a group ({expected_depths} entries for group_size={self.rollout.group_size})"
                )
            if any(
                not isinstance(depth, int) or isinstance(depth, bool) or depth < 1
                for depth in self.debate_rounds_per_group
            ):
                raise ValueError("debate_rounds_per_group depths must all be positive integers")
        min_rounds = self.effective_debate_min_rounds()
        if uses_configured_harness and min_rounds < judge_harness.required_rounds:
            raise ValueError(
                f"Judge harness {judge_harness.harness_id!r} requires at least "
                f"{judge_harness.required_rounds} rounds; configured minimum is {min_rounds}"
            )
        if (
            self.debate_external_judge_url is not None
            and judge_harness.harness_id != SOLUTION_R1_RATIONALE_V1
        ):
            raise ValueError(
                "External HTTP judge supports only "
                f"{SOLUTION_R1_RATIONALE_V1!r}; got {judge_harness.harness_id!r}"
            )
        if self.debate_judge_max_tokens < 0:
            raise ValueError("debate_judge_max_tokens must be non-negative")
        if self.debate_judge_constrain_single_token:
            if judge_harness.output_contract != "single_token_a_or_b":
                raise ValueError(
                    "debate_judge_constrain_single_token requires a single-token A/B harness"
                )
            if self.sampler_backend != "vllm" or self.debate_judge_server_url is not None:
                raise ValueError(
                    "debate_judge_constrain_single_token requires the in-process vLLM judge"
                )
        if self.debate_judge_score_mode not in ("hard_verdict", "order_sym_soft_logit"):
            raise ValueError("debate_judge_score_mode must be hard_verdict or order_sym_soft_logit")
        if self.judge_label_token_contract not in JUDGE_LABEL_TOKEN_CONTRACTS:
            raise ValueError(
                f"Unknown judge_label_token_contract={self.judge_label_token_contract!r}"
            )
        direct_js_judge = self.judge_training_objective == "supervised_label_ce_js"
        if direct_js_judge:
            if not self.train_judge:
                raise ValueError("supervised_label_ce_js requires train_judge=True")
            if self.debate_judge_score_mode != "order_sym_soft_logit":
                raise ValueError(
                    "supervised_label_ce_js requires debate_judge_score_mode="
                    "'order_sym_soft_logit'"
                )
            if not self.debate_judge_bidirectional:
                raise ValueError("supervised_label_ce_js requires bidirectional judge sampling")
            if not self.debate_judge_constrain_single_token:
                raise ValueError("supervised_label_ce_js requires constrained single-token judging")
            if self.judge_label_token_contract != LFM25_OPENBOOKQA_SPACED_AB_V1:
                raise ValueError(
                    "supervised_label_ce_js requires the strict two-token OpenBookQA contract"
                )
            if self.debate_judge_harness != CONSTITUTION_SINGLE_TOKEN_V1:
                raise ValueError(
                    "supervised_label_ce_js requires constitution_single_token_v1"
                )
            if self.debate_r23_reward != "soft_judge":
                raise ValueError(
                    "supervised_label_ce_js requires reliability-weighted soft_judge rewards"
                )
            if self.train_minibatch_size > 0 and self.train_minibatch_size % 2 != 0:
                raise ValueError(
                    "supervised_label_ce_js requires train_minibatch_size=0 or an even value"
                )
            active_round_adapters = self.resolved_debate_round_adapter_names()
            if "judge" in active_round_adapters:
                raise ValueError(
                    "supervised_label_ce_js reserves the judge adapter for direct judge rows; "
                    "active debate_round_adapter_names must not contain 'judge'"
                )
        uses_soft_score = self.debate_judge_score_mode == "order_sym_soft_logit"
        uses_soft_reward = (
            self.debate_r1_reward == "judge_soft_task_gap"
            or self.debate_r23_reward == "soft_judge"
        )
        if uses_soft_score:
            if self.judge_label_token_contract not in (
                LFM25_AB_WHITESPACE_COMPAT_V1,
                LFM25_OPENBOOKQA_SPACED_AB_V1,
            ):
                raise ValueError(
                    "order_sym_soft_logit requires an explicit tokenizer-bound A/B token contract"
                )
            if not self.debate_judge_bidirectional:
                raise ValueError("order_sym_soft_logit requires bidirectional judge sampling")
            if not self.debate_judge_constrain_single_token:
                raise ValueError("order_sym_soft_logit requires constrained single-token judging")
            if self.train_judge:
                if self.judge_label_token_contract != LFM25_OPENBOOKQA_SPACED_AB_V1:
                    raise ValueError(
                        "trainable soft judge requires the strict two-token OpenBookQA contract"
                    )
                if not direct_js_judge:
                    raise ValueError(
                        "trainable soft judge requires judge_training_objective="
                        "'supervised_label_ce_js'"
                    )
                if float(self.debate_judge_temperature) <= 0.0:
                    raise ValueError("trainable soft judge requires stochastic temperature > 0")
                if self.debate_judge_harness != CONSTITUTION_SINGLE_TOKEN_V1:
                    raise ValueError(
                        "strict OpenBookQA token boundary is bound to constitution_single_token_v1"
                    )
                if self.debate_r23_reward != "soft_judge":
                    raise ValueError(
                        "trainable soft judge requires reliability-weighted soft_judge debate rewards"
                    )
            elif float(self.debate_judge_temperature) != 0.0:
                raise ValueError("frozen order_sym_soft_logit requires debate_judge_temperature=0")
            if self.debate_judge_max_tokens not in (0, 1) or (
                self.debate_judge_max_tokens == 0 and judge_harness.default_max_tokens != 1
            ):
                raise ValueError("order_sym_soft_logit requires exactly one judge output token")
            if self.adapter_layout != "split":
                raise ValueError("order_sym_soft_logit currently requires adapter_layout='split'")
        elif self.judge_label_token_contract != JUDGE_LABEL_TOKEN_CONTRACT_NONE:
            raise ValueError(
                "judge_label_token_contract is temporary and may only be enabled with "
                "debate_judge_score_mode='order_sym_soft_logit'"
            )
        if uses_soft_reward and not uses_soft_score:
            raise ValueError("soft judge rewards require debate_judge_score_mode='order_sym_soft_logit'")
        if self.debate_r1_reward == "judge_soft_task_gap" and self.rollout.mode != "debate":
            raise ValueError("judge_soft_task_gap is only valid for debate rollouts")
        if self.debate_r23_reward == "soft_judge":
            if self.effective_debate_max_rounds() < 2:
                raise ValueError("soft_judge R2/R3 reward requires at least two debate rounds")
            if self.debate_r23_mode != "symmetric":
                raise ValueError("soft_judge is intrinsically symmetric/zero-sum")
        BehaviorPolicySpec(
            temperature=self.debate_judge_temperature,
            top_p=self.debate_judge_top_p,
            top_k=self.debate_judge_top_k,
            min_p=self.debate_judge_min_p,
            presence_penalty=self.debate_judge_presence_penalty,
            repetition_penalty=self.debate_judge_repetition_penalty,
        )
        if (
            judge_modes == 0
            and self.debate_judge_adapter == "judge"
            and judge_harness.serialization == "raw_base"
        ):
            if self.sampler_backend != "vllm":
                raise ValueError("Raw-Base judge harnesses currently require sampler_backend='vllm'")
        if self.rollout.env_name == "mmlu_pro_pairwise" and not self.mmlu_pro_data_path:
            raise ValueError("mmlu_pro_pairwise requires mmlu_pro_data_path")
        self.behavior_policy().assert_exact_trainer_reconstruction_supported()
        if not self.on_policy_logprob_check:
            raise ValueError(
                "PPO training requires the fail-closed on-policy logprob check; "
                "on_policy_logprob_check=False is not allowed."
            )

        if self.debate_stop_on_concluded:
            if self.rollout.mode != "debate":
                raise ValueError("debate_stop_on_concluded is only valid for debate rollouts")
            if self.effective_debate_max_rounds() < 2:
                raise ValueError("debate_stop_on_concluded requires at least two debate rounds")
            if self.debate_prompt_format != "qwen35_base_text_prefill":
                raise ValueError(
                    "debate_stop_on_concluded requires debate_prompt_format='qwen35_base_text_prefill'"
                )
            if self.sampler_backend != "sglang":
                raise ValueError("debate_stop_on_concluded requires sampler_backend='sglang'")

        if self.debate_r1_reward == "judge_rejection_task":
            if self.rollout.mode != "debate":
                raise ValueError("judge_rejection_task is only valid for debate rollouts")
            if self.adapter_layout != "split":
                raise ValueError("judge_rejection_task requires adapter_layout='split'")
            expected_round_adapters = ("solution",) + (
                "debate",
            ) * (self.effective_debate_max_rounds() - 1)
            configured_round_adapters = self.resolved_debate_round_adapter_names()
            if configured_round_adapters != expected_round_adapters:
                raise ValueError(
                    "judge_rejection_task requires round adapters "
                    f"{expected_round_adapters!r}, got {configured_round_adapters!r}"
                )
        if self.debate_r1_reward == "judge_delta_task":
            if self.rollout.mode != "debate":
                raise ValueError("judge_delta_task is only valid for debate rollouts")
            if self.adapter_layout != "split":
                raise ValueError("judge_delta_task requires adapter_layout='split'")
            if not math.isfinite(self.debate_r1_judge_delta_q) or self.debate_r1_judge_delta_q < 0:
                raise ValueError("debate_r1_judge_delta_q must be finite and non-negative")
            if not math.isfinite(self.debate_incoherent_r23_reward):
                raise ValueError("debate_incoherent_r23_reward must be finite")
            if not math.isfinite(self.debate_r23_format_failure_penalty):
                raise ValueError("debate_r23_format_failure_penalty must be finite")
            if self.debate_r23_format_failure_penalty > 0.0:
                raise ValueError("debate_r23_format_failure_penalty must be non-positive")
            if not self.debate_judge_bidirectional:
                raise ValueError("judge_delta_task requires bidirectional judge sampling")
        if self.debate_judge_bidirectional:
            if judge_modes != 0:
                raise ValueError("bidirectional judge sampling requires the in-process judge sampler")
            if self.rollout.mode != "debate" or self.effective_debate_max_rounds() < 1:
                raise ValueError(
                    "bidirectional judge sampling requires at least one debate round"
                )
        if (
            not math.isfinite(self.judge_coherence_js_weight)
            or self.judge_coherence_js_weight < 0.0
        ):
            raise ValueError("judge_coherence_js_weight must be finite and non-negative")
        if self.train_judge:
            if self.judge_training_objective not in ("grpo", "supervised_label_ce_js"):
                raise ValueError(
                    "judge_training_objective must be grpo or supervised_label_ce_js"
                )
            if (
                self.judge_training_objective == "grpo"
                and self.judge_grpo_reward_mode not in ("coherence", "label")
            ):
                raise ValueError("judge_grpo_reward_mode must be coherence or label")
            if not self.debate_judge_bidirectional:
                raise ValueError("judge training requires bidirectional judge sampling")
            if self.debate_judge_adapter != "judge":
                raise ValueError("judge training requires debate_judge_adapter='judge'")
            if self.adapter_layout != "split":
                raise ValueError("judge training requires adapter_layout='split'")
            if self.train_adapter_names and "judge" not in self.train_adapter_names:
                raise ValueError("judge training requires judge in train_adapter_names")
            judge_behavior_policy = BehaviorPolicySpec(
                temperature=self.debate_judge_temperature,
                top_p=self.debate_judge_top_p,
                top_k=self.debate_judge_top_k,
                min_p=self.debate_judge_min_p,
                presence_penalty=self.debate_judge_presence_penalty,
                repetition_penalty=self.debate_judge_repetition_penalty,
            )
            judge_behavior_policy.assert_exact_trainer_reconstruction_supported()
            if judge_behavior_policy != self.behavior_policy():
                raise ValueError(
                    "judge training requires the judge and policy rollout behavior "
                    "distributions to match until trainer behavior-policy configuration "
                    "is adapter-specific"
                )

    @classmethod
    def from_dict(cls, data: dict) -> "TrainRunConfig":
        rollout_data = data["rollout"]
        train_judge = bool(
            data.get("train_judge", data.get("train_judge_coherence_grpo", False))
        )
        judge_training_objective = str(data.get("judge_training_objective", "grpo"))
        judge_grpo_reward_mode = str(data.get("judge_grpo_reward_mode", "coherence"))
        if judge_grpo_reward_mode == "label_js":
            # Atomic migration from the previously documented scalar-reward
            # contract. The obsolete GRPO mode is removed even when judge
            # training is disabled; when enabled, it selects the replacement
            # direct CE+JS objective in the same conversion.
            judge_grpo_reward_mode = "coherence"
            if train_judge:
                judge_training_objective = "supervised_label_ce_js"
        return cls(
            model_path=data["model_path"],
            output_dir=data["output_dir"],
            tokenizer_path=data.get("tokenizer_path"),
            rollout=RolloutConfig(
                env_name=rollout_data.get("env_name", "ht_sequence"),
                mode=rollout_data.get("mode", "debate"),
                num_samples=rollout_data.get("num_samples", 16),
                num_rollouts_per_instance=rollout_data.get("num_rollouts_per_instance", 1),
                num_groups=rollout_data.get("num_groups", 2),
                group_size=rollout_data.get("group_size", 8),
                rollout_batch_size=rollout_data.get("rollout_batch_size", 0),
                max_tokens=rollout_data.get("max_tokens", 1024),
                temperature=rollout_data.get("temperature", 1.0),
                top_p=rollout_data.get("top_p", 1.0),
                min_p=rollout_data.get("min_p", 0.0),
                seed=rollout_data.get("seed"),
                request_seed_mode=rollout_data.get("request_seed_mode", "none"),
            ),
            steps=data["steps"],
            learning_rate=data["learning_rate"],
            weight_decay=data.get("weight_decay", 0.01),
            max_grad_norm=data.get("max_grad_norm", 1.0),
            adapter_layout=data["adapter_layout"],
            sequence_len=data["sequence_len"],
            reward_mode=data["reward_mode"],
            strict_ht_format=data.get("strict_ht_format", False),
            constrained_writing_rules_per_speaker=int(data.get("constrained_writing_rules_per_speaker", 2)),
            constrained_writing_reward_scope=data.get("constrained_writing_reward_scope", "both"),
            constrained_writing_sides=data.get("constrained_writing_sides", "both"),
            constrained_writing_rule_family=data.get("constrained_writing_rule_family", "generic"),
            constrained_writing_reward_mode=data.get("constrained_writing_reward_mode", "additive"),
            constrained_writing_letter_temperature=float(data.get("constrained_writing_letter_temperature", 1.0)),
            constrained_writing_anchors=data.get("constrained_writing_anchors", "on"),
            quality_data_dir=data.get("quality_data_dir"),
            quality_split=data.get("quality_split", "train"),
            quality_hard_only=data.get("quality_hard_only", True),
            quality_source=data.get("quality_source", "Gutenberg"),
            quality_topic_contains=data.get("quality_topic_contains", "Science fiction"),
            quality_download=data.get("quality_download", False),
            mmlu_pro_data_path=data.get("mmlu_pro_data_path"),
            thinking_mode=data.get("thinking_mode", "default"),
            advantage_mode=data.get("advantage_mode", "zscore"),
            ppo_clip_epsilon=data.get("ppo_clip_epsilon", 0.2),
            debate_rounds=data.get("debate_rounds", 3),
            debate_rounds_per_group=tuple(data.get("debate_rounds_per_group", ())),
            debate_r1_reward=data.get("debate_r1_reward", "task"),
            debate_r23_reward=data.get("debate_r23_reward", "constant"),
            debate_r23_constant=data.get("debate_r23_constant", 1.0),
            debate_r1_judge_delta_q=data.get("debate_r1_judge_delta_q", 1.0),
            debate_incoherent_r23_reward=data.get("debate_incoherent_r23_reward", -0.5),
            debate_r23_format_failure_penalty=data.get("debate_r23_format_failure_penalty", 0.0),
            debate_r23_mode=data.get("debate_r23_mode", "symmetric"),
            debate_r23_advantage_scope=data.get("debate_r23_advantage_scope", "per_round"),
            debate_judge_adapter=data.get("debate_judge_adapter", "policy"),
            debate_external_judge_url=data.get("debate_external_judge_url"),
            debate_judge_server_url=data.get("debate_judge_server_url"),
            debate_judge_server_adapter_path=data.get("debate_judge_server_adapter_path"),
            debate_mock_judge_seed=data.get("debate_mock_judge_seed"),
            debate_external_judge_timeout_s=data.get("debate_external_judge_timeout_s", 600.0),
            debate_judge_harness=resolve_judge_harness_id(
                harness_id=data.get("debate_judge_harness"),
                legacy_prompt_format=data.get("debate_judge_prompt_format"),
                num_rounds=max(
                    data.get("debate_rounds_per_group", ()),
                    default=int(data.get("debate_rounds", 3)),
                ),
            ),
            debate_judge_max_tokens=data.get("debate_judge_max_tokens", 0),
            debate_judge_temperature=data.get("debate_judge_temperature", 1.0),
            debate_judge_top_p=data.get("debate_judge_top_p", 1.0),
            debate_judge_top_k=data.get("debate_judge_top_k", -1),
            debate_judge_min_p=data.get("debate_judge_min_p", 0.0),
            debate_judge_presence_penalty=data.get("debate_judge_presence_penalty", 0.0),
            debate_judge_repetition_penalty=data.get("debate_judge_repetition_penalty", 1.0),
            debate_judge_seed=data.get("debate_judge_seed"),
            debate_judge_bidirectional=bool(data.get("debate_judge_bidirectional", False)),
            debate_judge_constrain_single_token=bool(
                data.get("debate_judge_constrain_single_token", False)
            ),
            debate_judge_score_mode=str(data.get("debate_judge_score_mode", "hard_verdict")),
            judge_label_token_contract=str(
                data.get("judge_label_token_contract", JUDGE_LABEL_TOKEN_CONTRACT_NONE)
            ),
            train_judge=train_judge,
            judge_grpo_reward_mode=judge_grpo_reward_mode,
            judge_training_objective=judge_training_objective,
            judge_coherence_js_weight=float(data.get("judge_coherence_js_weight", 1.0)),
            debate_round_adapter_names=tuple(data.get("debate_round_adapter_names", ("solution", "debate", "debate"))),
            debate_prompt_format=data.get("debate_prompt_format", "chat"),
            debate_stop_on_concluded=data.get("debate_stop_on_concluded", False),
            base_r2_prefill=data.get("base_r2_prefill", "The reasons that my solution is better than my opponent's are:\n1)"),
            base_r3_prefill=data.get("base_r3_prefill", "Responding to my opponent's criticism:\n1)"),
            debate_r1_max_tokens=data.get("debate_r1_max_tokens", 0),
            debate_r23_max_tokens=data.get("debate_r23_max_tokens", 0),
            debate_r2_max_tokens=data.get("debate_r2_max_tokens", 0),
            debate_r3_max_tokens=data.get("debate_r3_max_tokens", 0),
            rollout_grad_accum_steps=data.get("rollout_grad_accum_steps", 1),
            rollout_assistant_prefill=(
                data["rollout_assistant_prefill"] if "rollout_assistant_prefill" in data else ""
            ),
            train_minibatch_size=data.get("train_minibatch_size", 0),
            train_max_tokens=data.get("train_max_tokens", 0),
            train_length_bucket_batches=data.get("train_length_bucket_batches", False),
            train_logprob_backend=data.get("train_logprob_backend", "full_logits"),
            compile_train_logprob_helper=data.get("compile_train_logprob_helper", False),
            train_adapter_names=tuple(data.get("train_adapter_names", ())),
            stop_parsed_reward_hacking_min=data.get("stop_parsed_reward_hacking_min"),
            stop_parsed_reward_hacking_max=data.get("stop_parsed_reward_hacking_max"),
            gradient_checkpointing=data.get("gradient_checkpointing", True),
            on_policy_logprob_check=data.get("on_policy_logprob_check", True),
            on_policy_logprob_warn_only=data.get("on_policy_logprob_warn_only", False),
            on_policy_logprob_abs_tol=data.get("on_policy_logprob_abs_tol", 1e-3),
            on_policy_logprob_warning_path=data.get("on_policy_logprob_warning_path"),
            on_policy_logprob_max_records_per_batch=data.get("on_policy_logprob_max_records_per_batch", 8),
            sampler_gpu_memory_utilization=data.get("sampler_gpu_memory_utilization", 0.55),
            sampler_max_model_len=data.get("sampler_max_model_len", 512),
            sampler_max_num_seqs=data.get("sampler_max_num_seqs", 0),
            sampler_enforce_eager=data.get("sampler_enforce_eager", True),
            sampler_max_lora_rank=data.get("sampler_max_lora_rank", 32),
            sampler_max_loras=data.get("sampler_max_loras", 4),
            sampler_teardown_before_training=data.get("sampler_teardown_before_training", False),
            sampler_sleep_before_training=data.get("sampler_sleep_before_training", False),
            sampler_sleep_level=data.get("sampler_sleep_level", 1),
            sampler_backend=data.get("sampler_backend", "vllm"),
            sampler_sglang_base_url=data.get("sampler_sglang_base_url", "http://127.0.0.1:30000"),
            sampler_sglang_timeout_s=data.get("sampler_sglang_timeout_s", 600.0),
            sampler_sglang_pin_loras=data.get("sampler_sglang_pin_loras", False),
            sampler_sglang_unload_stale_adapters=data.get("sampler_sglang_unload_stale_adapters", True),
            init_adapter_dirs=data.get("init_adapter_dirs"),
            target_modules=tuple(data["target_modules"]),
            target_parameters=tuple(data.get("target_parameters", ())),
            lora_rank=data["lora_rank"],
            trace_model_io=data.get("trace_model_io", True),
            trace_model_io_dir=data.get("trace_model_io_dir"),
            trace_top_logprobs=data.get("trace_top_logprobs", 5),
            resource_logging=data.get("resource_logging", True),
            resource_log_interval_s=data.get("resource_log_interval_s", 5.0),
            wandb_enabled=data.get("wandb_enabled", True),
            wandb_project=data.get("wandb_project", "llm-local-rl"),
            wandb_entity=data.get("wandb_entity"),
            wandb_group=data.get("wandb_group"),
            wandb_run_name=data.get("wandb_run_name"),
            wandb_mode=data.get("wandb_mode", "online"),
            wandb_upload_artifacts=data.get("wandb_upload_artifacts", True),
            adapter_checkpoint_every=data.get("adapter_checkpoint_every", 10),
            optimizer_checkpoint_every=data.get("optimizer_checkpoint_every", 50),
            rollout_shard_every=data.get("rollout_shard_every", 10),
            wandb_table_samples_per_shard=data.get("wandb_table_samples_per_shard", 32),
            reference_kl_every=data.get("reference_kl_every", 10),
        )

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(temporary, path)


@dataclass(frozen=True)
class CheckpointManifest:
    run_config: dict
    current_step: int
    adapter_dirs: dict[str, str]
    step_records_path: str
    exact_resume_checkpoint: str | None = None
    reference_adapter_dirs: dict[str, str] | None = None

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2))
        os.replace(temporary, path)

    @classmethod
    def read_json(cls, path: str | Path) -> "CheckpointManifest":
        return cls(**json.loads(Path(path).read_text()))
