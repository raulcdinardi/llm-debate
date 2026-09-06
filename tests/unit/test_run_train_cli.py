from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shlex

import pytest

from scripts.run_train import (
    _parse_csv_tuple,
    _parse_init_adapter_dirs,
    _resolve_debate_depth_policy_args,
    parse_args,
)
from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.judge_harness import (
    CHAT_SOLUTION_TAGGED_V1,
    CONSTITUTION_SINGLE_TOKEN_V1,
    CONSULTANCY_SINGLE_TOKEN_V1,
    SOLUTION_R1_RATIONALE_V1,
    resolve_judge_harness_id,
)


def test_parse_named_init_adapter_dirs() -> None:
    assert _parse_init_adapter_dirs(
        [
            "solution=/adapters/countdown",
            "debate=/adapters/debate",
            "judge=/adapters/judge",
        ]
    ) == {
        "solution": "/adapters/countdown",
        "debate": "/adapters/debate",
        "judge": "/adapters/judge",
    }


def test_parse_named_init_adapter_dirs_rejects_missing_name() -> None:
    with pytest.raises(ValueError, match="NAME=PATH"):
        _parse_init_adapter_dirs(["/adapters/countdown"])


def test_parse_csv_tuple_strips_empty_items() -> None:
    assert _parse_csv_tuple("q_proj, v_proj,,") == ("q_proj", "v_proj")


def test_cli_accepts_judge_rejection_task_and_independent_concluded_stop() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--debate-r1-reward",
            "judge_rejection_task",
            "--debate-prompt-format",
            "qwen35_base_text_prefill",
            "--debate-stop-on-concluded",
        ]
    )

    assert args.debate_r1_reward == "judge_rejection_task"
    assert args.debate_prompt_format == "qwen35_base_text_prefill"
    assert args.debate_stop_on_concluded is True


def test_cli_defaults_all_sampling_temperatures_to_one() -> None:
    args = parse_args(["--model-path", "/tmp/model", "--output-dir", "/tmp/out"])

    assert args.adapter_layout == "shared"
    assert args.num_groups == 8
    assert args.group_size == 16
    assert args.temperature == 1.0
    assert args.debate_judge_temperature == 1.0
    assert args.debate_r1_reward == "task"
    assert args.debate_r1_judge_delta_q == 1.0
    assert args.debate_incoherent_r23_reward == -0.5
    assert args.debate_judge_bidirectional is False
    assert args.debate_judge_constrain_single_token is False
    assert args.judge_grpo_reward_mode == "coherence"
    assert args.debate_rounds_per_group == []
    assert args.debate_depth_policy == "fixed"
    assert args.debate_depth_policy_params_json == "{}"


def test_cli_accepts_arbitrary_heterogeneous_debate_depth() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--debate-rounds",
            "7",
            "--debate-rounds-per-group",
            "3",
            "7",
            "4",
            "6",
        ]
    )

    assert args.debate_rounds_per_group == [3, 7, 4, 6]
    assert _resolve_debate_depth_policy_args(args) == (
        "shuffled_multiset",
        {"depths": [3, 7, 4, 6]},
    )

    assert resolve_judge_harness_id(harness_id=None, num_rounds=7) == CHAT_SOLUTION_TAGGED_V1
    assert args.debate_round_adapter_names == ["solution", "debate", "debate"]


def test_config_repeats_last_adapter_for_arbitrary_round_depth() -> None:
    config = TrainRunConfig(
        model_path="/tmp/model",
        output_dir="/tmp/out",
        rollout=RolloutConfig(num_groups=2, group_size=8),
        debate_rounds=7,
        debate_depth_policy="shuffled_multiset",
        debate_depth_policy_params={"depths": [3, 7, 4, 6]},
    )

    assert config.effective_debate_min_rounds() == 3
    assert config.effective_debate_max_rounds() == 7
    restored = TrainRunConfig.from_dict(config.to_dict())
    assert restored.debate_depth_policy == "shuffled_multiset"
    assert restored.debate_depth_policy_params == {"depths": [3, 7, 4, 6]}
    assert config.resolved_debate_round_adapter_names() == (
        "solution",
        "debate",
        "debate",
        "debate",
        "debate",
        "debate",
        "debate",
    )


def test_config_rejects_depth_array_that_does_not_match_debates_per_group() -> None:
    with pytest.raises(ValueError, match="exactly one value per debate"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            debate_rounds=6,
            debate_depth_policy="shuffled_multiset",
            debate_depth_policy_params={"depths": [3, 4, 5]},
        )


def test_config_rejects_nonpositive_depth_in_group_array() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            debate_rounds=6,
            debate_depth_policy="shuffled_multiset",
            debate_depth_policy_params={"depths": [3, 4, 0, 6]},
        )


def test_cli_resolves_registered_policy_name_and_json_params() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--debate-depth-policy",
            "categorical",
            "--debate-depth-policy-params-json",
            '{"depths":[3,4],"weights":[1,2]}',
        ]
    )

    assert _resolve_debate_depth_policy_args(args) == (
        "categorical",
        {"depths": [3, 4], "weights": [1, 2]},
    )


def test_cli_rejects_conflicting_depth_policy_surfaces() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--debate-depth-policy",
            "categorical",
            "--debate-rounds-per-group",
            "3",
            "3",
            "4",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_debate_depth_policy_args(args)


def test_previous_depth_array_config_migrates_to_registered_policy() -> None:
    payload = TrainRunConfig(
        model_path="/tmp/model",
        output_dir="/tmp/out",
    ).to_dict()
    payload.pop("debate_depth_policy")
    payload.pop("debate_depth_policy_params")
    payload["debate_rounds"] = 3
    payload["rollout"]["group_size"] = 8
    payload["debate_rounds_per_group"] = [3, 7, 4, 6]

    restored = TrainRunConfig.from_dict(payload)

    assert restored.debate_rounds == 7
    assert restored.debate_depth_policy == "shuffled_multiset"
    assert restored.debate_depth_policy_params == {"depths": [3, 7, 4, 6]}


def test_rollout_config_defaults_to_apples_to_apples_8x16_debate_geometry() -> None:
    rollout = RolloutConfig()

    assert rollout.num_groups == 8
    assert rollout.group_size == 16


def test_cli_accepts_prompt_group_soft_judge_grpo() -> None:
    args = parse_args(
        [
            "--model-path", "/tmp/model",
            "--output-dir", "/tmp/out",
            "--debate-r23-reward", "soft_judge_prompt_grpo",
        ]
    )

    assert args.debate_r23_reward == "soft_judge_prompt_grpo"


def test_cli_accepts_frozen_single_token_judge_constraint() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--debate-judge-constrain-single-token",
        ]
    )
    assert args.debate_judge_constrain_single_token is True


def test_cli_accepts_temporary_soft_judge_contract_explicitly() -> None:
    args = parse_args([
        "--model-path", "/tmp/model",
        "--output-dir", "/tmp/out",
        "--debate-r1-reward", "judge_soft_task_gap",
        "--debate-r23-reward", "soft_judge",
        "--debate-judge-score-mode", "order_sym_soft_logit",
        "--judge-label-token-contract", "lfm25_ab_whitespace_compat_v1",
    ])
    assert args.debate_r1_reward == "judge_soft_task_gap"
    assert args.debate_r23_reward == "soft_judge"
    assert args.debate_judge_score_mode == "order_sym_soft_logit"
    assert args.judge_label_token_contract == "lfm25_ab_whitespace_compat_v1"


def test_soft_judge_config_is_opt_in_and_fail_closed() -> None:
    common = dict(
        model_path="/model",
        output_dir="/out",
        rollout=RolloutConfig(mode="debate"),
        adapter_layout="split",
        debate_rounds=2,
        debate_round_adapter_names=("solution", "debate", "debate"),
        debate_r1_reward="judge_soft_task_gap",
        debate_r23_reward="soft_judge",
        debate_judge_harness=CONSULTANCY_SINGLE_TOKEN_V1,
        debate_judge_temperature=0.0,
        debate_judge_max_tokens=1,
        debate_judge_bidirectional=True,
        debate_judge_constrain_single_token=True,
        debate_judge_score_mode="order_sym_soft_logit",
    )
    with pytest.raises(ValueError, match="explicit tokenizer-bound"):
        TrainRunConfig(**common)
    config = TrainRunConfig(
        **common,
        judge_label_token_contract="lfm25_ab_whitespace_compat_v1",
    )
    assert config.to_dict()["judge_label_token_contract"] == "lfm25_ab_whitespace_compat_v1"
    assert config.to_dict()["judge_label_token_contract_temporary"] is True
    restored = TrainRunConfig.from_dict(config.to_dict())
    assert restored.debate_judge_score_mode == "order_sym_soft_logit"


def test_prompt_group_soft_judge_grpo_requires_multiple_debates_and_frozen_judge() -> None:
    config = TrainRunConfig(
        model_path="/model",
        output_dir="/out",
        rollout=RolloutConfig(
            mode="debate",
            env_name="mmlu_pro_pairwise",
            num_groups=8,
            group_size=16,
        ),
        mmlu_pro_data_path="/corpus/mmlu.jsonl",
        adapter_layout="split",
        debate_rounds=3,
        debate_round_adapter_names=("solution", "debate", "debate"),
        debate_r1_reward="none",
        debate_r23_reward="soft_judge_prompt_grpo",
        debate_r23_mode="symmetric",
        debate_judge_adapter="judge",
        debate_judge_harness=CONSTITUTION_SINGLE_TOKEN_V1,
        debate_judge_temperature=0.0,
        debate_judge_max_tokens=1,
        debate_judge_bidirectional=True,
        debate_judge_constrain_single_token=True,
        debate_judge_score_mode="order_sym_soft_logit",
        judge_label_token_contract="lfm25_ab_whitespace_compat_v1",
        train_judge=False,
        train_adapter_names=("debate",),
    )

    assert config.rollout.num_groups == 8
    assert config.rollout.group_size == 16
    with pytest.raises(ValueError, match="at least two debates per prompt"):
        replace(config, rollout=replace(config.rollout, group_size=2))


def test_cli_accepts_label_judge_grpo_reward_mode() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--judge-grpo-reward-mode",
            "label",
        ]
    )
    assert args.judge_grpo_reward_mode == "label"


def test_cli_accepts_direct_js_judge_objective_and_legacy_enable_alias() -> None:
    direct = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--train-judge",
            "--judge-training-objective",
            "supervised_label_ce_js",
            "--judge-coherence-js-weight",
            "0.75",
        ]
    )
    legacy = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--train-judge-coherence-grpo",
            "--judge-grpo-reward-mode",
            "label_js",
        ]
    )
    assert direct.train_judge is True
    assert direct.judge_training_objective == "supervised_label_ce_js"
    assert direct.judge_coherence_js_weight == pytest.approx(0.75)
    assert legacy.train_judge is True
    assert legacy.judge_training_objective == "supervised_label_ce_js"
    assert legacy.judge_grpo_reward_mode == "coherence"


def test_openbookqa_supervised_label_ce_js_contract_is_trainable_and_granular() -> None:
    config = TrainRunConfig(
        model_path="/model",
        output_dir="/out",
        rollout=RolloutConfig(mode="debate", temperature=1.0),
        adapter_layout="split",
        debate_rounds=3,
        debate_round_adapter_names=("solution", "debate", "debate"),
        debate_r1_reward="none",
        debate_r23_reward="soft_judge",
        debate_r23_mode="symmetric",
        debate_judge_adapter="judge",
        debate_judge_harness=CONSTITUTION_SINGLE_TOKEN_V1,
        debate_judge_temperature=1.0,
        debate_judge_max_tokens=1,
        debate_judge_bidirectional=True,
        debate_judge_constrain_single_token=True,
        debate_judge_score_mode="order_sym_soft_logit",
        judge_label_token_contract="lfm25_openbookqa_spaced_ab_v1",
        train_judge=True,
        judge_training_objective="supervised_label_ce_js",
        judge_coherence_js_weight=0.75,
        train_adapter_names=("debate", "judge"),
    )
    assert config.judge_training_objective == "supervised_label_ce_js"
    assert config.judge_coherence_js_weight == pytest.approx(0.75)

    ce_only = replace(config, judge_coherence_js_weight=0.0, train_minibatch_size=32)
    restored = TrainRunConfig.from_dict(ce_only.to_dict())
    assert restored == ce_only
    assert restored.judge_coherence_js_weight == 0.0
    assert restored.debate_r23_reward == "soft_judge"
    assert restored.train_adapter_names == ("debate", "judge")
    with pytest.raises(ValueError, match="even"):
        replace(ce_only, train_minibatch_size=3)

    with pytest.raises(ValueError, match="order_sym_soft_logit"):
        replace(
            config,
            debate_judge_score_mode="hard_verdict",
            judge_label_token_contract="none",
        )
    with pytest.raises(ValueError, match="even"):
        replace(config, train_minibatch_size=3)
    with pytest.raises(ValueError, match="must not contain 'judge'"):
        replace(
            config,
            debate_round_adapter_names=("solution", "judge", "debate"),
        )

    legacy_data = config.to_dict()
    legacy_data.pop("train_judge")
    legacy_data.pop("judge_training_objective")
    legacy_data.pop("judge_coherence_js_weight")
    legacy_data["train_judge_coherence_grpo"] = True
    legacy_data["judge_grpo_reward_mode"] = "label_js"
    migrated = TrainRunConfig.from_dict(legacy_data)
    assert migrated.train_judge is True
    assert migrated.judge_training_objective == "supervised_label_ce_js"
    assert migrated.judge_grpo_reward_mode == "coherence"


def test_unlabeled_js_direct_objective_uses_same_strict_pair_contract() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--train-judge",
            "--judge-training-objective",
            "unsupervised_js",
            "--judge-coherence-js-weight",
            "1.0",
        ]
    )
    assert args.judge_training_objective == "unsupervised_js"

    config = TrainRunConfig(
        model_path="/model",
        output_dir="/out",
        rollout=RolloutConfig(mode="debate", env_name="constrained_writing", temperature=1.0),
        adapter_layout="split",
        debate_rounds=3,
        debate_round_adapter_names=("solution", "debate", "debate"),
        debate_r1_reward="none",
        debate_r23_reward="soft_judge",
        debate_r23_mode="symmetric",
        debate_judge_adapter="judge",
        debate_judge_harness=CONSTITUTION_SINGLE_TOKEN_V1,
        debate_judge_temperature=1.0,
        debate_judge_max_tokens=1,
        debate_judge_bidirectional=True,
        debate_judge_constrain_single_token=True,
        debate_judge_score_mode="order_sym_soft_logit",
        judge_label_token_contract="lfm25_openbookqa_spaced_ab_v1",
        train_judge=True,
        judge_training_objective="unsupervised_js",
        judge_coherence_js_weight=1.0,
        train_adapter_names=("debate", "judge"),
    )
    assert config.judge_training_objective == "unsupervised_js"

    with pytest.raises(ValueError, match="weight > 0"):
        replace(config, judge_coherence_js_weight=0.0)


def test_documented_judge_grpo_flags_parse_and_pass_config_validation() -> None:
    docs = (
        Path(__file__).parents[2] / "docs" / "debate_judge_grpo.md"
    ).read_text(encoding="utf-8")
    section = docs.split("## Judge coherence GRPO", 1)[1]
    flags_block = section.split("```text", 1)[1].split("```", 1)[0]
    documented_flags = shlex.split(flags_block)
    args = parse_args(
        ["--model-path", "/model", "--output-dir", "/out", *documented_flags]
    )

    config = TrainRunConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        rollout=RolloutConfig(
            mode=args.mode,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
        ),
        adapter_layout=args.adapter_layout,
        debate_judge_adapter=args.debate_judge_adapter,
        debate_judge_harness=args.debate_judge_harness,
        debate_judge_temperature=args.debate_judge_temperature,
        debate_judge_top_p=args.debate_judge_top_p,
        debate_judge_top_k=args.debate_judge_top_k,
        debate_judge_min_p=args.debate_judge_min_p,
        debate_judge_presence_penalty=args.debate_judge_presence_penalty,
        debate_judge_repetition_penalty=args.debate_judge_repetition_penalty,
        debate_judge_bidirectional=args.debate_judge_bidirectional,
        train_judge=args.train_judge,
        judge_grpo_reward_mode=args.judge_grpo_reward_mode,
        train_adapter_names=tuple(args.train_adapter_names),
    )

    assert config.adapter_layout == "split"
    assert config.debate_judge_adapter == "judge"
    assert config.debate_judge_harness == SOLUTION_R1_RATIONALE_V1
    assert config.debate_judge_bidirectional is True
    assert config.train_judge is True
    assert config.judge_grpo_reward_mode == "coherence"
    assert config.rollout.temperature == config.debate_judge_temperature == 1.0
    assert config.train_adapter_names == ("solution", "debate", "judge")


def test_rollout_assistant_prefill_default_distinguishes_omitted_from_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_train.py",
            "--model-path",
            "/models/base",
            "--output-dir",
            "/runs/out",
        ],
    )

    args = parse_args()

    assert args.rollout_assistant_prefill is None


def test_rollout_assistant_prefill_accepts_explicit_empty_override(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_train.py",
            "--model-path",
            "/models/base",
            "--output-dir",
            "/runs/out",
            "--rollout-assistant-prefill",
            "",
        ],
    )

    args = parse_args()

    assert args.rollout_assistant_prefill == ""


def test_parse_constrained_writing_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_train.py",
            "--model-path",
            "/models/base",
            "--output-dir",
            "/runs/out",
            "--env",
            "constrained_writing",
        ],
    )

    args = parse_args()

    assert args.env == "constrained_writing"


def test_parse_sglang_sampler_backend_args(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_train.py",
            "--model-path",
            "/models/base",
            "--output-dir",
            "/runs/out",
            "--sampler-backend",
            "sglang",
            "--sampler-sglang-base-url",
            "http://127.0.0.1:30123",
            "--sampler-sglang-timeout-s",
            "33",
            "--weight-decay",
            "0.02",
            "--max-grad-norm",
            "0.75",
            "--debate-r23-advantage-scope",
            "merged_r23",
            "--debate-r23-format-failure-penalty",
            "-1",
            "--sampler-sglang-pin-loras",
            "--no-sampler-sglang-unload-stale-adapters",
            "--on-policy-logprob-check",
            "--on-policy-logprob-abs-tol",
            "0.0002",
            "--on-policy-logprob-warning-path",
            "/runs/out/on_policy.jsonl",
            "--on-policy-logprob-max-records-per-batch",
            "3",
            "--train-adapter-names",
            "debate",
            "--train-logprob-backend",
            "selective_lm_head",
            "--lora-rank",
            "16",
            "--compile-train-logprob-helper",
            "--stop-parsed-reward-hacking-min",
            "0.45",
            "--stop-parsed-reward-hacking-max",
            "0.55",
        ],
    )

    args = parse_args()

    assert args.sampler_backend == "sglang"
    assert args.sampler_sglang_base_url == "http://127.0.0.1:30123"
    assert args.sampler_sglang_timeout_s == 33.0
    assert args.weight_decay == 0.02
    assert args.max_grad_norm == 0.75
    assert args.debate_r23_advantage_scope == "merged_r23"
    assert args.debate_r23_format_failure_penalty == -1.0
    assert args.sampler_sglang_pin_loras is True
    assert args.sampler_sglang_unload_stale_adapters is False
    assert args.on_policy_logprob_check is True
    assert args.on_policy_logprob_abs_tol == 0.0002
    assert args.on_policy_logprob_warning_path == "/runs/out/on_policy.jsonl"
    assert args.on_policy_logprob_max_records_per_batch == 3
    assert args.train_adapter_names == ["debate"]
    assert args.train_logprob_backend == "selective_lm_head"
    assert args.lora_rank == 16
    assert args.compile_train_logprob_helper is True
    assert args.stop_parsed_reward_hacking_min == 0.45
    assert args.stop_parsed_reward_hacking_max == 0.55


def test_cli_does_not_expose_a_parity_gate_disable_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-path",
                "/tmp/model",
                "--output-dir",
                "/tmp/out",
                "--no-on-policy-logprob-check",
            ]
        )


def test_cli_parses_same_engine_base_sft_judge_contract() -> None:
    args = parse_args(
        [
            "--model-path",
            "/model",
            "--output-dir",
            "/out",
            "--mode",
            "debate",
            "--adapter-layout",
            "split",
            "--debate-judge-adapter",
            "judge",
            "--debate-judge-harness",
            SOLUTION_R1_RATIONALE_V1,
            "--debate-judge-max-tokens",
            "512",
            "--debate-judge-temperature",
            "1",
            "--debate-judge-top-p",
            ".95",
            "--debate-judge-top-k",
            "20",
            "--debate-judge-presence-penalty",
            "1.5",
            "--debate-judge-seed",
            "0",
        ]
    )

    assert args.debate_judge_adapter == "judge"
    assert args.debate_judge_harness == SOLUTION_R1_RATIONALE_V1
    assert args.debate_judge_max_tokens == 512
    assert args.debate_judge_temperature == 1.0
    assert args.debate_judge_top_p == 0.95
    assert args.debate_judge_top_k == 20
    assert args.debate_judge_presence_penalty == 1.5
    assert args.debate_judge_seed == 0
