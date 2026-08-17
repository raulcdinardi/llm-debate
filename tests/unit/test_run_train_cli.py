from __future__ import annotations

from pathlib import Path
import shlex

import pytest

from scripts.run_train import _parse_csv_tuple, _parse_init_adapter_dirs, parse_args
from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.judge_harness import SOLUTION_R1_RATIONALE_V1


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
    assert args.temperature == 1.0
    assert args.debate_judge_temperature == 1.0
    assert args.debate_r1_reward == "task"
    assert args.debate_r1_judge_delta_q == 1.0
    assert args.debate_incoherent_r23_reward == -0.5
    assert args.debate_judge_bidirectional is False
    assert args.judge_grpo_reward_mode == "coherence"


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
        train_judge_coherence_grpo=args.train_judge_coherence_grpo,
        judge_grpo_reward_mode=args.judge_grpo_reward_mode,
        train_adapter_names=tuple(args.train_adapter_names),
    )

    assert config.adapter_layout == "split"
    assert config.debate_judge_adapter == "judge"
    assert config.debate_judge_harness == SOLUTION_R1_RATIONALE_V1
    assert config.debate_judge_bidirectional is True
    assert config.train_judge_coherence_grpo is True
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
