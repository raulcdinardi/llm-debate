from __future__ import annotations

import pytest

from scripts.run_train import _parse_csv_tuple, _parse_init_adapter_dirs, parse_args


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
