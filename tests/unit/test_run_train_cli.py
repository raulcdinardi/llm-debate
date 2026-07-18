from __future__ import annotations

from scripts.run_train import parse_args


def test_cli_accepts_judge_rejection_task_r1_mode() -> None:
    args = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/out",
            "--debate-r1-reward",
            "judge_rejection_task",
        ]
    )

    assert args.debate_r1_reward == "judge_rejection_task"
