from __future__ import annotations

import pytest

from llm_local_rl.debate_parity import (
    DebateResult,
    DebateTrajectory,
    Transition,
    assemble_split_train_examples,
    assemble_training_data_by_mode,
    assemble_training_data_r1_r23,
)


def _make_debate(
    *,
    question: str = "Q",
    verdict: str = "A",
    reward_a: float = 1.0,
    reward_b: float = 0.0,
    token_offset: int = 0,
) -> DebateResult:
    traj_a = DebateTrajectory(
        agent="A",
        transitions=[
            Transition(prompt_tokens=[1, 2], completion_tokens=[3 + token_offset, 4 + token_offset], completion_logprobs=[-0.1, -0.2], round_num=1),
            Transition(prompt_tokens=[1, 2, 3, 4, 5], completion_tokens=[6 + token_offset, 7 + token_offset], completion_logprobs=[-0.3, -0.4], round_num=2),
            Transition(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], completion_tokens=[9 + token_offset, 10 + token_offset], completion_logprobs=[-0.5, -0.6], round_num=3),
        ],
        frozen_solution="A",
        metrics={"task_reward": reward_a},
    )
    traj_b = DebateTrajectory(
        agent="B",
        transitions=[
            Transition(prompt_tokens=[1, 2], completion_tokens=[11 + token_offset, 12 + token_offset], completion_logprobs=[-0.1, -0.2], round_num=1),
            Transition(prompt_tokens=[1, 2, 11, 12, 13], completion_tokens=[14 + token_offset, 15 + token_offset], completion_logprobs=[-0.3, -0.4], round_num=2),
            Transition(prompt_tokens=[1, 2, 11, 12, 13, 14, 15, 16], completion_tokens=[17 + token_offset, 18 + token_offset], completion_logprobs=[-0.5, -0.6], round_num=3),
        ],
        frozen_solution="B",
        metrics={"task_reward": reward_b},
    )
    return DebateResult(
        question=question,
        ground_truth=None,
        trajectory_a=traj_a,
        trajectory_b=traj_b,
        verdict=verdict,
        judge_reasoning="judge",
    )


def test_split_projection_preserves_source_round_advantages() -> None:
    debate = _make_debate()
    shared_data = assemble_training_data_r1_r23(
        [debate],
        r1_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
        r23_reward=0.5,
        r23_symmetric=True,
    )
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="task",
        r23_reward_mode="constant",
        r23_constant=0.5,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )
    assert len(split["solution"]) == 2
    assert len(split["debate"]) == 4
    first_shared = shared_data[0]
    first_solution = split["solution"][0]
    assert first_solution.advantages[-2:] == first_shared.completion_advantages[:2]
    first_debate_r2 = split["debate"][0]
    assert first_debate_r2.advantages[-2:] == first_shared.completion_advantages[3:5]
    first_debate_r3 = split["debate"][1]
    assert first_debate_r3.advantages[-2:] == first_shared.completion_advantages[6:8]
    assert first_solution.metadata["source_exact_shared_equivalent"] is False


def test_shared_training_data_supports_judge_compare_for_three_rounds() -> None:
    debate = _make_debate()
    data = assemble_training_data_by_mode(
        debates=[debate],
        num_rounds=3,
        r1_reward_mode="judge",
        r23_reward_mode="constant",
        r23_constant=0.5,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )
    assert len(data) == 2
    winner = data[0]
    loser = data[1]
    assert winner.completion_advantages[:2] == [0.25, 0.25]
    assert winner.completion_advantages[3:5] == [0.25, 0.25]
    assert winner.completion_advantages[6:8] == [0.25, 0.25]
    assert loser.completion_advantages[:2] == [-0.25, -0.25]
    assert loser.completion_advantages[3:5] == [-0.25, -0.25]
    assert loser.completion_advantages[6:8] == [-0.25, -0.25]


def test_split_training_data_supports_judge_compare_for_three_rounds() -> None:
    debate = _make_debate()
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="judge",
        r23_reward_mode="constant",
        r23_constant=0.5,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )
    assert len(split["solution"]) == 2
    assert len(split["debate"]) == 4
    assert split["solution"][0].advantages[-2:] == [0.25, 0.25]
    assert split["solution"][1].advantages[-2:] == [-0.25, -0.25]
    assert split["debate"][0].advantages[-2:] == [0.25, 0.25]
    assert split["debate"][1].advantages[-2:] == [0.25, 0.25]
    assert split["debate"][2].advantages[-2:] == [-0.25, -0.25]
    assert split["debate"][3].advantages[-2:] == [-0.25, -0.25]


def test_judge_rejection_task_selects_only_winners_and_zscores_their_task_rewards() -> None:
    debates = [
        _make_debate(verdict="A", reward_a=1.0, reward_b=1000.0),
        _make_debate(verdict="B", reward_a=-1000.0, reward_b=3.0, token_offset=100),
    ]

    split = assemble_split_train_examples(
        debates=debates,
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="judge_rejection_task",
        r23_reward_mode="constant",
        r23_constant=0.5,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )

    assert len(split["solution"]) == 2
    assert [example.metadata["agent"] for example in split["solution"]] == ["A", "B"]
    assert [example.metadata["r1_rejected_agent"] for example in split["solution"]] == ["B", "A"]
    assert [example.metadata["r1_task_reward"] for example in split["solution"]] == [1.0, 3.0]
    assert [example.metadata["r1_zscore"] for example in split["solution"]] == [-1.0, 1.0]
    assert split["solution"][0].advantages[-2:] == [-0.5, -0.5]
    assert split["solution"][1].advantages[-2:] == [0.5, 0.5]

    # R2/R3 remain symmetric and include both speakers from both debates.
    assert len(split["debate"]) == 8
    assert [example.advantages[-2:] for example in split["debate"]] == [
        [0.25, 0.25],
        [0.25, 0.25],
        [-0.25, -0.25],
        [-0.25, -0.25],
        [-0.25, -0.25],
        [-0.25, -0.25],
        [0.25, 0.25],
        [0.25, 0.25],
    ]


def test_judge_rejection_task_zeroes_equal_winner_rewards_and_drops_invalid_debates() -> None:
    debates = [
        _make_debate(verdict="A", reward_a=2.0),
        _make_debate(verdict="B", reward_b=2.0, token_offset=100),
        _make_debate(verdict="INVALID", reward_a=999.0, reward_b=999.0, token_offset=200),
    ]

    split = assemble_split_train_examples(
        debates=debates,
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="judge_rejection_task",
        r23_reward_mode="constant",
        r23_constant=0.5,
        r23_symmetric=False,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )

    assert len(split["solution"]) == 2
    assert all(example.metadata["r1_group_live"] is False for example in split["solution"])
    assert all(example.advantages[-2:] == [0.0, 0.0] for example in split["solution"])
    assert len(split["debate"]) == 8
    assert sum(example.metadata["r23_reward"] == 0.0 for example in split["debate"]) == 4


def test_judge_rejection_task_requires_distinct_r1_and_debate_adapters() -> None:
    with pytest.raises(ValueError, match="R1 adapter distinct"):
        assemble_split_train_examples(
            debates=[_make_debate()],
            num_rounds=3,
            round_adapter_names=("solution", "solution", "solution"),
            r1_reward_mode="judge_rejection_task",
            r23_reward_mode="constant",
            r23_constant=0.5,
            r23_symmetric=True,
            task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
        )
