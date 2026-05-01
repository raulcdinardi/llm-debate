from __future__ import annotations

from llm_local_rl.debate_parity import (
    DebateResult,
    DebateTrajectory,
    Transition,
    assemble_split_train_examples,
    assemble_training_data_by_mode,
    assemble_training_data_r1_r23,
)


def _make_debate() -> DebateResult:
    traj_a = DebateTrajectory(
        agent="A",
        transitions=[
            Transition(prompt_tokens=[1, 2], completion_tokens=[3, 4], completion_logprobs=[-0.1, -0.2], round_num=1),
            Transition(prompt_tokens=[1, 2, 3, 4, 5], completion_tokens=[6, 7], completion_logprobs=[-0.3, -0.4], round_num=2),
            Transition(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], completion_tokens=[9, 10], completion_logprobs=[-0.5, -0.6], round_num=3),
        ],
        frozen_solution="A",
        metrics={"task_reward": 1.0},
    )
    traj_b = DebateTrajectory(
        agent="B",
        transitions=[
            Transition(prompt_tokens=[1, 2], completion_tokens=[11, 12], completion_logprobs=[-0.1, -0.2], round_num=1),
            Transition(prompt_tokens=[1, 2, 11, 12, 13], completion_tokens=[14, 15], completion_logprobs=[-0.3, -0.4], round_num=2),
            Transition(prompt_tokens=[1, 2, 11, 12, 13, 14, 15, 16], completion_tokens=[17, 18], completion_logprobs=[-0.5, -0.6], round_num=3),
        ],
        frozen_solution="B",
        metrics={"task_reward": 0.0},
    )
    return DebateResult(
        question="Q",
        ground_truth=None,
        trajectory_a=traj_a,
        trajectory_b=traj_b,
        verdict="A",
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
