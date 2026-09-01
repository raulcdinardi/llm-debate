from __future__ import annotations

from dataclasses import replace

import pytest

from llm_local_rl.debate_parity import (
    DebateResult,
    DebateTrajectory,
    Transition,
    audit_base_text_debate_format,
    assemble_judge_coherence_grpo_examples,
    assemble_judge_supervised_label_examples,
    assemble_split_train_examples,
    assemble_training_data_by_mode,
    assemble_training_data_r1_r23,
    summarize_judge_rejection_r1_projection,
)


def _judge_turns(offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "order": "forward",
            "prompt_tokens": [101 + offset, 102 + offset],
            "completion_tokens": [103 + offset, 104 + offset],
            "completion_logprobs": [-0.1, -0.2],
        },
        {
            "order": "reverse",
            "prompt_tokens": [201 + offset, 202 + offset],
            "completion_tokens": [203 + offset, 204 + offset],
            "completion_logprobs": [-0.3, -0.4],
        },
    ]


def test_judge_coherence_grpo_normalizes_across_all_judgments() -> None:
    coherent = _make_debate(
        question="coherent",
        judge_raw_response={
            "bidirectional_judge": True,
            "order_invariant": True,
            "_training_judge_turns": _judge_turns(),
        },
    )
    incoherent = _make_debate(
        question="incoherent",
        token_offset=30,
        judge_raw_response={
            "bidirectional_judge": True,
            "order_invariant": False,
            "_training_judge_turns": _judge_turns(30),
        },
    )

    examples, metrics = assemble_judge_coherence_grpo_examples([coherent, incoherent])

    assert len(examples) == 4
    assert metrics == {
        "judge_grpo_group_size": 4,
        "judge_grpo_reward_mean": 0.0,
        "judge_grpo_reward_std": 1.0,
        "judge_grpo_coherent_debates": 1,
        "judge_grpo_incoherent_debates": 1,
        "judge_grpo_reward_mode": "coherence",
    }
    assert [example.metadata["judge_coherence_reward"] for example in examples] == [1.0, 1.0, -1.0, -1.0]
    assert [example.metadata["judge_grpo_zscore"] for example in examples] == [1.0, 1.0, -1.0, -1.0]
    assert [example.metadata["judge_order"] for example in examples] == ["forward", "reverse", "forward", "reverse"]
    assert examples[0].advantages[-2:] == [0.5, 0.5]
    assert examples[2].advantages[-2:] == [-0.5, -0.5]


def test_judge_coherence_grpo_pairwise_normalization_would_not_be_used() -> None:
    debates = [
        _make_debate(
            judge_raw_response={
                "bidirectional_judge": True,
                "order_invariant": True,
                "_training_judge_turns": _judge_turns(),
            }
        )
    ]
    examples, metrics = assemble_judge_coherence_grpo_examples(debates)
    assert metrics["judge_grpo_reward_std"] == 0.0
    assert all(not any(example.advantages) for example in examples)


def test_judge_grpo_rejects_js_as_a_sampled_action_reward() -> None:
    with pytest.raises(ValueError, match="unsupported judge GRPO reward mode"):
        assemble_judge_coherence_grpo_examples([], reward_mode="label_js")


def test_supervised_judge_targets_gold_referent_in_both_orders() -> None:
    turns = [
        {
            "order": "forward",
            "verdict": "B",
            "prompt_tokens": [101, 102],
            "completion_tokens": [378],
            "completion_logprobs": [-0.8],
            "behavior_policy_allowed_token_ids": [334, 378],
        },
        {
            "order": "reverse",
            "verdict": "A",
            "prompt_tokens": [201, 202],
            "completion_tokens": [334],
            "completion_logprobs": [-0.7],
            "behavior_policy_allowed_token_ids": [334, 378],
        },
    ]
    debate = _make_debate(
        reward_a=1.0,
        reward_b=0.0,
        judge_raw_response={
            "bidirectional_judge": True,
            "order_invariant": False,
            "judge_label_token_contract": {
                "a_token_ids": [334],
                "b_token_ids": [378],
            },
            "soft_score": {"referent_js_divergence_normalized": 0.2},
            "_training_judge_turns": turns,
        },
    )

    examples, metrics = assemble_judge_supervised_label_examples([debate])

    assert [row.target_ids[-1] for row in examples] == [334, 378]
    assert [row.metadata["judge_label_visual_target"] for row in examples] == ["A", "B"]
    assert all(row.behavior_logprob_mask[-1] == 0 for row in examples)
    assert all(row.advantages[-1] == 0.0 for row in examples)
    assert metrics["judge_training_objective"] == "supervised_label_ce_js"
    assert [row.metadata["judge_coherence_pair_member"] for row in examples] == [
        "forward",
        "reverse",
    ]
    assert examples[0].metadata["judge_coherence_pair_id"] == examples[1].metadata[
        "judge_coherence_pair_id"
    ]
    assert metrics["judge_supervised_sampled_label_accuracy"] == 0.0
    assert metrics["judge_supervised_referent_js_mean"] == pytest.approx(0.2)


def _make_debate(
    *,
    question: str = "Q",
    verdict: str = "A",
    reward_a: float = 1.0,
    reward_b: float = 0.0,
    token_offset: int = 0,
    instance_id: str | None = None,
    judge_raw_response: dict[str, object] | None = None,
) -> DebateResult:
    traj_a = DebateTrajectory(
        agent="A",
        transitions=[
            Transition(prompt_tokens=[1, 2], completion_tokens=[3 + token_offset, 4 + token_offset], completion_logprobs=[-0.1, -0.2], round_num=1),
            Transition(prompt_tokens=[1, 2, 3, 4, 5], completion_tokens=[6 + token_offset, 7 + token_offset], completion_logprobs=[-0.3, -0.4], round_num=2),
            Transition(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], completion_tokens=[9 + token_offset, 10 + token_offset], completion_logprobs=[-0.5, -0.6], round_num=3),
        ],
        frozen_solution="A",
        metrics={"task_reward": reward_a, "instance_id": instance_id},
    )
    traj_b = DebateTrajectory(
        agent="B",
        transitions=[
            Transition(prompt_tokens=[1, 2], completion_tokens=[11 + token_offset, 12 + token_offset], completion_logprobs=[-0.1, -0.2], round_num=1),
            Transition(prompt_tokens=[1, 2, 11, 12, 13], completion_tokens=[14 + token_offset, 15 + token_offset], completion_logprobs=[-0.3, -0.4], round_num=2),
            Transition(prompt_tokens=[1, 2, 11, 12, 13, 14, 15, 16], completion_tokens=[17 + token_offset, 18 + token_offset], completion_logprobs=[-0.5, -0.6], round_num=3),
        ],
        frozen_solution="B",
        metrics={"task_reward": reward_b, "instance_id": instance_id},
    )
    return DebateResult(
        question=question,
        ground_truth=None,
        trajectory_a=traj_a,
        trajectory_b=traj_b,
        verdict=verdict,
        judge_reasoning="judge",
        judge_raw_response=judge_raw_response,
    )


def _formatted_round(round_num: int) -> str:
    header = (
        "The reasons that my solution is better than my opponent's are:\n1)"
        if round_num == 2
        else "Responding to my opponent's criticism:\n1)"
    )
    return header + " first\n2) second\n3) third\nCONCLUDED"


def _soft_judge_audit(score: float, *, js: float = 0.0) -> dict[str, object]:
    return {
        "bidirectional_judge": True,
        "soft_judge": True,
        "order_invariant": False,
        "soft_score": {
            "score": score,
            "referent_js_divergence_normalized": js,
            "coherence_reliability": 1.0 - js,
        },
    }


def test_soft_judge_r1_allocates_task_gap_and_preserves_pair_sum() -> None:
    debate = _make_debate(
        reward_a=1.0,
        reward_b=0.0,
        judge_raw_response=_soft_judge_audit(0.5, js=0.5),
    )
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=1,
        round_adapter_names=("solution",),
        r1_reward_mode="judge_soft_task_gap",
        r23_reward_mode="none",
        r23_constant=1.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )
    examples = split["solution"]
    rewards = {example.metadata["agent"]: example.metadata["r1_soft_reward"] for example in examples}
    assert rewards == {"A": pytest.approx(0.625), "B": pytest.approx(0.375)}
    assert sum(rewards.values()) == pytest.approx(1.0)
    assert examples[0].metadata["r1_task_reward_pair_sum"] == pytest.approx(1.0)
    assert examples[0].metadata["judge_coherence_reliability"] == pytest.approx(0.5)


def test_soft_judge_r23_is_exactly_zero_sum_even_when_hard_labels_disagree() -> None:
    debate = _make_debate(judge_raw_response=_soft_judge_audit(0.5, js=0.5))
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=2,
        round_adapter_names=("solution", "debate"),
        r1_reward_mode="task",
        r23_reward_mode="soft_judge",
        r23_constant=99.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )
    debate_examples = split["debate"]
    rewards = {example.metadata["agent"]: example.metadata["r23_reward"] for example in debate_examples}
    assert rewards == {"A": pytest.approx(0.25), "B": pytest.approx(-0.25)}
    assert sum(rewards.values()) == pytest.approx(0.0)
    assert all(example.metadata["r23_incoherent_reward_applied"] is False for example in debate_examples)
    assert all(
        example.metadata["judge_coherence_reliability"] == pytest.approx(0.5)
        for example in debate_examples
    )


def test_base_text_debate_format_audit_is_exact_and_terminal() -> None:
    assert audit_base_text_debate_format(text=_formatted_round(2), round_num=2)["strict_ok"] is True
    assert audit_base_text_debate_format(text=_formatted_round(3), round_num=3)["strict_ok"] is True
    extra = audit_base_text_debate_format(text=_formatted_round(2) + "\nextra", round_num=2)
    assert extra["strict_ok"] is False
    assert "terminal_CONCLUDED" in extra["failures"]


def test_base_text_debate_format_penalizes_legacy_truncation_trigger_without_removing_tail() -> None:
    raw = (
        "The reasons that my solution is better than my opponent's are:\n"
        "1) first point. This unnumbered sentence used to be removed.\n"
        "2) second point.\n"
        "3) third point.\n"
        "CONCLUDED"
    )
    audit = audit_base_text_debate_format(text=raw, round_num=2)

    assert audit["strict_ok"] is False
    assert audit["legacy_truncation_triggered"] is True
    assert "legacy_truncation_trigger" in audit["failures"]


def test_legacy_truncation_trigger_penalty_reaches_all_raw_round_tokens() -> None:
    debate = _make_debate(
        judge_raw_response={"bidirectional_judge": True, "order_invariant": True}
    )
    raw_r2 = (
        "The reasons that my solution is better than my opponent's are:\n"
        "1) first. Unnumbered offending tail.\n2) second.\n3) third.\nCONCLUDED"
    )
    for trajectory in (debate.trajectory_a, debate.trajectory_b):
        trajectory.metrics.update({"r2": raw_r2, "r3": _formatted_round(3)})

    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="task",
        r23_reward_mode="constant",
        r23_constant=1.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
        r23_advantage_scope="merged_r23",
        r23_format_failure_penalty=-1.0,
    )

    winner = split["debate"][0]
    assert winner.target_ids[-5:] == [6, 7, 8, 9, 10]
    assert winner.advantages[-5:] == [-0.25, -0.25, 0.0, 0.25, 0.25]
    assert winner.metadata["r2_legacy_truncation_triggered"] is True
    assert winner.metadata["r2_format_failure_penalty"] == -1.0


def test_merged_r23_adds_binary_format_penalty_only_to_failed_round() -> None:
    debate = _make_debate(
        judge_raw_response={"bidirectional_judge": True, "order_invariant": True}
    )
    debate.trajectory_a.metrics.update({"r2": _formatted_round(2), "r3": "malformed"})
    debate.trajectory_b.metrics.update({"r2": _formatted_round(2), "r3": _formatted_round(3)})
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="task",
        r23_reward_mode="constant",
        r23_constant=1.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
        r23_advantage_scope="merged_r23",
        r23_format_failure_penalty=-1.0,
    )

    winner = split["debate"][0]
    loser = split["debate"][1]
    assert winner.advantages[-5:] == [0.25, 0.25, 0.0, -0.25, -0.25]
    assert loser.advantages[-5:] == [-0.25, -0.25, 0.0, -0.25, -0.25]
    assert winner.metadata["r2_format_strict"] is True
    assert winner.metadata["r3_format_strict"] is False
    assert winner.metadata["r2_format_failure_penalty"] == 0.0
    assert winner.metadata["r3_format_failure_penalty"] == -1.0
    assert winner.metadata["r23_combined_reward"] == 0.0


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
    assert len(split["debate"]) == 2
    first_shared = shared_data[0]
    first_solution = split["solution"][0]
    assert first_solution.advantages[-2:] == first_shared.completion_advantages[:2]
    first_debate = split["debate"][0]
    assert first_debate.target_ids[-5:] == [6, 7, 8, 9, 10]
    assert first_debate.advantages[-5:] == [0.25, 0.25, 0.0, 0.25, 0.25]
    assert first_debate.old_logprobs[-5:] == [-0.3, -0.4, 0.0, -0.5, -0.6]
    assert first_debate.behavior_logprob_mask[-5:] == [1, 1, 0, 1, 1]
    assert first_debate.metadata["round_nums"] == [2, 3]
    assert first_debate.metadata["rounds_merged"] == 2
    assert first_debate.metadata["r23_advantage_scope"] == "per_round"
    assert first_solution.metadata["source_exact_shared_equivalent"] is False


def test_split_projection_keeps_different_round_adapters_separate() -> None:
    debate = _make_debate()
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "judge"),
        r1_reward_mode="task",
        r23_reward_mode="constant",
        r23_constant=0.5,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )
    assert len(split["solution"]) == 2
    assert len(split["debate"]) == 2
    assert len(split["judge"]) == 2
    assert split["debate"][0].metadata["round_num"] == 2
    assert split["judge"][0].metadata["round_num"] == 3


def test_split_projection_can_distribute_r23_reward_once_over_merged_rounds() -> None:
    debate = _make_debate()
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="task",
        r23_reward_mode="constant",
        r23_constant=1.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
        r23_advantage_scope="merged_r23",
    )

    winner_debate = split["debate"][0]
    loser_debate = split["debate"][1]
    assert winner_debate.advantages[-5:] == [0.25, 0.25, 0.0, 0.25, 0.25]
    assert loser_debate.advantages[-5:] == [-0.25, -0.25, 0.0, -0.25, -0.25]
    assert winner_debate.metadata["r23_advantage_scope"] == "merged_r23"
    assert winner_debate.metadata["r23_first_adv_value"] == 0.25
    assert winner_debate.metadata["r23_second_adv_value"] == 0.25


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
    assert winner.completion_logprob_mask == [1, 1, 0, 1, 1, 0, 1, 1]
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
    assert len(split["debate"]) == 2
    assert split["solution"][0].advantages[-2:] == [0.25, 0.25]
    assert split["solution"][1].advantages[-2:] == [-0.25, -0.25]
    assert split["debate"][0].advantages[-5:] == [0.25, 0.25, 0.0, 0.25, 0.25]
    assert split["debate"][1].advantages[-5:] == [-0.25, -0.25, 0.0, -0.25, -0.25]


def test_judge_delta_task_adds_delta_to_winner_and_subtracts_it_from_loser() -> None:
    debate = _make_debate(
        verdict="A",
        reward_a=5.0,
        reward_b=2.0,
        judge_raw_response={"bidirectional_judge": True, "order_invariant": True},
    )
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="judge_delta_task",
        r23_reward_mode="constant",
        r23_constant=1.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
        r1_judge_delta_q=1.0,
    )

    solution_a, solution_b = split["solution"]
    assert solution_a.metadata["r1_task_reward"] == 5.0
    assert solution_b.metadata["r1_task_reward"] == 2.0
    assert solution_a.metadata["r1_task_reward_delta"] == 3.0
    assert solution_b.metadata["r1_task_reward_delta"] == 3.0
    assert solution_a.metadata["r1_modulated_reward"] == 8.0
    assert solution_b.metadata["r1_modulated_reward"] == -1.0
    assert solution_a.advantages[-2:] == [0.5, 0.5]
    assert solution_b.advantages[-2:] == [-0.5, -0.5]


def test_judge_delta_task_follows_judge_even_when_judge_prefers_lower_task_reward() -> None:
    debate = _make_debate(
        verdict="B",
        reward_a=5.0,
        reward_b=2.0,
        judge_raw_response={"bidirectional_judge": True, "order_invariant": True},
    )
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="judge_delta_task",
        r23_reward_mode="constant",
        r23_constant=1.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )

    solution_a, solution_b = split["solution"]
    assert solution_a.metadata["r1_modulated_reward"] == 2.0
    assert solution_b.metadata["r1_modulated_reward"] == 5.0
    assert solution_a.advantages[-2:] == [-0.5, -0.5]
    assert solution_b.advantages[-2:] == [0.5, 0.5]


def test_incoherent_judgment_uses_coin_flip_winner_for_r1_and_penalizes_both_debate_paths() -> None:
    debate = _make_debate(
        verdict="A",
        reward_a=5.0,
        reward_b=2.0,
        judge_raw_response={"bidirectional_judge": True, "order_invariant": False},
    )
    split = assemble_split_train_examples(
        debates=[debate],
        num_rounds=3,
        round_adapter_names=("solution", "debate", "debate"),
        r1_reward_mode="judge_delta_task",
        r23_reward_mode="constant",
        r23_constant=1.0,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
        incoherent_r23_reward=-0.5,
    )

    solution_a, solution_b = split["solution"]
    # debate.verdict is the seeded coin-flip result produced on order disagreement.
    assert solution_a.metadata["r1_modulated_reward"] == 8.0
    assert solution_b.metadata["r1_modulated_reward"] == -1.0
    assert solution_a.metadata["judge_order_invariant"] is False
    assert solution_b.metadata["judge_order_invariant"] is False
    assert solution_a.metadata["r1_winner_source"] == "seeded_coin_flip"
    assert solution_b.metadata["r1_winner_source"] == "seeded_coin_flip"
    assert solution_a.advantages[-2:] == [0.5, 0.5]
    assert solution_b.advantages[-2:] == [-0.5, -0.5]
    assert [example.metadata["r23_reward"] for example in split["debate"]] == [-0.5, -0.5]
    assert all(example.metadata["r23_incoherent_reward_applied"] is True for example in split["debate"])
    assert all(example.advantages[-5:] == [-0.25, -0.25, 0.0, -0.25, -0.25] for example in split["debate"])


def test_judge_rejection_task_selects_only_winners_and_zscores_selected_task_rewards() -> None:
    debates = [
        _make_debate(verdict="A", reward_a=1.0, reward_b=1000.0, instance_id="group"),
        _make_debate(
            verdict="B",
            reward_a=-1000.0,
            reward_b=3.0,
            token_offset=100,
            instance_id="group",
        ),
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

    assert [example.metadata["agent"] for example in split["solution"]] == ["A", "B"]
    assert [example.metadata["r1_rejected_agent"] for example in split["solution"]] == ["B", "A"]
    assert [example.metadata["r1_task_reward"] for example in split["solution"]] == [1.0, 3.0]
    assert [example.metadata["r1_zscore"] for example in split["solution"]] == [-1.0, 1.0]
    assert split["solution"][0].advantages[-2:] == [-0.5, -0.5]
    assert split["solution"][1].advantages[-2:] == [0.5, 0.5]
    assert len(split["debate"]) == 4
    assert [example.metadata["r23_reward"] for example in split["debate"]] == [0.5, -0.5, -0.5, 0.5]


def test_judge_rejection_task_groups_by_instance_not_rules_blind_question() -> None:
    debates = [
        _make_debate(question="same topic", verdict="A", reward_a=1.0, instance_id="task-a"),
        _make_debate(
            question="same topic",
            verdict="B",
            reward_b=3.0,
            token_offset=100,
            instance_id="task-b",
        ),
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

    assert [example.metadata["r1_group_index"] for example in split["solution"]] == [0, 1]
    assert all(example.metadata["r1_selected_group_size"] == 1 for example in split["solution"])
    assert all(example.metadata["r1_zscore"] == 0.0 for example in split["solution"])


def test_judge_rejection_task_zeroes_equal_rewards_and_drops_invalid_debates() -> None:
    debates = [
        _make_debate(verdict="A", reward_a=2.0, instance_id="group"),
        _make_debate(verdict="B", reward_b=2.0, token_offset=100, instance_id="group"),
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
    assert len(split["debate"]) == 4


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


def test_judge_rejection_projection_summary_measures_emitted_losers() -> None:
    debates = [
        _make_debate(verdict="A", reward_a=1.0, instance_id="group"),
        _make_debate(verdict="B", reward_b=3.0, token_offset=100, instance_id="group"),
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

    summary = summarize_judge_rejection_r1_projection(r1_examples=split["solution"], debates=debates)
    assert summary["winner_r1_example_count"] == 2
    assert summary["loser_r1_example_count"] == 0
    assert summary["nonzero_advantage_r1_example_count"] == 2
    assert summary["zero_advantage_r1_example_count"] == 0
    assert summary["rejected_loser_count"] == 2

    loser_metadata = dict(split["solution"][0].metadata)
    loser_metadata.update({"agent": "B", "verdict": "A"})
    emitted_loser = replace(split["solution"][0], metadata=loser_metadata)
    measured = summarize_judge_rejection_r1_projection(
        r1_examples=[emitted_loser, split["solution"][1]], debates=debates
    )
    assert measured["winner_r1_example_count"] == 1
    assert measured["winner_r1_example_count_delta"] == -1
    assert measured["loser_r1_example_count"] == 1
    assert measured["nonzero_advantage_r1_example_count"] == 2


def test_split_projection_uses_configured_r1_adapter_for_every_reward_mode() -> None:
    split = assemble_split_train_examples(
        debates=[_make_debate()],
        num_rounds=3,
        round_adapter_names=("custom_solution", "debate", "debate"),
        r1_reward_mode="task",
        r23_reward_mode="constant",
        r23_constant=0.5,
        r23_symmetric=True,
        task_reward_fn=lambda traj, _debate: float(traj.metrics["task_reward"]),
    )

    assert len(split["custom_solution"]) == 2
    assert all(example.adapter_name == "custom_solution" for example in split["custom_solution"])
    assert "solution" not in split
