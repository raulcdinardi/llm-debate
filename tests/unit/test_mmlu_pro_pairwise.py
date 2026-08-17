from __future__ import annotations

import json

import pytest

from llm_local_rl.judge_harness import (
    AgentDebateText,
    CONSTITUTION_SINGLE_TOKEN_V1,
    JudgeTranscript,
    get_judge_harness,
)
from llm_local_rl.mmlu_pro_pairwise import MMLUProPairwiseDebateTask


def _write_corpus(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "question_id": "q1",
            "question": "Which process produces ATP?",
            "correct_answer": "Cellular respiration",
            "wrong_answer": "Transcription",
            "category": "biology",
        },
        {
            "question_id": "q2",
            "question": "What follows from P implies Q and P?",
            "correct_answer": "Q follows by modus ponens",
            "wrong_answer": "Not Q follows by contradiction",
            "category": "philosophy",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_mmlu_pair_expansion_creates_both_answer_orderings(tmp_path) -> None:
    task = MMLUProPairwiseDebateTask(data_path=str(_write_corpus(tmp_path)))
    base = task.sample_instances(n=1, seed=7)[0]
    expanded = task.expand_group_instances(inst=base, group_size=4, seed=11)

    assert [inst.payload["agent"] for inst in expanded] == ["A", "B", "A", "B"]
    assert [inst.payload["ordering_index"] for inst in expanded] == [0, 0, 1, 1]
    assert sum(bool(inst.payload["is_correct"]) for inst in expanded) == 2
    assert {inst.payload["gold_agent"] for inst in expanded} == {"A", "B"}
    first, mirrored = expanded[:2], expanded[2:]
    assert [inst.payload["fixed_answer"] for inst in first] == list(
        reversed([inst.payload["fixed_answer"] for inst in mirrored])
    )
    assert [inst.payload["is_correct"] for inst in first] == list(
        reversed([inst.payload["is_correct"] for inst in mirrored])
    )
    for inst in expanded:
        assert task.fixed_r1_completion_text(inst=inst) == inst.payload["fixed_answer"]
        reward = task.compute_reward(inst=inst, completion_tokens=[], tokenizer=None)
        assert reward.reward == float(inst.payload["is_correct"])


def test_mmlu_pair_task_requires_two_mirrored_debates_per_question(tmp_path) -> None:
    task = MMLUProPairwiseDebateTask(data_path=str(_write_corpus(tmp_path)))
    base = task.sample_instances(n=1, seed=1)[0]
    with pytest.raises(ValueError, match="multiple of 4"):
        task.expand_group_instances(inst=base, group_size=2, seed=1)


def test_mmlu_pair_expansion_balances_four_debates_in_group_of_eight(tmp_path) -> None:
    task = MMLUProPairwiseDebateTask(data_path=str(_write_corpus(tmp_path)))
    base = task.sample_instances(n=1, seed=7)[0]
    expanded = task.expand_group_instances(inst=base, group_size=8, seed=11)

    assert len(expanded) == 8
    assert [inst.payload["agent"] for inst in expanded] == ["A", "B"] * 4
    assert [inst.payload["ordering_index"] for inst in expanded] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert sum(bool(inst.payload["is_correct"]) for inst in expanded) == 4
    assert [inst.payload["gold_agent"] for inst in expanded].count("A") == 4
    assert [inst.payload["gold_agent"] for inst in expanded].count("B") == 4


def test_single_token_judge_prompt_matches_training_harness() -> None:
    harness = get_judge_harness(CONSTITUTION_SINGLE_TOKEN_V1)
    transcript = JudgeTranscript(
        question="Why is the sky blue?",
        constitution="Prefer the answer that is correct.",
        agent_a=AgentDebateText("Rayleigh scattering", "A2", "A3"),
        agent_b=AgentDebateText("Ocean reflection", "B2", "B3"),
    )
    rendered = harness.render_checked(transcript=transcript, base_system_text="ignored")
    assert rendered.raw_text is not None
    prompt = rendered.raw_text
    assert prompt.startswith("System:\nYou are a judge evaluating a debate between two solvers.")
    assert "Question: Why is the sky blue?" in prompt
    assert "Round 1 (Proposal):\nRayleigh scattering" in prompt
    assert prompt.endswith(harness.assistant_prefill)
    assert harness.parse_verdict(" A") == "A"
    assert harness.parse_verdict("B<|endoftext|>") == "B"
    assert harness.parse_verdict("A because") == "INVALID"
