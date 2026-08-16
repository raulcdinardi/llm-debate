from __future__ import annotations

import json

import pytest

from llm_local_rl.base_model_judge import (
    SINGLE_TOKEN_JUDGE_ASSISTANT_PREFILL,
    build_single_token_judge_prompt,
    extract_single_token_verdict,
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


def test_mmlu_pair_expansion_fixes_one_correct_and_one_wrong_r1(tmp_path) -> None:
    task = MMLUProPairwiseDebateTask(data_path=str(_write_corpus(tmp_path)))
    base = task.sample_instances(n=1, seed=7)[0]
    expanded = task.expand_group_instances(inst=base, group_size=2, seed=11)

    assert [inst.payload["agent"] for inst in expanded] == ["A", "B"]
    assert sum(bool(inst.payload["is_correct"]) for inst in expanded) == 1
    assert len({inst.payload["gold_agent"] for inst in expanded}) == 1
    for inst in expanded:
        assert task.fixed_r1_completion_text(inst=inst) == inst.payload["fixed_answer"]
        reward = task.compute_reward(inst=inst, completion_tokens=[], tokenizer=None)
        assert reward.reward == float(inst.payload["is_correct"])


def test_mmlu_pair_task_requires_exactly_one_debate_per_question(tmp_path) -> None:
    task = MMLUProPairwiseDebateTask(data_path=str(_write_corpus(tmp_path)))
    base = task.sample_instances(n=1, seed=1)[0]
    with pytest.raises(ValueError, match="group_size=2"):
        task.expand_group_instances(inst=base, group_size=4, seed=1)


def test_single_token_judge_prompt_matches_training_harness() -> None:
    prompt = build_single_token_judge_prompt(
        question="Why is the sky blue?",
        constitution="Prefer the answer that is correct.",
        r1_a="Rayleigh scattering",
        r1_b="Ocean reflection",
        r2_a="A2",
        r2_b="B2",
        r3_a="A3",
        r3_b="B3",
    )
    assert prompt.startswith("System:\nYou are a judge evaluating a debate between two solvers.")
    assert "Question: Why is the sky blue?" in prompt
    assert "Round 1 (Proposal):\nRayleigh scattering" in prompt
    assert prompt.endswith(SINGLE_TOKEN_JUDGE_ASSISTANT_PREFILL)
    assert extract_single_token_verdict(" A") == "A"
    assert extract_single_token_verdict("B<|endoftext|>") == "B"
    assert extract_single_token_verdict("A because") == "INVALID"
