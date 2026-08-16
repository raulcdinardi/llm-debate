from __future__ import annotations

from llm_local_rl.base_model_judge import build_base_judge_prompt, extract_strict_verdict


def test_extract_strict_verdict_requires_exact_tag() -> None:
    assert extract_strict_verdict("<VERDICT>A</VERDICT>") == "A"
    assert extract_strict_verdict("prefix <VERDICT>B</VERDICT> suffix") == "B"
    assert extract_strict_verdict("A") == "INVALID"


def test_build_base_judge_prompt_includes_context_constitution_and_all_rounds() -> None:
    prompt = build_base_judge_prompt(
        question="Question text",
        constitution="Follow the rules",
        r1_a="A1",
        r1_b="B1",
        r2_a="A2",
        r2_b="B2",
        r3_a="A3",
        r3_b="B3",
    )
    assert "Question text" in prompt
    assert "Follow the rules" in prompt
    assert "Round 1 (Proposal):\nA1" in prompt
    assert "Round 2 (Argument):\nB2" in prompt
    assert "Round 3 (Response):\nA3" in prompt
    assert "<VERDICT>A</VERDICT>" in prompt
    assert "which agent's Round 1 response best satisfies the user" in prompt
    assert "The rationale for my judgment is:\n" in prompt
    assert "more convincing case" not in prompt
    assert "argument quality" not in prompt
