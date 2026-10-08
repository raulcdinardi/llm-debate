from llm_local_rl.judge_harness import (
    QWEN35_CHAT_SINGLE_TOKEN_INTERLEAVED_V1,
    QWEN35_CHAT_SINGLE_TOKEN_V1,
    AgentDebateText,
    JudgeTranscript,
    get_judge_harness,
    harness_fingerprint,
)


def _transcript(rounds: int) -> JudgeTranscript:
    return JudgeTranscript(
        question="a wandering musician",
        constitution="Prefer the agent whose story best satisfies the user.",
        agent_a=AgentDebateText(rounds=tuple(f"A{k}" for k in range(1, rounds + 1))),
        agent_b=AgentDebateText(rounds=tuple(f"B{k}" for k in range(1, rounds + 1))),
    )


def _user(harness_id: str, transcript: JudgeTranscript) -> str:
    rendered = get_judge_harness(harness_id).render_checked(transcript=transcript, base_system_text="")
    return rendered.messages[1]["content"]


def test_interleaved_orders_rounds_then_agents():
    user = _user(QWEN35_CHAT_SINGLE_TOKEN_INTERLEAVED_V1, _transcript(3))
    assert user == (
        "Task:\na wandering musician\n\nCriterion:\nPrefer the agent whose story best satisfies the user.\n\n"
        "=== ROUND 1 ===\nAgent A:\nA1\n\nAgent B:\nB1\n\n"
        "=== ROUND 2 ===\nAgent A:\nA2\n\nAgent B:\nB2\n\n"
        "=== ROUND 3 ===\nAgent A:\nA3\n\nAgent B:\nB3"
    )
    # Every rebuttal now follows the argument it answers.
    assert user.index("B2") < user.index("A3")


def test_interleaved_shares_system_text_and_swaps_agents():
    t = _transcript(2)
    new = get_judge_harness(QWEN35_CHAT_SINGLE_TOKEN_INTERLEAVED_V1).render_checked(transcript=t, base_system_text="")
    old = get_judge_harness(QWEN35_CHAT_SINGLE_TOKEN_V1).render_checked(transcript=t, base_system_text="")
    assert new.messages[0] == old.messages[0]
    swapped = _user(QWEN35_CHAT_SINGLE_TOKEN_INTERLEAVED_V1, t.swapped())
    assert swapped.index("Agent A:\nB1") < swapped.index("Agent B:\nA1")


def test_agent_grouped_v1_contract_unchanged():
    # Existing manifests and judge checkpoints bind this fingerprint.
    assert harness_fingerprint(QWEN35_CHAT_SINGLE_TOKEN_V1) == (
        "de9d6c414536ea02e770f7eb9a1970ff2167becd0295ed6df0174093a873ef83"
    )
    assert harness_fingerprint(QWEN35_CHAT_SINGLE_TOKEN_INTERLEAVED_V1) != harness_fingerprint(QWEN35_CHAT_SINGLE_TOKEN_V1)
