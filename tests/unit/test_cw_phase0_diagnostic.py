from __future__ import annotations

from llm_local_rl.constrained_writing import ConstrainedWritingDebateTask
from scripts.cw_judge_signal_phase0_diagnostic import (
    R1_PREFILL,
    R2_PREFILL,
    R3_PREFILL,
    base_text_prompt,
    prompt_loss_mask_violation_count,
    r1_prompt,
    r2_continuation,
    r3_continuation,
)


class ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def _task() -> ConstrainedWritingDebateTask:
    return ConstrainedWritingDebateTask.from_args(
        rules_per_speaker=2,
        reward_scope="both",
        sides="both",
        rule_family="generic",
        reward_mode="additive",
        letter_temperature=1.0,
    )


def test_diagnostic_prompt_bytes_match_runtime_task_contract() -> None:
    task = _task()
    inst = task.sample_instances(n=1, seed=2026071401)[0]
    public_task = task.r1_context_text(inst=inst)
    constitution = task.judge_constitution_text(inst=inst)

    prompt_ids, prompt_text = r1_prompt(ByteTokenizer(), public_task)
    expected_r1 = base_text_prompt(
        system_text=None,
        user_text=public_task,
        assistant_prefill=R1_PREFILL,
    )
    assert prompt_text == expected_r1
    assert bytes(prompt_ids).decode("utf-8") == expected_r1

    runtime_r2 = task.build_base_text_debate_extension(
        inst=inst,
        opponent_round=1,
        opponent_answer="Opponent story.",
    )
    diagnostic_r2 = r2_continuation(
        public_task=public_task,
        constitution=constitution,
        own_r1="Own story.",
        opponent_r1="Opponent story.",
        rules_first=False,
    )
    assert diagnostic_r2 == "\n\n" + base_text_prompt(
        system_text=runtime_r2.system_text,
        user_text=runtime_r2.user_text,
        assistant_prefill=runtime_r2.assistant_prefill,
    )
    assert runtime_r2.assistant_prefill == R2_PREFILL

    runtime_r3 = task.build_base_text_debate_extension(
        inst=inst,
        opponent_round=2,
        opponent_answer="Opponent argument.",
    )
    diagnostic_r3 = r3_continuation(
        public_task=public_task,
        constitution=constitution,
        own_r1="Own story.",
        opponent_r1="Opponent story.",
        own_r2="Own argument.",
        opponent_r2="Opponent argument.",
        rules_first=False,
    )
    assert diagnostic_r3 == "\n\n" + base_text_prompt(
        system_text=runtime_r3.system_text,
        user_text=runtime_r3.user_text,
        assistant_prefill=runtime_r3.assistant_prefill,
    )
    assert runtime_r3.assistant_prefill == R3_PREFILL
    assert R3_PREFILL != R2_PREFILL

    assert prompt_loss_mask_violation_count([prompt_ids], completion_token_id=0) == 0
