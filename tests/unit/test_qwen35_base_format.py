from __future__ import annotations

import json

from llm_local_rl.countdown_code import build_countdown_files
from llm_local_rl.qwen35_base_format import (
    COUNTDOWN_JSON_PREAMBLE,
    base_text_prompt,
    countdown_assistant_completion,
    countdown_user_text,
    encode_countdown_prefix_tokens,
    encode_countdown_sft_example,
    resolve_countdown_assistant_prefill,
)
from llm_local_rl.task_types import TaskInstance


class CharTokenizer:
    eos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_countdown_assistant_completion_has_single_json_fence() -> None:
    numbers = [1, 2, 3, 4]
    target = 10
    pure = json.dumps(build_countdown_files(numbers=numbers, target=target))
    assistant = countdown_assistant_completion(completion_text=pure)
    assert assistant.startswith(COUNTDOWN_JSON_PREAMBLE)
    assert assistant.count("```json") == 1
    assert assistant.endswith("\n```")
    assert pure in assistant


def test_countdown_assistant_completion_strips_duplicate_fence() -> None:
    numbers = [1, 2, 3, 4]
    target = 10
    pure = json.dumps(build_countdown_files(numbers=numbers, target=target))
    wrapped = f"```json\n{pure}\n```"
    assistant = countdown_assistant_completion(completion_text=wrapped)
    assert assistant.count("```json") == 1
    assert assistant.endswith("\n```")


def test_encode_countdown_sft_example_trains_eos_after_closed_fence() -> None:
    tokenizer = CharTokenizer()
    user = countdown_user_text(numbers=[1, 2, 3, 4], target=10)
    pure = json.dumps(build_countdown_files(numbers=[1, 2, 3, 4], target=10))

    input_ids, labels = encode_countdown_sft_example(
        tokenizer=tokenizer,
        user_text=user,
        assistant_completion=pure,
    )

    prefix_len = len(encode_countdown_prefix_tokens(tokenizer=tokenizer, user_text=user))
    assert input_ids[-1] == tokenizer.eos_token_id
    assert labels[-1] == tokenizer.eos_token_id
    assert labels[:prefix_len] == [-100] * prefix_len
    assert labels[prefix_len:-1] == input_ids[prefix_len:-1]
    assert tokenizer.decode(input_ids[:-1]).endswith("\n```")


def test_base_text_prompt_no_system_for_countdown() -> None:
    user = countdown_user_text(numbers=[1, 2, 3, 4], target=10)
    text = base_text_prompt(system_text=None, user_text=user, assistant_prefill=COUNTDOWN_JSON_PREAMBLE)
    assert text.startswith("User:\n")
    assert "Assistant:\n" in text
    assert "System:\n" not in text


def test_resolve_countdown_assistant_prefill_defaults_base_text_to_json_preamble() -> None:
    assert (
        resolve_countdown_assistant_prefill(
            prompt_format="qwen35_base_text_prefill",
            configured_prefill=None,
        )
        == COUNTDOWN_JSON_PREAMBLE
    )
    assert resolve_countdown_assistant_prefill(prompt_format="chat", configured_prefill=None) == ""


def test_resolve_countdown_assistant_prefill_preserves_explicit_empty_override() -> None:
    assert (
        resolve_countdown_assistant_prefill(
            prompt_format="qwen35_base_text_prefill",
            configured_prefill="",
        )
        == ""
    )
