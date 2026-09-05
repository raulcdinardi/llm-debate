from __future__ import annotations

from llm_local_rl.countdown_code import (
    COUNTDOWN_JSON_PREAMBLE,
    build_countdown_user_prompt,
    normalize_countdown_completion_files,
)


def base_text_prompt(*, system_text: str | None, user_text: str, assistant_prefill: str = "") -> str:
    if system_text:
        return f"System:\n{system_text}\n\nUser:\n{user_text}\nAssistant:\n{assistant_prefill}"
    return f"User:\n{user_text}\nAssistant:\n{assistant_prefill}"


def default_countdown_assistant_prefill(*, prompt_format: str) -> str:
    if prompt_format == "qwen35_base_text_prefill":
        return COUNTDOWN_JSON_PREAMBLE
    if prompt_format == "chat":
        return ""
    raise ValueError(f"Unsupported Countdown prompt_format={prompt_format!r}")


def resolve_countdown_assistant_prefill(*, prompt_format: str, configured_prefill: str | None) -> str:
    if configured_prefill is not None:
        return configured_prefill
    return default_countdown_assistant_prefill(prompt_format=prompt_format)


def countdown_user_text(*, numbers: list[int], target: int) -> str:
    return build_countdown_user_prompt(numbers=numbers, target=target)


def strip_leading_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :].lstrip()
    if stripped.startswith("```"):
        stripped = stripped[3:].lstrip()
    if stripped.endswith("```"):
        stripped = stripped[: -3].rstrip()
    return stripped


def countdown_pure_json_completion(*, completion_text: str) -> str:
    return normalize_countdown_completion_files(completion_text=strip_leading_json_fence(completion_text))


def countdown_assistant_completion(*, completion_text: str) -> str:
    pure_json = countdown_pure_json_completion(completion_text=completion_text)
    if "```json" in pure_json:
        raise ValueError("Pure JSON completion must not contain a ```json fence.")
    assistant = COUNTDOWN_JSON_PREAMBLE + pure_json + "\n```"
    if assistant.count("```json") != 1:
        raise ValueError(f"Expected exactly one ```json fence, got {assistant.count('```json')}.")
    return assistant


def encode_base_text(*, tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode_countdown_prefix_tokens(*, tokenizer, user_text: str) -> list[int]:
    prefix_text = base_text_prompt(
        system_text=None,
        user_text=user_text,
        assistant_prefill=COUNTDOWN_JSON_PREAMBLE,
    )
    return encode_base_text(tokenizer=tokenizer, text=prefix_text)


def encode_countdown_full_tokens(*, tokenizer, user_text: str, assistant_completion: str) -> list[int]:
    full_text = base_text_prompt(
        system_text=None,
        user_text=user_text,
        assistant_prefill=assistant_completion,
    )
    return encode_base_text(tokenizer=tokenizer, text=full_text)


def encode_countdown_sft_example(
    *,
    tokenizer,
    user_text: str,
    assistant_completion: str,
) -> tuple[list[int], list[int]]:
    if assistant_completion.startswith(COUNTDOWN_JSON_PREAMBLE):
        assistant = assistant_completion
    else:
        assistant = countdown_assistant_completion(completion_text=assistant_completion)
    if assistant.count("```json") != 1:
        raise ValueError(f"Expected exactly one ```json fence, got {assistant.count('```json')}.")
    prefix_tokens = encode_countdown_prefix_tokens(tokenizer=tokenizer, user_text=user_text)
    full_tokens = encode_countdown_full_tokens(
        tokenizer=tokenizer,
        user_text=user_text,
        assistant_completion=assistant,
    )
    if full_tokens[: len(prefix_tokens)] != prefix_tokens:
        raise ValueError("Countdown assistant span is not a token suffix of the full base-text prompt.")
    label_tokens = full_tokens[len(prefix_tokens) :]
    if len(label_tokens) == 0:
        raise ValueError("Countdown assistant span is empty after the JSON preamble.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Countdown base-text SFT requires tokenizer.eos_token_id.")
    full_tokens = full_tokens + [int(eos_token_id)]
    labels = [-100] * len(prefix_tokens) + label_tokens + [int(eos_token_id)]
    return full_tokens, labels
