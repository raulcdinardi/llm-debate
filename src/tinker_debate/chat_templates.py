from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ChatTemplateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatTemplateAdapter:
    tokenizer: Any
    name: str
    stop_token_id: int | None
    sentinel_tokens: tuple[str, str] | None
    sentinel_token_ids: tuple[int, int] | None
    chatml_token_ids: tuple[int, int] | None

    def encode_messages(self, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ChatTemplateError("Tokenizer lacks apply_chat_template; cannot build chat prompts.")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def get_stop_sequences(self) -> list[int] | None:
        if self.stop_token_id is None:
            return None
        return [int(self.stop_token_id)]

    def build_user_continuation_tokens(self, *, user_pre: str, user_post: str) -> tuple[list[int], list[int]]:
        if self.sentinel_tokens is not None and self.sentinel_token_ids is not None:
            # Sentinel tokens must be single-token, template-reserved specials so boundaries are stable.
            return _continuation_with_sentinels(
                self,
                user_pre=user_pre,
                user_post=user_post,
                assistant_sentinel=self.sentinel_tokens[0],
                user_sentinel=self.sentinel_tokens[1],
                assistant_sentinel_id=self.sentinel_token_ids[0],
                user_sentinel_id=self.sentinel_token_ids[1],
            )
        if self.chatml_token_ids is not None:
            return _continuation_chatml(self, user_pre=user_pre, user_post=user_post)
        raise ChatTemplateError(
            "Cannot build continuation tokens: no sentinel tokens available and ChatML tokens not found. "
            "Use a tokenizer with reserved special tokens or a ChatML-compatible model."
        )


_ADAPTER_CACHE: dict[int, ChatTemplateAdapter] = {}


def get_chat_adapter(tokenizer: Any) -> ChatTemplateAdapter:
    key = id(tokenizer)
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]

    stop_token_id = _infer_stop_token_id(tokenizer)
    sentinel = _select_sentinels(tokenizer)
    chatml = _infer_chatml_token_ids(tokenizer)

    name = getattr(tokenizer, "name_or_path", "unknown")
    adapter = ChatTemplateAdapter(
        tokenizer=tokenizer,
        name=str(name),
        stop_token_id=stop_token_id,
        sentinel_tokens=None if sentinel is None else (sentinel[0][0], sentinel[1][0]),
        sentinel_token_ids=None if sentinel is None else (sentinel[0][1], sentinel[1][1]),
        chatml_token_ids=chatml,
    )
    _ADAPTER_CACHE[key] = adapter
    return adapter


def _infer_stop_token_id(tokenizer: Any) -> int | None:
    for tok in ("<|im_end|>", "<|eot_id|>"):
        if tok in getattr(tokenizer, "all_special_tokens", []):
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        ids = tokenizer.encode(eos, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def _infer_chatml_token_ids(tokenizer: Any) -> tuple[int, int] | None:
    try:
        im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    except Exception:
        return None
    if len(im_start) != 1 or len(im_end) != 1:
        return None
    return int(im_start[0]), int(im_end[0])


def _select_sentinels(tokenizer: Any) -> list[tuple[str, int]] | None:
    template = getattr(tokenizer, "chat_template", "") or ""
    candidates: list[tuple[str, int]] = []
    # Prefer reserved special tokens not referenced in the chat template to ensure uniqueness.
    for tok in getattr(tokenizer, "additional_special_tokens", []) or []:
        if tok in template:
            continue
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            continue
        candidates.append((tok, int(ids[0])))
    if len(candidates) >= 2:
        return candidates[:2]
    return None


def _continuation_with_sentinels(
    adapter: ChatTemplateAdapter,
    *,
    user_pre: str,
    user_post: str,
    assistant_sentinel: str,
    user_sentinel: str,
    assistant_sentinel_id: int,
    user_sentinel_id: int,
) -> tuple[list[int], list[int]]:
    messages = [
        {"role": "assistant", "content": assistant_sentinel},
        {"role": "user", "content": f"{user_pre}{user_sentinel}{user_post}"},
    ]
    tokens = adapter.encode_messages(messages, add_generation_prompt=True)
    # We expect each sentinel to appear exactly once in the rendered token stream.
    a_idxs = [i for i, t in enumerate(tokens) if t == assistant_sentinel_id]
    u_idxs = [i for i, t in enumerate(tokens) if t == user_sentinel_id]
    if len(a_idxs) != 1 or len(u_idxs) != 1 or a_idxs[0] >= u_idxs[0]:
        raise ChatTemplateError(
            "Sentinel tokens not found uniquely in chat template. "
            "Try a model with reserved special tokens or use a ChatML-compatible tokenizer."
        )
    a_idx = a_idxs[0]
    u_idx = u_idxs[0]
    prefix = tokens[a_idx + 1 : u_idx]
    suffix = tokens[u_idx + 1 :]
    return list(prefix), list(suffix)


def _continuation_chatml(
    adapter: ChatTemplateAdapter,
    *,
    user_pre: str,
    user_post: str,
) -> tuple[list[int], list[int]]:
    _im_start = "<|im_start|>"
    _im_end = "<|im_end|>"

    prefix_str = _im_end + _im_start + "user\n" + user_pre
    suffix_str = user_post + "\n" + _im_end + _im_start + "assistant"

    prefix = adapter.tokenizer.encode(prefix_str, add_special_tokens=False)
    suffix = adapter.tokenizer.encode(suffix_str, add_special_tokens=False)
    return list(prefix), list(suffix)
