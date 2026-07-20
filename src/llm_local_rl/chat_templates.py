from __future__ import annotations

import inspect
import weakref
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

    def encode_messages(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ChatTemplateError("Tokenizer lacks apply_chat_template; cannot build chat prompts.")
        kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        sig = inspect.signature(self.tokenizer.apply_chat_template)
        supports_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
        if enable_thinking is not None and ("enable_thinking" in sig.parameters or supports_var_kwargs):
            kwargs["enable_thinking"] = enable_thinking
        text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def get_stop_sequences(self) -> list[int] | None:
        if self.stop_token_id is None:
            return None
        return [int(self.stop_token_id)]

    def build_user_continuation_tokens(
        self,
        *,
        user_pre: str,
        user_post: str,
        enable_thinking: bool | None = None,
    ) -> tuple[list[int], list[int]]:
        if self.sentinel_tokens is not None and self.sentinel_token_ids is not None:
            return _continuation_with_sentinels(
                self,
                user_pre=user_pre,
                user_post=user_post,
                enable_thinking=enable_thinking,
                assistant_sentinel=self.sentinel_tokens[0],
                user_sentinel=self.sentinel_tokens[1],
                assistant_sentinel_id=self.sentinel_token_ids[0],
                user_sentinel_id=self.sentinel_token_ids[1],
            )
        if self.chatml_token_ids is not None:
            return _continuation_chatml(self, user_pre=user_pre, user_post=user_post)
        raise ChatTemplateError(
            "Cannot build continuation tokens: no sentinel tokens available and ChatML tokens not found."
        )


_ADAPTER_CACHE: weakref.WeakKeyDictionary[Any, ChatTemplateAdapter] = weakref.WeakKeyDictionary()


def get_chat_adapter(tokenizer: Any) -> ChatTemplateAdapter:
    try:
        return _ADAPTER_CACHE[tokenizer]
    except KeyError:
        pass
    except TypeError:
        return _build_chat_adapter(tokenizer=tokenizer, adapter_tokenizer=tokenizer)
    try:
        adapter_tokenizer = weakref.proxy(tokenizer)
    except TypeError:
        return _build_chat_adapter(tokenizer=tokenizer, adapter_tokenizer=tokenizer)
    adapter = _build_chat_adapter(tokenizer=tokenizer, adapter_tokenizer=adapter_tokenizer)
    try:
        _ADAPTER_CACHE[tokenizer] = adapter
    except TypeError:
        return _build_chat_adapter(tokenizer=tokenizer, adapter_tokenizer=tokenizer)
    return adapter


def _build_chat_adapter(*, tokenizer: Any, adapter_tokenizer: Any) -> ChatTemplateAdapter:
    adapter = ChatTemplateAdapter(
        tokenizer=adapter_tokenizer,
        name=str(getattr(tokenizer, "name_or_path", "unknown")),
        stop_token_id=_infer_stop_token_id(tokenizer),
        sentinel_tokens=None,
        sentinel_token_ids=None,
        chatml_token_ids=_infer_chatml_token_ids(tokenizer),
    )
    sentinel = _select_sentinels(tokenizer)
    if sentinel is not None:
        object.__setattr__(adapter, "sentinel_tokens", (sentinel[0][0], sentinel[1][0]))
        object.__setattr__(adapter, "sentinel_token_ids", (sentinel[0][1], sentinel[1][1]))
    return adapter


def _infer_stop_token_id(tokenizer: Any) -> int | None:
    for tok in ("<|im_end|>", "<|eot_id|>"):
        if tok in getattr(tokenizer, "all_special_tokens", []):
            token_ids = tokenizer.encode(tok, add_special_tokens=False)
            if len(token_ids) == 1:
                return int(token_ids[0])
    eos = getattr(tokenizer, "eos_token", None)
    if eos:
        token_ids = tokenizer.encode(eos, add_special_tokens=False)
        if len(token_ids) == 1:
            return int(token_ids[0])
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
    for tok in getattr(tokenizer, "additional_special_tokens", []) or []:
        if tok in template:
            continue
        token_ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        candidates.append((tok, int(token_ids[0])))
    if len(candidates) >= 2:
        return candidates[:2]
    return None


def _continuation_with_sentinels(
    adapter: ChatTemplateAdapter,
    *,
    user_pre: str,
    user_post: str,
    enable_thinking: bool | None,
    assistant_sentinel: str,
    user_sentinel: str,
    assistant_sentinel_id: int,
    user_sentinel_id: int,
) -> tuple[list[int], list[int]]:
    messages = [
        {"role": "assistant", "content": assistant_sentinel},
        {"role": "user", "content": f"{user_pre}{user_sentinel}{user_post}"},
    ]
    tokens = adapter.encode_messages(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
    assistant_indices = [idx for idx, token in enumerate(tokens) if token == assistant_sentinel_id]
    user_indices = [idx for idx, token in enumerate(tokens) if token == user_sentinel_id]
    if len(assistant_indices) != 1 or len(user_indices) != 1 or assistant_indices[0] >= user_indices[0]:
        raise ChatTemplateError("Sentinel tokens not found uniquely in chat template.")
    return list(tokens[assistant_indices[0] + 1 : user_indices[0]]), list(tokens[user_indices[0] + 1 :])


def _continuation_chatml(
    adapter: ChatTemplateAdapter,
    *,
    user_pre: str,
    user_post: str,
) -> tuple[list[int], list[int]]:
    prefix = adapter.tokenizer.encode("<|im_end|>\n<|im_start|>user\n" + user_pre, add_special_tokens=False)
    suffix = adapter.tokenizer.encode(user_post + "\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    return list(prefix), list(suffix)
