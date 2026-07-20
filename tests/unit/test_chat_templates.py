from __future__ import annotations

import gc
import weakref

from llm_local_rl.chat_templates import _ADAPTER_CACHE, get_chat_adapter


class ChatMLTokenizer:
    name_or_path = "chatml-test"
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]
    additional_special_tokens: list[str] = []
    chat_template = "<|im_start|>{role}\n{content}<|im_end|>\n"
    eos_token = "<|im_end|>"

    _SPECIAL_TO_ID = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        out: list[int] = []
        idx = 0
        while idx < len(text):
            matched = False
            for token, token_id in self._SPECIAL_TO_ID.items():
                if text.startswith(token, idx):
                    out.append(token_id)
                    idx += len(token)
                    matched = True
                    break
            if matched:
                continue
            out.append(ord(text[idx]))
            idx += 1
        return out

    def decode(self, token_ids: list[int]) -> str:
        inverse = {value: key for key, value in self._SPECIAL_TO_ID.items()}
        return "".join(inverse.get(token_id, chr(token_id)) for token_id in token_ids)


class UnhashableChatMLTokenizer(ChatMLTokenizer):
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        return self is other


def test_chat_adapter_cache_reuses_and_evicts_by_tokenizer_object() -> None:
    _ADAPTER_CACHE.clear()
    tokenizer = ChatMLTokenizer()
    tokenizer_ref = weakref.ref(tokenizer)

    adapter = get_chat_adapter(tokenizer)

    assert get_chat_adapter(tokenizer) is adapter
    assert len(_ADAPTER_CACHE) == 1
    del tokenizer
    gc.collect()
    assert tokenizer_ref() is None
    assert len(_ADAPTER_CACHE) == 0


def test_chat_adapter_cache_falls_back_to_no_cache_for_unhashable_tokenizer() -> None:
    _ADAPTER_CACHE.clear()
    tokenizer = UnhashableChatMLTokenizer()

    first = get_chat_adapter(tokenizer)
    second = get_chat_adapter(tokenizer)

    assert first is not second
    assert len(_ADAPTER_CACHE) == 0


def test_chatml_continuation_matches_qwen_newlines() -> None:
    _ADAPTER_CACHE.clear()
    tokenizer = ChatMLTokenizer()
    adapter = get_chat_adapter(tokenizer)

    prefix, suffix = adapter.build_user_continuation_tokens(user_pre="pre", user_post="post")

    assert tokenizer.decode(prefix) == "<|im_end|>\n<|im_start|>user\npre"
    assert tokenizer.decode(suffix) == "post\n<|im_end|>\n<|im_start|>assistant\n"
