from __future__ import annotations


def infer_chat_preamble(tokenizer: object) -> str:
    bos = getattr(tokenizer, "bos_token", None)
    if bos is None:
        return ""
    tokens = tokenizer.encode(bos, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"Expected single-token bos_token, got {len(tokens)} for {bos!r}")
    return str(bos)
