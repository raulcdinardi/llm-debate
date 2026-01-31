from __future__ import annotations


class Qwen3InstructStopRenderer:
    """Minimal stop-sequence provider for Qwen3 instruct-style prompts.

    We use token IDs (not strings) because tokenization and detokenization are not bijective,
    and token-level stop conditions are what the Tinker API expects for RL environments.
    """

    def __init__(self, tokenizer: object):
        self.tokenizer = tokenizer

    def get_stop_sequences(self) -> list[int]:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Expected single token for <|im_end|>, got {len(tokens)}")
        return [int(tokens[0])]


def infer_chat_preamble(tokenizer: object) -> str:
    bos = tokenizer.bos_token
    if bos is None:
        return ""
    tokens = tokenizer.encode(bos, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"Expected single-token bos_token, got {len(tokens)} for {bos!r}")
    return bos


def select_instruct_renderer_name(tokenizer: object) -> str:
    preamble = infer_chat_preamble(tokenizer)
    if preamble == "<|startoftext|>":
        return "lfm2_instruct"
    return "qwen3_instruct"
