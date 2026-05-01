from __future__ import annotations

from dataclasses import dataclass
import random

from llm_local_rl.chat_templates import get_chat_adapter
from llm_local_rl.ht_sequence_format import parse_ht_sequence
from llm_local_rl.interfaces import Tokenizer
from llm_local_rl.prompts import format_prompt, load_prompt
from llm_local_rl.types import CoinFlipInstance, HTSequenceInstance

_COIN_SYSTEM = load_prompt("tasks/coin_system.md")
_COIN_USER = load_prompt("tasks/coin_user.md")
_HT_USER = load_prompt("tasks/ht_sequence_user.md")


@dataclass(frozen=True)
class HTSequenceEnv:
    name: str = "ht_sequence"
    sequence_len: int = 8
    reward_mode: str = "num_h"
    strict_format: bool = False

    def __post_init__(self) -> None:
        if self.sequence_len <= 0:
            raise ValueError(f"sequence_len must be > 0, got {self.sequence_len}")
        if self.reward_mode not in {"num_h", "num_transitions"}:
            raise ValueError(f"Unsupported reward_mode={self.reward_mode!r}")

    def sample_instances(self, *, n: int, seed: int | None) -> list[HTSequenceInstance]:
        _ = seed
        return [
            HTSequenceInstance(
                instance_id=f"ht_{offset}",
                sequence_len=self.sequence_len,
            )
            for offset in range(n)
        ]

    def build_initial_prompt(self, *, instance: HTSequenceInstance) -> str:
        return format_prompt(_HT_USER, sequence_len=str(int(instance.sequence_len)))

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: HTSequenceInstance,
        tokenizer: Tokenizer,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        prompt = self.build_initial_prompt(instance=instance)
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return get_chat_adapter(tokenizer).encode_messages(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.encode(prompt, add_special_tokens=False)

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
        if hasattr(tokenizer, "apply_chat_template"):
            stop = get_chat_adapter(tokenizer).get_stop_sequences()
            if stop is not None and len(stop) == 1:
                return [int(stop[0])]
        stop = tokenizer.encode("\n", add_special_tokens=False)
        if len(stop) != 1:
            raise ValueError("Expected newline stop sequence to tokenize to one token.")
        return stop

    def score_completion(
        self,
        *,
        instance: HTSequenceInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
        parsed, parse_success = parse_ht_sequence(
            text=text,
            target_len=instance.sequence_len,
            strict_format=self.strict_format,
        )
        num_h = parsed.count("H")
        num_t = parsed.count("T")
        num_transitions = sum(1 for idx in range(1, len(parsed)) if parsed[idx] != parsed[idx - 1])

        if not parse_success:
            reward = 0.0
        elif self.reward_mode == "num_h":
            reward = float(num_h)
        else:
            reward = float(num_transitions)

        return reward, {
            "parse_success": 1.0 if parse_success else 0.0,
            "parsed_sequence": parsed,
            "num_h": num_h,
            "num_t": num_t,
            "num_transitions": num_transitions,
            "reward_mode": self.reward_mode,
            "full_text": text,
        }


@dataclass(frozen=True)
class CoinFlipEnv:
    name: str = "coin_flip"
    target_color: str = "Blue"

    def sample_instances(self, *, n: int, seed: int | None) -> list[CoinFlipInstance]:
        rng = random.Random(seed)
        offsets = list(range(n))
        rng.shuffle(offsets)
        return [CoinFlipInstance(instance_id=f"coin_{offset}") for offset in offsets]

    def build_initial_prompt(self, *, instance: CoinFlipInstance) -> str:
        _ = instance
        return _COIN_USER

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: CoinFlipInstance,
        tokenizer: Tokenizer,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        _ = instance
        if hasattr(tokenizer, "apply_chat_template"):
            messages = []
            if _COIN_SYSTEM:
                messages.append({"role": "system", "content": _COIN_SYSTEM})
            messages.append({"role": "user", "content": _COIN_USER})
            return get_chat_adapter(tokenizer).encode_messages(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.encode(_COIN_USER, add_special_tokens=False)

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
        if hasattr(tokenizer, "apply_chat_template"):
            stop = get_chat_adapter(tokenizer).get_stop_sequences()
            if stop is not None and len(stop) == 1:
                return [int(stop[0])]
        stop = tokenizer.encode("\n", add_special_tokens=False)
        if len(stop) != 1:
            raise ValueError("Expected newline stop sequence to tokenize to one token.")
        return stop

    def score_completion(
        self,
        *,
        instance: CoinFlipInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        _ = instance
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True).strip()
        first_word = text.split()[0] if text else ""
        normalized = first_word.capitalize()
        parse_success = 1.0 if normalized in ("Red", "Blue") else 0.0
        reward = 1.0 if normalized == self.target_color else 0.0
        red_tok = tokenizer.encode("Red", add_special_tokens=False)
        blue_tok = tokenizer.encode("Blue", add_special_tokens=False)
        if len(red_tok) != 1 or len(blue_tok) != 1:
            raise ValueError(f"Expected single-token Red/Blue, got {red_tok} / {blue_tok}")
        return reward, {
            "parse_success": parse_success,
            "choice": normalized if parse_success else None,
            "choice_token_id": int(completion_token_ids[0]) if completion_token_ids else None,
            "candidate_token_ids": {"Red": int(red_tok[0]), "Blue": int(blue_tok[0])},
            "target": self.target_color,
            "full_text": text,
        }
