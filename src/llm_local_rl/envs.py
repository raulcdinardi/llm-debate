from __future__ import annotations

from dataclasses import dataclass
import random

from llm_local_rl.interfaces import Tokenizer
from llm_local_rl.types import HTSequenceInstance


@dataclass(frozen=True)
class HTSequenceEnv:
    name: str = "ht_sequence"
    sequence_len: int = 8
    reward_mode: str = "num_h"

    def __post_init__(self) -> None:
        if self.sequence_len <= 0:
            raise ValueError(f"sequence_len must be > 0, got {self.sequence_len}")
        if self.reward_mode not in {"num_h", "num_transitions"}:
            raise ValueError(f"Unsupported reward_mode={self.reward_mode!r}")

    def sample_instances(self, *, n: int, seed: int | None) -> list[HTSequenceInstance]:
        rng = random.Random(seed)
        offsets = list(range(n))
        rng.shuffle(offsets)
        return [
            HTSequenceInstance(
                instance_id=f"ht_{offset}",
                sequence_len=self.sequence_len,
            )
            for offset in offsets
        ]

    def build_initial_prompt(self, *, instance: HTSequenceInstance) -> str:
        return (
            "Write exactly one sequence of "
            f"{instance.sequence_len} symbols using only H and T. "
            "Do not add explanation or extra text."
        )

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
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
        parsed = self._extract_ht_prefix(text=text, target_len=instance.sequence_len)
        num_h = parsed.count("H")
        num_t = parsed.count("T")
        num_transitions = sum(1 for idx in range(1, len(parsed)) if parsed[idx] != parsed[idx - 1])

        if self.reward_mode == "num_h":
            reward = float(num_h)
        else:
            reward = float(num_transitions)

        return reward, {
            "parse_success": 1.0 if len(parsed) == instance.sequence_len else 0.0,
            "parsed_sequence": parsed,
            "num_h": num_h,
            "num_t": num_t,
            "num_transitions": num_transitions,
            "reward_mode": self.reward_mode,
            "full_text": text,
        }

    @staticmethod
    def _extract_ht_prefix(*, text: str, target_len: int) -> str:
        out: list[str] = []
        for ch in text:
            up = ch.upper()
            if up in {"H", "T"}:
                out.append(up)
                if len(out) == target_len:
                    break
        return "".join(out)
