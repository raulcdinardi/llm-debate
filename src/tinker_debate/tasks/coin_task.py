from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinker_debate.prompts import load_prompt
from tinker_debate.chat_templates import get_chat_adapter

from .task_types import TaskInstance, TaskReward, TaskSpec

_SYSTEM_PROMPT = load_prompt("tasks/coin_system.md")
_USER_PROMPT = load_prompt("tasks/coin_user.md")


@dataclass(frozen=True)
class CoinTask(TaskSpec):
    """Coin-flip canary task for GRPO sanity checks."""

    name: str
    target_color: str

    @classmethod
    def create(cls, *, target_color: str = "Blue") -> "CoinTask":
        color = target_color.capitalize()
        if color not in ("Red", "Blue"):
            raise ValueError(f"target_color must be Red or Blue, got {target_color!r}")
        return cls(name="coin", target_color=color)

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        return [TaskInstance(instance_id=f"coin_{i}", payload={}) for i in range(n)]

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer: Any) -> list[int]:
        messages = []
        if _SYSTEM_PROMPT:
            messages.append({"role": "system", "content": _SYSTEM_PROMPT})
        messages.append({"role": "user", "content": _USER_PROMPT})
        adapter = get_chat_adapter(tokenizer)
        return adapter.encode_messages(messages, add_generation_prompt=True)

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        adapter = get_chat_adapter(tokenizer)
        stop = adapter.get_stop_sequences()
        if stop is None or len(stop) != 1:
            raise ValueError("Stop token must be a single token for coin task.")
        return [int(stop[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        return _USER_PROMPT

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward:
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
        first_word = text.split()[0] if text else ""
        normalized = first_word.capitalize()
        parse_success = 1.0 if normalized in ("Red", "Blue") else 0.0
        reward = 1.0 if normalized == self.target_color else 0.0
        red_tok = tokenizer.encode("Red", add_special_tokens=False)
        blue_tok = tokenizer.encode("Blue", add_special_tokens=False)
        if len(red_tok) != 1 or len(blue_tok) != 1:
            raise ValueError(f"Expected single-token Red/Blue, got {red_tok} / {blue_tok}")
        return TaskReward(
            reward=reward,
            metrics={
                "parse_success": parse_success,
                "choice": normalized if parse_success else None,
                "choice_token_id": int(completion_tokens[0]) if completion_tokens else None,
                "candidate_token_ids": {"Red": int(red_tok[0]), "Blue": int(blue_tok[0])},
                "target": self.target_color,
                "full_text": text,
            },
        )
