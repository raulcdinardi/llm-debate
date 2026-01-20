from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TaskInstance:
    instance_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class TaskReward:
    reward: float
    metrics: dict[str, Any]


class TaskSpec(Protocol):
    name: str

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]: ...

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer: Any) -> list[int]: ...

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]: ...

    def judge_context_text(self, *, inst: TaskInstance) -> str: ...

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward: ...
