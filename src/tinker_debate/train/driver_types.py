from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from tinker_debate.debate_types import TrainingDatum


@dataclass(frozen=True)
class RolloutOutput:
    training_data: list[TrainingDatum]
    rollout_time_s: float
    num_rollouts: int
    info_lines: list[str]


class RolloutDriver(Protocol):
    async def rollout_step(self, *, step: int) -> RolloutOutput: ...


@dataclass(frozen=True)
class TrainLogFns:
    save_debate_log: Callable[..., Any]
    save_baseline_log: Callable[..., Any]
    save_summary_log: Callable[..., Any]
