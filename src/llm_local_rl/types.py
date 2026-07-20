from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

AdapterName = Literal["shared", "solution", "debate", "judge"]


@dataclass(frozen=True)
class SamplingRequest:
    adapter_name: AdapterName
    prompt_token_ids: list[int]
    stop_token_ids: list[int]
    max_tokens: int
    temperature: float
    seed: int | None = None
    min_p: float = 0.0
    top_p: float = 1.0
    stop_strings: tuple[str, ...] = ()
    include_stop_str_in_output: bool = False


@dataclass(frozen=True)
class SamplingResult:
    adapter_name: AdapterName
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeTurn:
    turn_name: str
    adapter_name: AdapterName
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    trainable: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeSample:
    instance_id: str
    turns: list[EpisodeTurn]
    reward: float
    reward_metrics: dict[str, Any]


@dataclass(frozen=True)
class TrainExample:
    adapter_name: AdapterName
    input_ids: list[int]
    target_ids: list[int]
    loss_mask: list[int]
    old_logprobs: list[float]
    advantages: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HTSequenceInstance:
    instance_id: str
    sequence_len: int


@dataclass(frozen=True)
class CoinFlipInstance:
    instance_id: str
    options: tuple[str, str] = ("Heads", "Tails")


@dataclass(frozen=True)
class ShortStoryInstance:
    instance_id: str
    secret_word: str


class RolloutSampler(Protocol):
    def set_adapter_paths(self, *, adapter_paths: dict[str, str]) -> None: ...

    def wake_up(self, *, level: int = 1) -> None: ...

    def sleep(self, *, level: int = 1) -> None: ...

    def close(self) -> None: ...

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]: ...
