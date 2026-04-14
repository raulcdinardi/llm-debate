from __future__ import annotations

from typing import Protocol

from llm_local_rl.types import HTSequenceInstance, SamplingRequest, SamplingResult, TrainExample


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


class Environment(Protocol):
    name: str

    def sample_instances(self, *, n: int, seed: int | None) -> list[HTSequenceInstance]: ...
    def build_initial_prompt(self, *, instance: HTSequenceInstance) -> str: ...
    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]: ...
    def score_completion(
        self,
        *,
        instance: HTSequenceInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]: ...


class Sampler(Protocol):
    def sample(self, request: SamplingRequest) -> SamplingResult: ...


class Trainer(Protocol):
    def forward(self, batch: list[TrainExample]) -> dict: ...
