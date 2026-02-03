from __future__ import annotations

from dataclasses import dataclass

from .training_client import TrainingClient


@dataclass(frozen=True)
class ServiceClient:
    """Local replacement for `tinker.ServiceClient`."""

    async def create_lora_training_client_async(
        self,
        *,
        base_model: str,
        rank: int = 32,
    ) -> TrainingClient:
        return TrainingClient.create(base_model=base_model, rank=rank)

