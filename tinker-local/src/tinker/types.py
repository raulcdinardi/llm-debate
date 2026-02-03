from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .model_input import ModelInput

LossFn = Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]

if TYPE_CHECKING:  # pragma: no cover
    import torch

    TorchTensor = torch.Tensor
else:  # pragma: no cover
    TorchTensor = object


@dataclass(frozen=True)
class TensorData:
    """Tiny wrapper matching the subset used by `tinker_debate`."""

    _tensor: TorchTensor

    @classmethod
    def from_torch(cls, tensor: TorchTensor) -> "TensorData":
        return cls(tensor)

    @property
    def tensor(self) -> TorchTensor:
        return self._tensor

    @property
    def data(self) -> list[float] | list[int]:
        import torch

        if not isinstance(self._tensor, torch.Tensor):
            raise TypeError("TensorData only supports torch.Tensor in tinker-local.")
        return self._tensor.detach().cpu().tolist()


@dataclass(frozen=True)
class Datum:
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int | None = None
    stop: str | list[str] | list[int] | None = None
    seed: int | None = None
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0


@dataclass(frozen=True)
class SampleSequence:
    tokens: list[int]
    logprobs: list[float] | None


@dataclass(frozen=True)
class SampleResponse:
    sequences: list[SampleSequence]


@dataclass(frozen=True)
class ForwardBackwardResult:
    loss_fn_outputs: list[dict[str, TensorData]]
    metrics: dict[str, float]
