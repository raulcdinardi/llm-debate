from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInput:
    """Minimal token-only ModelInput compatible with `tinker_debate` usage."""

    _tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        return cls(list(tokens))

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls([])

    def to_ints(self) -> list[int]:
        return list(self._tokens)

