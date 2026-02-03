from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class LocalFuture(Generic[T]):
    """A tiny awaitable wrapper to mimic the Tinker SDK's double-await pattern."""

    _task: "asyncio.Task[T]"

    def __await__(self):
        return self._task.__await__()

    @classmethod
    def from_awaitable(cls, awaitable: Awaitable[T]) -> "LocalFuture[T]":
        return cls(asyncio.create_task(awaitable))

