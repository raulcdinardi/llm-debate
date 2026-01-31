from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console

from tinker_debate.tinker_client import TinkerDebateClient

from .driver_types import TrainLogFns


@dataclass(frozen=True)
class DriverContext:
    args: Any
    client: TinkerDebateClient
    console: Console
    log_dir: Any
    log_fns: TrainLogFns

