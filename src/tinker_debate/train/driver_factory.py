from __future__ import annotations

from .driver_context import DriverContext
from .driver_types import RolloutDriver
from .orthogonal_driver import OrthogonalDriver


def build_driver(*, ctx: DriverContext) -> RolloutDriver:
    return OrthogonalDriver(ctx=ctx)
