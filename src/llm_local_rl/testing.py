from __future__ import annotations


def max_abs_diff(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Length mismatch.")
    if len(xs) == 0:
        return 0.0
    return max(abs(x - y) for x, y in zip(xs, ys, strict=True))


def assert_close_lists(xs: list[float], ys: list[float], *, atol: float) -> None:
    diff = max_abs_diff(xs, ys)
    if diff > atol:
        raise AssertionError(f"Lists differ by max_abs_diff={diff}, atol={atol}")


def warning_rate(*, total: int, warnings: int) -> float:
    if total <= 0:
        raise ValueError("total must be > 0")
    if warnings < 0 or warnings > total:
        raise ValueError("warnings must be in [0, total]")
    return warnings / total
