"""Exact trailing training-step means, independent of W&B display smoothing."""
from collections import deque
import math


def trailing_rows(rows, window=5):
    """Average finite observations in [step-window+1, step], never future values.

    Rows must be unique and increasing by source training step. At startup or
    gaps, use available observations; emit only metrics present at the current
    step, so a missing measurement never becomes zero or a carried-forward value.
    Each step has equal weight (not weighted by that step's evaluation sample N).
    """
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    history = deque()
    previous = None
    for row in rows:
        step, metrics = row["step"], row["metrics"]
        if type(step) is not int or (previous is not None and step <= previous):
            raise ValueError("Source steps must be unique, increasing integers")
        previous = step
        finite = {k: float(v) for k, v in metrics.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
        history.append((step, finite))
        while history[0][0] < step - window + 1:
            history.popleft()
        result = {}
        for key in finite:
            values = [m[key] for _, m in history if key in m]
            result[f"trailing{window}/{key}"] = math.fsum(values) / len(values)
        yield {"averaging_step": step, **result}
