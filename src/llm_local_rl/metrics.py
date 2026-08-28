from __future__ import annotations

from statistics import mean


def mean_numeric_metrics(metrics: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for metric in metrics:
        for key, value in metric.items():
            if isinstance(value, bool):
                grouped.setdefault(key, []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                grouped.setdefault(key, []).append(float(value))
    return {f"mean_{key}": mean(values) for key, values in sorted(grouped.items()) if values}
