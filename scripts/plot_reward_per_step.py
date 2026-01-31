#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from bisect import bisect_left


TS_FMT = "%Y%m%d_%H%M%S_%f"


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FMT)


@dataclass(frozen=True)
class StepStats:
    step: int
    n: int
    mean: float
    std: float
    min: float
    max: float


def _load_training_steps(run_dir: Path) -> list[tuple[int, datetime]]:
    steps: list[tuple[int, datetime]] = []
    for p in run_dir.glob("training_step_*.json"):
        obj = json.load(open(p))
        step = int(obj["step"])
        ts = _parse_ts(str(obj["timestamp"]))
        steps.append((step, ts))
    steps.sort(key=lambda x: x[0])
    if not steps:
        raise ValueError(f"No training_step_*.json found in {run_dir}")
    return steps


def _load_summary_rewards(run_dir: Path) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    for p in run_dir.glob("summary_*.json"):
        obj = json.load(open(p))
        ts = _parse_ts(str(obj["timestamp"]))
        reward = float(obj["reward"])
        out.append((ts, reward))
    out.sort(key=lambda x: x[0])
    if not out:
        raise ValueError(f"No summary_*.json found in {run_dir}")
    return out


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        raise ValueError("Empty list")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, var**0.5


def compute_reward_per_step(*, run_dir: Path) -> list[StepStats]:
    steps = _load_training_steps(run_dir)
    step_ts = [ts for _step, ts in steps]
    step_ids = [step for step, _ts in steps]

    rewards = _load_summary_rewards(run_dir)

    by_step: dict[int, list[float]] = {step: [] for step in step_ids}
    for ts, r in rewards:
        i = bisect_left(step_ts, ts)
        if i >= len(step_ts):
            continue
        by_step[step_ids[i]].append(r)

    stats: list[StepStats] = []
    for step in step_ids:
        rs = by_step[step]
        if len(rs) == 0:
            continue
        mean, std = _mean_std(rs)
        stats.append(StepStats(step=step, n=len(rs), mean=mean, std=std, min=min(rs), max=max(rs)))
    if not stats:
        raise RuntimeError("No rewards assigned to any training step.")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Plot reward vs step from a run directory.")
    p.add_argument("--run-dir", type=str, required=True, help="Path containing training_step_*.json and summary_*.json")
    p.add_argument("--out", type=str, default=None, help="Output image path (default: <run-dir>/reward_per_step.png)")
    p.add_argument("--title", type=str, default=None, help="Plot title")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    out_path = Path(args.out).expanduser().resolve() if args.out else (run_dir / "reward_per_step.png")

    stats = compute_reward_per_step(run_dir=run_dir)

    xs = [s.step for s in stats]
    ys = [s.mean for s in stats]
    ylo = [s.mean - s.std for s in stats]
    yhi = [s.mean + s.std for s in stats]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label="mean reward")
    plt.fill_between(xs, ylo, yhi, alpha=0.2, label="±1 std")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    if args.title is None:
        title = f"Reward vs Step ({run_dir.name})"
    else:
        title = args.title
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("step,n,mean,std,min,max\n")
        for s in stats:
            f.write(f"{s.step},{s.n},{s.mean},{s.std},{s.min},{s.max}\n")

    print(str(out_path))
    print(str(csv_path))


if __name__ == "__main__":
    main()

