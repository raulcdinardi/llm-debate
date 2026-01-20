#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute within-group reward variance per step (summary mode).")
    p.add_argument("--run-dir", required=True, help="Run directory containing summary_*.json logs.")
    p.add_argument("--out", default=None, help="Optional output CSV path (default: <run_dir>/summary_group_variance.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run dir not found: {run_dir}")

    files = sorted(run_dir.glob("summary_*.json"))
    if not files:
        raise ValueError(f"No summary_*.json files in {run_dir}")

    records: list[dict] = []
    for f in files:
        data = json.loads(f.read_text())
        if "step" not in data:
            raise ValueError(f"Missing step in {f.name}")
        if "group_id" not in data:
            raise ValueError(f"Missing group_id in {f.name}")
        if "reward" not in data:
            raise ValueError(f"Missing reward in {f.name}")
        records.append(
            {
                "step": int(data["step"]),
                "group_id": int(data["group_id"]),
                "reward": float(data["reward"]),
            }
        )

    by_step: dict[int, dict[int, list[float]]] = {}
    for r in records:
        by_step.setdefault(r["step"], {}).setdefault(r["group_id"], []).append(r["reward"])

    rows: list[dict] = []
    all_vars: list[float] = []
    for step in sorted(by_step.keys()):
        group_vars: list[float] = []
        group_sizes = []
        for _gid, rewards in by_step[step].items():
            group_sizes.append(len(rewards))
            if len(rewards) < 2:
                raise ValueError(f"Group size < 2 at step {step} (group_id={_gid})")
            group_vars.append(statistics.pvariance(rewards))
        mean_var = statistics.mean(group_vars)
        rows.append(
            {
                "step": step,
                "groups": len(group_vars),
                "mean_var": mean_var,
                "min_var": min(group_vars),
                "max_var": max(group_vars),
                "min_group_size": min(group_sizes),
                "max_group_size": max(group_sizes),
            }
        )
        all_vars.extend(group_vars)

    overall_mean = statistics.mean(all_vars)
    overall_min = min(all_vars)
    overall_max = max(all_vars)

    out_path = Path(args.out) if args.out else (run_dir / "summary_group_variance.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["step", "groups", "mean_var", "min_var", "max_var", "min_group_size", "max_group_size"],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(out_path)
    print(f"steps={len(rows)} groups_per_step={rows[0]['groups'] if rows else 0}")
    print(f"overall_mean_var={overall_mean:.6g} overall_min_var={overall_min:.6g} overall_max_var={overall_max:.6g}")


if __name__ == "__main__":
    main()
