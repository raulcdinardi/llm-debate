#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot group variance distribution from triage summary.")
    p.add_argument("--log-dir", type=str, required=True, help="Triage run directory")
    p.add_argument("--out-hist", type=str, default=None, help="Histogram output path")
    p.add_argument("--out-steps", type=str, default=None, help="Per-step mean variance output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    summary_path = log_dir / "triage_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(str(summary_path))

    data = json.loads(summary_path.read_text())
    variances = data.get("group_variances_all", [])
    per_step = data.get("per_step", [])
    if len(variances) == 0:
        raise ValueError("No group_variances_all found in triage_summary.json")

    out_hist = Path(args.out_hist).expanduser().resolve() if args.out_hist else (log_dir / "group_variance_hist.png")
    out_steps = Path(args.out_steps).expanduser().resolve() if args.out_steps else (log_dir / "group_variance_by_step.png")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    plt.hist(variances, bins=40, alpha=0.8, color="#1f77b4")
    plt.title(f"Group Variance Distribution ({log_dir.name})")
    plt.xlabel("Intra-group reward variance")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_hist, dpi=160)

    if per_step:
        xs = [int(row["step"]) for row in per_step]
        ys = [float(row["group_variance_mean"]) for row in per_step]
        ylo = [float(row["group_variance_mean"]) - float(row["group_variance_std"]) for row in per_step]
        yhi = [float(row["group_variance_mean"]) + float(row["group_variance_std"]) for row in per_step]

        plt.figure(figsize=(9, 4.5))
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label="mean variance")
        plt.fill_between(xs, ylo, yhi, alpha=0.2, label="±1 std")
        plt.title(f"Group Variance by Step ({log_dir.name})")
        plt.xlabel("Step")
        plt.ylabel("Intra-group reward variance")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_steps, dpi=160)

    csv_path = out_hist.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("variance\n")
        for v in variances:
            f.write(f"{v}\n")

    print(str(out_hist))
    if per_step:
        print(str(out_steps))
    print(str(csv_path))


if __name__ == "__main__":
    main()
