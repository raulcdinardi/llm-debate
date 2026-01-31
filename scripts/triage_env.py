#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollouts only and log per-group reward variance.")
    parser.add_argument(
        "--mode",
        type=str,
        default="single_turn",
        choices=["single_turn"],
        help="Rollout mode (triage only supports single_turn).",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["qa", "confidence", "summary"],
        help="Task/env for rollouts.",
    )
    parser.add_argument("-n", "--num-rollouts", type=int, default=128, help="Total rollouts per step")
    parser.add_argument("-s", "--steps", type=int, default=1, help="Rollout steps")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        choices=["gpqa_diamond", "gpqa_extended", "gpqa_main", "cnn_dailymail", "test"],
        help="Dataset to sample questions from (summary uses cnn_dailymail).",
    )
    parser.add_argument(
        "--reward-fn",
        type=str,
        default="compression",
        help="(summary only) Reward function. Can use weights like 'compression:0.5,rouge:0.3'.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max new tokens.",
    )
    parser.add_argument(
        "--accept-min-reward",
        type=float,
        default=0.0,
        help="(baseline only) Keep rollouts with reward >= this threshold.",
    )
    parser.add_argument(
        "--accept-require-parse",
        action="store_true",
        help="(baseline only) Only keep rollouts where parsing succeeded.",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="(single_turn only) Replay cached rollouts from summary_/baseline_ logs in this directory.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Sampling min_p (local backend only; 0 disables).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--num-groups",
        type=int,
        default=16,
        help="Number of unique prompts per step (rollouts are evenly split across groups).",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name (required). Logs to logs/<timestamp>_<params>_<name>/",
    )
    return parser.parse_args()


def write_run_metadata(*, log_dir: Path, args: argparse.Namespace) -> None:
    import sys

    env_keys = [
        "TINKER_LOCAL_BACKEND",
        "TINKER_LOCAL_SEED",
        "TINKER_LOCAL_GRAD_ACCUM_STEPS",
        "TINKER_LOCAL_DEVICE",
        "TINKER_LOCAL_LOAD_IN_4BIT",
        "TINKER_LOCAL_MAX_SEQ_LENGTH",
        "TINKER_DEBATE_BASE_MODEL",
        "USE_TF",
        "TRANSFORMERS_NO_TF",
        "TRANSFORMERS_NO_FLAX",
        "PYTHONUNBUFFERED",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]

    env: dict[str, str] = {}
    for k in env_keys:
        if k in os.environ:
            env[k] = os.environ[k]

    versions: dict[str, str] = {"python": sys.version.replace("\n", " ")}
    if "TINKER_LOCAL_BACKEND" in os.environ:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda_available"] = str(torch.cuda.is_available())
        versions["torch_cuda"] = str(torch.version.cuda)

        import transformers

        versions["transformers"] = transformers.__version__

    meta = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "args": vars(args),
        "env": env,
        "versions": versions,
    }

    path = log_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def get_log_dir(
    *,
    name: str,
    mode: str,
    env: str | None,
    num_rollouts: int,
    num_groups: int,
    dataset: str | None = None,
) -> Path:
    base = Path(os.environ.get("TINKER_DEBATE_LOG_ROOT", "logs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = dataset if dataset else "custom"
    env_tag = env if env else ""
    mode_env = f"{mode}_{env_tag}" if env_tag else mode
    dir_name = f"{timestamp}_n{num_rollouts}_g{num_groups}_{mode_env}_{dataset_tag}_{name}"
    return base / dir_name


def save_baseline_log(*, record: dict, log_dir: Path, model_name: str | None = None) -> Path:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"baseline_{timestamp}.json"
    record_with_meta = {
        "timestamp": timestamp,
        "model": model_name,
        **record,
    }
    with open(log_path, "w") as f:
        json.dump(record_with_meta, f, indent=2)
    return log_path


def save_summary_log(*, record: dict, log_dir: Path, model_name: str | None = None) -> Path:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"summary_{timestamp}.json"
    record_with_meta = {
        "timestamp": timestamp,
        "model": model_name,
        **record,
    }
    with open(log_path, "w") as f:
        json.dump(record_with_meta, f, indent=2)
    return log_path


@dataclass
class RecordCapture:
    current_step: int = 0
    records: list[dict] = None

    def reset(self, step: int) -> None:
        self.current_step = int(step)
        self.records = []

    def add(self, record: dict) -> None:
        if self.records is None:
            self.records = []
        self.records.append(record)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        raise ValueError("Empty list")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return mean, var**0.5


def _group_variances(records: list[dict]) -> list[float]:
    groups: dict[int, list[float]] = {}
    for r in records:
        gid = r.get("group_id")
        if gid is None:
            raise ValueError("Missing group_id in rollout record")
        groups.setdefault(int(gid), []).append(float(r["reward"]))

    variances: list[float] = []
    for values in groups.values():
        if len(values) == 0:
            continue
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        variances.append(var)
    return variances


async def main() -> None:
    args = parse_args()
    console = Console()

    if args.seed is not None:
        random.seed(args.seed)

    log_dir = get_log_dir(
        name=args.name,
        mode=args.mode,
        env=args.env,
        num_rollouts=args.num_rollouts,
        num_groups=args.num_groups,
        dataset=args.dataset,
    )
    os.makedirs(log_dir, exist_ok=True)
    write_run_metadata(log_dir=log_dir, args=args)

    console.print("\n[bold cyan]TINKER TRIAGE[/bold cyan]")
    console.print(f"Mode: {args.mode} ({args.env})")
    console.print(f"Dataset: {args.dataset}" if args.dataset else "Dataset: custom")
    console.print(f"Rollouts: {args.num_rollouts}")
    console.print(f"Steps: {args.steps}")
    console.print(f"Seed: {args.seed}")
    console.print(f"Run dir → {log_dir}")

    from tinker_debate.tinker_client import TinkerDebateClient
    from tinker_debate.train.driver_context import DriverContext
    from tinker_debate.train.driver_factory import build_driver
    from tinker_debate.train.driver_types import TrainLogFns

    console.print("\n[dim]Setting up Tinker client...[/dim]")
    client = await TinkerDebateClient.create()

    capture = RecordCapture()

    def save_record(*, record: dict, log_dir: Path, model_name: str | None = None) -> None:
        capture.add(record)
        if args.env == "summary":
            save_summary_log(record=record, log_dir=log_dir, model_name=model_name)
        else:
            save_baseline_log(record=record, log_dir=log_dir, model_name=model_name)

    log_fns = TrainLogFns(
        save_debate_log=lambda *_args, **_kwargs: None,
        save_baseline_log=save_record,
        save_summary_log=save_record,
    )

    driver = build_driver(
        ctx=DriverContext(
            args=args,
            client=client,
            console=console,
            log_dir=log_dir,
            log_fns=log_fns,
        )
    )

    per_step: list[dict] = []
    all_group_variances: list[float] = []

    for step in range(1, args.steps + 1):
        capture.reset(step)
        console.rule(f"[bold]Step {step}/{args.steps}[/bold]")
        t0 = time.time()
        out = await driver.rollout_step(step=step)
        rollout_time = time.time() - t0
        for line in out.info_lines:
            console.print(f"[dim]{line}[/dim]")

        records = capture.records or []
        rewards = [float(r["reward"]) for r in records]
        if len(rewards) == 0:
            raise RuntimeError("No rollout records captured.")

        group_vars = _group_variances(records)
        if len(group_vars) == 0:
            raise RuntimeError("No group variances computed.")
        all_group_variances.extend(group_vars)

        reward_mean, reward_std = _mean_std(rewards)
        gv_mean, gv_std = _mean_std(group_vars)
        per_step.append(
            {
                "step": step,
                "num_rollouts": len(records),
                "num_groups": len(group_vars),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "group_variance_mean": gv_mean,
                "group_variance_std": gv_std,
                "group_variances": group_vars,
                "rollout_time_s": float(rollout_time),
            }
        )

        console.print(
            f"[dim]Step stats: reward_mean={reward_mean:.6g} reward_std={reward_std:.6g} "
            f"group_var_mean={gv_mean:.6g} group_var_std={gv_std:.6g}[/dim]"
        )

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "args": vars(args),
        "per_step": per_step,
        "group_variances_all": all_group_variances,
    }
    summary_path = log_dir / "triage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold green]Saved triage summary:[/bold green] {summary_path}")
    console.print(f"[dim]Plot: python scripts/plot_group_variance.py --log-dir {log_dir}[/dim]")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
