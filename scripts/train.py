#!/usr/bin/env python3
"""Minimal training run using Tinker API.

Runs debates, trains on winners (rejection sampling), logs losses.

Usage:
    python train.py                  # Run 2 debates, 1 train step
    python train.py -n 4 -s 2        # 4 debates, 2 train steps
    python train.py --dry-run        # Run debates but don't train

HTTP Inspection (live):
    Run `mitmweb --listen-port 8080` in another terminal, then open
    http://127.0.0.1:8081 to inspect all API traffic.

HTTP Recording:
    All API calls are recorded to logs/<run>/http_traffic.yaml via VCR.py
    These cassettes are human-readable and can be replayed for testing.
"""

from __future__ import annotations

import argparse
import math
import asyncio
import atexit
from contextlib import nullcontext
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Allow running as a script without installing the package.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

load_dotenv()

# Proxy configuration - opt-in only via --proxy
PROXY_PORT = 8080
WEB_PORT = 8081
USE_PROXY = "--proxy" in sys.argv

if "--proxy" in sys.argv:
    sys.argv.remove("--proxy")
if "--no-proxy" in sys.argv:
    sys.argv.remove("--no-proxy")

if USE_PROXY:
    PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"

    print(f"\n[mitmproxy] Starting mitmweb on port {PROXY_PORT}...")
    mitmweb_path = Path(sys.executable).parent / "mitmweb"
    assert mitmweb_path.exists(), f"mitmweb not found at {mitmweb_path}. Install with: pip install mitmproxy"
    mitm_proc = subprocess.Popen(
        [str(mitmweb_path), "--listen-port", str(PROXY_PORT), "--web-port", str(WEB_PORT), "--set", "ssl_insecure=true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(mitm_proc.terminate)
    time.sleep(1)

    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

    # Use mitmproxy's CA certificate
    mitmproxy_ca = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
    if mitmproxy_ca.exists():
        os.environ["SSL_CERT_FILE"] = str(mitmproxy_ca)
        os.environ["REQUESTS_CA_BUNDLE"] = str(mitmproxy_ca)

    print(f"[mitmproxy] Web UI: http://127.0.0.1:{WEB_PORT}\n")
else:
    print("\n[mitmproxy] Disabled (use --proxy to enable).\n")

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal debate training")
    parser.add_argument(
        "--mode",
        type=str,
        default="debate",
        choices=["debate", "single_turn"],
        help="Training paradigm. 'debate' = multi-turn with judge; 'single_turn' = simple rollout.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["qa", "confidence", "summary", "coin", "secret_word", "constrained_writing", "graph_path"],
        help="Task/env. Required if --mode=single_turn. Optional if --mode=debate (default: qa).",
    )
    parser.add_argument("-n", "--num-rollouts", type=int, default=16, help="Total rollouts per step")
    parser.add_argument("-s", "--steps", type=int, default=1, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Run debates but don't train")
    parser.add_argument("--mock-judge", action="store_true", help="Use random mock judge instead of LLM")
    parser.add_argument(
        "--confidence-judge",
        action="store_true",
        help="(debate only) Use R1 <CONFIDENCE> tags to decide verdict (higher confidence wins).",
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
        help="(baseline only) Only keep rollouts where confidence parsing succeeded.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name (required). Logs to logs/<timestamp>_<params>_<name>/",
    )
    parser.add_argument("--question", "-q", default="What is 7 * 8?", help="Question (ignored if --dataset)")
    parser.add_argument("--ground-truth", "-g", default="56", help="Ground truth (ignored if --dataset)")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        choices=["gpqa_diamond", "gpqa_extended", "gpqa_main", "cnn_dailymail", "test"],
        help="Dataset to sample questions from (overrides --question). 'test' = simple arithmetic.",
    )
    parser.add_argument(
        "--reward-fn",
        type=str,
        default="compression",
        help="(summary only) Reward function for training. Can also use comma-sep weights like 'compression:0.5,rouge:0.3'.",
    )
    parser.add_argument(
        "--constraint-rules-per-speaker",
        type=int,
        default=2,
        help="(constrained_writing only) Number of rules sampled per speaker.",
    )
    parser.add_argument(
        "--constraint-reward-scope",
        type=str,
        default="both",
        choices=["alice", "bob", "both"],
        help="(constrained_writing only) Which rules contribute to reward.",
    )
    parser.add_argument(
        "--constraint-sides",
        type=str,
        default="both",
        choices=["alice", "bob", "both"],
        help="(constrained_writing only) Which speakers appear in the prompt.",
    )
    parser.add_argument(
        "--graph-num-nodes",
        type=int,
        default=8,
        help="(graph_path only) Total nodes including S and G.",
    )
    parser.add_argument(
        "--graph-path-length",
        type=int,
        default=3,
        help="(graph_path only) Prefix length K (number of steps to simulate).",
    )
    parser.add_argument(
        "--graph-edge-prob",
        type=float,
        default=0.3,
        help="(graph_path only) Probability of adding a directed edge between nodes.",
    )
    parser.add_argument(
        "--graph-max-edge-cost",
        type=int,
        default=9,
        help="(graph_path only) Maximum edge cost (min is 1).",
    )
    parser.add_argument(
        "--graph-trap-nodes",
        type=int,
        default=2,
        help="(graph_path only) Number of trap nodes with low heuristic and no outgoing edges.",
    )
    parser.add_argument(
        "--graph-budget-factor",
        type=float,
        default=1.3,
        help="(graph_path only) Budget multiplier over optimal S->G cost.",
    )
    parser.add_argument(
        "--graph-proxy-alpha",
        type=float,
        default=0.1,
        help="(graph_path only) Proxy cost weight: reward = -h(last) - alpha*prefix_cost.",
    )
    parser.add_argument(
        "--graph-heuristic-noise",
        type=float,
        default=0.0,
        help="(graph_path only) Additive noise on heuristics (clipped at 0).",
    )
    parser.add_argument(
        "--graph-trap-h-scale",
        type=float,
        default=0.3,
        help="(graph_path only) Multiplier applied to trap node heuristics.",
    )
    parser.add_argument(
        "--graph-strict-sampling",
        action="store_true",
        default=True,
        help="(graph_path only) Resample until output matches schema (default: enabled).",
    )
    parser.add_argument(
        "--graph-no-strict-sampling",
        action="store_false",
        dest="graph_strict_sampling",
        help="(graph_path only) Disable schema-based resampling.",
    )
    parser.add_argument(
        "--graph-strict-max-attempts",
        type=int,
        default=4,
        help="(graph_path only) Max resampling attempts per rollout.",
    )
    parser.add_argument(
        "--coin-target",
        type=str,
        default="Blue",
        choices=["Red", "Blue"],
        help="(coin env) Color treated as reward=1.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for rollouts.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max new tokens for rollouts.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Sampling min_p (local backend only; 0 disables).",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="(single_turn only) Replay cached rollouts from summary_/baseline_ logs in this directory.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--num-groups",
        type=int,
        default=2,
        help="Number of unique questions (groups) per step.",
    )
    parser.add_argument(
        "--debate-reward",
        type=str,
        default="task",
        choices=["task"],
        help="(debate only) Reward definition for debate training data (centered across winners).",
    )
    parser.add_argument(
        "--debate-r1-reward",
        type=str,
        default="task",
        choices=["task", "none"],
        help="(debate only) R1 reward source. 'task' trains winner R1 tokens with centered task reward.",
    )
    parser.add_argument(
        "--debate-r23-reward",
        type=str,
        default="constant",
        choices=["none", "constant"],
        help="(debate only) R2/R3 reward. 'constant' applies win/loss reward to R2/R3 tokens.",
    )
    parser.add_argument(
        "--debate-r23-constant",
        type=float,
        default=1.0,
        help="(debate only) Constant reward magnitude for R2/R3 (used when --debate-r23-reward=constant).",
    )
    parser.add_argument(
        "--debate-r23-mode",
        type=str,
        default="symmetric",
        choices=["symmetric", "winner_only"],
        help="(debate only) R2/R3 reward mode. symmetric => winner=+c, loser=-c. winner_only => loser=0.",
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
    """Build log directory with params in name for easy identification."""
    base = Path(os.environ.get("TINKER_DEBATE_LOG_ROOT", "logs"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build descriptive name: timestamp_params_mode_env_name
    # e.g., 20250106_1930_n16_g2_debate_gpqa_my_experiment
    # or    20250106_1930_n16_g2_single_turn_summary_cnn_my_experiment
    dataset_tag = dataset if dataset else "custom"
    env_tag = env if env else ""
    mode_env = f"{mode}_{env_tag}" if env_tag else mode
    dir_name = f"{timestamp}_n{num_rollouts}_g{num_groups}_{mode_env}_{dataset_tag}_{name}"
    
    return base / dir_name


def save_debate_log(result, log_dir: Path, config=None, model_name: str | None = None) -> Path:
    """Save debate result to JSON log with full API call details."""

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"debate_{timestamp}.json"

    def serialize_trajectory(traj):
        return {
            "frozen_solution": traj.frozen_solution,
            "r1": traj.metrics.get("r1", ""),
            "r2": traj.metrics.get("r2", ""),
            "r3": traj.metrics.get("r3", ""),
            "metrics": traj.metrics,
            "transitions": [
                {
                    "round": t.round_num,
                    "prompt_tokens": t.prompt_tokens,
                    "completion_tokens": t.completion_tokens,
                    "completion_logprobs": t.completion_logprobs,
                    # Token counts directly from API (for verifying KV cache extension)
                    "prompt_token_count": len(t.prompt_tokens),
                    "completion_token_count": len(t.completion_tokens),
                    # Raw API response (everything returned by tinker)
                    "raw_response": t.raw_response,
                }
                for t in traj.transitions
            ],
        }

    config_dict = None
    if config is not None:
        config_dict = {
            "num_rounds": config.num_rounds,
            "max_tokens_per_turn": config.max_tokens_per_turn,
            "temperature": config.temperature,
            "kl_coef": config.kl_coef,
            "learning_rate": config.learning_rate,
        }

    log_data = {
        "timestamp": timestamp,
        "model": model_name,
        "config": config_dict,
        "question": result.question,
        "ground_truth": result.ground_truth,
        "verdict": result.verdict,
        "judge_reasoning": result.judge_reasoning,
        "agent_a": serialize_trajectory(result.trajectory_a),
        "agent_b": serialize_trajectory(result.trajectory_b),
        "metrics": result.metrics,
        # Judge tokens (None if mock judge)
        "judge": {
            "prompt_tokens": result.judge_prompt_tokens,
            "completion_tokens": result.judge_completion_tokens,
            "completion_logprobs": result.judge_completion_logprobs,
            "prompt_token_count": len(result.judge_prompt_tokens) if result.judge_prompt_tokens else None,
            "completion_token_count": len(result.judge_completion_tokens) if result.judge_completion_tokens else None,
            # Raw API response (everything returned by tinker)
            "raw_response": result.judge_raw_response,
        },
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_path


def save_training_step_log(
    *,
    step: int,
    training_data: list,
    fwd_bwd_result: dict,
    learning_rate: float,
    log_dir: Path,
    debug_metrics: dict | None = None,
) -> Path:
    """Save training step log with exact data sent to API."""

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"training_step_{step:04d}_{timestamp}.json"

    datums = []
    for d in training_data:
        prompt_toks = d.prompt_tokens
        completion_toks = d.completion_tokens
        prompt_len = len(prompt_toks)

        full_tokens = prompt_toks + completion_toks
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]

        advantages_tensor = [0.0] * (prompt_len - 1) + list(d.completion_advantages)
        sampling_logprobs_tensor = [0.0] * (prompt_len - 1) + list(d.completion_logprobs)

        datums.append(
            {
                "input_tokens": input_tokens,
                "target_tokens": target_tokens,
                "advantages": advantages_tensor,
                "sampling_logprobs": sampling_logprobs_tensor,
                "source": d.metadata,
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "completion_advantages": d.completion_advantages,
            }
        )

    log_data = {
        "step": step,
        "timestamp": timestamp,
        "num_datums": len(datums),
        "datums": datums,
        "results": fwd_bwd_result,
        "learning_rate": learning_rate,
    }
    if debug_metrics is not None:
        log_data["replay_debug"] = debug_metrics

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_path


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

async def main():
    args = parse_args()

    # Validate mode/env combination
    if args.mode == "single_turn" and args.env is None:
        console.print("[red]Error: --env is required when --mode=single_turn[/red]")
        sys.exit(1)
    if args.replay_dir is not None and args.mode != "single_turn":
        console.print("[red]Error: --replay-dir is only supported for --mode=single_turn[/red]")
        sys.exit(1)
    if args.replay_dir is not None:
        replay_path = Path(args.replay_dir)
        if not replay_path.exists():
            console.print(f"[red]Error: --replay-dir not found: {replay_path}[/red]")
            sys.exit(1)

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

    # VCR cassette for recording all HTTP traffic
    cassette_path = log_dir / "http_traffic.yaml"
    is_local = ("TINKER_LOCAL_BACKEND" in os.environ) or (os.environ.get("TINKER_BACKEND") == "local")
    if is_local:
        console.print("[dim]Local backend detected; skipping HTTP traffic recording.[/dim]")
    else:
        console.print(f"[dim]Recording HTTP traffic to {cassette_path}[/dim]")

    console.print("\n[bold cyan]TINKER TRAINING[/bold cyan]")
    mode_desc = f"{args.mode}" + (f" ({args.env})" if args.env else "")
    console.print(f"Mode: {mode_desc}")
    if args.mode == "debate":
        console.print(f"Dataset: {args.dataset}" if args.dataset else f"Question: {args.question}")
    elif args.env == "confidence":
        console.print("Dataset: confidence/questions.json")
    elif args.env == "summary":
        console.print(f"Dataset: cnn_dailymail, reward_fn: {args.reward_fn}")
    elif args.env == "constrained_writing":
        console.print(
            "Constrained writing: "
            f"rules_per_speaker={args.constraint_rules_per_speaker}, "
            f"reward_scope={args.constraint_reward_scope}, "
            f"sides={args.constraint_sides}"
        )
    elif args.env == "graph_path":
        console.print(
            "Graph path: "
            f"nodes={args.graph_num_nodes}, K={args.graph_path_length}, edge_prob={args.graph_edge_prob}, "
            f"max_cost={args.graph_max_edge_cost}, traps={args.graph_trap_nodes}, "
            f"budget_factor={args.graph_budget_factor}, proxy_alpha={args.graph_proxy_alpha}, "
            f"heur_noise={args.graph_heuristic_noise}, trap_h_scale={args.graph_trap_h_scale}, "
            f"strict_sampling={args.graph_strict_sampling} (max_attempts={args.graph_strict_max_attempts})"
        )

    if args.mode == "debate":
        if args.num_rollouts % args.num_groups != 0:
            console.print(f"[red]Error: --num-rollouts ({args.num_rollouts}) must be divisible by --num-groups ({args.num_groups})[/red]")
            sys.exit(1)
        group_size = args.num_rollouts // args.num_groups
        if group_size % 2 != 0:
            console.print(f"[red]Error: group_size ({group_size}) must be even for debate (2 rollouts per debate)[/red]")
            sys.exit(1)
        debates_per_question = group_size // 2
        num_debates = args.num_groups * debates_per_question
        console.print(f"Rollouts: {args.num_rollouts} total, {args.num_groups} groups of {group_size}")
        console.print(f"Debates: {num_debates} total ({debates_per_question} per question)")
    else:
        num_debates = args.num_rollouts  # baseline env: 1 rollout = 1 "datum"
        console.print(f"Rollouts: {args.num_rollouts}")

    console.print(f"Training steps: {args.steps}")
    console.print(f"Learning rate: {args.lr}")
    console.print(f"Seed: {args.seed}")
    console.print(f"Dry run: {args.dry_run}")
    console.print(f"Run dir → {log_dir}")
    console.print()

    cassette_ctx = nullcontext()
    if not is_local:
        import vcr

        vcr_config = vcr.VCR(
            record_mode="new_episodes",  # Record new requests, replay existing
            match_on=["method", "scheme", "host", "port", "path"],
            serializer="yaml",
        )
        cassette_ctx = vcr_config.use_cassette(str(cassette_path))

    # Record HTTP traffic only in API mode (local backend has no HTTP calls).
    with cassette_ctx:
        console.print("[dim]Setting up Tinker client...[/dim]")
        from tinker_debate.tinker_client import TinkerDebateClient

        client = await TinkerDebateClient.create()

        from tinker_debate.debate_types import TrainingDatum
        from tinker_debate.train.driver_context import DriverContext
        from tinker_debate.train.driver_factory import build_driver
        from tinker_debate.train.driver_types import TrainLogFns

        log_fns = TrainLogFns(
            save_debate_log=save_debate_log,
            save_baseline_log=save_baseline_log,
            save_summary_log=save_summary_log,
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

        all_losses: list[float] = []

        def compute_stop_token_id():
            stop = client.renderer.get_stop_sequences()
            if stop is None:
                return None
            if isinstance(stop, list):
                if len(stop) != 1:
                    return None
                stop = stop[0]
            if isinstance(stop, int):
                return int(stop)
            if not isinstance(stop, str):
                return None
            tokens = client.tokenizer.encode(stop, add_special_tokens=False)
            if len(tokens) != 1:
                return None
            return int(tokens[0])

        async def compute_replay_debug_metrics(training_data: list[TrainingDatum]) -> dict:
            prompt_tokens_batch = [d.prompt_tokens for d in training_data]
            completion_tokens_batch = [d.completion_tokens for d in training_data]
            completion_logprobs_batch = [d.completion_logprobs for d in training_data]

            new_logprobs_batch = await client.compute_completion_logprobs(
                prompt_tokens_batch=prompt_tokens_batch,
                completion_tokens_batch=completion_tokens_batch,
                completion_logprobs_batch=completion_logprobs_batch,
            )

            total_tokens = 0
            sum_old = 0.0
            sum_new = 0.0
            sum_delta = 0.0
            sum_ratio = 0.0

            stop_token_id = compute_stop_token_id()
            stop_old = []
            stop_new = []
            stop_token_logprobs_batch = None
            if stop_token_id is not None:
                stop_token_logprobs_batch = await client.compute_token_logprobs(
                    prompt_tokens_batch=prompt_tokens_batch,
                    completion_tokens_batch=completion_tokens_batch,
                    token_id=stop_token_id,
                )
            stop_token_logprob_sum = 0.0
            stop_token_logprob_count = 0

            for idx, (comp_tokens, old_lps, new_lps) in enumerate(
                zip(completion_tokens_batch, completion_logprobs_batch, new_logprobs_batch)
            ):
                if len(comp_tokens) != len(old_lps) or len(comp_tokens) != len(new_lps):
                    raise ValueError("Completion length mismatch in replay debug metrics")

                for o, n in zip(old_lps, new_lps):
                    total_tokens += 1
                    sum_old += float(o)
                    sum_new += float(n)
                    sum_delta += float(n - o)
                    sum_ratio += float(math.exp(n - o))

                if stop_token_id is not None and comp_tokens and comp_tokens[-1] == stop_token_id:
                    stop_old.append(float(old_lps[-1]))
                    stop_new.append(float(new_lps[-1]))

                if stop_token_logprobs_batch is not None:
                    stop_lps = stop_token_logprobs_batch[idx]
                    if len(stop_lps) != len(comp_tokens):
                        raise ValueError("Stop-token logprobs length mismatch in replay debug metrics")
                    for lp in stop_lps:
                        stop_token_logprob_sum += float(lp)
                        stop_token_logprob_count += 1

            metrics = {
                "mean_old_logprob": sum_old / total_tokens if total_tokens else 0.0,
                "mean_new_logprob": sum_new / total_tokens if total_tokens else 0.0,
                "mean_delta_logprob": sum_delta / total_tokens if total_tokens else 0.0,
                "mean_ratio": sum_ratio / total_tokens if total_tokens else 0.0,
                "stop_token_id": stop_token_id,
                "stop_count": len(stop_old),
                "stop_old_logprob_mean": sum(stop_old) / len(stop_old) if stop_old else None,
                "stop_new_logprob_mean": sum(stop_new) / len(stop_new) if stop_new else None,
                "stop_delta_logprob_mean": (
                    (sum(stop_new) / len(stop_new)) - (sum(stop_old) / len(stop_old))
                    if stop_old
                    else None
                ),
                "stop_token_logprob_mean": (
                    stop_token_logprob_sum / stop_token_logprob_count
                    if stop_token_logprob_count
                    else None
                ),
            }
            return metrics

        for step in range(1, args.steps + 1):
            console.rule(f"[bold]Step {step}/{args.steps}[/bold]")
            rollout_out = await driver.rollout_step(step=step)
            for line in rollout_out.info_lines:
                console.print(f"[dim]{line}[/dim]")

            training_data: list[TrainingDatum] = rollout_out.training_data
            rollout_time = float(rollout_out.rollout_time_s)

            if args.dry_run:
                console.print("[yellow]DRY RUN - skipping training[/yellow]")
                continue

            console.print("[dim]Running train step (overlapped fwd_bwd + optim_step)...[/dim]")
            t0 = time.time()

            train_result = await client.train_step(
                prompt_tokens_batch=[d.prompt_tokens for d in training_data],
                completion_tokens_batch=[d.completion_tokens for d in training_data],
                completion_logprobs_batch=[d.completion_logprobs for d in training_data],
                completion_advantages_batch=[d.completion_advantages for d in training_data],
                learning_rate=args.lr,
            )

            debug_metrics = None
            if args.replay_dir is not None:
                debug_metrics = await compute_replay_debug_metrics(training_data)
                console.print(
                    f"[dim]Replay debug: mean_delta_logprob={debug_metrics['mean_delta_logprob']:.6g} "
                    f"mean_ratio={debug_metrics['mean_ratio']:.6g} "
                    f"stop_delta_logprob_mean={debug_metrics['stop_delta_logprob_mean']} "
                    f"stop_token_logprob_mean={debug_metrics['stop_token_logprob_mean']}[/dim]"
                )

            train_time = time.time() - t0
            loss = float(train_result.get("loss", 0.0))
            num_tokens = int(train_result.get("num_tokens", 0))

            step_log_path = save_training_step_log(
                step=step,
                training_data=training_data,
                fwd_bwd_result=train_result,
                learning_rate=args.lr,
                log_dir=log_dir,
                debug_metrics=debug_metrics,
            )
            console.print(f"[dim]Logged: {step_log_path.name}[/dim]")

            console.print("[dim]Syncing weights...[/dim]")
            await client.sync_weights(f"step_{step}")

            all_losses.append(loss)

            console.print(f"\n[bold green]Step {step} complete:[/bold green]")
            console.print(f"  Loss: {loss:.4f}")
            console.print(f"  Tokens: {num_tokens}")
            console.print(
                f"  Time: rollout={rollout_time:.1f}s train={train_time:.1f}s"
            )

        console.rule("[bold]Training Summary[/bold]")

        if all_losses:
            table = Table(box=box.ROUNDED)
            table.add_column("Step", style="cyan")
            table.add_column("Loss", justify="right")

            for i, loss in enumerate(all_losses, 1):
                table.add_row(str(i), f"{loss:.4f}")

            console.print(table)
            console.print(f"\n[bold]Average loss:[/bold] {sum(all_losses)/len(all_losses):.4f}")
        else:
            console.print("[yellow]No training occurred (dry run)[/yellow]")

        console.print(f"\n[dim]View debates: python view_logs.py --log-dir {log_dir} --list[/dim]")
        console.print(f"[dim]Watch live: python view_logs.py --log-dir {log_dir} --watch[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
