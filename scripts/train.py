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
import asyncio
import atexit
import json
import os
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

# Proxy configuration - auto-launch mitmweb (disabled by default due to SSL issues)
PROXY_PORT = 8080
WEB_PORT = 8081
USE_PROXY = "--proxy" in sys.argv  # Enable with --proxy flag

if USE_PROXY:
    sys.argv.remove("--proxy")
    PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"

    print(f"\n[mitmproxy] Starting mitmweb on port {PROXY_PORT}...")
    mitm_proc = subprocess.Popen(
        ["mitmweb", "--listen-port", str(PROXY_PORT), "--web-port", str(WEB_PORT), "--set", "ssl_insecure=true"],
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
    print("\n[mitmproxy] Disabled. Use --proxy flag to enable live HTTP inspection.\n")

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
import vcr  # noqa: E402

console = Console()

# VCR configuration for recording HTTP traffic
vcr_config = vcr.VCR(
    record_mode="new_episodes",  # Record new requests, replay existing
    match_on=["method", "scheme", "host", "port", "path"],
    serializer="yaml",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal debate training")
    parser.add_argument(
        "--task",
        type=str,
        default="debate",
        choices=["debate", "confidence"],
        help="Training task/env to run. 'debate' is multi-turn; 'confidence' is single-turn.",
    )
    parser.add_argument("-n", "--num-debates", type=int, default=2, help="Debates per step")
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
        help="(confidence only) Keep rollouts with reward >= this threshold.",
    )
    parser.add_argument(
        "--accept-require-parse",
        action="store_true",
        help="(confidence only) Only keep rollouts where confidence parsing succeeded.",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment name (logs to logs/experiments/<name>/)",
    )
    parser.add_argument("--question", "-q", default="What is 7 * 8?", help="Question (ignored if --dataset)")
    parser.add_argument("--ground-truth", "-g", default="56", help="Ground truth (ignored if --dataset)")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        choices=["gpqa_diamond", "gpqa_extended", "gpqa_main", "test"],
        help="Dataset to sample questions from (overrides --question). 'test' = simple arithmetic.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def get_log_dir(experiment_name: str | None, dataset: str | None = None) -> Path:
    base = Path("logs")
    if experiment_name:
        return base / "experiments" / experiment_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dataset:
        return base / "gpqa_runs" / dataset / timestamp
    return base / "test_runs" / timestamp


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

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_path


def save_confidence_log(*, record: dict, log_dir: Path, model_name: str | None = None) -> Path:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"confidence_{timestamp}.json"
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

    if args.seed is not None:
        random.seed(args.seed)

    log_dir = get_log_dir(args.experiment, args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    # VCR cassette for recording all HTTP traffic
    cassette_path = log_dir / "http_traffic.yaml"
    console.print(f"[dim]Recording HTTP traffic to {cassette_path}[/dim]")

    # Load dataset if specified
    dataset = None
    if args.dataset:
        console.print(f"[dim]Loading {args.dataset}...[/dim]")
        if args.dataset == "test":
            from tinker_debate.datasets import load_test_dataset
            dataset = load_test_dataset()
        else:
            from tinker_debate.datasets import load_gpqa
            dataset = load_gpqa(args.dataset, seed=args.seed)
        console.print(f"[dim]Loaded {len(dataset)} questions[/dim]")

    console.print("\n[bold cyan]TINKER TRAINING[/bold cyan]")
    console.print(f"Task: {args.task}")
    if args.task == "debate":
        if args.dataset:
            console.print(f"Dataset: {args.dataset}")
        else:
            console.print(f"Question: {args.question}")
    else:
        console.print("Dataset: confidence/questions.json")
    console.print(f"Debates per step: {args.num_debates}")
    console.print(f"Training steps: {args.steps}")
    console.print(f"Learning rate: {args.lr}")
    console.print(f"Seed: {args.seed}")
    console.print(f"Dry run: {args.dry_run}")
    console.print(f"Run dir → {log_dir}")
    console.print()

    # Record all HTTP traffic with VCR
    with vcr_config.use_cassette(str(cassette_path)):
        console.print("[dim]Setting up Tinker client...[/dim]")
        from tinker_debate.tinker_client import TinkerDebateClient

        client = await TinkerDebateClient.create()

        from tinker_debate.debate_types import TrainingDatum

        if args.task == "debate":
            from tinker_debate.debate_env import (
                DebateRolloutClient,
                DebateTokenRolloutClient,
                confidence_judge_r1,
                mock_judge_random,
                run_debate_batch_token_only,
            )
            from tinker_debate.debate_types import DebateConfig, assemble_training_data, compute_training_stats
        else:
            from tinker_debate.confidence.confidence_env import ConfidenceDataset, load_questions
            from tinker_cookbook import renderers
            import tinker

        async def generate_fn(prompts, max_tokens, temp):
            return await client.generate(prompts, max_tokens=max_tokens, temperature=temp)

        rollout_client = None
        token_rollout_client = None
        config = None
        judge_fn = None
        confidence_dataset = None
        confidence_renderer = None

        if args.task == "debate":
            rollout_client = DebateRolloutClient(generate_fn=generate_fn)

            async def sample_tokens_fn(prompt_tokens_list, max_tokens, temp):
                return await client.sample_token_prompts(
                    prompt_tokens_list=prompt_tokens_list,
                    max_tokens=max_tokens,
                    temperature=temp,
                )

            token_rollout_client = DebateTokenRolloutClient(
                sample_fn=sample_tokens_fn,
                decode_fn=lambda toks: client.tokenizer.decode(toks, skip_special_tokens=True),
            )

            config = DebateConfig.cheap()
            if args.confidence_judge:
                judge_fn = confidence_judge_r1
            else:
                judge_fn = mock_judge_random if args.mock_judge else None
        else:
            questions_confidence = load_questions()
            if len(questions_confidence) == 0:
                raise ValueError("confidence/questions.json is empty")
            confidence_renderer = renderers.get_renderer("qwen3_instruct", client.tokenizer)
            confidence_dataset = ConfidenceDataset(
                questions_confidence,
                batch_size=args.num_debates,
                group_size=1,
                renderer=confidence_renderer,
            )

        all_losses: list[float] = []

        for step in range(1, args.steps + 1):
            console.rule(f"[bold]Step {step}/{args.steps}[/bold]")

            if args.task == "debate":
                console.print(f"[dim]Running {args.num_debates} debates (batched per-round, token-only)...[/dim]")
            else:
                console.print(f"[dim]Running {args.num_debates} rollouts (single-turn confidence env)...[/dim]")
            t0 = time.time()

            training_data: list[TrainingDatum] = []
            rollout_time = 0.0

            if args.task == "debate":
                if dataset:
                    from tinker_debate.datasets import sample_questions
                    step_seed = args.seed + step if args.seed is not None else None
                    batch = sample_questions(dataset, args.num_debates, seed=step_seed)
                else:
                    batch = [(args.question, args.ground_truth) for _ in range(args.num_debates)]

                debates = await run_debate_batch_token_only(
                    batch,
                    token_rollout_client,
                    client.tokenizer,
                    config,
                    rollout_client,
                    judge_fn=judge_fn,
                )

                for r in debates:
                    save_debate_log(r, log_dir, config=config, model_name=client.model_name)

                rollout_time = time.time() - t0
                stats = compute_training_stats(debates)
                console.print(
                    f"[dim]Rollout time: {rollout_time:.1f}s | A:{stats['a_wins']} B:{stats['b_wins']} Invalid:{stats['invalid']}[/dim]"
                )

                training_data = assemble_training_data(debates)
                console.print(f"Training data: {len(training_data)} datums from {len(debates)} debates")
            else:
                assert confidence_dataset is not None
                assert confidence_renderer is not None

                batch_idx = (step - 1) % len(confidence_dataset)
                env_group_builders = confidence_dataset.get_batch(batch_idx)
                num_rollouts = len(env_group_builders)

                for builder in env_group_builders:
                    envs = await builder.make_envs()
                    if len(envs) != 1:
                        raise ValueError(f"Expected group_size=1, got {len(envs)}")
                    env = envs[0]
                    ob, stop = await env.initial_observation()
                    sampling_params = tinker.SamplingParams(
                        temperature=0.7,
                        stop=stop,
                        max_tokens=128,
                    )
                    resp = await client.sampling_client.sample_async(
                        prompt=ob,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    seq = resp.sequences[0]
                    if seq.logprobs is None:
                        raise RuntimeError("Tinker sampling did not return completion logprobs (seq.logprobs is None)")
                    completion_tokens = list(seq.tokens)
                    completion_logprobs = list(seq.logprobs)

                    step_result = await env.step(completion_tokens)

                    record = {
                        "tags": builder.logging_tags(),
                        "prompt_tokens": ob.to_ints(),
                        "completion_tokens": completion_tokens,
                        "completion_logprobs": completion_logprobs,
                        "reward": step_result.reward,
                        "metrics": step_result.metrics,
                    }
                    save_confidence_log(record=record, log_dir=log_dir, model_name=client.model_name)

                    if args.accept_require_parse and float(step_result.metrics["parse_success"]) != 1.0:
                        continue
                    if float(step_result.reward) < args.accept_min_reward:
                        continue
                    if len(completion_tokens) == 0:
                        continue

                    advantage = float(step_result.reward) / len(completion_tokens)
                    training_data.append(
                        TrainingDatum(
                            prompt_tokens=ob.to_ints(),
                            completion_tokens=completion_tokens,
                            completion_logprobs=completion_logprobs,
                            completion_advantages=[advantage] * len(completion_tokens),
                            metadata={
                                "task": "confidence",
                                "tags": builder.logging_tags(),
                                "parse_success": float(step_result.metrics["parse_success"]),
                                "reward": float(step_result.reward),
                            },
                        )
                    )

                rollout_time = time.time() - t0
                console.print(f"[dim]Rollout time: {rollout_time:.1f}s[/dim]")
                console.print(f"Training data: {len(training_data)} datums (accepted) from {num_rollouts} rollouts")

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

            train_time = time.time() - t0
            loss = float(train_result.get("loss", 0.0))
            num_tokens = int(train_result.get("num_tokens", 0))

            step_log_path = save_training_step_log(
                step=step,
                training_data=training_data,
                fwd_bwd_result=train_result,
                learning_rate=args.lr,
                log_dir=log_dir,
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
