#!/usr/bin/env python3
"""Interactive TUI viewer for debate logs.

Usage:
    python view_logs.py                     # Interactive menu
    python view_logs.py --list              # List all logs (non-interactive)
    python view_logs.py --watch             # Live reload (watch for new logs)
    python view_logs.py logs/debate_xxx.json  # View specific log directly
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy output from libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv

load_dotenv(verbose=False)

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

console = Console()


def clear_screen_full():
    """Clear screen AND scrollback buffer."""
    print("\033[2J\033[3J\033[H", end="", flush=True)


# Lazy-loaded tokenizer for token mode
_tokenizer = None
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def get_tokenizer():
    """Lazy load tokenizer directly from HuggingFace."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    return _tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="View debate logs")
    parser.add_argument("file", nargs="?", help="Specific log file to view")
    parser.add_argument("--list", "-l", action="store_true", help="List all logs (non-interactive)")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch for new logs (live reload)")
    parser.add_argument("--tokens", "-t", action="store_true", help="Show token visualization (color=logprob)")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--interval", type=float, default=2.0, help="Watch interval in seconds")
    return parser.parse_args()


def get_log_files(log_dir: str, file_type: str = "debate", recursive: bool = False) -> list[Path]:
    """Get log files sorted by modification time (newest first)."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    prefix = "**/" if recursive else ""

    if file_type == "debate":
        pattern = f"{prefix}debate_*.json"
    elif file_type == "training":
        pattern = f"{prefix}training_step_*.json"
    elif file_type == "all":
        debates = list(log_path.glob(f"{prefix}debate_*.json"))
        training = list(log_path.glob(f"{prefix}training_step_*.json"))
        return sorted(debates + training, key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        pattern = f"{prefix}{file_type}_*.json"

    return sorted(log_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def get_training_runs(log_dir: str) -> list[Path]:
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    runs = []
    for item in log_path.iterdir():
        if item.is_dir() and list(item.glob("**/debate_*.json")):
            runs.append(item)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def get_logprob_style(logprob: float) -> str:
    """Green=high prob, red=low prob."""
    import math

    prob = math.exp(logprob)
    if prob > 0.9:
        return "bright_green"
    if prob > 0.7:
        return "green"
    if prob > 0.5:
        return "yellow"
    if prob > 0.3:
        return "rgb(255,165,0)"
    if prob > 0.1:
        return "red"
    return "bright_red"


def visualize_token_mask_with_separators(
    tokenizer,
    title: str,
    *,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float] | None = None,
):
    text = Text()

    text.append("║ PROMPT (not trained) ║\n", style="bold blue")
    for i, tok in enumerate(prompt_tokens):
        decoded = tokenizer.decode([tok])
        has_newline = "\n" in decoded
        decoded = decoded.replace("\n", "↵\n").replace("\t", "→")
        text.append(decoded, style="dim")
        if i < len(prompt_tokens) - 1 and not has_newline:
            text.append("│", style="dim blue")

    text.append("\n║ COMPLETION (trained) ║\n", style="bold yellow")

    for i, tok in enumerate(completion_tokens):
        decoded = tokenizer.decode([tok])
        has_newline = "\n" in decoded
        decoded = decoded.replace("\n", "↵\n").replace("\t", "→")

        if completion_logprobs and i < len(completion_logprobs):
            style = get_logprob_style(completion_logprobs[i])
        else:
            style = "green"

        text.append(decoded, style=style)
        if i < len(completion_tokens) - 1 and not has_newline:
            text.append("│", style="dim")

    stats = f"prompt={len(prompt_tokens)} │ completion={len(completion_tokens)}"
    if completion_logprobs:
        import math

        avg_lp = sum(completion_logprobs) / len(completion_logprobs)
        avg_prob = math.exp(avg_lp) * 100
        stats += f" │ avg_prob={avg_prob:.1f}%"

    console.print(
        Panel(
            text,
            title=f"[cyan]{title}[/cyan]",
            subtitle=f"[dim]{stats}[/dim]",
            border_style="blue",
        )
    )


def visualize_training_datum_with_advantages(
    tokenizer,
    title: str,
    *,
    prompt_tokens: list[int] | None,
    target_tokens: list[int],
    advantages: list[float],
    sampling_logprobs: list[float] | None = None,
) -> None:
    """Visualize a training datum using the *advantages mask*.

    - prompt_tokens is optional, but when provided we can distinguish:
      - prompt tokens (mask=0, before prompt_len-1)
      - injected/continuation tokens (mask=0, after prompt_len-1)
    - target_tokens/advantages are aligned (same length).
    """

    prompt_len_minus_1 = None
    if prompt_tokens is not None:
        prompt_len_minus_1 = max(0, len(prompt_tokens) - 1)

    text = Text()

    # Legend
    text.append("Legend: ", style="dim")
    text.append("prompt(mask=0)", style="dim blue")
    text.append(" │ ", style="dim")
    text.append("injected(mask=0)", style="dim yellow")
    text.append(" │ ", style="dim")
    text.append("trained(mask>0)", style="green")
    text.append("\n\n", style="dim")

    n = min(len(target_tokens), len(advantages))
    if sampling_logprobs is not None:
        n = min(n, len(sampling_logprobs))

    for i in range(n):
        tok = target_tokens[i]
        adv = advantages[i]
        lp = sampling_logprobs[i] if sampling_logprobs is not None else None

        decoded = tokenizer.decode([tok])
        has_newline = "\n" in decoded
        decoded = decoded.replace("\n", "↵\n").replace("\t", "→")

        if adv != 0.0:
            # trained token
            style = get_logprob_style(lp) if lp is not None else "green"
        else:
            # masked token: prompt vs injected continuation
            if prompt_len_minus_1 is not None and i >= prompt_len_minus_1:
                style = "dim yellow"
            else:
                style = "dim blue"

        text.append(decoded, style=style)
        if i < n - 1 and not has_newline:
            text.append("│", style="dim")

    num_trained = sum(1 for a in advantages[:n] if a != 0.0)
    num_masked = n - num_trained
    subtitle = f"len={n} │ trained={num_trained} │ masked={num_masked}"

    console.print(
        Panel(
            text,
            title=f"[cyan]{title}[/cyan]",
            subtitle=f"[dim]{subtitle}[/dim]",
            border_style="blue",
        )
    )


def render_log(log_data: dict, show_tokens: bool = False) -> None:
    console.clear()

    tokenizer = get_tokenizer() if show_tokens else None

    console.print(
        Panel.fit(
            f"[bold cyan]DEBATE LOG VIEWER[/bold cyan]\n"
            f"[dim]{log_data.get('timestamp', 'unknown')}[/dim]",
            box=box.DOUBLE,
        )
    )

    question = log_data.get("question", "?")
    ground_truth = log_data.get("ground_truth", "?")
    verdict = log_data.get("verdict", "?")

    verdict_color = {"A": "blue", "B": "green", "TIE": "yellow"}.get(verdict, "white")

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[dim]Ground truth: {ground_truth}[/dim]")
    console.print(
        Panel(
            f"[bold {verdict_color}]VERDICT: {verdict}[/bold {verdict_color}]",
            border_style=verdict_color,
        )
    )
    console.print(f"[dim]Judge: {log_data.get('judge_reasoning', '')}[/dim]")

    # Show judge token info if available
    judge_data = log_data.get("judge", {})
    if judge_data and judge_data.get("prompt_token_count"):
        console.print(
            f"[dim]Judge tokens: {judge_data.get('prompt_token_count', 0)} prompt + "
            f"{judge_data.get('completion_token_count', 0)} completion[/dim]"
        )

    agent_a = log_data.get("agent_a", {})
    agent_b = log_data.get("agent_b", {})

    for round_name, key in [("R1: PROPOSE", "r1"), ("R2: ARGUE", "r2"), ("R3: RESPOND", "r3")]:
        console.rule(f"[bold]{round_name}[/bold]")
        panel_a = Panel(Text(agent_a.get(key, "")), title="[blue]Agent A[/blue]", border_style="blue")
        panel_b = Panel(Text(agent_b.get(key, "")), title="[green]Agent B[/green]", border_style="green")
        console.print(Columns([panel_a, panel_b], equal=True, expand=True))

    sol_a = agent_a.get("frozen_solution", "?")
    sol_b = agent_b.get("frozen_solution", "?")
    console.print(f"\n[bold]Frozen Solutions:[/bold] A=[cyan]{sol_a}[/cyan]  B=[green]{sol_b}[/green]")

    console.rule("[bold]Token Counts (from API)[/bold]")

    def get_token_counts(agent_data):
        transitions = agent_data.get("transitions")
        if transitions:
            # Use explicit counts if available, otherwise compute from token lists
            return {
                f"r{t['round']}": {
                    "prompt": t.get("prompt_token_count", len(t.get("prompt_tokens", []))),
                    "completion": t.get("completion_token_count", len(t.get("completion_tokens", []))),
                }
                for t in transitions
            }
        return agent_data.get("token_counts", {})

    tc_a = get_token_counts(agent_a)
    tc_b = get_token_counts(agent_b)

    table = Table(box=box.SIMPLE)
    table.add_column("Round", style="cyan")
    table.add_column("Agent A (prompt + completion)", justify="right")
    table.add_column("Agent B (prompt + completion)", justify="right")
    table.add_column("Extension Check", justify="center")

    # Track for extension verification
    prev_total_a = 0
    prev_total_b = 0

    for round_key in ["r1", "r2", "r3"]:
        a_counts = tc_a.get(round_key, {})
        b_counts = tc_b.get(round_key, {})

        a_prompt = a_counts.get("prompt", 0)
        a_comp = a_counts.get("completion", 0)
        b_prompt = b_counts.get("prompt", 0)
        b_comp = b_counts.get("completion", 0)

        # Check if this round's prompt equals previous round's total (extension property)
        if round_key == "r1":
            ext_check = "[dim]-[/dim]"
        else:
            # R2 prompt should = R1 prompt + R1 completion + continuation tokens
            # R3 prompt should = R2 prompt + R2 completion + continuation tokens
            # Simplified check: prompt should be > prev_total
            a_ok = a_prompt > prev_total_a
            b_ok = b_prompt > prev_total_b
            if a_ok and b_ok:
                ext_check = "[green]OK[/green]"
            else:
                ext_check = f"[red]A:{a_ok} B:{b_ok}[/red]"

        prev_total_a = a_prompt + a_comp
        prev_total_b = b_prompt + b_comp

        table.add_row(
            round_key.upper(),
            f"{a_prompt} + {a_comp} = {a_prompt + a_comp}",
            f"{b_prompt} + {b_comp} = {b_prompt + b_comp}",
            ext_check,
        )

    # Add judge row if available
    judge_data = log_data.get("judge", {})
    if judge_data and judge_data.get("prompt_token_count"):
        j_prompt = judge_data.get("prompt_token_count", 0)
        j_comp = judge_data.get("completion_token_count", 0)
        table.add_row(
            "JUDGE",
            f"{j_prompt} + {j_comp} = {j_prompt + j_comp}",
            "[dim]-[/dim]",
            "[dim]-[/dim]",
        )

    console.print(table)

    console.rule("[bold]Training Status[/bold]")

    # New code should never emit TIE, but keep backward compatibility for old logs.
    is_winner_a = verdict == "A"
    is_winner_b = verdict == "B"
    is_tie = verdict == "TIE"

    correct_a = sol_a == ground_truth
    correct_b = sol_b == ground_truth

    status_table = Table(box=box.ROUNDED)
    status_table.add_column("Agent")
    status_table.add_column("Solution")
    status_table.add_column("Correct?")
    status_table.add_column("Training")

    def train_status(is_winner, is_tie):
        if is_tie:
            return Text("REJECT (tie)", style="yellow")
        if is_winner:
            return Text("TRAIN", style="green bold")
        return Text("REJECT (lost)", style="red")

    status_table.add_row("A", str(sol_a), "[green]Yes[/green]" if correct_a else "[red]No[/red]", train_status(is_winner_a, is_tie))
    status_table.add_row("B", str(sol_b), "[green]Yes[/green]" if correct_b else "[red]No[/red]", train_status(is_winner_b, is_tie))
    console.print(status_table)

    if show_tokens and tokenizer:
        console.rule("[bold]Token Masks (Winner Only)[/bold]")

        if is_tie:
            console.print("[yellow]TIE log - no single winner, showing Agent A for reference[/yellow]")
            winner_agent = agent_a
            winner_label = "A (rejected)"
        elif is_winner_a:
            winner_agent = agent_a
            winner_label = "A (TRAINED)"
        else:
            winner_agent = agent_b
            winner_label = "B (TRAINED)"

        transitions = winner_agent.get("transitions")
        if not transitions:
            console.print("[red]No raw token data in log.[/red]")
        else:
            round_names = {1: "PROPOSE", 2: "ARGUE", 3: "RESPOND"}
            for t in transitions:
                visualize_token_mask_with_separators(
                    tokenizer,
                    f"R{t['round']} {round_names.get(t['round'], '?')} - Agent {winner_label}",
                    prompt_tokens=t["prompt_tokens"],
                    completion_tokens=t["completion_tokens"],
                    completion_logprobs=t.get("completion_logprobs"),
                )


def render_training_step(log_data: dict, show_tokens: bool = False) -> None:
    console.clear()
    tokenizer = get_tokenizer() if show_tokens else None

    console.print(
        Panel.fit(
            f"[bold cyan]TRAINING STEP LOG[/bold cyan]\n"
            f"[dim]Step {log_data.get('step', '?')} - {log_data.get('timestamp', 'unknown')}[/dim]",
            box=box.DOUBLE,
        )
    )

    results = log_data.get("results", {})
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Loss: {results.get('loss', '?')}")
    console.print(f"  Num tokens: {results.get('num_tokens', '?')}")
    console.print(f"  Num trained tokens: {results.get('num_trained_tokens', '?')}")
    console.print(f"  Learning rate: {log_data.get('learning_rate', '?')}")

    datums = log_data.get("datums", [])
    console.print(f"\n[bold]Datums:[/bold] {len(datums)}")

    for i, datum in enumerate(datums):
        source = datum.get("source", {})
        console.rule(f"[bold]Datum {i+1}: Agent {source.get('agent', '?')} R{source.get('round', '?')}[/bold]")

        num_prompt = sum(1 for a in datum.get("advantages", []) if a == 0.0)
        num_completion = sum(1 for a in datum.get("advantages", []) if a != 0.0)

        console.print(f"  Question: {source.get('question', '?')[:50]}")
        console.print(f"  Verdict: {source.get('verdict', '?')}")
        console.print(f"  Tokens: {num_prompt} prompt + {num_completion} completion")

        if show_tokens and tokenizer:
            # Prefer advantage-mask visualization if available (this correctly masks injected continuation tokens).
            advantages = datum.get("advantages")
            target_tokens = datum.get("target_tokens")
            if isinstance(advantages, list) and isinstance(target_tokens, list) and advantages and target_tokens:
                visualize_training_datum_with_advantages(
                    tokenizer,
                    f"Datum {i+1}",
                    prompt_tokens=datum.get("prompt_tokens"),
                    target_tokens=target_tokens,
                    advantages=advantages,
                    sampling_logprobs=datum.get("sampling_logprobs"),
                )
            else:
                # Fallback: simple prompt vs completion split (less accurate for merged datums).
                visualize_token_mask_with_separators(
                    tokenizer,
                    f"Datum {i+1}",
                    prompt_tokens=datum.get("prompt_tokens", []),
                    completion_tokens=datum.get("completion_tokens", []),
                    completion_logprobs=None,
                )


def compute_run_stats(log_files: list[Path]) -> dict:
    stats = {"total": len(log_files), "a_wins": 0, "b_wins": 0, "ties": 0, "correct_wins": 0, "wrong_wins": 0}

    for f in log_files:
        with open(f) as fp:
            data = json.load(fp)

        verdict = data.get("verdict", "?")
        if verdict == "A":
            stats["a_wins"] += 1
            winner_sol = data.get("agent_a", {}).get("frozen_solution")
        elif verdict == "B":
            stats["b_wins"] += 1
            winner_sol = data.get("agent_b", {}).get("frozen_solution")
        elif verdict == "TIE":
            stats["ties"] += 1
            winner_sol = None
        else:
            winner_sol = None

        gt = data.get("ground_truth")
        if winner_sol and gt:
            if winner_sol == gt:
                stats["correct_wins"] += 1
            else:
                stats["wrong_wins"] += 1

    return stats


def list_logs_table(log_dir: str) -> list[tuple[Path, dict]]:
    files = get_log_files(log_dir)
    result = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        result.append((f, data))
    return result


def show_debate_browser(log_dir: str, show_tokens: bool = False):
    logs = list_logs_table(log_dir)
    if not logs:
        console.print(f"[yellow]No logs found in {log_dir}/[/yellow]")
        return

    current_idx = 0
    first_render = True

    while True:
        if first_render:
            clear_screen_full()
            first_render = False
        else:
            console.clear()

        console.print(
            Panel.fit(
                "[bold cyan]DEBATE BROWSER[/bold cyan]\n" f"[dim]{len(logs)} debates in {log_dir}/[/dim]",
                box=box.DOUBLE,
            )
        )

        table = Table(box=box.ROUNDED, highlight=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Timestamp", style="cyan", width=20)
        table.add_column("Question", width=25)
        table.add_column("Verdict", justify="center", width=7)
        table.add_column("Correct?", justify="center", width=8)
        table.add_column("LP", justify="center", width=3)

        start = max(0, current_idx - 5)
        end = min(len(logs), current_idx + 10)

        for i in range(start, end):
            f, data = logs[i]
            question = data.get("question", "?")[:23]
            verdict = data.get("verdict", "?")
            verdict_color = {"A": "blue", "B": "green", "TIE": "yellow"}.get(verdict, "white")

            gt = data.get("ground_truth")
            winner_agent = "agent_a" if verdict == "A" else "agent_b" if verdict == "B" else None
            if winner_agent and gt:
                winner_sol = data.get(winner_agent, {}).get("frozen_solution")
                correct = "[green]✓[/green]" if winner_sol == gt else "[red]✗[/red]"
            else:
                correct = "-"

            agent_data = data.get("agent_a", {})
            transitions = agent_data.get("transitions", [])
            has_lp = transitions and len(transitions) > 0 and "completion_logprobs" in transitions[0]
            lp_indicator = "[green]✓[/green]" if has_lp else "[dim]-[/dim]"

            row_style = "bold reverse" if i == current_idx else None
            table.add_row(
                str(i + 1),
                data.get("timestamp", "?")[:20],
                question,
                Text(verdict, style=verdict_color),
                correct,
                lp_indicator,
                style=row_style,
            )

        console.print(table)
        console.print(
            "\n[dim]Navigation:[/dim] [cyan]j/↓[/cyan] down  [cyan]k/↑[/cyan] up  [cyan]Enter[/cyan] view  [cyan]#[/cyan] jump"
        )
        console.print("[dim]View options:[/dim] [cyan]t[/cyan] toggle tokens (color=logprob)")
        console.print("[dim]Actions:[/dim] [cyan]b[/cyan] back to menu  [cyan]q[/cyan] quit")

        if show_tokens:
            console.print("[dim]Current:[/dim] [green]tokens:ON[/green]")

        try:
            choice = Prompt.ask("\n[bold]>>[/bold]", default="")
        except (KeyboardInterrupt, EOFError):
            return

        choice = choice.strip().lower()

        if choice in ("q", "quit"):
            sys.exit(0)
        if choice in ("b", "back"):
            return
        if choice in ("j", "down"):
            current_idx = min(current_idx + 1, len(logs) - 1)
            continue
        if choice in ("k", "up"):
            current_idx = max(current_idx - 1, 0)
            continue
        if choice == "t":
            show_tokens = not show_tokens
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(logs):
                current_idx = idx
            continue

        f, data = logs[current_idx]
        render_log(data, show_tokens)
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_training_step_browser(log_dir: str, show_tokens: bool = False):
    logs = []
    for f in get_log_files(log_dir, "training"):
        with open(f) as fp:
            data = json.load(fp)
        logs.append((f, data))

    if not logs:
        console.print(f"[yellow]No training step logs found in {log_dir}/[/yellow]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    current_idx = 0
    first_render = True

    while True:
        if first_render:
            clear_screen_full()
            first_render = False
        else:
            console.clear()

        console.print(
            Panel.fit(
                "[bold cyan]TRAINING STEP BROWSER[/bold cyan]\n" f"[dim]{len(logs)} training steps in {log_dir}/[/dim]",
                box=box.DOUBLE,
            )
        )

        table = Table(box=box.ROUNDED, highlight=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Step", style="cyan", width=6)
        table.add_column("Timestamp", width=20)
        table.add_column("Datums", justify="right", width=7)
        table.add_column("Loss", justify="right", width=10)
        table.add_column("Trained Toks", justify="right", width=12)

        start = max(0, current_idx - 5)
        end = min(len(logs), current_idx + 10)

        for i in range(start, end):
            f, data = logs[i]
            results = data.get("results", {})
            loss = results.get("loss", 0)
            loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)

            row_style = "bold reverse" if i == current_idx else None
            table.add_row(
                str(i + 1),
                str(data.get("step", "?")),
                data.get("timestamp", "?")[:20],
                str(data.get("num_datums", len(data.get("datums", [])))),
                loss_str,
                str(results.get("num_trained_tokens", "?")),
                style=row_style,
            )

        console.print(table)

        console.print(
            "\n[dim]Navigation:[/dim] [cyan]j/↓[/cyan] down  [cyan]k/↑[/cyan] up  [cyan]Enter[/cyan] view  [cyan]#[/cyan] jump"
        )
        console.print("[dim]View options:[/dim] [cyan]t[/cyan] toggle tokens (color=logprob)")
        console.print("[dim]Actions:[/dim] [cyan]b[/cyan] back  [cyan]q[/cyan] quit")

        if show_tokens:
            console.print("[dim]Current:[/dim] [green]tokens:ON[/green]")

        try:
            choice = Prompt.ask("\n[bold]>>[/bold]", default="")
        except (KeyboardInterrupt, EOFError):
            return

        choice = choice.strip().lower()

        if choice in ("q", "quit"):
            sys.exit(0)
        if choice in ("b", "back"):
            return
        if choice in ("j", "down"):
            current_idx = min(current_idx + 1, len(logs) - 1)
            continue
        if choice in ("k", "up"):
            current_idx = max(current_idx - 1, 0)
            continue
        if choice == "t":
            show_tokens = not show_tokens
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(logs):
                current_idx = idx
            continue

        f, data = logs[current_idx]
        render_training_step(data, show_tokens)
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_training_runs_browser(log_dir: str):
    runs = get_training_runs(log_dir)

    if not runs:
        console.print(f"[yellow]No training runs found in {log_dir}/[/yellow]")
        console.print("[dim]Training runs are subdirectories containing debate_*.json files[/dim]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    first_render = True
    while True:
        if first_render:
            clear_screen_full()
            first_render = False
        else:
            console.clear()

        console.print(
            Panel.fit(
                "[bold cyan]TRAINING RUNS BROWSER[/bold cyan]\n" f"[dim]{len(runs)} runs in {log_dir}/[/dim]",
                box=box.DOUBLE,
            )
        )

        table = Table(box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column("Run Name", style="cyan", width=30)
        table.add_column("Debates", justify="right", width=8)
        table.add_column("A Wins", justify="right", width=8)
        table.add_column("B Wins", justify="right", width=8)
        table.add_column("Ties", justify="right", width=8)
        table.add_column("Win Accuracy", justify="right", width=12)

        run_stats = []
        for i, run_path in enumerate(runs):
            logs = list(run_path.glob("**/debate_*.json"))
            stats = compute_run_stats(logs)
            run_stats.append((run_path, stats))

            total_decided = stats["correct_wins"] + stats["wrong_wins"]
            accuracy = f"{stats['correct_wins']/total_decided*100:.1f}%" if total_decided > 0 else "-"
            accuracy_style = (
                "green" if total_decided > 0 and stats["correct_wins"] / total_decided > 0.5 else "red" if total_decided > 0 else "dim"
            )

            table.add_row(
                str(i + 1),
                run_path.relative_to(log_dir).as_posix(),
                str(stats["total"]),
                str(stats["a_wins"]),
                str(stats["b_wins"]),
                str(stats["ties"]),
                Text(accuracy, style=accuracy_style),
            )

        console.print(table)
        console.print("\n[dim]Commands:[/dim] [cyan]#[/cyan] view run details  [cyan]b[/cyan] back  [cyan]q[/cyan] quit")

        try:
            choice = Prompt.ask("\n[bold]>>[/bold]", default="b")
        except (KeyboardInterrupt, EOFError):
            return

        choice = choice.strip().lower()

        if choice in ("q", "quit"):
            sys.exit(0)
        if choice in ("b", "back"):
            return
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                run_path, _ = run_stats[idx]
                show_debate_browser(str(run_path))


def show_main_menu(log_dir: str, show_tokens: bool = False):
    clear_screen_full()
    while True:
        console.clear()

        debate_logs = get_log_files(log_dir, "debate")
        training_logs = get_log_files(log_dir, "training")
        runs = get_training_runs(log_dir)

        console.print(
            Panel.fit(
                "[bold cyan]DEBATE LOG VIEWER[/bold cyan]\n"
                f"[dim]{len(debate_logs)} debates │ {len(training_logs)} training steps │ {len(runs)} runs[/dim]",
                box=box.DOUBLE,
            )
        )

        console.print("\n[bold]Main Menu[/bold]\n")
        console.print("  [cyan][1][/cyan] Browse debates (inference logs)")
        console.print("  [cyan][2][/cyan] Browse training steps")
        console.print("  [cyan][3][/cyan] Browse training runs (directories)")
        console.print("  [cyan][4][/cyan] Watch for new logs (live)")
        console.print("  [cyan][q][/cyan] Quit")

        try:
            choice = Prompt.ask("\n[bold]Select[/bold]", choices=["1", "2", "3", "4", "q"], default="1")
        except (KeyboardInterrupt, EOFError):
            return

        if choice == "1":
            show_debate_browser(log_dir, show_tokens)
        elif choice == "2":
            show_training_step_browser(log_dir, show_tokens)
        elif choice == "3":
            show_training_runs_browser(log_dir)
        elif choice == "4":
            console.print("[cyan]Watching for new logs (Ctrl+C to stop)...[/cyan]")
            try:
                watch_logs(log_dir, 2.0, show_tokens)
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped watching.[/yellow]")
        elif choice == "q":
            return


def list_logs(log_dir: str) -> None:
    files = get_log_files(log_dir)
    if not files:
        console.print(f"[yellow]No logs found in {log_dir}/[/yellow]")
        return

    table = Table(title=f"Debate Logs ({log_dir}/)", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Question", width=30)
    table.add_column("Verdict", justify="center")
    table.add_column("Winner Correct?", justify="center")

    for i, f in enumerate(files):
        with open(f) as fp:
            data = json.load(fp)

        question = data.get("question", "?")[:30]
        verdict = data.get("verdict", "?")
        verdict_color = {"A": "blue", "B": "green", "TIE": "yellow"}.get(verdict, "white")

        gt = data.get("ground_truth")
        winner_agent = "agent_a" if verdict == "A" else "agent_b" if verdict == "B" else None
        if winner_agent and gt:
            winner_sol = data.get(winner_agent, {}).get("frozen_solution")
            correct = "[green]Yes[/green]" if winner_sol == gt else "[red]No[/red]"
        else:
            correct = "-"

        table.add_row(str(i + 1), f.name, question, Text(verdict, style=verdict_color), correct)

    console.print(table)
    console.print(f"\n[dim]View with: python view_logs.py {files[0]}[/dim]")


def watch_logs(log_dir: str, interval: float, show_tokens: bool) -> None:
    console.print(f"[cyan]Watching {log_dir}/ for new logs (Ctrl+C to stop)...[/cyan]")

    last_file = None
    last_mtime = 0

    while True:
        files = get_log_files(log_dir, file_type="all")

        if files:
            newest = files[0]
            mtime = newest.stat().st_mtime

            if newest != last_file or mtime > last_mtime:
                last_file = newest
                last_mtime = mtime

                with open(newest) as f:
                    data = json.load(f)

                if "training_step" in newest.name or "datums" in data:
                    render_training_step(data, show_tokens)
                else:
                    render_log(data, show_tokens)
                console.print(f"\n[dim]Watching... (last update: {newest.name})[/dim]")
        else:
            console.print(f"[yellow]No logs yet in {log_dir}/[/yellow]")

        time.sleep(interval)


def main():
    args = parse_args()

    if os.environ.get("VIRTUAL_ENV") is None:
        console.print(
            "[yellow]Warning: not running inside a virtualenv. "
            "Activate one (e.g., `source venv/bin/activate`) to ensure dependencies are available.[/yellow]"
        )

    if args.list:
        list_logs(args.log_dir)
        return

    if args.watch:
        try:
            watch_logs(args.log_dir, args.interval, args.tokens)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped watching.[/yellow]")
        return

    if args.file:
        log_file = Path(args.file)
        if not log_file.exists():
            console.print(f"[red]File not found: {log_file}[/red]")
            return

        with open(log_file) as f:
            data = json.load(f)

        if "training_step" in log_file.name or "datums" in data:
            render_training_step(data, args.tokens)
        else:
            render_log(data, args.tokens)
        return

    try:
        show_main_menu(args.log_dir, args.tokens)
    except KeyboardInterrupt:
        console.print("\n[yellow]Bye![/yellow]")


if __name__ == "__main__":
    main()
