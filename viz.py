"""
TUI Visualization for debate training.

Uses rich library to display tokens, masks, loss, rewards in terminal.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich import box
from typing import Any


console = Console()


def visualize_tokens_and_mask(
    tokens: list[int],
    mask: list[float],
    tokenizer,
    title: str = "Token Visualization",
    max_display: int = 50,
):
    """
    Display tokens with color-coded mask.

    Green = mask=1 (trained on)
    Gray = mask=0 (not trained on)
    """
    text = Text()

    for i, (tok, m) in enumerate(zip(tokens[:max_display], mask[:max_display])):
        decoded = tokenizer.decode([tok])
        # Escape special chars for display
        decoded = decoded.replace("\n", "\\n").replace("\t", "\\t")

        if m > 0:
            text.append(decoded, style="green bold")
        else:
            text.append(decoded, style="dim")

    if len(tokens) > max_display:
        text.append(f" ... (+{len(tokens) - max_display} more)", style="italic dim")

    panel = Panel(text, title=title, border_style="blue")
    console.print(panel)


def visualize_debate_turn(
    agent: str,
    round_num: int,
    prompt: str,
    completion: str,
    confidence: float | None,
    solution: str | None,
    reward: float | None = None,
):
    """Display a single debate turn."""
    table = Table(
        title=f"Agent {agent} - Round {round_num}",
        box=box.ROUNDED,
        show_header=False,
        width=100,
    )
    table.add_column("Field", style="cyan", width=12)
    table.add_column("Value", width=85)

    # Truncate prompt for display
    prompt_display = prompt[:200] + "..." if len(prompt) > 200 else prompt
    prompt_display = prompt_display.replace("\n", " ")

    table.add_row("Prompt", Text(prompt_display, style="dim"))
    table.add_row("Response", Text(completion[:300] + "..." if len(completion) > 300 else completion))

    if confidence is not None:
        conf_style = "green" if confidence < 50 else "yellow" if confidence < 80 else "red"
        table.add_row("Confidence", Text(f"{confidence:.1f}", style=conf_style))

    if solution is not None:
        table.add_row("Solution", Text(str(solution), style="bold"))

    if reward is not None:
        reward_style = "green" if reward > 0 else "red" if reward < 0 else "white"
        table.add_row("Reward", Text(f"{reward:.4f}", style=reward_style))

    console.print(table)


def visualize_trajectory_group(
    question: str,
    ground_truth: str | None,
    trajectories: list[dict],
    advantages: list[float],
):
    """Display a full trajectory group with all agents."""
    console.print()
    console.rule(f"[bold blue]Question[/bold blue]")
    console.print(Panel(question[:500], title="Question", border_style="blue"))

    if ground_truth:
        console.print(f"[dim]Ground truth: {ground_truth}[/dim]")

    console.print()

    # Summary table
    table = Table(title="Trajectory Summary", box=box.DOUBLE_EDGE)
    table.add_column("Agent", style="cyan")
    table.add_column("R1 Conf", justify="right")
    table.add_column("R2 Conf", justify="right")
    table.add_column("Solution", justify="center")
    table.add_column("Reward", justify="right")
    table.add_column("Advantage", justify="right")

    for traj, adv in zip(trajectories, advantages):
        agent = traj.get("agent", "?")
        r1_conf = traj.get("r1_confidence")
        r2_conf = traj.get("r2_confidence")
        solution = traj.get("r2_solution", "?")
        reward = traj.get("reward", 0)

        # Color code advantage
        adv_style = "green" if adv > 0 else "red" if adv < 0 else "white"

        table.add_row(
            agent,
            f"{r1_conf:.0f}" if r1_conf else "-",
            f"{r2_conf:.0f}" if r2_conf else "-",
            str(solution)[:20],
            f"{reward:.3f}",
            Text(f"{adv:.4f}", style=adv_style),
        )

    console.print(table)


def visualize_training_step(
    step: int,
    loss: float,
    kl: float,
    avg_reward: float,
    avg_confidence_r1: float,
    avg_confidence_r2: float,
    num_trajectories: int,
):
    """Display training step metrics."""
    table = Table(title=f"Step {step}", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Loss", f"{loss:.4f}")
    table.add_row("KL", f"{kl:.4f}")
    table.add_row("Avg Reward", f"{avg_reward:.4f}")
    table.add_row("Avg Conf R1", f"{avg_confidence_r1:.1f}")
    table.add_row("Avg Conf R2", f"{avg_confidence_r2:.1f}")
    table.add_row("Trajectories", str(num_trajectories))

    console.print(table)


def visualize_grpo_advantages(
    rewards: list[float],
    advantages: list[float],
    labels: list[str] | None = None,
):
    """Visualize GRPO advantage computation."""
    if labels is None:
        labels = [f"T{i}" for i in range(len(rewards))]

    mean_reward = sum(rewards) / len(rewards)

    table = Table(title="GRPO Advantage Computation", box=box.ROUNDED)
    table.add_column("Trajectory", style="cyan")
    table.add_column("Reward", justify="right")
    table.add_column("vs Mean", justify="right")
    table.add_column("Advantage (z-score)", justify="right")

    for label, reward, adv in zip(labels, rewards, advantages):
        diff = reward - mean_reward
        diff_style = "green" if diff > 0 else "red" if diff < 0 else "white"
        adv_style = "green bold" if adv > 0 else "red bold" if adv < 0 else "white"

        table.add_row(
            label,
            f"{reward:.4f}",
            Text(f"{diff:+.4f}", style=diff_style),
            Text(f"{adv:+.4f}", style=adv_style),
        )

    table.add_row("", "", "", "")
    table.add_row("[bold]Mean[/bold]", f"{mean_reward:.4f}", "-", "-")

    console.print(table)


def visualize_token_level_training(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    advantage: float,
    tokenizer,
    title: str = "Token-Level Training Data",
):
    """Show exactly what tokens are being trained with what advantage."""
    console.print()
    console.rule(title)

    # Show prompt (not trained)
    prompt_text = tokenizer.decode(prompt_tokens)
    console.print(Panel(
        Text(prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text, style="dim"),
        title="[dim]Prompt (mask=0, not trained)[/dim]",
        border_style="dim",
    ))

    # Show completion (trained)
    completion_text = tokenizer.decode(completion_tokens)
    adv_style = "green" if advantage > 0 else "red" if advantage < 0 else "white"
    console.print(Panel(
        Text(completion_text, style=adv_style),
        title=f"[bold]Completion (mask=1, advantage={advantage:.4f})[/bold]",
        border_style=adv_style,
    ))

    # Token count
    console.print(f"[dim]Prompt: {len(prompt_tokens)} tokens | Completion: {len(completion_tokens)} tokens[/dim]")
