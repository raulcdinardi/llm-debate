"""
Core types for tinker-style debate training.

Key design:
- 3 rounds: propose -> argue -> respond
- Solutions FROZEN after R1
- Judge declares winner (A or B)
- Rejection sampling: train on winners only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import math

from tinker_debate.prompts import load_prompt

Verdict = Literal["A", "B", "INVALID"]


@dataclass
class Transition:
    """A single turn in a debate trajectory."""

    # Tokenized prompt (includes history)
    prompt_tokens: list[int]

    # Tokenized model response
    completion_tokens: list[int]

    # Logprobs for each completion token (required for importance sampling / PPO)
    completion_logprobs: list[float]

    # 1, 2, 3
    round_num: int

    metrics: dict[str, Any] = field(default_factory=dict)

    # Raw API response (all fields) for debugging
    raw_response: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)


@dataclass
class DebateTrajectory:
    """A complete debate trajectory for one agent (R1 + R2 + R3)."""

    agent: Literal["A", "B"]
    transitions: list[Transition]  # [R1, R2, R3]
    frozen_solution: str | None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_completion_tokens(self) -> int:
        return sum(len(t.completion_tokens) for t in self.transitions)


@dataclass
class DebateResult:
    """Result of a single debate between Agent A and Agent B."""

    question: str
    ground_truth: str | None
    trajectory_a: DebateTrajectory
    trajectory_b: DebateTrajectory
    verdict: Verdict
    judge_reasoning: str
    metrics: dict[str, Any] = field(default_factory=dict)

    # Judge token info (None if mock judge used)
    judge_prompt_tokens: list[int] | None = None
    judge_completion_tokens: list[int] | None = None
    judge_completion_logprobs: list[float] | None = None
    judge_raw_response: dict[str, Any] | None = None

    def get_winner_trajectory(self) -> DebateTrajectory:
        if self.verdict == "A":
            return self.trajectory_a
        return self.trajectory_b

    def get_loser_trajectory(self) -> DebateTrajectory:
        if self.verdict == "A":
            return self.trajectory_b
        return self.trajectory_a


@dataclass
class DebateConfig:
    """Configuration for debate environment."""

    num_rounds: int = 3  # currently only 3 is supported
    # Avoid hard caps; prefer stop sequences. If you *must* cap, set an int.
    max_tokens_per_turn: int | None = None
    temperature: float = 0.8

    # KL penalty (future use)
    kl_coef: float = 0.01

    # Training
    learning_rate: float = 1e-5

    # Prompts / templates (move hardcoding out of env)
    system_propose: str = load_prompt("debate/system_propose.md")
    system_argue: str = load_prompt("debate/system_argue.md")
    system_judge: str = load_prompt("debate/system_judge.md")
    r2_user_template: str = load_prompt("debate/r2_user_template.md")
    r3_user_template: str = load_prompt("debate/r3_user_template.md")
    chat_preamble: str = ""

    @staticmethod
    def cheap(*, chat_preamble: str = "") -> "DebateConfig":
        return DebateConfig(max_tokens_per_turn=None, temperature=0.7, chat_preamble=chat_preamble)


@dataclass
class TrainingDatum:
    """A single training example (from a winning trajectory)."""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_logprobs: list[float]
    completion_advantages: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _merge_rounds_with_centered_reward(
    *,
    debate: DebateResult,
    winner: DebateTrajectory,
    reward: float,
    mean_reward: float,
    group_size: int | None,
    std_reward: float | None = None,
) -> TrainingDatum:
    """Merge a 3-round trajectory into one TrainingDatum using centered rewards."""

    if len(winner.transitions) != 3:
        raise ValueError(f"Expected 3 rounds, got {len(winner.transitions)}")

    t1, t2, t3 = winner.transitions

    for t in (t1, t2, t3):
        if len(t.completion_tokens) != len(t.completion_logprobs):
            raise ValueError(
                f"Completion/logprob length mismatch in round {t.round_num}: "
                f"{len(t.completion_tokens)} vs {len(t.completion_logprobs)}"
            )

    # Extension property checks + continuation slices
    r1_full_len = len(t1.prompt_tokens) + len(t1.completion_tokens)
    if len(t2.prompt_tokens) < r1_full_len:
        raise ValueError("R2 prompt shorter than R1 history; extension property violated.")
    r2_continuation_tokens = t2.prompt_tokens[r1_full_len:]

    r2_full_len = len(t2.prompt_tokens) + len(t2.completion_tokens)
    if len(t3.prompt_tokens) < r2_full_len:
        raise ValueError("R3 prompt shorter than R2 history; extension property violated.")
    r3_continuation_tokens = t3.prompt_tokens[r2_full_len:]

    merged_completion = (
        t1.completion_tokens
        + r2_continuation_tokens
        + t2.completion_tokens
        + r3_continuation_tokens
        + t3.completion_tokens
    )

    merged_logprobs = (
        list(t1.completion_logprobs)
        + [0.0] * len(r2_continuation_tokens)
        + list(t2.completion_logprobs)
        + [0.0] * len(r3_continuation_tokens)
        + list(t3.completion_logprobs)
    )

    total_generated_tokens = (
        len(t1.completion_tokens) + len(t2.completion_tokens) + len(t3.completion_tokens)
    )
    if total_generated_tokens <= 0:
        raise ValueError("Winner trajectory has zero generated tokens.")

    centered_reward = reward - mean_reward
    if std_reward is not None:
        if std_reward > 0:
            centered_reward = centered_reward / std_reward
        else:
            centered_reward = 0.0
    advantage_value = centered_reward / total_generated_tokens

    merged_advantages = (
        [advantage_value] * len(t1.completion_tokens)
        + [0.0] * len(r2_continuation_tokens)
        + [advantage_value] * len(t2.completion_tokens)
        + [0.0] * len(r3_continuation_tokens)
        + [advantage_value] * len(t3.completion_tokens)
    )

    return TrainingDatum(
        prompt_tokens=t1.prompt_tokens,
        completion_tokens=merged_completion,
        completion_logprobs=merged_logprobs,
        completion_advantages=merged_advantages,
        metadata={
            "question": debate.question[:100],
            "agent": winner.agent,
            "verdict": debate.verdict,
            "reward": reward,
            "centered_reward": centered_reward,
            "group_mean_reward": mean_reward,
            "group_std_reward": std_reward,
            "group_size": group_size,
            "rounds_merged": 3,
        },
    )


def _merge_rounds_with_adv_values(
    *,
    debate: DebateResult,
    traj: DebateTrajectory,
    r1_adv_value: float,
    r2_adv_value: float,
    r3_adv_value: float,
    metadata: dict[str, Any],
) -> TrainingDatum:
    if len(traj.transitions) != 3:
        raise ValueError(f"Expected 3 rounds, got {len(traj.transitions)}")

    t1, t2, t3 = traj.transitions
    for t in (t1, t2, t3):
        if len(t.completion_tokens) != len(t.completion_logprobs):
            raise ValueError(
                f"Completion/logprob length mismatch in round {t.round_num}: "
                f"{len(t.completion_tokens)} vs {len(t.completion_logprobs)}"
            )

    if r1_adv_value != 0.0 and len(t1.completion_tokens) == 0:
        raise ValueError("Non-zero R1 advantage with zero R1 completion tokens.")
    if r2_adv_value != 0.0 and len(t2.completion_tokens) == 0:
        raise ValueError("Non-zero R2 advantage with zero R2 completion tokens.")
    if r3_adv_value != 0.0 and len(t3.completion_tokens) == 0:
        raise ValueError("Non-zero R3 advantage with zero R3 completion tokens.")

    r1_full_len = len(t1.prompt_tokens) + len(t1.completion_tokens)
    if len(t2.prompt_tokens) < r1_full_len:
        raise ValueError("R2 prompt shorter than R1 history; extension property violated.")
    r2_continuation_tokens = t2.prompt_tokens[r1_full_len:]

    r2_full_len = len(t2.prompt_tokens) + len(t2.completion_tokens)
    if len(t3.prompt_tokens) < r2_full_len:
        raise ValueError("R3 prompt shorter than R2 history; extension property violated.")
    r3_continuation_tokens = t3.prompt_tokens[r2_full_len:]

    merged_completion = (
        t1.completion_tokens
        + r2_continuation_tokens
        + t2.completion_tokens
        + r3_continuation_tokens
        + t3.completion_tokens
    )

    merged_logprobs = (
        list(t1.completion_logprobs)
        + [0.0] * len(r2_continuation_tokens)
        + list(t2.completion_logprobs)
        + [0.0] * len(r3_continuation_tokens)
        + list(t3.completion_logprobs)
    )

    merged_advantages = (
        [r1_adv_value] * len(t1.completion_tokens)
        + [0.0] * len(r2_continuation_tokens)
        + [r2_adv_value] * len(t2.completion_tokens)
        + [0.0] * len(r3_continuation_tokens)
        + [r3_adv_value] * len(t3.completion_tokens)
    )

    return TrainingDatum(
        prompt_tokens=t1.prompt_tokens,
        completion_tokens=merged_completion,
        completion_logprobs=merged_logprobs,
        completion_advantages=merged_advantages,
        metadata={
            "question": debate.question[:100],
            "agent": traj.agent,
            "verdict": debate.verdict,
            **metadata,
        },
    )


def assemble_training_data_grpo(
    debates: list[DebateResult],
    reward_fn: Callable[[DebateTrajectory, DebateResult], float],
) -> list[TrainingDatum]:
    """Winner-only assembly with reward centering per question (GRPO-style).

    Args:
        debates: List of debate results (should contain multiple debates per question for grouping)
        reward_fn: Reward function (trajectory, debate) -> float. Must be explicit.
    """

    # Group winners by question.
    groups: dict[str, list[tuple[DebateTrajectory, DebateResult, float]]] = {}
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        winner = debate.get_winner_trajectory()
        reward = reward_fn(winner, debate)
        groups.setdefault(debate.question, []).append((winner, debate, reward))

    data: list[TrainingDatum] = []
    for question, group in groups.items():
        if len(group) == 0:
            continue

        rewards = [r for _, _, r in group]
        mean_reward = sum(rewards) / len(rewards)
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)
        group_size = len(group)

        for winner, debate, reward in group:
            datum = _merge_rounds_with_centered_reward(
                debate=debate,
                winner=winner,
                reward=reward,
                mean_reward=mean_reward,
                std_reward=std,
                group_size=group_size,
            )
            data.append(datum)

    return data


def assemble_training_data_r1_r23(
    debates: list[DebateResult],
    r1_reward_fn: Callable[[DebateTrajectory, DebateResult], float],
    *,
    r23_reward: float,
    r23_symmetric: bool,
) -> list[TrainingDatum]:
    """Split rewards: R1 gets task reward, R2/R3 get debate win/loss reward.

    - R1: both agents trained; reward z-scored across all R1 solutions per question.
    - R2/R3: both winner and loser trained; symmetric by default (winner=+c, loser=-c).
    """
    groups: dict[str, list[tuple[DebateTrajectory, DebateResult, float]]] = {}
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        traj_a = debate.trajectory_a
        traj_b = debate.trajectory_b
        r1_a = r1_reward_fn(traj_a, debate)
        r1_b = r1_reward_fn(traj_b, debate)
        groups.setdefault(debate.question, []).extend(
            [
                (traj_a, debate, r1_a),
                (traj_b, debate, r1_b),
            ]
        )

    data: list[TrainingDatum] = []
    for question, group in groups.items():
        if len(group) == 0:
            continue

        def _per_token_adv(reward: float, tokens: list[int], label: str) -> float:
            if len(tokens) == 0:
                raise ValueError(f"{label} completion tokens empty; cannot assign reward.")
            return reward / len(tokens)

        rewards = [r for _, _, r in group]
        mean_reward = sum(rewards) / len(rewards)
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)
        group_size = len(group)

        r23_w = float(r23_reward)
        r23_l = -float(r23_reward) if r23_symmetric else 0.0

        for traj, debate, r1_reward in group:
            if std > 0:
                r1_centered = (r1_reward - mean_reward) / std
            else:
                r1_centered = 0.0
            r1_adv = _per_token_adv(r1_centered, traj.transitions[0].completion_tokens, "R1")

            is_winner = debate.get_winner_trajectory().agent == traj.agent
            r23_reward_signed = r23_w if is_winner else r23_l

            r2_adv = _per_token_adv(r23_reward_signed, traj.transitions[1].completion_tokens, "R2")
            r3_adv = _per_token_adv(r23_reward_signed, traj.transitions[2].completion_tokens, "R3")

            data.append(
                _merge_rounds_with_adv_values(
                    debate=debate,
                    traj=traj,
                    r1_adv_value=r1_adv,
                    r2_adv_value=r2_adv,
                    r3_adv_value=r3_adv,
                    metadata={
                        "r1_reward": r1_reward,
                        "r1_centered_reward": r1_centered,
                        "r1_adv_value": r1_adv,
                        "r1_group_mean_reward": mean_reward,
                        "r1_group_std_reward": std,
                        "r1_group_size": group_size,
                        "r23_reward": r23_reward_signed,
                        "r23_symmetric": r23_symmetric,
                        "r23_adv_value": r23_reward_signed,
                        "rounds_merged": 3,
                        "r1_trained": True,
                        "r23_trained": r23_reward_signed != 0.0,
                    },
                )
            )

    return data


def compute_training_stats(debates: list[DebateResult]) -> dict[str, Any]:
    total = len(debates)
    if total == 0:
        return {"total": 0}

    a_wins = sum(1 for d in debates if d.verdict == "A")
    b_wins = sum(1 for d in debates if d.verdict == "B")
    invalid = sum(1 for d in debates if d.verdict == "INVALID")

    correct_wins = 0
    wrong_wins = 0
    accuracy_calculable = 0

    for d in debates:
        if d.ground_truth is None or d.verdict not in ("A", "B"):
            continue
        winner = d.get_winner_trajectory()
        accuracy_calculable += 1
        if winner.frozen_solution == d.ground_truth:
            correct_wins += 1
        else:
            wrong_wins += 1

    valid_total = a_wins + b_wins
    return {
        "total": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "invalid": invalid,
        "win_rate_a": a_wins / valid_total if valid_total > 0 else None,
        "win_rate_b": b_wins / valid_total if valid_total > 0 else None,
        "correct_wins": correct_wins,
        "wrong_wins": wrong_wins,
        "accuracy_from_debates": correct_wins / accuracy_calculable if accuracy_calculable > 0 else None,
    }
