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
from typing import Any, Literal


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
    system_propose: str = (
        "You are solving a problem. Provide your answer in the following format:\n\n"
        "<SOLUTION>[Your final answer]</SOLUTION>"
    )

    system_argue: str = (
        "You are in a debate. Your solution is FIXED and cannot change.\n"
        "Your goal: convince the judge your solution is correct.\n\n"
        "- Defend your reasoning\n"
        "- Point out flaws in opponent's approach\n"
        "- Be persuasive but accurate\n\n"
        "Do NOT change your solution. Only argue for it."
    )

    system_judge: str = (
        "You are a judge evaluating a debate between two solvers.\n\n"
        "Read the full transcript:\n"
        "- Round 1: Both solvers propose solutions\n"
        "- Round 2: Both argue for their solutions\n"
        "- Round 3: Both respond to criticism\n\n"
        "Evaluate:\n"
        "1. Which solution is more likely correct?\n"
        "2. Which debater made stronger arguments?\n"
        "3. Who addressed criticisms better?\n\n"
        "Output your verdict:\n"
        "<VERDICT>[A or B]</VERDICT>\n"
        "<REASONING>[Brief explanation]</REASONING>"
    )

    r2_user_template: str = (
        "Another solver proposed this solution:\n{opponent_r1}\n\n"
        "Argue why your solution is correct and critique their approach."
    )

    r3_user_template: str = (
        "They responded:\n{opponent_r2}\n\n"
        "Make your final case."
    )

    @staticmethod
    def cheap() -> "DebateConfig":
        return DebateConfig(max_tokens_per_turn=None, temperature=0.7)


@dataclass
class TrainingDatum:
    """A single training example (from a winning trajectory)."""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_logprobs: list[float]
    completion_advantages: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def assemble_training_data(debates: list[DebateResult]) -> list[TrainingDatum]:
    """Convert debate results into training data.

    Winner-only (rejection sampling). Merges all rounds into a single datum per winner using
    the extension property:

      r1_comp + r2_cont + r2_comp + r3_cont + r3_comp

    Continuation tokens get advantage=0 and logprob=0.
    """

    data: list[TrainingDatum] = []

    for debate in debates:
        # Safety: only accept A/B
        if debate.verdict not in ("A", "B"):
            continue

        winner = debate.get_winner_trajectory()

        total_model_tokens = winner.total_completion_tokens
        if total_model_tokens <= 0:
            continue

        advantage = 1.0 / total_model_tokens

        if len(winner.transitions) != 3:
            raise ValueError(f"Expected 3 rounds, got {len(winner.transitions)}")
        t1, t2, t3 = winner.transitions

        # Extract continuation tokens via extension property.
        r1_full_len = len(t1.prompt_tokens) + len(t1.completion_tokens)
        r2_continuation_tokens = t2.prompt_tokens[r1_full_len:]

        r2_full_len = len(t2.prompt_tokens) + len(t2.completion_tokens)
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
            [advantage] * len(t1.completion_tokens)
            + [0.0] * len(r2_continuation_tokens)
            + [advantage] * len(t2.completion_tokens)
            + [0.0] * len(r3_continuation_tokens)
            + [advantage] * len(t3.completion_tokens)
        )

        data.append(
            TrainingDatum(
                prompt_tokens=t1.prompt_tokens,
                completion_tokens=merged_completion,
                completion_logprobs=merged_logprobs,
                completion_advantages=merged_advantages,
                metadata={
                    "question": debate.question[:100],
                    "agent": winner.agent,
                    "verdict": debate.verdict,
                    "rounds_merged": 3,
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
