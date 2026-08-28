from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Literal

from llm_local_rl.prompts import load_prompt
from llm_local_rl.types import TrainExample

Verdict = Literal["A", "B", "INVALID"]

_BASE_R2_HEADER = "The reasons that my solution is better than my opponent's are:\n1)"
_BASE_R3_HEADER = "Responding to my opponent's criticism:\n1)"
_DEBATE_POINT_RE = re.compile(r"(?m)^\s*([1-9])\)")
_DEBATE_CONCLUDED_RE = re.compile(r"(?m)^\s*CONCLUDED\s*$")
_LEGACY_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_LEGACY_THINK_TAIL_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_LEGACY_CLOSING_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
_LEGACY_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_LEGACY_NUMBERED_SENTENCE_RE = re.compile(r"^\s*\d+\)")
_EOS_LITERAL = "<|endoftext|>"


def _legacy_numbered_completion_would_truncate(text: str) -> bool:
    """Return whether the removed runtime truncator would have shortened ``text``."""
    clean = _LEGACY_THINK_BLOCK_RE.sub("", text)
    clean = _LEGACY_THINK_TAIL_RE.sub("", clean)
    clean = _LEGACY_CLOSING_THINK_RE.sub("", clean)
    clean = clean.strip().replace(_EOS_LITERAL, "").split("CONCLUDED", 1)[0].strip()
    if not clean:
        return False
    kept_end = 0
    raw_start = 0
    boundaries = [match.end() for match in _LEGACY_SENTENCE_BOUNDARY_RE.finditer(clean)]
    boundaries.append(len(clean))
    for raw_end in boundaries:
        raw_segment = clean[raw_start:raw_end]
        if raw_segment.strip():
            right_trimmed_segment = raw_segment.rstrip()
            is_numbered = kept_end == 0 or bool(
                _LEGACY_NUMBERED_SENTENCE_RE.match(right_trimmed_segment)
            )
            if is_numbered:
                kept_end = raw_start + len(right_trimmed_segment)
            else:
                break
        raw_start = raw_end
    truncated = clean[:kept_end].strip()
    return bool(truncated and truncated != clean)


def audit_base_text_debate_format(*, text: str, round_num: int) -> dict[str, Any]:
    """Audit the exact visible R2/R3 contract used by the base-text harness.

    The canonical header and ``1)`` are prompt-side prefill.  They are checked
    to ensure the expected harness was used, but only sampled completion tokens
    receive advantages, so no reward is assigned to the prefill itself.
    """
    if round_num not in (2, 3):
        raise ValueError(f"round_num must be 2 or 3, got {round_num!r}")
    header = _BASE_R2_HEADER if round_num == 2 else _BASE_R3_HEADER
    visible = text or ""
    markers = list(_DEBATE_POINT_RE.finditer(visible))
    marker_numbers = [int(match.group(1)) for match in markers]
    conclusions = list(_DEBATE_CONCLUDED_RE.finditer(visible))
    header_ok = visible.startswith(header)
    completion_text = visible[len(header):] if header_ok else visible
    would_have_truncated = _legacy_numbered_completion_would_truncate(completion_text)
    numbering_ok = marker_numbers == [1, 2, 3]
    concluded_once = len(conclusions) == 1
    concluded_terminal = concluded_once and not visible[conclusions[0].end():].strip()
    nonempty_points = False
    if numbering_ok:
        end = conclusions[0].start() if concluded_once else len(visible)
        spans = [
            visible[markers[index].end():(markers[index + 1].start() if index < 2 else end)].strip()
            for index in range(3)
        ]
        nonempty_points = all(re.search(r"[A-Za-z0-9]", span) for span in spans)
    strict_ok = (
        header_ok
        and numbering_ok
        and nonempty_points
        and concluded_terminal
        and not would_have_truncated
    )
    failures = []
    if not header_ok:
        failures.append("canonical_header")
    if not numbering_ok:
        failures.append("exact_1_2_3_numbering")
    if not nonempty_points:
        failures.append("three_nonempty_points")
    if not concluded_terminal:
        failures.append("terminal_CONCLUDED")
    if would_have_truncated:
        failures.append("legacy_truncation_trigger")
    return {
        "strict_ok": strict_ok,
        "header_ok": header_ok,
        "numbering_ok": numbering_ok,
        "nonempty_points_ok": nonempty_points,
        "concluded_terminal_ok": concluded_terminal,
        "legacy_truncation_triggered": would_have_truncated,
        "point_markers": marker_numbers,
        "failures": failures,
    }


@dataclass
class Transition:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_logprobs: list[float]
    round_num: int
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)


@dataclass
class DebateTrajectory:
    agent: Literal["A", "B"]
    transitions: list[Transition]
    frozen_solution: str | None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_completion_tokens(self) -> int:
        return sum(len(t.completion_tokens) for t in self.transitions)


@dataclass
class DebateResult:
    question: str
    ground_truth: str | None
    trajectory_a: DebateTrajectory
    trajectory_b: DebateTrajectory
    verdict: Verdict
    judge_reasoning: str
    metrics: dict[str, Any] = field(default_factory=dict)
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
    num_rounds: int = 3
    enable_thinking: bool | None = None
    max_tokens_per_turn: int | None = None
    max_tokens_r1: int | None = None
    max_tokens_r23: int | None = None
    max_tokens_r2: int | None = None
    max_tokens_r3: int | None = None
    temperature: float = 1.0
    kl_coef: float = 0.01
    learning_rate: float = 1e-5
    system_propose: str = load_prompt("debate/system_propose.md")
    system_argue: str = load_prompt("debate/system_argue.md")
    system_judge: str = load_prompt("debate/system_judge.md")
    r2_user_template: str = load_prompt("debate/r2_user_template.md")
    r3_user_template: str = load_prompt("debate/r3_user_template.md")
    chat_preamble: str = ""

    @staticmethod
    def cheap(*, chat_preamble: str = "") -> "DebateConfig":
        return DebateConfig(max_tokens_per_turn=None, temperature=1.0, chat_preamble=chat_preamble)


@dataclass
class TrainingDatum:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_logprobs: list[float]
    completion_logprob_mask: list[int]
    completion_advantages: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"


def _im_end() -> str:
    return "<|im_end|>\n"


def build_r1_prompt(question: str, config: DebateConfig) -> str:
    return (
        config.chat_preamble
        + _im_start("system")
        + config.system_propose
        + "\n"
        + _im_end()
        + _im_start("user")
        + question
        + "\n"
        + _im_end()
        + _im_start("assistant")
    )


def build_r2_continuation(opponent_r1: str, config: DebateConfig) -> str:
    user_msg = config.r2_user_template.format(opponent_r1=opponent_r1)
    return _im_end() + _im_start("user") + user_msg + "\n" + _im_end() + _im_start("assistant")


def build_r3_continuation(opponent_r2: str, config: DebateConfig) -> str:
    user_msg = config.r3_user_template.format(opponent_r2=opponent_r2)
    return _im_end() + _im_start("user") + user_msg + "\n" + _im_end() + _im_start("assistant")


def _merge_rounds_with_centered_reward(
    *,
    debate: DebateResult,
    winner: DebateTrajectory,
    reward: float,
    mean_reward: float,
    group_size: int | None,
    std_reward: float | None = None,
) -> TrainingDatum:
    if len(winner.transitions) != 3:
        raise ValueError(f"Expected 3 rounds, got {len(winner.transitions)}")

    t1, t2, t3 = winner.transitions
    for t in (t1, t2, t3):
        if len(t.completion_tokens) != len(t.completion_logprobs):
            raise ValueError(
                f"Completion/logprob length mismatch in round {t.round_num}: "
                f"{len(t.completion_tokens)} vs {len(t.completion_logprobs)}"
            )

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
    merged_logprob_mask = (
        [1] * len(t1.completion_tokens)
        + [0] * len(r2_continuation_tokens)
        + [1] * len(t2.completion_tokens)
        + [0] * len(r3_continuation_tokens)
        + [1] * len(t3.completion_tokens)
    )

    total_generated_tokens = len(t1.completion_tokens) + len(t2.completion_tokens) + len(t3.completion_tokens)
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
        completion_logprob_mask=merged_logprob_mask,
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
    merged_logprob_mask = (
        [1] * len(t1.completion_tokens)
        + [0] * len(r2_continuation_tokens)
        + [1] * len(t2.completion_tokens)
        + [0] * len(r3_continuation_tokens)
        + [1] * len(t3.completion_tokens)
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
        completion_logprob_mask=merged_logprob_mask,
        completion_advantages=merged_advantages,
        metadata={
            "question": debate.question[:100],
            "agent": traj.agent,
            "verdict": debate.verdict,
            **metadata,
        },
    )


def _merge_two_rounds_with_adv_values(
    *,
    debate: DebateResult,
    traj: DebateTrajectory,
    r1_adv_value: float,
    r2_adv_value: float,
    metadata: dict[str, Any],
) -> TrainingDatum:
    if len(traj.transitions) != 2:
        raise ValueError(f"Expected 2 rounds, got {len(traj.transitions)}")
    t1, t2 = traj.transitions
    for t in (t1, t2):
        if len(t.completion_tokens) != len(t.completion_logprobs):
            raise ValueError(
                f"Completion/logprob length mismatch in round {t.round_num}: "
                f"{len(t.completion_tokens)} vs {len(t.completion_logprobs)}"
            )
    if r1_adv_value != 0.0 and len(t1.completion_tokens) == 0:
        raise ValueError("Non-zero R1 advantage with zero R1 completion tokens.")
    if r2_adv_value != 0.0 and len(t2.completion_tokens) == 0:
        raise ValueError("Non-zero R2 advantage with zero R2 completion tokens.")

    r1_full_len = len(t1.prompt_tokens) + len(t1.completion_tokens)
    if len(t2.prompt_tokens) < r1_full_len:
        raise ValueError("R2 prompt shorter than R1 history; extension property violated.")
    r2_continuation_tokens = t2.prompt_tokens[r1_full_len:]
    merged_completion = t1.completion_tokens + r2_continuation_tokens + t2.completion_tokens
    merged_logprobs = list(t1.completion_logprobs) + [0.0] * len(r2_continuation_tokens) + list(t2.completion_logprobs)
    merged_logprob_mask = (
        [1] * len(t1.completion_tokens)
        + [0] * len(r2_continuation_tokens)
        + [1] * len(t2.completion_tokens)
    )
    merged_advantages = [r1_adv_value] * len(t1.completion_tokens) + [0.0] * len(r2_continuation_tokens) + [r2_adv_value] * len(t2.completion_tokens)

    return TrainingDatum(
        prompt_tokens=t1.prompt_tokens,
        completion_tokens=merged_completion,
        completion_logprobs=merged_logprobs,
        completion_logprob_mask=merged_logprob_mask,
        completion_advantages=merged_advantages,
        metadata={
            "question": debate.question[:100],
            "agent": traj.agent,
            "verdict": debate.verdict,
            **metadata,
        },
    )


def _merge_transition_pair_with_adv_values(
    *,
    debate: DebateResult,
    traj: DebateTrajectory,
    first: Transition,
    second: Transition,
    first_adv_value: float,
    second_adv_value: float,
    metadata: dict[str, Any],
) -> TrainingDatum:
    for t in (first, second):
        if len(t.completion_tokens) != len(t.completion_logprobs):
            raise ValueError(
                f"Completion/logprob length mismatch in round {t.round_num}: "
                f"{len(t.completion_tokens)} vs {len(t.completion_logprobs)}"
            )
    if first_adv_value != 0.0 and len(first.completion_tokens) == 0:
        raise ValueError(f"Non-zero R{first.round_num} advantage with zero completion tokens.")
    if second_adv_value != 0.0 and len(second.completion_tokens) == 0:
        raise ValueError(f"Non-zero R{second.round_num} advantage with zero completion tokens.")

    first_full_len = len(first.prompt_tokens) + len(first.completion_tokens)
    if len(second.prompt_tokens) < first_full_len:
        raise ValueError(
            f"R{second.round_num} prompt shorter than R{first.round_num} history; "
            "extension property violated."
        )
    continuation_tokens = second.prompt_tokens[first_full_len:]
    merged_completion = first.completion_tokens + continuation_tokens + second.completion_tokens
    merged_logprobs = (
        list(first.completion_logprobs)
        + [0.0] * len(continuation_tokens)
        + list(second.completion_logprobs)
    )
    merged_logprob_mask = (
        [1] * len(first.completion_tokens)
        + [0] * len(continuation_tokens)
        + [1] * len(second.completion_tokens)
    )
    merged_advantages = (
        [first_adv_value] * len(first.completion_tokens)
        + [0.0] * len(continuation_tokens)
        + [second_adv_value] * len(second.completion_tokens)
    )

    return TrainingDatum(
        prompt_tokens=first.prompt_tokens,
        completion_tokens=merged_completion,
        completion_logprobs=merged_logprobs,
        completion_logprob_mask=merged_logprob_mask,
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
    groups: dict[str, list[tuple[DebateTrajectory, DebateResult, float]]] = {}
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        winner = debate.get_winner_trajectory()
        reward = reward_fn(winner, debate)
        groups.setdefault(debate.question, []).append((winner, debate, reward))

    data: list[TrainingDatum] = []
    for question, group in groups.items():
        _ = question
        rewards = [r for _, _, r in group]
        mean_reward = sum(rewards) / len(rewards)
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)
        group_size = len(group)
        for winner, debate, reward in group:
            data.append(
                _merge_rounds_with_centered_reward(
                    debate=debate,
                    winner=winner,
                    reward=reward,
                    mean_reward=mean_reward,
                    std_reward=std,
                    group_size=group_size,
                )
            )
    return data


def assemble_training_data_r1_r23(
    debates: list[DebateResult],
    r1_reward_fn: Callable[[DebateTrajectory, DebateResult], float],
    *,
    r23_reward: float,
    r23_symmetric: bool,
) -> list[TrainingDatum]:
    groups: dict[str, list[tuple[DebateTrajectory, DebateResult, float]]] = {}
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        traj_a = debate.trajectory_a
        traj_b = debate.trajectory_b
        groups.setdefault(debate.question, []).extend(
            [
                (traj_a, debate, r1_reward_fn(traj_a, debate)),
                (traj_b, debate, r1_reward_fn(traj_b, debate)),
            ]
        )

    data: list[TrainingDatum] = []
    for question, group in groups.items():
        _ = question

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


def assemble_training_data_r1_only_centered(
    debates: list[DebateResult],
    reward_fn: Callable[[DebateTrajectory, DebateResult], float],
) -> list[TrainingDatum]:
    groups: dict[str, list[tuple[DebateTrajectory, DebateResult, float]]] = {}
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        traj_a = debate.trajectory_a
        traj_b = debate.trajectory_b
        groups.setdefault(debate.question, []).extend(
            [
                (traj_a, debate, float(reward_fn(traj_a, debate))),
                (traj_b, debate, float(reward_fn(traj_b, debate))),
            ]
        )

    data: list[TrainingDatum] = []
    for question, group in groups.items():
        _ = question
        rewards = [r for _, _, r in group]
        mean_reward = sum(rewards) / len(rewards)
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)
        group_size = len(group)
        for traj, debate, reward in group:
            t1 = traj.transitions[0]
            if len(t1.completion_tokens) == 0:
                raise ValueError("R1 completion tokens empty; cannot assign centered R1 reward.")
            centered = (reward - mean_reward) / std if std > 0 else 0.0
            adv = centered / len(t1.completion_tokens)
            data.append(
                TrainingDatum(
                    prompt_tokens=t1.prompt_tokens,
                    completion_tokens=t1.completion_tokens,
                    completion_logprobs=t1.completion_logprobs,
                    completion_logprob_mask=[1] * len(t1.completion_tokens),
                    completion_advantages=[adv] * len(t1.completion_tokens),
                    metadata={
                        "question": debate.question[:100],
                        "agent": traj.agent,
                        "verdict": debate.verdict,
                        "reward": reward,
                        "centered_reward": centered,
                        "group_mean_reward": mean_reward,
                        "group_std_reward": std,
                        "group_size": group_size,
                        "rounds_merged": 1,
                        "r1_trained": True,
                        "r23_trained": False,
                    },
                )
            )
    return data


def assemble_training_data_r1_only_compare(
    debates: list[DebateResult],
    *,
    r1_reward: float,
    r1_symmetric: bool,
) -> tuple[list[TrainingDatum], int]:
    data: list[TrainingDatum] = []
    skipped_empty_r1_debates = 0
    loser_reward = -float(r1_reward) if r1_symmetric else 0.0
    winner_reward = float(r1_reward)
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        if any(len(traj.transitions[0].completion_tokens) == 0 for traj in (debate.trajectory_a, debate.trajectory_b)):
            skipped_empty_r1_debates += 1
            continue
        for traj in (debate.trajectory_a, debate.trajectory_b):
            t1 = traj.transitions[0]
            is_winner = debate.get_winner_trajectory().agent == traj.agent
            reward = winner_reward if is_winner else loser_reward
            adv = reward / len(t1.completion_tokens)
            data.append(
                TrainingDatum(
                    prompt_tokens=t1.prompt_tokens,
                    completion_tokens=t1.completion_tokens,
                    completion_logprobs=t1.completion_logprobs,
                    completion_logprob_mask=[1] * len(t1.completion_tokens),
                    completion_advantages=[adv] * len(t1.completion_tokens),
                    metadata={
                        "question": debate.question[:100],
                        "agent": traj.agent,
                        "verdict": debate.verdict,
                        "r1_reward": reward,
                        "r1_adv_value": adv,
                        "r1_symmetric": r1_symmetric,
                        "rounds_merged": 1,
                        "r1_trained": reward != 0.0,
                        "r23_trained": False,
                    },
                )
            )
    return data, skipped_empty_r1_debates


def assemble_training_data_r1_r2(
    debates: list[DebateResult],
    r1_reward_fn: Callable[[DebateTrajectory, DebateResult], float],
    *,
    r2_reward: float,
    r2_symmetric: bool,
) -> list[TrainingDatum]:
    groups: dict[str, list[tuple[DebateTrajectory, DebateResult, float]]] = {}
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        traj_a = debate.trajectory_a
        traj_b = debate.trajectory_b
        groups.setdefault(debate.question, []).extend(
            [
                (traj_a, debate, r1_reward_fn(traj_a, debate)),
                (traj_b, debate, r1_reward_fn(traj_b, debate)),
            ]
        )

    data: list[TrainingDatum] = []
    for question, group in groups.items():
        _ = question

        def _per_token_adv(reward: float, tokens: list[int], label: str) -> float:
            if len(tokens) == 0:
                raise ValueError(f"{label} completion tokens empty; cannot assign reward.")
            return reward / len(tokens)

        rewards = [r for _, _, r in group]
        mean_reward = sum(rewards) / len(rewards)
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)
        group_size = len(group)
        r2_w = float(r2_reward)
        r2_l = -float(r2_reward) if r2_symmetric else 0.0

        for traj, debate, r1_reward in group:
            if std > 0:
                r1_centered = (r1_reward - mean_reward) / std
            else:
                r1_centered = 0.0
            r1_adv = _per_token_adv(r1_centered, traj.transitions[0].completion_tokens, "R1")
            is_winner = debate.get_winner_trajectory().agent == traj.agent
            r2_reward_signed = r2_w if is_winner else r2_l
            r2_adv = _per_token_adv(r2_reward_signed, traj.transitions[1].completion_tokens, "R2")
            data.append(
                _merge_two_rounds_with_adv_values(
                    debate=debate,
                    traj=traj,
                    r1_adv_value=r1_adv,
                    r2_adv_value=r2_adv,
                    metadata={
                        "r1_reward": r1_reward,
                        "r1_centered_reward": r1_centered,
                        "r1_adv_value": r1_adv,
                        "r1_group_mean_reward": mean_reward,
                        "r1_group_std_reward": std,
                        "r1_group_size": group_size,
                        "r2_reward": r2_reward_signed,
                        "r2_symmetric": r2_symmetric,
                        "r2_adv_value": r2_adv,
                        "rounds_merged": 2,
                        "r1_trained": True,
                        "r23_trained": r2_reward_signed != 0.0,
                    },
                )
            )
    return data


def assemble_training_data_r1_compare_r2(
    debates: list[DebateResult],
    *,
    r1_reward: float,
    r1_symmetric: bool,
    r2_reward: float,
    r2_symmetric: bool,
) -> list[TrainingDatum]:
    data: list[TrainingDatum] = []
    r1_winner_reward = float(r1_reward)
    r1_loser_reward = -float(r1_reward) if r1_symmetric else 0.0
    r2_winner_reward = float(r2_reward)
    r2_loser_reward = -float(r2_reward) if r2_symmetric else 0.0
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        for traj in (debate.trajectory_a, debate.trajectory_b):
            t1, t2 = traj.transitions
            if len(t1.completion_tokens) == 0:
                raise ValueError("R1 completion tokens empty; cannot assign judge-compare reward.")
            if len(t2.completion_tokens) == 0:
                raise ValueError("R2 completion tokens empty; cannot assign reward.")
            is_winner = debate.get_winner_trajectory().agent == traj.agent
            signed_r1_reward = r1_winner_reward if is_winner else r1_loser_reward
            signed_r2_reward = r2_winner_reward if is_winner else r2_loser_reward
            data.append(
                _merge_two_rounds_with_adv_values(
                    debate=debate,
                    traj=traj,
                    r1_adv_value=signed_r1_reward / len(t1.completion_tokens),
                    r2_adv_value=signed_r2_reward / len(t2.completion_tokens),
                    metadata={
                        "r1_reward": signed_r1_reward,
                        "r1_compare": True,
                        "r1_symmetric": r1_symmetric,
                        "r1_adv_value": signed_r1_reward / len(t1.completion_tokens),
                        "r2_reward": signed_r2_reward,
                        "r2_symmetric": r2_symmetric,
                        "r2_adv_value": signed_r2_reward / len(t2.completion_tokens),
                        "rounds_merged": 2,
                        "r1_trained": signed_r1_reward != 0.0,
                        "r23_trained": signed_r2_reward != 0.0,
                    },
                )
            )
    return data


def assemble_training_data_r1_compare_r23(
    debates: list[DebateResult],
    *,
    r1_reward: float,
    r1_symmetric: bool,
    r23_reward: float,
    r23_symmetric: bool,
) -> list[TrainingDatum]:
    data: list[TrainingDatum] = []
    r1_winner_reward = float(r1_reward)
    r1_loser_reward = -float(r1_reward) if r1_symmetric else 0.0
    r23_winner_reward = float(r23_reward)
    r23_loser_reward = -float(r23_reward) if r23_symmetric else 0.0
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        for traj in (debate.trajectory_a, debate.trajectory_b):
            t1, t2, t3 = traj.transitions
            if len(t1.completion_tokens) == 0:
                raise ValueError("R1 completion tokens empty; cannot assign judge-compare reward.")
            if len(t2.completion_tokens) == 0:
                raise ValueError("R2 completion tokens empty; cannot assign reward.")
            if len(t3.completion_tokens) == 0:
                raise ValueError("R3 completion tokens empty; cannot assign reward.")
            is_winner = debate.get_winner_trajectory().agent == traj.agent
            signed_r1_reward = r1_winner_reward if is_winner else r1_loser_reward
            signed_r23_reward = r23_winner_reward if is_winner else r23_loser_reward
            data.append(
                _merge_rounds_with_adv_values(
                    debate=debate,
                    traj=traj,
                    r1_adv_value=signed_r1_reward / len(t1.completion_tokens),
                    r2_adv_value=signed_r23_reward / len(t2.completion_tokens),
                    r3_adv_value=signed_r23_reward / len(t3.completion_tokens),
                    metadata={
                        "r1_reward": signed_r1_reward,
                        "r1_compare": True,
                        "r1_symmetric": r1_symmetric,
                        "r1_adv_value": signed_r1_reward / len(t1.completion_tokens),
                        "r23_reward": signed_r23_reward,
                        "r23_symmetric": r23_symmetric,
                        "r23_adv_value": signed_r23_reward / len(t2.completion_tokens),
                        "rounds_merged": 3,
                        "r1_trained": signed_r1_reward != 0.0,
                        "r23_trained": signed_r23_reward != 0.0,
                    },
                )
            )
    return data


def training_datum_to_train_example(*, datum: TrainingDatum, adapter_name: str) -> TrainExample:
    if len(datum.completion_tokens) == 0:
        raise ValueError("Cannot train on an empty completion.")
    if len(datum.completion_tokens) != len(datum.completion_logprobs):
        raise ValueError("Completion tokens and logprobs must have equal length.")
    if len(datum.completion_tokens) != len(datum.completion_logprob_mask):
        raise ValueError("Completion tokens and logprob mask must have equal length.")
    if len(datum.completion_tokens) != len(datum.completion_advantages):
        raise ValueError("Completion tokens and advantages must have equal length.")
    if any(value not in (0, 1) for value in datum.completion_logprob_mask):
        raise ValueError("Completion logprob mask must contain only 0 or 1.")
    if any(
        advantage != 0.0 and not has_behavior_logprob
        for advantage, has_behavior_logprob in zip(
            datum.completion_advantages,
            datum.completion_logprob_mask,
            strict=True,
        )
    ):
        raise ValueError("Every nonzero-advantage token must have a behavior-policy logprob.")

    full_token_ids = datum.prompt_tokens + datum.completion_tokens
    input_ids = full_token_ids[:-1]
    target_ids = full_token_ids[1:]
    prompt_prefix_len = len(datum.prompt_tokens) - 1
    if prompt_prefix_len < 0:
        raise ValueError("Prompt must contain at least one token.")

    return TrainExample(
        adapter_name=adapter_name,
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=([0] * prompt_prefix_len) + ([1] * len(datum.completion_tokens)),
        behavior_logprob_mask=([0] * prompt_prefix_len) + list(datum.completion_logprob_mask),
        old_logprobs=([0.0] * prompt_prefix_len) + list(datum.completion_logprobs),
        advantages=([0.0] * prompt_prefix_len) + list(datum.completion_advantages),
        metadata=dict(datum.metadata),
    )


def assemble_training_data_by_mode(
    *,
    debates: list[DebateResult],
    num_rounds: int,
    r1_reward_mode: str,
    r23_reward_mode: str,
    r23_constant: float,
    r23_symmetric: bool,
    task_reward_fn: Callable[[DebateTrajectory, DebateResult], float],
    pointwise_reward_map: dict[int, float] | None = None,
) -> list[TrainingDatum]:
    if num_rounds == 1:
        if r23_reward_mode != "none":
            raise ValueError("r23 reward must be none when num_rounds=1")
        if r1_reward_mode == "task":
            return assemble_training_data_r1_only_centered(debates, task_reward_fn)
        if r1_reward_mode == "judge_pointwise":
            if pointwise_reward_map is None:
                raise ValueError("judge_pointwise requires pointwise_reward_map")
            return assemble_training_data_r1_only_centered(debates, lambda traj, _debate: pointwise_reward_map[id(traj)])
        if r1_reward_mode == "judge":
            result = assemble_training_data_r1_only_compare(
                debates,
                r1_reward=float(r23_constant),
                r1_symmetric=r23_symmetric,
            )
            if isinstance(result, tuple):
                return result[0]
            return result
        raise ValueError(f"Unsupported r1_reward_mode={r1_reward_mode!r}")
    if num_rounds == 2:
        if r1_reward_mode == "task":
            r1_reward_fn = task_reward_fn
        elif r1_reward_mode == "judge_pointwise":
            if pointwise_reward_map is None:
                raise ValueError("judge_pointwise requires pointwise_reward_map")
            r1_reward_fn = lambda traj, _debate: pointwise_reward_map[id(traj)]
        elif r1_reward_mode == "judge":
            r2_reward = 0.0 if r23_reward_mode == "none" else float(r23_constant)
            return assemble_training_data_r1_compare_r2(
                debates,
                r1_reward=float(r23_constant),
                r1_symmetric=r23_symmetric,
                r2_reward=r2_reward,
                r2_symmetric=r23_symmetric,
            )
        elif r1_reward_mode == "none":
            r1_reward_fn = lambda _traj, _debate: 0.0
        else:
            raise ValueError(f"Unsupported r1_reward_mode={r1_reward_mode!r}")
        r2_reward = 0.0 if r23_reward_mode == "none" else float(r23_constant)
        return assemble_training_data_r1_r2(
            debates,
            r1_reward_fn=r1_reward_fn,
            r2_reward=r2_reward,
            r2_symmetric=r23_symmetric,
        )
    if num_rounds == 3:
        if r1_reward_mode == "task":
            r1_reward_fn = task_reward_fn
        elif r1_reward_mode == "judge_pointwise":
            if pointwise_reward_map is None:
                raise ValueError("judge_pointwise requires pointwise_reward_map")
            r1_reward_fn = lambda traj, _debate: pointwise_reward_map[id(traj)]
        elif r1_reward_mode == "judge":
            r23_reward = 0.0 if r23_reward_mode == "none" else float(r23_constant)
            return assemble_training_data_r1_compare_r23(
                debates,
                r1_reward=float(r23_constant),
                r1_symmetric=r23_symmetric,
                r23_reward=r23_reward,
                r23_symmetric=r23_symmetric,
            )
        elif r1_reward_mode == "none":
            r1_reward_fn = lambda _traj, _debate: 0.0
        else:
            raise ValueError(f"Unsupported r1_reward_mode={r1_reward_mode!r}")
        r23_reward = 0.0 if r23_reward_mode == "none" else float(r23_constant)
        return assemble_training_data_r1_r23(
            debates,
            r1_reward_fn=r1_reward_fn,
            r23_reward=r23_reward,
            r23_symmetric=r23_symmetric,
        )
    raise ValueError(f"Unsupported num_rounds={num_rounds!r}")


def assemble_split_train_examples(
    *,
    debates: list[DebateResult],
    num_rounds: int,
    round_adapter_names: tuple[str, ...],
    r1_reward_mode: str,
    r23_reward_mode: str,
    r23_constant: float,
    r23_symmetric: bool,
    task_reward_fn: Callable[[DebateTrajectory, DebateResult], float],
    r1_judge_delta_q: float = 1.0,
    incoherent_r23_reward: float = -0.5,
    r23_format_failure_penalty: float = 0.0,
    pointwise_reward_map: dict[int, float] | None = None,
    r23_advantage_scope: Literal["per_round", "merged_r23"] = "per_round",
) -> dict[str, list[TrainExample]]:
    if len(round_adapter_names) < num_rounds:
        raise ValueError(f"Need at least {num_rounds} round adapter names, got {len(round_adapter_names)}")
    if r23_advantage_scope not in ("per_round", "merged_r23"):
        raise ValueError(f"Unsupported r23_advantage_scope={r23_advantage_scope!r}.")
    if not math.isfinite(r1_judge_delta_q) or r1_judge_delta_q < 0.0:
        raise ValueError("r1_judge_delta_q must be finite and non-negative")
    if not math.isfinite(incoherent_r23_reward):
        raise ValueError("incoherent_r23_reward must be finite")
    if not math.isfinite(r23_format_failure_penalty) or r23_format_failure_penalty > 0.0:
        raise ValueError("r23_format_failure_penalty must be finite and non-positive")
    grouped: dict[str, list[TrainExample]] = {}

    def _soft_score(debate: DebateResult) -> float:
        audit = debate.judge_raw_response if isinstance(debate.judge_raw_response, dict) else {}
        score_record = audit.get("soft_score")
        if audit.get("soft_judge") is not True or not isinstance(score_record, dict):
            raise ValueError("Soft judge reward requires an order-symmetric soft-score audit")
        score = float(score_record.get("score"))
        if not math.isfinite(score) or score < -1.0 or score > 1.0:
            raise ValueError(f"Invalid soft judge score: {score!r}")
        return score

    def _append_turn(*, adapter_name: str, prompt_tokens: list[int], completion_tokens: list[int], completion_logprobs: list[float], advantages: list[float], metadata: dict[str, Any]) -> None:
        datum = TrainingDatum(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_logprobs=completion_logprobs,
            completion_logprob_mask=[1] * len(completion_tokens),
            completion_advantages=advantages,
            metadata=metadata,
        )
        grouped.setdefault(adapter_name, []).append(
            training_datum_to_train_example(datum=datum, adapter_name=adapter_name)
        )

    if num_rounds == 1 and r1_reward_mode == "judge":
        result = assemble_training_data_r1_only_compare(
            debates,
            r1_reward=float(r23_constant),
            r1_symmetric=r23_symmetric,
        )
        data = result[0] if isinstance(result, tuple) else result
        for datum in data:
            _append_turn(
                adapter_name=round_adapter_names[0],
                prompt_tokens=datum.prompt_tokens,
                completion_tokens=datum.completion_tokens,
                completion_logprobs=datum.completion_logprobs,
                advantages=datum.completion_advantages,
                metadata=dict(datum.metadata),
            )
        return grouped

    selected_r1_trajectory_ids: set[int] | None = None
    if r1_reward_mode == "judge_rejection_task":
        if round_adapter_names[0] in round_adapter_names[1:num_rounds]:
            raise ValueError(
                "judge_rejection_task requires an R1 adapter distinct from all later-round adapters"
            )
        selected_r1_trajectory_ids = set()

    groups: dict[str, list[tuple[DebateTrajectory, DebateResult, float]]] = {}
    for debate in debates:
        if debate.verdict not in ("A", "B"):
            continue
        traj_a = debate.trajectory_a
        traj_b = debate.trajectory_b
        if r1_reward_mode == "task":
            r1_a = task_reward_fn(traj_a, debate)
            r1_b = task_reward_fn(traj_b, debate)
        elif r1_reward_mode == "judge_rejection_task":
            winner = debate.get_winner_trajectory()
            winner_task_reward = float(task_reward_fn(winner, debate))
            if not math.isfinite(winner_task_reward):
                raise ValueError(f"Non-finite winner task reward for question={debate.question!r}")
            assert selected_r1_trajectory_ids is not None
            selected_r1_trajectory_ids.add(id(winner))
            r1_a = winner_task_reward if winner.agent == traj_a.agent else 0.0
            r1_b = winner_task_reward if winner.agent == traj_b.agent else 0.0
        elif r1_reward_mode == "judge_delta_task":
            task_a = float(task_reward_fn(traj_a, debate))
            task_b = float(task_reward_fn(traj_b, debate))
            if not math.isfinite(task_a) or not math.isfinite(task_b):
                raise ValueError(f"Non-finite task reward for question={debate.question!r}")
            judge_audit = debate.judge_raw_response if isinstance(debate.judge_raw_response, dict) else {}
            delta = abs(task_a - task_b)
            modulation = r1_judge_delta_q * delta
            # On a coherent bidirectional verdict this is the agreed referent. On
            # disagreement, debate_runtime has already replaced the verdict with
            # the deterministic seeded coin flip requested by the experiment.
            winner_agent = debate.get_winner_trajectory().agent
            r1_a = task_a + modulation if winner_agent == traj_a.agent else task_a - modulation
            r1_b = task_b + modulation if winner_agent == traj_b.agent else task_b - modulation
        elif r1_reward_mode == "judge_soft_task_gap":
            task_a = float(task_reward_fn(traj_a, debate))
            task_b = float(task_reward_fn(traj_b, debate))
            if not math.isfinite(task_a) or not math.isfinite(task_b):
                raise ValueError(f"Non-finite task reward for question={debate.question!r}")
            score = _soft_score(debate)
            midpoint = 0.5 * (task_a + task_b)
            half_gap = 0.5 * abs(task_a - task_b)
            r1_a = midpoint + score * half_gap
            r1_b = midpoint - score * half_gap
            residual = (r1_a + r1_b) - (task_a + task_b)
            if abs(residual) > 1e-12:
                raise AssertionError(f"Soft R1 reward failed pair-sum conservation: {residual}")
        elif r1_reward_mode == "judge_pointwise":
            if pointwise_reward_map is None:
                raise ValueError("judge_pointwise requires pointwise_reward_map")
            r1_a = pointwise_reward_map[id(traj_a)]
            r1_b = pointwise_reward_map[id(traj_b)]
        elif r1_reward_mode == "judge":
            winner_reward = float(r23_constant)
            loser_reward = -winner_reward if r23_symmetric else 0.0
            winner_agent = debate.get_winner_trajectory().agent
            r1_a = winner_reward if winner_agent == traj_a.agent else loser_reward
            r1_b = winner_reward if winner_agent == traj_b.agent else loser_reward
        else:
            r1_a = 0.0
            r1_b = 0.0
        instance_id_a = traj_a.metrics.get("instance_id")
        instance_id_b = traj_b.metrics.get("instance_id")
        if instance_id_a is not None and instance_id_b is not None and instance_id_a != instance_id_b:
            raise ValueError(
                "Debate trajectories disagree on instance_id: "
                f"A={instance_id_a!r}, B={instance_id_b!r}"
            )
        instance_id = instance_id_a if instance_id_a is not None else instance_id_b
        group_key = debate.question if instance_id is None else f"{debate.question}\0{instance_id}"
        groups.setdefault(group_key, []).extend([(traj_a, debate, float(r1_a)), (traj_b, debate, float(r1_b))])

    for group_index, (question, group) in enumerate(groups.items()):
        _ = question
        selected_group = [
            (traj, debate, reward)
            for traj, debate, reward in group
            if selected_r1_trajectory_ids is None or id(traj) in selected_r1_trajectory_ids
        ]
        rewards = [reward for _traj, _debate, reward in selected_group]
        mean_reward = sum(rewards) / len(rewards)
        var = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        std = math.sqrt(var)
        winner_reward = 0.0 if r23_reward_mode == "none" else float(r23_constant)
        loser_reward = -winner_reward if r23_symmetric else 0.0

        for traj, debate, r1_reward in group:
            r1_selected = selected_r1_trajectory_ids is None or id(traj) in selected_r1_trajectory_ids
            if r1_selected:
                t1 = traj.transitions[0]
                if len(t1.completion_tokens) == 0:
                    raise ValueError("R1 completion tokens empty.")
                if r1_reward_mode == "judge":
                    r1_value = r1_reward
                    r1_advantages = [r1_value / len(t1.completion_tokens)] * len(t1.completion_tokens)
                else:
                    r1_value = (r1_reward - mean_reward) / std if std > 0 else 0.0
                    r1_advantages = [r1_value / len(t1.completion_tokens)] * len(t1.completion_tokens)
                if r1_reward_mode == "judge_rejection_task":
                    r1_metadata = {
                        "question": debate.question[:100],
                        "agent": traj.agent,
                        "verdict": debate.verdict,
                        "source_exact_shared_equivalent": False,
                        "reason": "split_layout_judge_rejection_task_projection",
                        "round_num": 1,
                        "r1_reward_mode": "judge_rejection_task",
                        "r1_selected_by_judge": True,
                        "r1_rejected_agent": debate.get_loser_trajectory().agent,
                        "r1_task_reward": r1_reward,
                        "r1_group_index": group_index,
                        "r1_selected_group_size": len(selected_group),
                        "r1_group_mean_reward": mean_reward,
                        "r1_group_std_reward": std,
                        "r1_group_live": std > 0.0,
                        "r1_zscore": r1_value,
                    }
                else:
                    r1_metadata = {
                        "question": debate.question[:100],
                        "agent": traj.agent,
                        "verdict": debate.verdict,
                        "source_exact_shared_equivalent": False,
                        "reason": "split_layout_per_round_projection",
                        "round_num": 1,
                        "r1_reward": r1_reward,
                        "r1_compare": r1_reward_mode == "judge",
                        "r1_centered_reward": None if r1_reward_mode == "judge" else r1_value,
                    }
                    if r1_reward_mode == "judge_delta_task":
                        judge_audit = debate.judge_raw_response if isinstance(debate.judge_raw_response, dict) else {}
                        task_a = float(task_reward_fn(debate.trajectory_a, debate))
                        task_b = float(task_reward_fn(debate.trajectory_b, debate))
                        r1_metadata.update({
                            "reason": "split_layout_judge_delta_task_projection",
                            "r1_reward_mode": "judge_delta_task",
                            "judge_order_invariant": judge_audit.get("order_invariant") is True,
                            "r1_winner_source": (
                                "order_invariant_judge"
                                if judge_audit.get("order_invariant") is True
                                else "seeded_coin_flip"
                            ),
                            "r1_task_reward": float(task_reward_fn(traj, debate)),
                            "r1_task_reward_delta": abs(task_a - task_b),
                            "r1_judge_delta_q": r1_judge_delta_q,
                            "r1_modulated_reward": r1_reward,
                        })
                    elif r1_reward_mode == "judge_soft_task_gap":
                        task_a = float(task_reward_fn(debate.trajectory_a, debate))
                        task_b = float(task_reward_fn(debate.trajectory_b, debate))
                        score = _soft_score(debate)
                        r1_metadata.update({
                            "reason": "split_layout_judge_soft_task_gap_projection",
                            "r1_reward_mode": "judge_soft_task_gap",
                            "judge_soft_score": score,
                            "r1_task_reward": float(task_reward_fn(traj, debate)),
                            "r1_task_reward_gap": abs(task_a - task_b),
                            "r1_task_reward_pair_sum": task_a + task_b,
                            "r1_soft_reward": r1_reward,
                            "r1_soft_reward_pair_sum_residual": 0.0,
                        })
                _append_turn(
                    adapter_name=round_adapter_names[0],
                    prompt_tokens=t1.prompt_tokens,
                    completion_tokens=t1.completion_tokens,
                    completion_logprobs=t1.completion_logprobs,
                    advantages=r1_advantages,
                    metadata=r1_metadata,
                )
            if num_rounds >= 2:
                t2 = traj.transitions[1]
                judge_audit = debate.judge_raw_response if isinstance(debate.judge_raw_response, dict) else {}
                coherent = judge_audit.get("order_invariant") is True
                if r23_reward_mode == "soft_judge":
                    score = _soft_score(debate)
                    signed = score if traj.agent == "A" else -score
                else:
                    signed = (
                        winner_reward if debate.get_winner_trajectory().agent == traj.agent else loser_reward
                    ) if coherent or not judge_audit.get("bidirectional_judge") else incoherent_r23_reward
                r2_format = audit_base_text_debate_format(
                    text=str(traj.metrics.get("r2", "")), round_num=2
                ) if r23_format_failure_penalty != 0.0 else {"strict_ok": True, "failures": []}
                r2_format_penalty = 0.0 if r2_format["strict_ok"] else r23_format_failure_penalty
                if len(t2.completion_tokens) == 0:
                    raise ValueError("R2 completion tokens empty.")
                if num_rounds >= 3 and round_adapter_names[1] == round_adapter_names[2]:
                    t3 = traj.transitions[2]
                    if len(t3.completion_tokens) == 0:
                        raise ValueError("R3 completion tokens empty.")
                    r3_format = audit_base_text_debate_format(
                        text=str(traj.metrics.get("r3", "")), round_num=3
                    ) if r23_format_failure_penalty != 0.0 else {"strict_ok": True, "failures": []}
                    r3_format_penalty = 0.0 if r3_format["strict_ok"] else r23_format_failure_penalty
                    if r23_advantage_scope == "per_round":
                        first_adv_value = (signed + r2_format_penalty) / len(t2.completion_tokens)
                        second_adv_value = (signed + r3_format_penalty) / len(t3.completion_tokens)
                    else:
                        r23_token_count = len(t2.completion_tokens) + len(t3.completion_tokens)
                        base_adv_value = signed / r23_token_count
                        first_adv_value = base_adv_value + r2_format_penalty / len(t2.completion_tokens)
                        second_adv_value = base_adv_value + r3_format_penalty / len(t3.completion_tokens)
                    datum = _merge_transition_pair_with_adv_values(
                        debate=debate,
                        traj=traj,
                        first=t2,
                        second=t3,
                        first_adv_value=first_adv_value,
                        second_adv_value=second_adv_value,
                        metadata={
                            "question": debate.question[:100],
                            "agent": traj.agent,
                            "verdict": debate.verdict,
                            "source_exact_shared_equivalent": False,
                            "reason": "split_layout_same_adapter_round_merge",
                            "round_nums": [2, 3],
                            "rounds_merged": 2,
                            "r23_reward": signed,
                            "r23_base_judge_reward": signed,
                            "r2_format_strict": r2_format["strict_ok"],
                            "r3_format_strict": r3_format["strict_ok"],
                            "r2_format_failures": r2_format["failures"],
                            "r3_format_failures": r3_format["failures"],
                            "r2_legacy_truncation_triggered": r2_format.get("legacy_truncation_triggered", False),
                            "r3_legacy_truncation_triggered": r3_format.get("legacy_truncation_triggered", False),
                            "r2_format_failure_penalty": r2_format_penalty,
                            "r3_format_failure_penalty": r3_format_penalty,
                            "r23_combined_reward": signed + r2_format_penalty + r3_format_penalty,
                            "r23_advantage_scope": r23_advantage_scope,
                            "r23_first_adv_value": first_adv_value,
                            "r23_second_adv_value": second_adv_value,
                            "judge_order_invariant": coherent,
                            "r23_incoherent_reward_applied": (
                                r23_reward_mode != "soft_judge"
                                and not coherent
                                and judge_audit.get("bidirectional_judge") is True
                            ),
                            "r23_reward_mode": r23_reward_mode,
                            "judge_soft_score": _soft_score(debate) if r23_reward_mode == "soft_judge" else None,
                        },
                    )
                    grouped.setdefault(round_adapter_names[1], []).append(
                        training_datum_to_train_example(datum=datum, adapter_name=round_adapter_names[1])
                    )
                    continue
                _append_turn(
                    adapter_name=round_adapter_names[1],
                    prompt_tokens=t2.prompt_tokens,
                    completion_tokens=t2.completion_tokens,
                    completion_logprobs=t2.completion_logprobs,
                    advantages=[(signed + r2_format_penalty) / len(t2.completion_tokens)] * len(t2.completion_tokens),
                    metadata={
                        "question": debate.question[:100],
                        "agent": traj.agent,
                        "verdict": debate.verdict,
                        "source_exact_shared_equivalent": False,
                        "reason": "split_layout_per_round_projection",
                        "round_num": 2,
                        "r23_reward": signed,
                        "r23_base_judge_reward": signed,
                        "r2_format_strict": r2_format["strict_ok"],
                        "r2_format_failures": r2_format["failures"],
                        "r2_legacy_truncation_triggered": r2_format.get("legacy_truncation_triggered", False),
                        "r2_format_failure_penalty": r2_format_penalty,
                        "r2_effective_reward": signed + r2_format_penalty,
                        "r23_advantage_scope": r23_advantage_scope,
                        "judge_order_invariant": coherent,
                        "r23_incoherent_reward_applied": (
                            r23_reward_mode != "soft_judge"
                            and not coherent
                            and judge_audit.get("bidirectional_judge") is True
                        ),
                        "r23_reward_mode": r23_reward_mode,
                        "judge_soft_score": _soft_score(debate) if r23_reward_mode == "soft_judge" else None,
                    },
                )
            if num_rounds >= 3:
                t3 = traj.transitions[2]
                judge_audit = debate.judge_raw_response if isinstance(debate.judge_raw_response, dict) else {}
                coherent = judge_audit.get("order_invariant") is True
                if r23_reward_mode == "soft_judge":
                    score = _soft_score(debate)
                    signed = score if traj.agent == "A" else -score
                else:
                    signed = (
                        winner_reward if debate.get_winner_trajectory().agent == traj.agent else loser_reward
                    ) if coherent or not judge_audit.get("bidirectional_judge") else incoherent_r23_reward
                r3_format = audit_base_text_debate_format(
                    text=str(traj.metrics.get("r3", "")), round_num=3
                ) if r23_format_failure_penalty != 0.0 else {"strict_ok": True, "failures": []}
                r3_format_penalty = 0.0 if r3_format["strict_ok"] else r23_format_failure_penalty
                if len(t3.completion_tokens) == 0:
                    raise ValueError("R3 completion tokens empty.")
                _append_turn(
                    adapter_name=round_adapter_names[2],
                    prompt_tokens=t3.prompt_tokens,
                    completion_tokens=t3.completion_tokens,
                    completion_logprobs=t3.completion_logprobs,
                    advantages=[(signed + r3_format_penalty) / len(t3.completion_tokens)] * len(t3.completion_tokens),
                    metadata={
                        "question": debate.question[:100],
                        "agent": traj.agent,
                        "verdict": debate.verdict,
                        "source_exact_shared_equivalent": False,
                        "reason": "split_layout_per_round_projection",
                        "round_num": 3,
                        "r23_reward": signed,
                        "r23_base_judge_reward": signed,
                        "r3_format_strict": r3_format["strict_ok"],
                        "r3_format_failures": r3_format["failures"],
                        "r3_legacy_truncation_triggered": r3_format.get("legacy_truncation_triggered", False),
                        "r3_format_failure_penalty": r3_format_penalty,
                        "r3_effective_reward": signed + r3_format_penalty,
                        "r23_advantage_scope": r23_advantage_scope,
                        "judge_order_invariant": coherent,
                        "r23_incoherent_reward_applied": (
                            r23_reward_mode != "soft_judge"
                            and not coherent
                            and judge_audit.get("bidirectional_judge") is True
                        ),
                        "r23_reward_mode": r23_reward_mode,
                        "judge_soft_score": _soft_score(debate) if r23_reward_mode == "soft_judge" else None,
                    },
                )
    return grouped


def assemble_judge_coherence_grpo_examples(
    debates: list[DebateResult],
    *,
    adapter_name: str = "judge",
    reward_mode: str = "coherence",
) -> tuple[list[TrainExample], dict[str, float | int | str]]:
    """Build one global judge-GRPO group from both transcript orderings.

    ``label_js`` keeps objective OpenBookQA label supervision and replaces the
    old binary order-coherence term with a continuous, referent-aligned
    Jensen-Shannon penalty.  JS is normalized by ln(2), so no scale coefficient
    is needed: ``raw_reward = label_reward - JS/ln(2)``.
    """
    if reward_mode not in ("coherence", "label", "label_js"):
        raise ValueError(f"unsupported judge GRPO reward mode: {reward_mode!r}")
    turns: list[
        tuple[DebateResult, dict[str, Any], float, str | None, str, float, float]
    ] = []
    coherent_debates = 0
    label_correct_judgments = 0
    js_values: list[float] = []
    for debate in debates:
        audit = debate.judge_raw_response if isinstance(debate.judge_raw_response, dict) else {}
        if audit.get("bidirectional_judge") is not True:
            raise ValueError("judge coherence GRPO requires bidirectional judge audit data")
        training_turns = audit.get("_training_judge_turns")
        if not isinstance(training_turns, list) or len(training_turns) != 2:
            raise ValueError("judge coherence GRPO requires exactly two sampled judgment turns per debate")
        coherent = audit.get("order_invariant") is True
        coherent_debates += int(coherent)
        if reward_mode == "coherence":
            pair_reward = 1.0 if coherent else -1.0
            for turn in training_turns:
                if not isinstance(turn, dict):
                    raise ValueError("judge training turn must be a mapping")
                turns.append((debate, turn, pair_reward, None, "INVALID", pair_reward, 0.0))
            continue

        reward_a = float(debate.trajectory_a.metrics["task_reward"])
        reward_b = float(debate.trajectory_b.metrics["task_reward"])
        gold_agent = "A" if reward_a > reward_b else "B" if reward_b > reward_a else None
        js_penalty = 0.0
        if reward_mode == "label_js":
            soft_record = audit.get("soft_score")
            if audit.get("soft_judge") is not True or not isinstance(soft_record, dict):
                raise ValueError("label_js judge GRPO requires a soft-score audit")
            js_penalty = float(soft_record.get("referent_js_divergence_normalized"))
            if not math.isfinite(js_penalty) or not 0.0 <= js_penalty <= 1.0 + 1e-12:
                raise ValueError(f"invalid normalized referent JS penalty: {js_penalty!r}")
            js_penalty = min(1.0, js_penalty)
            js_values.append(js_penalty)
        for turn in training_turns:
            if not isinstance(turn, dict):
                raise ValueError("judge training turn must be a mapping")
            verdict = turn.get("verdict")
            if turn.get("order") == "reverse":
                verdict = "B" if verdict == "A" else "A" if verdict == "B" else "INVALID"
            label_reward = (
                0.0 if gold_agent is None else 1.0 if verdict == gold_agent else -1.0
            )
            label_correct_judgments += int(label_reward > 0.0)
            reward = label_reward - js_penalty
            turns.append(
                (debate, turn, reward, gold_agent, str(verdict), label_reward, js_penalty)
            )

    rewards = [reward for _debate, _turn, reward, *_rest in turns]
    if not rewards:
        return [], {
            "judge_grpo_group_size": 0,
            "judge_grpo_reward_mean": 0.0,
            "judge_grpo_reward_std": 0.0,
            "judge_grpo_coherent_debates": 0,
            "judge_grpo_incoherent_debates": 0,
            "judge_grpo_reward_mode": reward_mode,
        }
    reward_mean = sum(rewards) / len(rewards)
    reward_var = sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
    reward_std = math.sqrt(reward_var)
    examples: list[TrainExample] = []
    for judgment_index, (
        debate,
        turn,
        reward,
        gold_agent,
        referent_verdict,
        label_reward,
        js_penalty,
    ) in enumerate(turns):
        prompt_tokens = list(turn.get("prompt_tokens", []))
        completion_tokens = list(turn.get("completion_tokens", []))
        completion_logprobs = [float(value) for value in turn.get("completion_logprobs", [])]
        if not completion_tokens:
            raise ValueError("judge coherence GRPO completion tokens empty")
        if len(completion_tokens) != len(completion_logprobs):
            raise ValueError("judge coherence GRPO completion/logprob lengths differ")
        allowed_token_ids = tuple(
            int(token_id) for token_id in turn.get("behavior_policy_allowed_token_ids", ())
        )
        if reward_mode == "label_js":
            if len(allowed_token_ids) != 2 or len(set(allowed_token_ids)) != 2:
                raise ValueError(
                    "label_js judge GRPO requires exactly two behavior-policy label tokens"
                )
            if any(token_id not in allowed_token_ids for token_id in completion_tokens):
                raise ValueError("judge completion escaped its two-token behavior policy")
        zscore = (reward - reward_mean) / reward_std if reward_std > 0.0 else 0.0
        datum = TrainingDatum(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_logprobs=completion_logprobs,
            completion_logprob_mask=[1] * len(completion_tokens),
            completion_advantages=[zscore / len(completion_tokens)] * len(completion_tokens),
            metadata={
                "reason": f"judge_bidirectional_{reward_mode}_grpo",
                "judge_grpo_reward_mode": reward_mode,
                "question": debate.question[:100],
                "judge_order": turn.get("order"),
                "judge_order_invariant": (
                    debate.judge_raw_response.get("order_invariant") is True
                    if isinstance(debate.judge_raw_response, dict)
                    else False
                ),
                "judge_label_gold_agent": gold_agent,
                "judge_label_referent_verdict": referent_verdict,
                "judge_label_correct": label_reward > 0.0,
                "judge_label_reward": label_reward,
                "judge_referent_js_penalty": js_penalty,
                "judge_label_js_reward": reward if reward_mode == "label_js" else None,
                "judge_coherence_reward": reward if reward_mode == "coherence" else None,
                "behavior_policy_allowed_token_ids": list(allowed_token_ids),
                "judge_grpo_group_index": 0,
                "judge_grpo_group_size": len(turns),
                "judge_grpo_reward_mean": reward_mean,
                "judge_grpo_reward_std": reward_std,
                "judge_grpo_zscore": zscore,
                "judgment_index": judgment_index,
            },
        )
        examples.append(training_datum_to_train_example(datum=datum, adapter_name=adapter_name))
    return examples, {
        "judge_grpo_group_size": len(turns),
        "judge_grpo_reward_mean": reward_mean,
        "judge_grpo_reward_std": reward_std,
        "judge_grpo_coherent_debates": coherent_debates,
        "judge_grpo_incoherent_debates": len(debates) - coherent_debates,
        "judge_grpo_reward_mode": reward_mode,
        **(
            {
                "judge_grpo_label_correct_judgments": label_correct_judgments,
                "judge_grpo_label_total_judgments": len(turns),
                "judge_grpo_label_accuracy": label_correct_judgments / len(turns),
            }
            if reward_mode in ("label", "label_js")
            else {}
        ),
        **(
            {
                "judge_grpo_referent_js_mean": sum(js_values) / len(js_values),
                "judge_grpo_referent_js_max": max(js_values),
                "judge_grpo_reward_formula": "label_reward - referent_js_divergence/ln(2)",
            }
            if reward_mode == "label_js" and js_values
            else {}
        ),
    }


def summarize_judge_rejection_r1_projection(
    *,
    r1_examples: list[TrainExample],
    debates: list[DebateResult],
) -> dict[str, int]:
    valid_verdict_count = sum(debate.verdict in ("A", "B") for debate in debates)
    winner_r1_example_count = 0
    loser_r1_example_count = 0
    nonzero_advantage_r1_example_count = 0
    zero_advantage_r1_example_count = 0
    for example in r1_examples:
        agent = example.metadata.get("agent")
        verdict = example.metadata.get("verdict")
        if agent not in ("A", "B") or verdict not in ("A", "B"):
            continue
        if agent == verdict:
            winner_r1_example_count += 1
        else:
            loser_r1_example_count += 1
        if any(float(value) != 0.0 for value in example.advantages):
            nonzero_advantage_r1_example_count += 1
        else:
            zero_advantage_r1_example_count += 1

    classified_r1_example_count = winner_r1_example_count + loser_r1_example_count
    indexed_examples = [
        example
        for example in r1_examples
        if "r1_group_index" in example.metadata
    ]
    group_ids = {
        int(example.metadata["r1_group_index"])
        for example in indexed_examples
    }
    live_group_ids = {
        int(example.metadata["r1_group_index"])
        for example in indexed_examples
        if bool(example.metadata.get("r1_group_live", False))
    }
    return {
        "valid_verdict_count": valid_verdict_count,
        "invalid_verdict_count": len(debates) - valid_verdict_count,
        "expected_emitted_r1_example_count": valid_verdict_count,
        "emitted_r1_example_count": len(r1_examples),
        "emitted_r1_example_count_delta": len(r1_examples) - valid_verdict_count,
        "winner_r1_example_count": winner_r1_example_count,
        "winner_r1_example_count_delta": winner_r1_example_count - valid_verdict_count,
        "loser_r1_example_count": loser_r1_example_count,
        "nonzero_advantage_r1_example_count": nonzero_advantage_r1_example_count,
        "zero_advantage_r1_example_count": zero_advantage_r1_example_count,
        "rejected_loser_count": valid_verdict_count - loser_r1_example_count,
        "unclassified_r1_example_count": len(r1_examples) - classified_r1_example_count,
        "missing_group_metadata_count": len(r1_examples) - len(indexed_examples),
        "group_count": len(group_ids),
        "live_group_count": len(live_group_ids),
        "zero_variance_group_count": len(group_ids - live_group_ids),
    }
