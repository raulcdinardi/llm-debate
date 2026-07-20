from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_local_rl.types import AdapterName, TrainExample


@dataclass(frozen=True)
class OnPolicyLogprobCheckResult:
    num_checked_tokens: int
    num_zero_advantage_loss_mask_tokens_skipped: int
    num_violations: int
    sum_abs_logprob_diff: float
    max_abs_logprob_diff: float
    first_offending_trained_token: dict[str, Any] | None
    records: list[dict[str, Any]]

    def metrics(self) -> dict[str, float]:
        mean_abs = (
            self.sum_abs_logprob_diff / self.num_checked_tokens
            if self.num_checked_tokens > 0
            else 0.0
        )
        return {
            "trained_tokens_checked": float(self.num_checked_tokens),
            "zero_advantage_loss_mask_tokens_skipped": float(
                self.num_zero_advantage_loss_mask_tokens_skipped
            ),
            "trained_token_mean_abs_diff": float(mean_abs),
            "trained_token_max_abs_diff": float(self.max_abs_logprob_diff),
            "on_policy_logprob_checked_tokens": float(self.num_checked_tokens),
            "on_policy_logprob_violations": float(self.num_violations),
            "on_policy_logprob_mean_abs_diff": float(mean_abs),
            "on_policy_logprob_max_abs_diff": float(self.max_abs_logprob_diff),
            "on_policy_logprob_trained_tokens_checked": float(self.num_checked_tokens),
            "on_policy_logprob_zero_advantage_loss_mask_tokens_skipped": float(
                self.num_zero_advantage_loss_mask_tokens_skipped
            ),
            "on_policy_logprob_trained_token_mean_abs_diff": float(mean_abs),
            "on_policy_logprob_trained_token_max_abs_diff": float(self.max_abs_logprob_diff),
        }


def truncation_start(*, token_count: int, max_tokens: int) -> int:
    if max_tokens <= 0 or token_count <= max_tokens:
        return 0
    return token_count - max_tokens


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=False))


def check_on_policy_logprobs(
    *,
    adapter_name: AdapterName,
    examples: list[TrainExample],
    current_logprob_rows: list[list[float]],
    tokenizer: Any,
    abs_tol: float,
    max_tokens: int,
    max_records: int,
    minibatch_start: int,
) -> OnPolicyLogprobCheckResult:
    if len(examples) != len(current_logprob_rows):
        raise ValueError("examples and current_logprob_rows must have equal length.")

    num_checked_tokens = 0
    num_zero_advantage_loss_mask_tokens_skipped = 0
    num_violations = 0
    sum_abs_logprob_diff = 0.0
    max_abs_logprob_diff = 0.0
    first_offending_trained_token: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []

    for row_idx, example in enumerate(examples):
        start = truncation_start(token_count=len(example.input_ids), max_tokens=max_tokens)
        input_ids = example.input_ids[start:]
        target_ids = example.target_ids[start:]
        loss_mask = example.loss_mask[start:]
        old_logprobs = example.old_logprobs[start:]
        advantages = example.advantages[start:]
        current_logprobs = current_logprob_rows[row_idx]
        if not (
            len(input_ids)
            == len(target_ids)
            == len(loss_mask)
            == len(old_logprobs)
            == len(advantages)
            == len(current_logprobs)
        ):
            raise ValueError("On-policy logprob check rows must align after truncation.")

        sequence_token_ids = list(input_ids)
        if target_ids:
            sequence_token_ids.append(target_ids[-1])
        sequence_text = _decode(tokenizer, sequence_token_ids)

        first_loss_mask_pos = loss_mask.index(1) if 1 in loss_mask else None
        prompt_token_count = start + first_loss_mask_pos + 1 if first_loss_mask_pos is not None else len(input_ids)

        for sliced_pos, should_check in enumerate(loss_mask):
            if not should_check:
                continue
            advantage = float(advantages[sliced_pos])
            if advantage == 0.0:
                num_zero_advantage_loss_mask_tokens_skipped += 1
                continue
            old_logprob = float(old_logprobs[sliced_pos])
            current_logprob = float(current_logprobs[sliced_pos])
            diff = current_logprob - old_logprob
            abs_diff = abs(diff)
            num_checked_tokens += 1
            sum_abs_logprob_diff += abs_diff
            max_abs_logprob_diff = max(max_abs_logprob_diff, abs_diff)
            if abs_diff <= abs_tol:
                continue
            num_violations += 1
            record = {
                "event": "on_policy_logprob_drift",
                "adapter_name": adapter_name,
                "minibatch_start": minibatch_start,
                "minibatch_row": row_idx,
                "example_index": minibatch_start + row_idx,
                "original_token_position": start + sliced_pos,
                "sliced_token_position": sliced_pos,
                "input_token_id": int(input_ids[sliced_pos]),
                "target_token_id": int(target_ids[sliced_pos]),
                "target_token_text": _decode(tokenizer, [int(target_ids[sliced_pos])]),
                "sequence_text": sequence_text,
                "prompt_token_count": int(prompt_token_count),
                "completion_token_position": (
                    int(sliced_pos - first_loss_mask_pos) if first_loss_mask_pos is not None else None
                ),
                "old_logprob": old_logprob,
                "current_logprob": current_logprob,
                "diff": diff,
                "abs_diff": abs_diff,
                "abs_tol": float(abs_tol),
                "advantage": advantage,
                "trained_token": True,
                "metadata": dict(example.metadata),
            }
            if first_offending_trained_token is None:
                first_offending_trained_token = dict(record)
            if len(records) < max_records:
                records.append(record)

    return OnPolicyLogprobCheckResult(
        num_checked_tokens=num_checked_tokens,
        num_zero_advantage_loss_mask_tokens_skipped=num_zero_advantage_loss_mask_tokens_skipped,
        num_violations=num_violations,
        sum_abs_logprob_diff=sum_abs_logprob_diff,
        max_abs_logprob_diff=max_abs_logprob_diff,
        first_offending_trained_token=first_offending_trained_token,
        records=records,
    )
