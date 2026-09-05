from __future__ import annotations

import math
from typing import Any

import torch


def _percentile(values: torch.Tensor, quantile: float) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(torch.quantile(values.float(), quantile).detach().cpu().item())


def _selected_logprobs(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}.")
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError(
            f"target_ids shape {tuple(target_ids.shape)} does not match logits prefix "
            f"{tuple(logits.shape[:-1])}."
        )
    return torch.log_softmax(logits.float() / float(temperature), dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)


def _cell_metrics(
    *,
    old_logprobs: torch.Tensor,
    current_logprobs: torch.Tensor,
    abs_tol: float,
    ppo_clip_epsilon: float,
) -> dict[str, float | int | bool]:
    diffs = current_logprobs.float() - old_logprobs.float()
    abs_diffs = diffs.abs()
    ratios = diffs.exp()
    clipped = (ratios < (1.0 - ppo_clip_epsilon)) | (
        ratios > (1.0 + ppo_clip_epsilon)
    )
    return {
        "n": int(diffs.numel()),
        "mean_signed_diff": float(diffs.mean().detach().cpu().item()),
        "mean_abs_diff": float(abs_diffs.mean().detach().cpu().item()),
        "p50_abs_diff": _percentile(abs_diffs, 0.50),
        "p95_abs_diff": _percentile(abs_diffs, 0.95),
        "p99_abs_diff": _percentile(abs_diffs, 0.99),
        "max_abs_diff": float(abs_diffs.max().detach().cpu().item()),
        "violation_fraction": float((abs_diffs > abs_tol).float().mean().detach().cpu().item()),
        "zero_update_clip_fraction": float(clipped.float().mean().detach().cpu().item()),
        "ratio_mean": float(ratios.mean().detach().cpu().item()),
        "numeric_parity_pass": bool(torch.all(abs_diffs <= abs_tol).detach().cpu().item()),
    }


def run_temperature_four_cell_microprobe(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    behavior_temperature: float,
    abs_tol: float = 1e-6,
    ppo_clip_epsilon: float = 0.1,
) -> dict[str, Any]:
    """Evaluate the post-T/original × trainer-T/T1 counterfactual matrix.

    This is a zero-update semantic probe. The original/T1 cell is expected to
    have numerical parity, but it is explicitly not strict on-policy when the
    actions were sampled at a non-unit behavior temperature.
    """

    if not 0.0 < float(ppo_clip_epsilon) < 1.0:
        raise ValueError("ppo_clip_epsilon must be in (0, 1).")
    if float(abs_tol) < 0.0:
        raise ValueError("abs_tol must be non-negative.")

    post_temperature = _selected_logprobs(
        logits=logits,
        target_ids=target_ids,
        temperature=behavior_temperature,
    )
    original = _selected_logprobs(
        logits=logits,
        target_ids=target_ids,
        temperature=1.0,
    )
    sampler_cells = {
        "post_temperature": post_temperature,
        "original": original,
    }
    trainer_cells = {
        "behavior_temperature": post_temperature,
        "temperature_1": original,
    }

    cells: dict[str, dict[str, Any]] = {}
    for sampler_name, old_logprobs in sampler_cells.items():
        for trainer_name, current_logprobs in trainer_cells.items():
            key = f"{sampler_name}__trainer_{trainer_name}"
            strict_behavior_alignment = (
                sampler_name == "post_temperature"
                and trainer_name == "behavior_temperature"
            ) or (
                float(behavior_temperature) == 1.0
                and sampler_name == "original"
                and trainer_name == "temperature_1"
            )
            cells[key] = {
                "sampler_semantics": sampler_name,
                "trainer_semantics": trainer_name,
                "strict_behavior_policy_alignment": strict_behavior_alignment,
                **_cell_metrics(
                    old_logprobs=old_logprobs,
                    current_logprobs=current_logprobs,
                    abs_tol=abs_tol,
                    ppo_clip_epsilon=ppo_clip_epsilon,
                ),
            }

    raw_log_probs = torch.log_softmax(logits.float(), dim=-1)
    behavior_log_probs = torch.log_softmax(
        logits.float() / float(behavior_temperature),
        dim=-1,
    )
    sorted_raw, sorted_ids = torch.sort(raw_log_probs, dim=-1)
    sorted_behavior = behavior_log_probs.gather(dim=-1, index=sorted_ids)
    raw_pair_delta = sorted_raw[..., 1] - sorted_raw[..., 0]
    behavior_pair_delta = sorted_behavior[..., 1] - sorted_behavior[..., 0]
    finite_pairs = raw_pair_delta.abs() > 1e-12
    rare_pair_slopes = behavior_pair_delta[finite_pairs] / raw_pair_delta[finite_pairs]

    return {
        "behavior_temperature": float(behavior_temperature),
        "expected_inverse_temperature": 1.0 / float(behavior_temperature),
        "abs_tol": float(abs_tol),
        "ppo_clip_epsilon": float(ppo_clip_epsilon),
        "rare_token_pair_slope_p50": _percentile(rare_pair_slopes, 0.50),
        "rare_token_pair_slope_p99": _percentile(rare_pair_slopes, 0.99),
        "cells": cells,
    }
