"""Selective chunked projection with fused unrestricted scoring and tiny A/B heads."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def require_cuda(hidden_states):
    if hidden_states.device.type != "cuda":
        raise ValueError("train_lm_head_kernel='triton' requires CUDA; no implicit Torch fallback")
    if hidden_states.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("triton LM-head kernel supports float32/bfloat16")
    if torch.is_autocast_enabled():
        raise ValueError("triton LM-head kernel expects explicit dtypes, not autocast")


def _chunk(hidden, weight, bias, targets, temperature, restricted):
    # Preserve the reference's GEMM rounding followed by bias addition.
    logits = hidden @ weight.t()
    if bias is not None:
        logits = logits + bias
    if restricted:
        # For a two-token judge, projecting the whole vocabulary wastes almost
        # all work. Small dense Torch CE is cheaper than a multi-kernel reduction.
        logits = logits.float() / temperature
        values = -F.cross_entropy(logits, targets, reduction="none")
        with torch.no_grad():
            lp = logits.log_softmax(-1)
            entropy = -(lp.exp() * lp).sum(-1)
    else:
        from llm_local_rl.triton_logprobs import logprobs_entropy
        values, entropy = logprobs_entropy(logits, targets, temperature)
    return values, entropy


def selected_logprobs(*, hidden, weight, bias, targets, entropy_positions,
                      temperature, allowed_tokens, chunk_size):
    require_cuda(hidden)
    if targets.dtype != torch.long:
        raise ValueError("fused LM-head targets must be int64")
    if weight.dtype != hidden.dtype or (bias is not None and bias.dtype != hidden.dtype):
        raise ValueError("fused LM-head inputs and parameters must share dtype")
    if weight.device != hidden.device or (bias is not None and bias.device != hidden.device):
        raise ValueError("fused LM-head inputs and parameters must share device")
    if bool(((targets < 0) | (targets >= weight.shape[0])).any().item()):
        raise ValueError("LM-head target outside vocabulary")
    groups = {}
    for row, allowed in enumerate(allowed_tokens if allowed_tokens is not None else [()] * len(targets)):
        groups.setdefault(allowed, []).append(row)
    values = torch.empty(len(targets), dtype=torch.float32, device=hidden.device)
    entropy_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for allowed, rows in groups.items():
        indices = torch.tensor(rows, dtype=torch.long, device=hidden.device)
        group_targets = targets[indices]
        group_weight, group_bias = weight, bias
        if allowed:
            if len(set(allowed)) != len(allowed) or min(allowed) < 0 or max(allowed) >= weight.shape[0]:
                raise ValueError("allowed-token set must contain unique vocabulary IDs")
            allowed_ids = torch.tensor(allowed, dtype=torch.long, device=hidden.device)
            matches = group_targets[:, None] == allowed_ids[None, :]
            if not bool(matches.any(-1).all().item()):
                raise ValueError("Target token escaped the configured allowed-token set")
            group_targets = matches.long().argmax(-1)
            group_weight = weight[allowed_ids]
            group_bias = None if bias is None else bias[allowed_ids]
        for start in range(0, len(rows), chunk_size):
            end = start + chunk_size
            chunk_indices = indices[start:end]
            args = (hidden[chunk_indices], group_weight, group_bias,
                    group_targets[start:end], temperature, bool(allowed))
            if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in args[:3]):
                lp, ent = checkpoint(_chunk, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                lp, ent = _chunk(*args)
            values = values.index_copy(0, chunk_indices, lp)
            entropy_sum += ent.detach()[entropy_positions[chunk_indices]].sum()
    # One CPU synchronization, rather than one for every vocabulary chunk.
    return values, float(entropy_sum.item())
