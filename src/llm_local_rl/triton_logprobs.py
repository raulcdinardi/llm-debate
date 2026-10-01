"""Chunk-local log probability + diagnostic entropy, without FP32 vocabulary buffers.

Projection remains a Torch/cuBLAS GEMM. These kernels fuse temperature scaling,
normalization, target gathering, entropy and the arbitrary per-token VJP. The
caller checkpoints each projection/loss chunk to bound saved vocabulary memory.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _partial_stats(X, P, V: tl.constexpr, TILES: tl.constexpr,
                   TEMPERATURE: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(X + row * V + col, col < V, other=-float("inf")).to(tl.float32) / TEMPERATURE
    maximum = tl.max(z, 0)
    shifted = z - maximum
    p = tl.exp(shifted)
    total = tl.sum(p, 0)
    # Mask before multiplication: zero times the padded -inf would be NaN.
    moment = tl.sum(p * tl.where(col < V, shifted, 0.0), 0)
    offset = (row * TILES + tile) * 3
    tl.store(P + offset, maximum)
    tl.store(P + offset + 1, total)
    tl.store(P + offset + 2, moment)


@triton.jit
def _finish_stats(X, Y, P, S, LP, ENT, V: tl.constexpr, TILES: tl.constexpr,
                  TEMPERATURE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    offset = (row * TILES + tile) * 3
    maxima = tl.load(P + offset, tile < TILES, other=-float("inf"))
    sums = tl.load(P + offset + 1, tile < TILES, other=0.0)
    moments = tl.load(P + offset + 2, tile < TILES, other=0.0)
    maximum = tl.max(maxima, 0)
    shift = tl.where(tile < TILES, maxima - maximum, 0.0)
    scale = tl.exp(shift)
    total = tl.sum(scale * sums, 0)
    moment = tl.sum(scale * (moments + shift * sums), 0)
    log_total = tl.log(total)
    target = tl.load(Y + row)
    target_logit = tl.load(X + row * V + target).to(tl.float32) / TEMPERATURE
    tl.store(LP + row, (target_logit - maximum) - log_total)
    tl.store(ENT + row, log_total - moment / total)
    tl.store(S + row * 2, maximum)
    tl.store(S + row * 2 + 1, 1.0 / total)


@triton.jit
def _backward(X, Y, S, GO, DX, V: tl.constexpr, GO_STRIDE: tl.constexpr,
              TEMPERATURE: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(X + row * V + col, col < V, other=-float("inf")).to(tl.float32) / TEMPERATURE
    maximum = tl.load(S + row * 2)
    inverse_sum = tl.load(S + row * 2 + 1)
    p = tl.exp(z - maximum) * inverse_sum
    target = tl.load(Y + row)
    upstream = tl.load(GO + row * GO_STRIDE).to(tl.float32)
    gradient = ((col == target).to(tl.float32) - p) * upstream / TEMPERATURE
    tl.store(DX + row * V + col, gradient, col < V)


class _LogprobsEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, temperature):
        logits, targets = logits.contiguous(), targets.contiguous()
        rows, vocab = logits.shape
        block = min(4096, triton.next_power_of_2(vocab))
        tiles = triton.cdiv(vocab, block)
        partial = torch.empty((rows, tiles, 3), device=logits.device, dtype=torch.float32)
        stats = torch.empty((rows, 2), device=logits.device, dtype=torch.float32)
        values = torch.empty(rows, device=logits.device, dtype=torch.float32)
        entropy = torch.empty_like(values)
        _partial_stats[(rows, tiles)](
            logits, partial, vocab, tiles, temperature, block, enable_fp_fusion=False,
        )
        _finish_stats[(rows,)](
            logits, targets, partial, stats, values, entropy, vocab, tiles,
            temperature, triton.next_power_of_2(tiles), enable_fp_fusion=False,
        )
        ctx.save_for_backward(logits, targets, stats)
        ctx.temperature, ctx.block, ctx.tiles = temperature, block, tiles
        # Entropy is a diagnostic in the trainer, not an entropy regularizer.
        ctx.mark_non_differentiable(entropy)
        ctx.set_materialize_grads(False)
        return values, entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_values, grad_entropy):
        if grad_values is None:
            return None, None, None
        logits, targets, stats = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        _backward[(logits.shape[0], ctx.tiles)](
            logits, targets, stats, grad_values, grad_logits, logits.shape[1],
            grad_values.stride(0), ctx.temperature, ctx.block, enable_fp_fusion=False,
        )
        return grad_logits, None, None


def logprobs_entropy(logits, targets, temperature):
    """Internal validated-input primitive; also exercised by the CPU interpreter."""
    return _LogprobsEntropy.apply(logits, targets, temperature)
