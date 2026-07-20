"""Per-step GPU memory attribution for MultiAdapterTrainer.

Records structured CSV rows describing CUDA memory statistics sampled at
distinct points in a training minibatch:
  - just before pad_batch tensors are materialized (constant baseline holds)
  - after the backbone forward (selective_lm_head path: backbone hidden_states)
  - after the lm-head projection over trained positions
      (selective_lm_head only; for full_logits the backbone+lm_head call is
      the full model(...) forward, so "backbone_selective_*" columns stay empty
      and the "forward" peak captures the combined cost)
  - immediately after `loss.backward()` (autograd context peak)
  - once per train_batch, after `optimizer.step()` (captures optim state lazy
    allocation delta on the first step)

Each row also carries metadata (step, adapter_name, minibatch index,
seq_len_max, num_examples, num_trained_tokens, num_padded_positions,
train_logprob_backend) so a single CSV across the whole run can be
aggregated into a stacked-area plot per training step.

Strictly opt-in via env var `TINKER_LOCAL_MEMORY_TRACE_PATH`. When that env
var is unset, `MaybeMemoryTraceRecorder` returns None and the trainer code
path skips every assignment that depends on it (the trainer checks the
returned recorder truthiness before touching sampler helpers, so importing
this module costs absolutely nothing in the default code path).
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Optional

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def trace_path_from_env() -> Optional[str]:
    return os.environ.get("TINKER_LOCAL_MEMORY_TRACE_PATH") or None


def current_alloc_bytes(device: str) -> int:
    if device != "cuda" or not _HAS_TORCH:
        return 0
    import torch
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def reserved_bytes(device: str) -> int:
    if device != "cuda" or not _HAS_TORCH:
        return 0
    import torch
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_reserved())


def reset_peak(device: str) -> None:
    if device != "cuda" or not _HAS_TORCH:
        return
    import torch
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


def peak_alloc_bytes(device: str) -> int:
    if device != "cuda" or not _HAS_TORCH:
        return 0
    import torch
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


class MemoryTraceRecorder:
    """Append-only CSV recorder for per-minibatch CUDA memory attribution.

    One recorder instance is constructed per trainer (lazily, only when
    `TINKER_LOCAL_MEMORY_TRACE_PATH` is set). The recorder writes a single CSV
    file: one row per (step, adapter, minibatch). Per-step aggregation for the
    stacked-area chart is the plot script's responsibility.

    `weights_bytes_constant` is captured lazily at the first minibatch
    (allocator snapshot right before pad_batch, when model+optimizer are the
    only thing resident). `optim_state_bytes_constant` is captured once after
    the first `optimizer.step()`. Both are then echoed on every subsequent
    row so the plot script can fold the constant component into the stack
    without recomputing it from the same env.
    """

    FIELDS = [
        "step",
        "adapter_name",
        "minibatch_idx",
        "seq_len_max",
        "num_examples",
        "num_trained_tokens",
        "num_padded_positions",
        "train_logprob_backend",
        "weights_bytes_constant",
        "optim_state_bytes_constant",
        "alloc_before_mb",
        "alloc_after_forward_full_logits",
        "peak_after_forward_full_logits",
        "alloc_after_backbone_selective",
        "peak_after_backbone_selective",
        "alloc_after_lm_head_selective",
        "peak_after_lm_head_selective",
        "alloc_after_backward",
        "peak_after_backward",
        "alloc_after_optim_step",
        "peak_during_optim_step",
        "reserved_bytes_end",
        "wall_clock_s",
        "phase_elapsed_forward_s",
        "phase_elapsed_lm_head_s",
        "phase_elapsed_backward_s",
    ]

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.weights_bytes_constant = 0
        self.weights_bytes_constant_captured = False
        self.optim_state_bytes_constant = 0
        self.optim_state_bytes_constant_captured = False
        self._step = -1
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        write_header = not Path(self.path).exists() or Path(self.path).stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()
        self._fh.flush()

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def maybe_capture_weights_baseline(self, alloc_bytes: int) -> None:
        if not self.weights_bytes_constant_captured:
            self.weights_bytes_constant = int(alloc_bytes)
            self.weights_bytes_constant_captured = True

    def maybe_capture_optim_state_baseline(self, alloc_bytes: int) -> None:
        if not self.optim_state_bytes_constant_captured:
            # Optim state is everything allocated beyond the frozen weights
            # baseline (LoRA master fp32 copies + Adam m/v). No transient
            # activation should be alive at this point: grads are zeroed
            # before this call, and we sample immediately after step().
            self.optim_state_bytes_constant = max(0, int(alloc_bytes) - self.weights_bytes_constant)
            self.optim_state_bytes_constant_captured = True

    def append(self, row: dict[str, object]) -> None:
        clean = {key: row.get(key, "") for key in self.FIELDS}
        clean["weights_bytes_constant"] = self.weights_bytes_constant
        clean["optim_state_bytes_constant"] = self.optim_state_bytes_constant
        self._writer.writerow(clean)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def maybe_create_recorder() -> Optional["MemoryTraceRecorder"]:
    path = trace_path_from_env()
    if path is None:
        return None
    return MemoryTraceRecorder(path=path)


def now_seconds() -> float:
    return time.perf_counter()