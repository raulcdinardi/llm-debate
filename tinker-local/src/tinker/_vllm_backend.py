from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VllmConfig:
    gpu_memory_utilization: float
    max_model_len: int | None
    tensor_parallel_size: int
    enforce_eager: bool

    @classmethod
    def from_env(cls) -> "VllmConfig":
        gpu_mem = os.environ.get("TINKER_LOCAL_VLLM_GPU_MEMORY_UTILIZATION")
        max_len = os.environ.get("TINKER_LOCAL_VLLM_MAX_MODEL_LEN")
        tp = os.environ.get("TINKER_LOCAL_VLLM_TENSOR_PARALLEL_SIZE")
        eager = os.environ.get("TINKER_LOCAL_VLLM_ENFORCE_EAGER")

        gpu_memory_utilization = 0.90 if gpu_mem is None else float(gpu_mem)
        max_model_len = None if max_len is None else int(max_len)
        tensor_parallel_size = 1 if tp is None else int(tp)
        enforce_eager = False if eager is None else (eager == "1")

        if tensor_parallel_size <= 0:
            raise ValueError(f"TINKER_LOCAL_VLLM_TENSOR_PARALLEL_SIZE must be > 0, got {tensor_parallel_size}")
        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"TINKER_LOCAL_VLLM_GPU_MEMORY_UTILIZATION must be in (0,1], got {gpu_memory_utilization}"
            )
        if max_model_len is not None and max_model_len <= 0:
            raise ValueError(f"TINKER_LOCAL_VLLM_MAX_MODEL_LEN must be > 0, got {max_model_len}")

        return cls(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
        )


_VLLM_LLM = None
_VLLM_BASE_MODEL: str | None = None


def get_vllm_llm(*, base_model: str):
    global _VLLM_LLM, _VLLM_BASE_MODEL
    if _VLLM_LLM is not None:
        if _VLLM_BASE_MODEL != base_model:
            raise ValueError(
                f"vLLM LLM already initialized with base_model={_VLLM_BASE_MODEL!r}, cannot switch to {base_model!r}"
            )
        return _VLLM_LLM

    from vllm import LLM

    cfg = VllmConfig.from_env()
    max_lora_rank_env = os.environ.get("TINKER_LOCAL_VLLM_MAX_LORA_RANK")
    max_lora_rank = 32 if max_lora_rank_env is None else int(max_lora_rank_env)
    if max_lora_rank <= 0:
        raise ValueError(f"TINKER_LOCAL_VLLM_MAX_LORA_RANK must be > 0, got {max_lora_rank}")
    _VLLM_BASE_MODEL = base_model
    _VLLM_LLM = LLM(
        model=base_model,
        enable_lora=True,
        max_loras=int(os.environ.get("TINKER_LOCAL_VLLM_MAX_LORAS", "128")),
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        enforce_eager=cfg.enforce_eager,
    )
    return _VLLM_LLM
