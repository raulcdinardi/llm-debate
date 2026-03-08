from __future__ import annotations

import os
from dataclasses import dataclass, replace

from .local_config import LocalConfig


@dataclass(frozen=True)
class BackendConfig:
    base_model: str
    device: str
    rank: int


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Decoder-only generation with batched padding should be left-padded.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model_and_tokenizer(*, model_name: str, rank: int, config: LocalConfig):
    """Load a LoRA-wrapped causal LM.

    Backend selection is explicit via `TINKER_LOCAL_BACKEND`:
    - `transformers`: uses Transformers + PEFT
    """
    if config.backend == "unsloth":
        print("[warn] TINKER_LOCAL_BACKEND=unsloth is unsupported; using transformers.", flush=True)
        config = replace(config, backend="transformers")

    if config.backend != "transformers":
        raise ValueError(f"Unknown TINKER_LOCAL_BACKEND={config.backend!r}")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if config.device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    base.to(config.device)

    target_modules = None
    model_type = getattr(base.config, "model_type", None)
    if model_type == "lfm2":
        # LFM2.5 doesn't provide PEFT target module hints; specify common attention/MLP linears.
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "in_proj"]
    elif model_type == "qwen3_5":
        # Qwen3.5 may not auto-map in PEFT; target common attention + MLP projections explicitly.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if target_modules is None:
        # Fail-safe for model families that PEFT cannot auto-map in this environment.
        # "all-linear" is supported by PEFT and avoids brittle architecture name checks.
        target_modules = "all-linear"

    lora = LoraConfig(
        r=rank,
        lora_alpha=max(8, rank * 2),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(base, lora)
    model.train()

    # Reasonable default for small local runs.
    model.config.use_cache = False

    if config.device == "cuda":
        model.to(torch.device("cuda"))
    else:
        model.to(torch.device("cpu"))

    tokenizer = load_tokenizer(model_name)
    return model, tokenizer


def build_optimizer(model, learning_rate: float):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = os.environ.get("TINKER_LOCAL_OPTIMIZER", "adamw").strip().lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    if optimizer_name == "sgd":
        momentum_env = os.environ.get("TINKER_LOCAL_SGD_MOMENTUM")
        momentum = 0.0 if momentum_env is None else float(momentum_env)
        nesterov = os.environ.get("TINKER_LOCAL_SGD_NESTEROV", "0") == "1"
        if nesterov and momentum <= 0.0:
            raise ValueError("TINKER_LOCAL_SGD_NESTEROV=1 requires TINKER_LOCAL_SGD_MOMENTUM > 0.")
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, nesterov=nesterov)
    raise ValueError(
        f"Unknown TINKER_LOCAL_OPTIMIZER={optimizer_name!r}. Expected 'adamw' or 'sgd'."
    )
