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

    trust_remote_code = os.environ.get("TINKER_LOCAL_TRUST_REMOTE_CODE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if os.environ.get("TINKER_LOCAL_ADD_SENTINELS") == "1":
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|tinker_sentinel_a|>", "<|tinker_sentinel_b|>"]}
        )
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
    from transformers import AutoConfig, AutoModelForCausalLM, Mistral3ForConditionalGeneration

    if config.device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tokenizer = load_tokenizer(model_name)

    trust_remote_code = os.environ.get("TINKER_LOCAL_TRUST_REMOTE_CODE") == "1"
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if getattr(cfg, "model_type", None) == "mistral3":
        base = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))
    # Mistral3 weights are FP8 by default; cast to the requested dtype to avoid FP8 matmul failures.
    base.to(device=config.device, dtype=dtype)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    target_modules = None
    model_type = getattr(base.config, "model_type", None)
    if model_type == "lfm2":
        # LFM2.5 doesn't provide PEFT target module hints; specify common attention/MLP linears.
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "in_proj"]
    if model_type in ("mistral3", "ministral3"):
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

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

    return model, tokenizer


def build_optimizer(model, learning_rate: float):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=learning_rate)
