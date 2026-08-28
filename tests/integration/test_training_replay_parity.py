from __future__ import annotations

import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_local_rl.integration_assets import IntegrationAssets
from llm_local_rl.replay import (
    build_replay_example,
    collect_trainable_grads,
    replay_loss_loop,
    replay_loss_vectorized,
    target_logprobs_from_model,
)


def _load_trainable_model(*, base_model_path: str, adapter_path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.train()
    model.config.use_cache = False
    model.to(device)
    return model


def _assert_grad_sets_close(
    grads_a: dict[str, torch.Tensor],
    grads_b: dict[str, torch.Tensor],
    *,
    atol: float,
) -> None:
    if grads_a.keys() != grads_b.keys():
        raise AssertionError("Gradient parameter-name sets differ.")
    worst_name = None
    worst_diff = -1.0
    for name in grads_a:
        diff = float(torch.max(torch.abs(grads_a[name] - grads_b[name])).item())
        if diff > worst_diff:
            worst_diff = diff
            worst_name = name
    if worst_diff > atol:
        raise AssertionError(
            f"Gradient mismatch worst_param={worst_name} max_abs_diff={worst_diff} atol={atol}"
        )


def test_training_replay_gradient_matches_reference_loop() -> None:
    assets = IntegrationAssets.from_env()
    tokenizer = AutoTokenizer.from_pretrained(assets.base_model_path, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_token_ids = tokenizer.encode(
        "Write exactly one sequence of four symbols using only H and T.",
        add_special_tokens=False,
    )
    completion_token_ids = tokenizer.encode(" HHTT", add_special_tokens=False)
    behavior_temperature = 0.8

    seed = 0
    torch.manual_seed(seed)
    model_for_baseline = _load_trainable_model(
        base_model_path=assets.base_model_path,
        adapter_path=assets.adapter_a_path,
        device=device,
    )
    with torch.no_grad():
        baseline_logprobs = target_logprobs_from_model(
            model=model_for_baseline,
            input_ids=(prompt_token_ids + completion_token_ids)[:-1],
            target_ids=(prompt_token_ids + completion_token_ids)[1:],
            device=device,
            behavior_temperature=behavior_temperature,
        ).detach().cpu().tolist()
    prompt_prefix_len = len(prompt_token_ids) - 1
    completion_logprobs = baseline_logprobs[prompt_prefix_len:]
    example = build_replay_example(
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        completion_logprobs=completion_logprobs,
        advantage_per_token=1.0,
    )
    del model_for_baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(seed)
    model_loop = _load_trainable_model(
        base_model_path=assets.base_model_path,
        adapter_path=assets.adapter_a_path,
        device=device,
    )
    loss_loop = replay_loss_loop(
        model=model_loop,
        example=example,
        device=device,
        behavior_temperature=behavior_temperature,
    )
    loss_loop.backward()
    grads_loop = collect_trainable_grads(model_loop)
    loss_loop_value = float(loss_loop.detach().cpu().item())
    del model_loop
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(seed)
    model_vectorized = _load_trainable_model(
        base_model_path=assets.base_model_path,
        adapter_path=assets.adapter_a_path,
        device=device,
    )
    loss_vectorized = replay_loss_vectorized(
        model=model_vectorized,
        example=example,
        device=device,
        behavior_temperature=behavior_temperature,
    )
    loss_vectorized.backward()
    grads_vectorized = collect_trainable_grads(model_vectorized)
    loss_vectorized_value = float(loss_vectorized.detach().cpu().item())

    assert abs(loss_loop_value - loss_vectorized_value) <= 1e-5
    _assert_grad_sets_close(grads_loop, grads_vectorized, atol=1e-5)
