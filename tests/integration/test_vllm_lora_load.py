from __future__ import annotations

from transformers import AutoTokenizer

from llm_local_rl.integration_assets import IntegrationAssets
from llm_local_rl.types import SamplingRequest
from llm_local_rl.vllm_sampling import VllmRuntimeConfig, VllmSampler


def test_vllm_loads_two_real_loras_on_one_engine() -> None:
    assets = IntegrationAssets.from_env()
    tokenizer = AutoTokenizer.from_pretrained(assets.base_model_path, use_fast=True)
    sampler = VllmSampler(
        runtime=VllmRuntimeConfig(model_path=assets.base_model_path),
        adapter_paths={
            "adapter_a": assets.adapter_a_path,
            "adapter_b": assets.adapter_b_path,
        },
    )
    prompt_token_ids = tokenizer.encode(
        "Answer with exactly four symbols using only H and T.",
        add_special_tokens=False,
    )
    stop_token_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(stop_token_ids) != 1:
        raise ValueError("Expected newline stop token to be a single token.")

    out_a = sampler.sample(
        SamplingRequest(
            adapter_name="adapter_a",
            prompt_token_ids=prompt_token_ids,
            stop_token_ids=stop_token_ids,
            max_tokens=8,
            temperature=0.0,
            seed=0,
        )
    )
    out_b = sampler.sample(
        SamplingRequest(
            adapter_name="adapter_b",
            prompt_token_ids=prompt_token_ids,
            stop_token_ids=stop_token_ids,
            max_tokens=8,
            temperature=0.0,
            seed=0,
        )
    )

    assert len(out_a.completion_token_ids) > 0
    assert len(out_b.completion_token_ids) > 0
    assert len(out_a.completion_token_ids) == len(out_a.completion_logprobs)
    assert len(out_b.completion_token_ids) == len(out_b.completion_logprobs)
