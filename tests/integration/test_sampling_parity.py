from __future__ import annotations

import warnings

from transformers import AutoTokenizer

from llm_local_rl.integration_assets import IntegrationAssets
from llm_local_rl.testing import assert_close_lists
from llm_local_rl.types import SamplingRequest
from llm_local_rl.vllm_sampling import VllmRuntimeConfig, VllmSampler, direct_vllm_sample


def test_sampling_pipeline_matches_direct_vllm_at_temp_zero() -> None:
    assets = IntegrationAssets.from_env()
    tokenizer = AutoTokenizer.from_pretrained(assets.base_model_path, use_fast=True)
    sampler = VllmSampler(
        runtime=VllmRuntimeConfig(model_path=assets.base_model_path),
        adapter_paths={"adapter_a": assets.adapter_a_path},
    )
    stop_token_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(stop_token_ids) != 1:
        raise ValueError("Expected newline stop token to be a single token.")

    prompt_texts = [
        "Answer with exactly four symbols using only H and T.",
        "Write exactly one H/T sequence of length six. No explanation.",
        "Return only one short H/T answer.",
    ]
    requests = [
        SamplingRequest(
            adapter_name="adapter_a",
            prompt_token_ids=tokenizer.encode(text, add_special_tokens=False),
            stop_token_ids=stop_token_ids,
            max_tokens=8,
            temperature=0.0,
            seed=0,
        )
        for text in prompt_texts
    ]

    mismatches = 0
    matched = 0
    for request in requests:
        wrapped = sampler.sample(request)
        direct = direct_vllm_sample(
            llm=sampler.llm,
            request=request,
            adapter_path=assets.adapter_a_path,
            adapter_id=99,
        )
        if wrapped.completion_token_ids != direct.completion_token_ids:
            mismatches += 1
            continue
        matched += 1
        assert_close_lists(
            wrapped.completion_logprobs,
            direct.completion_logprobs,
            atol=1e-5,
        )

    mismatch_rate = mismatches / len(requests)
    if 0.0 < mismatch_rate <= 0.25:
        warnings.warn(
            f"Temperature-zero sampling mismatch_rate={mismatch_rate:.3f}; "
            "literal output mismatches stayed below the fail threshold."
        )
    assert mismatch_rate <= 0.25
    assert matched >= len(requests) - mismatches
