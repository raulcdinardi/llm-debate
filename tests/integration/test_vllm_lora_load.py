from __future__ import annotations

from pathlib import Path
import shutil

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


def test_vllm_sleep_cycle_refreshes_trainable_lora_and_keeps_frozen_lora(tmp_path: Path) -> None:
    assets = IntegrationAssets.from_env()
    refreshed_adapter = tmp_path / "adapter_a_refreshed"
    shutil.copytree(assets.adapter_a_path, refreshed_adapter)
    sampler = VllmSampler(
        runtime=VllmRuntimeConfig(model_path=assets.base_model_path, enable_sleep_mode=True),
        adapter_paths={"solution": assets.adapter_a_path, "judge": assets.adapter_b_path},
    )
    tokenizer = AutoTokenizer.from_pretrained(assets.base_model_path, use_fast=True)
    prompt_token_ids = tokenizer.encode("Choose A or B.", add_special_tokens=False)
    request_kwargs = {
        "prompt_token_ids": prompt_token_ids,
        "max_tokens": 2,
        "temperature": 0.0,
        "seed": 0,
    }

    sampler.sample(SamplingRequest(adapter_name="solution", **request_kwargs))
    sampler.sample(SamplingRequest(adapter_name="judge", **request_kwargs))
    old_solution_id = sampler._adapter_ids["solution"]
    frozen_judge_id = sampler._adapter_ids["judge"]

    sampler.unload_adapters(adapter_names={"solution"})
    assert old_solution_id not in sampler.llm.llm_engine.list_loras()
    assert frozen_judge_id in sampler.llm.llm_engine.list_loras()
    sampler.sleep(level=1)
    sampler.set_adapter_paths(
        adapter_paths={"solution": str(refreshed_adapter), "judge": assets.adapter_b_path}
    )
    sampler.wake_up()
    sampler.sample(SamplingRequest(adapter_name="solution", **request_kwargs))
    sampler.sample(SamplingRequest(adapter_name="judge", **request_kwargs))

    assert sampler._adapter_ids["solution"] != old_solution_id
    assert sampler._adapter_ids["judge"] == frozen_judge_id
    assert old_solution_id not in sampler.llm.llm_engine.list_loras()
    assert sampler._adapter_ids["solution"] in sampler.llm.llm_engine.list_loras()
    assert frozen_judge_id in sampler.llm.llm_engine.list_loras()
