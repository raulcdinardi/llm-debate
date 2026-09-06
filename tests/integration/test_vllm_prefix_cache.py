"""Opt-in real-GPU cache test; use LFM2.5 with two compatible LoRA artifacts."""
from dataclasses import replace
import json
import shutil

import pytest
from transformers import AutoTokenizer

from llm_local_rl.integration_assets import IntegrationAssets
from llm_local_rl.types import SamplingRequest
from llm_local_rl.vllm_sampling import VllmRuntimeConfig, VllmSampler


def test_same_lora_round_extension_reuses_cache_and_matches_cold(tmp_path):
    assets = IntegrationAssets.from_env()
    tokenizer = AutoTokenizer.from_pretrained(assets.base_model_path)
    sampler = VllmSampler(
        runtime=VllmRuntimeConfig(
            model_path=assets.base_model_path, max_model_len=4096, max_num_seqs=1,
            gpu_memory_utilization=0.75, max_loras=2, enable_prefix_caching=True,
        ),
        adapter_paths={"debate": assets.adapter_a_path, "judge": assets.adapter_b_path},
    )
    # Capture the backend's real cache counter while exercising sample(), not a mock.
    outputs = []
    generate = sampler.llm.generate
    def capture(*args, **kwargs):
        result = generate(*args, **kwargs)
        outputs.extend(result)
        return result
    sampler.llm.generate = capture
    try:
        prefix = tokenizer.encode("A debate about a story.\n" * 256, add_special_tokens=False)
        request = SamplingRequest(adapter_name="debate", prompt_token_ids=prefix,
                                  max_tokens=8, temperature=0.0, seed=17)
        r2 = sampler.sample(request)
        extension = tokenizer.encode("\nOpponent: explain your reasoning.\nResponse:", add_special_tokens=False)
        r3_request = replace(request, prompt_token_ids=prefix + r2.completion_token_ids + extension)
        cached = sampler.sample(r3_request)
        cache_hits = outputs[-1].num_cached_tokens
        assert cache_hits is not None and cache_hits > 0
        assert cache_hits <= len(prefix) + len(r2.completion_token_ids)

        # Distinct adapters must not consume one another's cached state.
        assert sampler._adapter_ids["judge"] != sampler._adapter_ids["debate"]
        sampler.sample(replace(r3_request, adapter_name="judge"))
        assert outputs[-1].num_cached_tokens == 0

        sampler.llm.reset_prefix_cache()
        cold = sampler.sample(r3_request)
        assert outputs[-1].num_cached_tokens == 0
        assert cached.completion_token_ids == cold.completion_token_ids
        assert cached.completion_logprobs == pytest.approx(cold.completion_logprobs, abs=0.01, rel=0)

        # A freshly saved adapter identity must not hit stale pre-update cache entries.
        refreshed = tmp_path / "refreshed_debate"
        shutil.copytree(assets.adapter_a_path, refreshed)
        old_id = sampler._adapter_ids["debate"]
        sampler.unload_adapters(adapter_names={"debate"})
        sampler.set_adapter_paths(adapter_paths={"debate": str(refreshed), "judge": assets.adapter_b_path})
        assert sampler._adapter_ids["debate"] != old_id
        sampler.sample(r3_request)
        assert outputs[-1].num_cached_tokens == 0
        print(json.dumps({"r3_prompt_tokens": len(r3_request.prompt_token_ids),
                          "r3_cached_tokens": cache_hits,
                          "cached_cold_greedy_tokens_equal": True,
                          "cross_adapter_and_refreshed_identity_cache_hits": 0}))
    finally:
        sampler.close()
