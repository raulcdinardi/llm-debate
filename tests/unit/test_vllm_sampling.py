from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from llm_local_rl.behavior_policy import (
    BEHAVIOR_POLICY_LOGPROBS,
    RAW_MODEL_LOGPROBS,
    TEMPERATURE_SCALED_MODEL_LOGPROBS,
    UNSPECIFIED_LOGPROBS,
    BehaviorPolicySpec,
)
from llm_local_rl import vllm_sampling
from llm_local_rl.types import SamplingRequest


class SamplingParamsWithLogprobsMode:
    def __init__(
        self,
        *,
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        repetition_penalty,
        max_tokens,
        stop_token_ids,
        logprobs,
        seed,
        logprobs_mode,
    ) -> None:
        self.kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "stop_token_ids": stop_token_ids,
            "logprobs": logprobs,
            "seed": seed,
            "logprobs_mode": logprobs_mode,
        }


class SamplingParamsWithoutLogprobsMode:
    def __init__(
        self,
        *,
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        repetition_penalty,
        max_tokens,
        stop_token_ids,
        logprobs,
        seed,
    ) -> None:
        self.kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "stop_token_ids": stop_token_ids,
            "logprobs": logprobs,
            "seed": seed,
        }


class SamplingParamsWithAllowedTokenIds:
    def __init__(
        self,
        *,
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        repetition_penalty,
        max_tokens,
        stop_token_ids,
        logprobs,
        seed,
        logprobs_mode,
        allowed_token_ids=None,
    ) -> None:
        self.kwargs = locals().copy()
        self.kwargs.pop("self")


def test_vllm_sampling_params_pin_processed_behavior_logprobs_mode_when_supported() -> None:
    params = vllm_sampling._build_sampling_params(
        SamplingParamsWithLogprobsMode,
        temperature=0.7,
        top_p=1.0,
        min_p=0.0,
        max_tokens=8,
        stop_token_ids=(99,),
        seed=123,
        trace_top_logprobs=2,
    )

    assert params.kwargs["logprobs_mode"] == "processed_logprobs"
    assert params.kwargs["top_p"] == 1.0
    assert params.kwargs["top_k"] == -1
    assert params.kwargs["repetition_penalty"] == 1.0
    assert params.kwargs["logprobs"] == 2
    assert params.kwargs["stop_token_ids"] == [99]


def test_vllm_sampling_params_preserve_nontrainable_judge_processors_without_claiming_parity() -> None:
    params = vllm_sampling._build_sampling_params(
        SamplingParamsWithLogprobsMode,
        temperature=0.7,
        top_p=0.95,
        min_p=0.1,
        top_k=20,
        presence_penalty=1.5,
        repetition_penalty=1.1,
        max_tokens=8,
        stop_token_ids=(99,),
        seed=123,
        trace_top_logprobs=2,
    )

    assert params.kwargs["top_p"] == 0.95
    assert params.kwargs["min_p"] == 0.1
    assert params.kwargs["top_k"] == 20
    assert params.kwargs["presence_penalty"] == 1.5
    assert params.kwargs["repetition_penalty"] == 1.1
    assert vllm_sampling._completion_logprob_semantics(
        policy=BehaviorPolicySpec(
            temperature=0.7,
            top_p=0.95,
            top_k=20,
            min_p=0.1,
            presence_penalty=1.5,
            repetition_penalty=1.1,
        ),
        backend_mode="processed_logprobs",
    ) == TEMPERATURE_SCALED_MODEL_LOGPROBS


def test_vllm_sampling_params_record_and_apply_allowed_token_ids() -> None:
    params = vllm_sampling._build_sampling_params(
        SamplingParamsWithAllowedTokenIds,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        max_tokens=1,
        stop_token_ids=(2,),
        seed=123,
        trace_top_logprobs=1,
        allowed_token_ids=(41, 334, 42, 378),
    )

    assert params.kwargs["allowed_token_ids"] == [41, 334, 42, 378]
    policy = BehaviorPolicySpec(allowed_token_ids=(41, 334, 42, 378))
    assert policy.exact_trainer_reconstruction_supported()
    assert policy.to_dict()["allowed_token_ids"] == (41, 334, 42, 378)


def test_candidate_logprob_contract_requests_and_extracts_all_allowed_labels() -> None:
    params = vllm_sampling._build_sampling_params(
        SamplingParamsWithAllowedTokenIds,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        max_tokens=1,
        stop_token_ids=(2,),
        seed=123,
        trace_top_logprobs=0,
        allowed_token_ids=(41, 334, 42, 378),
        candidate_logprob_token_ids=(41, 334, 42, 378),
    )
    assert params.kwargs["logprobs"] == 4
    rows = vllm_sampling._extract_candidate_logprobs(
        token_ids=[334],
        token_logprobs=[{
            41: SimpleNamespace(logprob=-1.0),
            334: SimpleNamespace(logprob=-0.2),
            42: SimpleNamespace(logprob=-2.0),
            378: SimpleNamespace(logprob=-1.5),
        }],
        candidate_token_ids=(41, 334, 42, 378),
    )
    assert rows == [{41: -1.0, 334: -0.2, 42: -2.0, 378: -1.5}]


def test_candidate_logprob_contract_rejects_missing_candidate() -> None:
    with pytest.raises(RuntimeError, match="Missing requested candidate token ids"):
        vllm_sampling._extract_candidate_logprobs(
            token_ids=[41],
            token_logprobs=[{41: SimpleNamespace(logprob=-0.1)}],
            candidate_token_ids=(41, 334, 42, 378),
        )


def test_vllm_logprob_semantics_distinguish_policy_and_greedy_judge_requests() -> None:
    assert vllm_sampling._completion_logprob_semantics(
        policy=BehaviorPolicySpec(temperature=0.8),
        backend_mode="processed_logprobs",
    ) == BEHAVIOR_POLICY_LOGPROBS
    assert vllm_sampling._completion_logprob_semantics(
        policy=BehaviorPolicySpec(temperature=0.0),
        backend_mode="processed_logprobs",
    ) == RAW_MODEL_LOGPROBS


def test_vllm_sampling_params_allow_legacy_versions_without_logprobs_mode(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.9.0"))

    params = vllm_sampling._build_sampling_params(
        SamplingParamsWithoutLogprobsMode,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        max_tokens=8,
        stop_token_ids=(),
        seed=None,
        trace_top_logprobs=0,
    )

    assert "logprobs_mode" not in params.kwargs
    assert params.kwargs["logprobs"] == 1
    assert vllm_sampling._completion_logprob_semantics(
        policy=BehaviorPolicySpec(temperature=1.0),
        backend_mode="legacy_unverified_logprobs",
    ) == UNSPECIFIED_LOGPROBS


def test_vllm_sampling_params_reject_modern_versions_without_logprobs_mode(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.10.0"))

    with pytest.raises(RuntimeError, match="does not expose logprobs_mode"):
        vllm_sampling._build_sampling_params(
            SamplingParamsWithoutLogprobsMode,
            temperature=1.0,
            top_p=1.0,
            min_p=0.0,
            max_tokens=8,
            stop_token_ids=(),
            seed=None,
            trace_top_logprobs=0,
        )


def test_vllm_sampling_params_accept_verified_engine_level_mode() -> None:
    params = vllm_sampling._build_sampling_params(
        SamplingParamsWithoutLogprobsMode,
        temperature=0.8,
        top_p=1.0,
        min_p=0.0,
        max_tokens=8,
        stop_token_ids=(),
        seed=None,
        trace_top_logprobs=0,
        engine_logprobs_mode="processed_logprobs",
    )

    assert "logprobs_mode" not in params.kwargs


@pytest.mark.parametrize("mode", [None, "raw_logits", "processed_logits", "unknown"])
def test_vllm_engine_level_mode_still_fails_closed_when_unverified(monkeypatch, mode) -> None:
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.26.0"))

    with pytest.raises(RuntimeError, match="cannot be pinned"):
        vllm_sampling._build_sampling_params(
            SamplingParamsWithoutLogprobsMode,
            temperature=0.8,
            top_p=1.0,
            min_p=0.0,
            max_tokens=8,
            stop_token_ids=(),
            seed=None,
            trace_top_logprobs=0,
            engine_logprobs_mode=mode,
        )


def test_sample_many_batches_distinct_per_request_seeds(monkeypatch) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, prompts, *, sampling_params, lora_request, use_tqdm):
            self.calls.append((prompts, sampling_params, lora_request, use_tqdm))
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[100 + idx],
                            logprobs=[{100 + idx: -0.1}],
                            text=str(100 + idx),
                            finish_reason="stop",
                        )
                    ]
                )
                for idx in range(len(prompts))
            ]

    monkeypatch.setattr(
        vllm_sampling,
        "_import_vllm_symbols",
        lambda: (None, SamplingParamsWithLogprobsMode, None),
    )
    sampler = object.__new__(vllm_sampling.VllmSampler)
    sampler._llm = FakeLLM()
    sampler._engine_logprobs_mode = "processed_logprobs"
    sampler.adapter_paths = {}

    requests = [
        SamplingRequest(
            adapter_name="debate",
            prompt_token_ids=[idx],
            stop_token_ids=[2],
            max_tokens=8,
            temperature=1.0,
            seed=seed,
        )
        for idx, seed in enumerate((11, 22, 33))
    ]
    results = sampler.sample_many(requests)

    assert len(sampler._llm.calls) == 1
    _prompts, params, _lora, _tqdm = sampler._llm.calls[0]
    assert [item.kwargs["seed"] for item in params] == [11, 22, 33]
    assert [result.prompt_token_ids for result in results] == [[0], [1], [2]]


def test_unload_adapters_evicts_only_trainable_loras_and_preserves_frozen_judge() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.loaded = {11, 12, 13}
            self.removed = []

        def list_loras(self):
            return set(self.loaded)

        def remove_lora(self, adapter_id):
            self.removed.append(adapter_id)
            self.loaded.remove(adapter_id)
            return True

    sampler = object.__new__(vllm_sampling.VllmSampler)
    sampler._llm = SimpleNamespace(llm_engine=FakeEngine())
    sampler._adapter_ids = {"solution": 11, "debate": 12, "judge": 13}

    sampler.unload_adapters(adapter_names={"solution", "debate"})

    assert sampler._llm.llm_engine.removed == [11, 12]
    assert sampler._llm.llm_engine.list_loras() == {13}


def test_unload_adapters_fails_closed_when_vllm_cannot_verify_eviction() -> None:
    class RefusingEngine:
        def list_loras(self):
            return {7}

        def remove_lora(self, adapter_id):
            _ = adapter_id
            return False

    sampler = object.__new__(vllm_sampling.VllmSampler)
    sampler._llm = SimpleNamespace(llm_engine=RefusingEngine())
    sampler._adapter_ids = {"solution": 7}

    with pytest.raises(RuntimeError, match="refused to unload"):
        sampler.unload_adapters(adapter_names={"solution"})


@pytest.mark.parametrize("enabled", [None, False, True])
def test_prefix_cache_opt_in_preserves_engine_default_and_selects_align(monkeypatch, enabled):
    calls = []
    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(cache_config=SimpleNamespace(
                enable_prefix_caching=enabled, mamba_cache_mode="align")))
    monkeypatch.setattr(vllm_sampling, "_import_vllm_symbols", lambda: (FakeLLM, None, None))
    monkeypatch.setattr(vllm_sampling, "_engine_accepts_logprobs_mode", lambda: False)
    vllm_sampling.VllmSampler(runtime=vllm_sampling.VllmRuntimeConfig(
        model_path="/model", enable_prefix_caching=enabled))
    if enabled is None:
        assert "enable_prefix_caching" not in calls[0]
        assert "mamba_cache_mode" not in calls[0]
    else:
        assert calls[0]["enable_prefix_caching"] is enabled
        if enabled:
            assert calls[0]["mamba_cache_mode"] == "align"
        else:
            assert "mamba_cache_mode" not in calls[0]


@pytest.mark.parametrize("actual_enabled,actual_mode", [(False, "align"), (True, "none")])
def test_prefix_cache_requested_settings_must_take_effect(monkeypatch, actual_enabled, actual_mode):
    class FakeLLM:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(cache_config=SimpleNamespace(
                enable_prefix_caching=actual_enabled, mamba_cache_mode=actual_mode)))
    monkeypatch.setattr(vllm_sampling, "_import_vllm_symbols", lambda: (FakeLLM, None, None))
    monkeypatch.setattr(vllm_sampling, "_engine_accepts_logprobs_mode", lambda: False)
    with pytest.raises(RuntimeError, match="did not honor"):
        vllm_sampling.VllmSampler(runtime=vllm_sampling.VllmRuntimeConfig(
            model_path="/model", enable_prefix_caching=True))
