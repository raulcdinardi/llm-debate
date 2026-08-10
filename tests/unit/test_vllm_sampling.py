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
