from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from llm_local_rl import vllm_sampling


class SamplingParamsWithLogprobsMode:
    def __init__(
        self,
        *,
        temperature,
        top_p,
        min_p,
        max_tokens,
        stop_token_ids,
        logprobs,
        seed,
        logprobs_mode,
    ) -> None:
        self.kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
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
        min_p,
        max_tokens,
        stop_token_ids,
        logprobs,
        seed,
    ) -> None:
        self.kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "stop_token_ids": stop_token_ids,
            "logprobs": logprobs,
            "seed": seed,
        }


def test_vllm_sampling_params_pin_raw_logprobs_mode_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(vllm_sampling, "_WARNED_RAW_LOGPROB_SAMPLING_POLICY", False)

    with pytest.warns(RuntimeWarning, match="raw model logprobs"):
        params = vllm_sampling._build_sampling_params(
            SamplingParamsWithLogprobsMode,
            temperature=0.7,
            top_p=0.95,
            min_p=0.1,
            max_tokens=8,
            stop_token_ids=(99,),
            seed=123,
            trace_top_logprobs=2,
        )

    assert params.kwargs["logprobs_mode"] == "raw_logprobs"
    assert params.kwargs["top_p"] == 0.95
    assert params.kwargs["logprobs"] == 2
    assert params.kwargs["stop_token_ids"] == [99]


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
