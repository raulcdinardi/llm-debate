from __future__ import annotations

import pytest

from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig


def test_default_request_seed_mode_does_not_seed_each_request() -> None:
    runtime = object.__new__(DebateRuntime)
    runtime.runtime_config = DebateRuntimeConfig(request_seed_mode="none")
    assert runtime._round_seed(step_seed=123, request_idx=4, round_num=2) is None


def test_per_request_seed_mode_preserves_unique_request_seeds() -> None:
    runtime = object.__new__(DebateRuntime)
    runtime.runtime_config = DebateRuntimeConfig(request_seed_mode="per_request")
    assert runtime._round_seed(step_seed=123, request_idx=4, round_num=2) == 200127


def test_unknown_request_seed_mode_fails_loudly() -> None:
    runtime = object.__new__(DebateRuntime)
    runtime.runtime_config = DebateRuntimeConfig(request_seed_mode="batch")
    with pytest.raises(ValueError, match="request_seed_mode"):
        runtime._round_seed(step_seed=123, request_idx=4, round_num=2)
