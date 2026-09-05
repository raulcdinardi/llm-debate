from __future__ import annotations

import pytest

from llm_local_rl.behavior_policy import (
    BEHAVIOR_POLICY_LOGPROBS,
    RAW_MODEL_LOGPROBS,
    BehaviorPolicySpec,
    behavior_policy_contract_record,
    validate_sampling_result_contract,
)
from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.types import SamplingRequest, SamplingResult


def _request(*, temperature: float = 0.8) -> SamplingRequest:
    return SamplingRequest(
        adapter_name="solution",
        prompt_token_ids=[1, 2],
        stop_token_ids=[3],
        max_tokens=4,
        temperature=temperature,
        top_p=1.0,
        min_p=0.0,
    )


def _result(
    request: SamplingRequest,
    *,
    policy: BehaviorPolicySpec | None = None,
    semantics: str = BEHAVIOR_POLICY_LOGPROBS,
) -> SamplingResult:
    return SamplingResult(
        adapter_name=request.adapter_name,
        prompt_token_ids=request.prompt_token_ids,
        completion_token_ids=[4],
        completion_logprobs=[-0.5],
        text="x",
        behavior_policy=policy or BehaviorPolicySpec.from_sampling_request(request),
        completion_logprob_semantics=semantics,  # type: ignore[arg-type]
    )


def test_train_config_serializes_one_explicit_behavior_policy_contract() -> None:
    config = TrainRunConfig(
        model_path="/model",
        output_dir="/output",
        rollout=RolloutConfig(temperature=0.8, top_p=1.0, min_p=0.0),
    )

    assert config.behavior_policy() == BehaviorPolicySpec(temperature=0.8)
    assert config.to_dict()["behavior_policy_contract"] == {
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "allowed_token_ids": (),
    }


def test_trainer_contract_supports_finite_allowed_token_normalization() -> None:
    policy = BehaviorPolicySpec(temperature=1.0, allowed_token_ids=(334, 378))
    policy.assert_exact_trainer_reconstruction_supported()


@pytest.mark.parametrize(
    ("rollout", "fragment"),
    [
        (RolloutConfig(temperature=0.0), "temperature must be > 0"),
        (RolloutConfig(temperature=1e-7), "temperature must be at least 1e-06"),
        (RolloutConfig(top_p=0.95), "top_p=0.95"),
        (RolloutConfig(min_p=0.02), "min_p=0.02"),
    ],
)
def test_train_config_rejects_behavior_processors_the_trainer_cannot_reconstruct(
    rollout: RolloutConfig,
    fragment: str,
) -> None:
    with pytest.raises(ValueError, match=fragment):
        TrainRunConfig(model_path="/model", output_dir="/output", rollout=rollout)


def test_train_config_cannot_disable_fail_closed_parity_gate() -> None:
    with pytest.raises(ValueError, match="on_policy_logprob_check=False"):
        TrainRunConfig(
            model_path="/model",
            output_dir="/output",
            on_policy_logprob_check=False,
        )


def test_sampling_result_contract_accepts_exact_behavior_distribution() -> None:
    request = _request()
    result = _result(request)

    validate_sampling_result_contract(request=request, result=result)


def test_sampling_result_contract_rejects_original_logprobs_at_nonunit_temperature() -> None:
    request = _request()
    result = _result(request, semantics=RAW_MODEL_LOGPROBS)

    with pytest.raises(ValueError, match="normalized behavior policy"):
        validate_sampling_result_contract(request=request, result=result)


def test_sampling_result_contract_rejects_parameter_mismatch() -> None:
    request = _request()
    result = _result(request, policy=BehaviorPolicySpec(temperature=1.0))

    with pytest.raises(ValueError, match="differs from its request"):
        validate_sampling_result_contract(request=request, result=result)


def test_contract_record_names_normalization_and_backend_semantics() -> None:
    policy = BehaviorPolicySpec(temperature=0.8)
    record = behavior_policy_contract_record(
        policy=policy,
        backend="sglang",
        backend_mode="standard_sampler",
        return_original_logprobs=False,
        scoring_dtype="float32",
    )

    assert record["version"] == 2
    assert record["semantics"] == BEHAVIOR_POLICY_LOGPROBS
    assert record["normalization"] == "log_softmax"
    assert record["policy"] == policy.to_dict()
    assert record["return_original_logprobs"] is False
