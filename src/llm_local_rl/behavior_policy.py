from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from llm_local_rl.types import SamplingRequest, SamplingResult


BEHAVIOR_POLICY_CONTRACT_VERSION = 1
BEHAVIOR_POLICY_LOGPROBS = "normalized_behavior_policy_logprobs"
RAW_MODEL_LOGPROBS = "normalized_raw_model_logprobs"
TEMPERATURE_SCALED_MODEL_LOGPROBS = "normalized_temperature_scaled_model_logprobs"
UNSPECIFIED_LOGPROBS = "unspecified"
LogprobSemantics = Literal[
    "normalized_behavior_policy_logprobs",
    "normalized_raw_model_logprobs",
    "normalized_temperature_scaled_model_logprobs",
    "unspecified",
]


@dataclass(frozen=True)
class BehaviorPolicySpec:
    """Normalized distribution used to sample policy actions.

    The current trainer can exactly reconstruct temperature scaling. It does
    not yet reproduce truncation or history-dependent logit processors, so
    those settings are rejected for PPO rather than silently scored under a
    different distribution.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        numeric_values = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }
        for name, value in numeric_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}.")
        if float(self.temperature) < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if int(self.top_k) == 0 or int(self.top_k) < -1:
            raise ValueError(f"top_k must be -1 (disabled) or positive, got {self.top_k}.")
        if not 0.0 <= float(self.min_p) <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if float(self.repetition_penalty) <= 0.0:
            raise ValueError(
                f"repetition_penalty must be positive, got {self.repetition_penalty}."
            )

    @classmethod
    def from_sampling_request(cls, request: SamplingRequest) -> BehaviorPolicySpec:
        return cls(
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            top_k=-1,
            min_p=float(request.min_p),
            repetition_penalty=1.0,
        )

    @classmethod
    def from_rollout_config(cls, rollout: Any) -> BehaviorPolicySpec:
        return cls(
            temperature=float(rollout.temperature),
            top_p=float(rollout.top_p),
            top_k=-1,
            min_p=float(rollout.min_p),
            repetition_penalty=1.0,
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def is_raw_model_distribution(self) -> bool:
        return (
            float(self.temperature) == 1.0
            and float(self.top_p) == 1.0
            and int(self.top_k) == -1
            and float(self.min_p) == 0.0
            and float(self.repetition_penalty) == 1.0
        )

    def assert_exact_trainer_reconstruction_supported(self) -> None:
        unsupported: list[str] = []
        if float(self.temperature) <= 0.0:
            unsupported.append("temperature must be > 0 for stochastic PPO")
        if float(self.top_p) != 1.0:
            unsupported.append(f"top_p={self.top_p} (only 1.0 is supported)")
        if int(self.top_k) != -1:
            unsupported.append(f"top_k={self.top_k} (only -1/disabled is supported)")
        if float(self.min_p) != 0.0:
            unsupported.append(f"min_p={self.min_p} (only 0.0 is supported)")
        if float(self.repetition_penalty) != 1.0:
            unsupported.append(
                f"repetition_penalty={self.repetition_penalty} (only 1.0 is supported)"
            )
        if unsupported:
            raise ValueError(
                "The PPO trainer cannot exactly reconstruct this behavior policy: "
                + "; ".join(unsupported)
                + ". Disable unsupported processors or implement and parity-test the identical "
                "normalized transformation in both sampler and trainer."
            )

    def exact_trainer_reconstruction_supported(self) -> bool:
        try:
            self.assert_exact_trainer_reconstruction_supported()
        except ValueError:
            return False
        return True


def behavior_policy_contract_record(
    *,
    policy: BehaviorPolicySpec,
    backend: str,
    backend_mode: str,
    return_original_logprobs: bool,
    semantics: LogprobSemantics = BEHAVIOR_POLICY_LOGPROBS,
    scoring_dtype: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "version": BEHAVIOR_POLICY_CONTRACT_VERSION,
        "semantics": semantics,
        "normalization": "log_softmax",
        "policy": policy.to_dict(),
        "sampler_backend": str(backend),
        "backend_mode": str(backend_mode),
        "return_original_logprobs": bool(return_original_logprobs),
    }
    if scoring_dtype is not None:
        record["scoring_dtype"] = str(scoring_dtype)
    return record


def validate_sampling_result_contract(
    *,
    request: SamplingRequest,
    result: SamplingResult,
) -> None:
    expected_policy = BehaviorPolicySpec.from_sampling_request(request)
    if result.behavior_policy != expected_policy:
        raise ValueError(
            "Sampler result behavior-policy contract differs from its request: "
            f"expected={expected_policy.to_dict()}, "
            f"got={None if result.behavior_policy is None else result.behavior_policy.to_dict()}."
        )
    if result.completion_logprob_semantics != BEHAVIOR_POLICY_LOGPROBS:
        raise ValueError(
            "PPO rollout logprobs must describe the normalized behavior policy; "
            f"got semantics={result.completion_logprob_semantics!r}."
        )
    if len(result.completion_token_ids) != len(result.completion_logprobs):
        raise ValueError(
            "Sampler result completion-token/logprob lengths differ: "
            f"{len(result.completion_token_ids)} != {len(result.completion_logprobs)}."
        )
