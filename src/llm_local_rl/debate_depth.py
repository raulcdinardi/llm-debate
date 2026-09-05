from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
import random
from typing import Any


DebateDepthParams = Mapping[str, Any]


@dataclass(frozen=True)
class DebateDepthContext:
    rollout_seed: int | None
    optimizer_step: int | None
    rollout_index: int
    group_index: int
    num_groups: int
    debates_per_group: int
    max_rounds: int
    group_instance_id: str
    group_payload: Mapping[str, object]
    rng: random.Random


DebateDepthGenerator = Callable[
    [DebateDepthContext, DebateDepthParams],
    Sequence[int],
]
DebateDepthValidator = Callable[[DebateDepthParams, int, int], None]
DebateDepthMinimum = Callable[[DebateDepthParams, int], int]


@dataclass(frozen=True)
class RegisteredDebateDepthPolicy:
    name: str
    generator: DebateDepthGenerator
    minimum_rounds: DebateDepthMinimum
    validate: DebateDepthValidator


_POLICIES: dict[str, RegisteredDebateDepthPolicy] = {}


def _noop_validator(
    _params: DebateDepthParams,
    _max_rounds: int,
    _debates_per_group: int,
) -> None:
    return None


def register_debate_depth_policy(
    name: str,
    generator: DebateDepthGenerator,
    *,
    minimum_rounds: int | DebateDepthMinimum,
    validate: DebateDepthValidator | None = None,
    replace: bool = False,
) -> None:
    """Register a reproducible policy implementation under a serialized name.

    Registration must happen before constructing or restoring a configuration that
    references the policy. The serialized name and parameters identify the policy;
    its implementation is pinned by the source revision used for the run.
    """
    normalized = name.strip()
    if not normalized:
        raise ValueError("Debate-depth policy name must not be empty")
    if normalized in _POLICIES and not replace:
        raise ValueError(f"Debate-depth policy {normalized!r} is already registered")
    minimum_fn = (
        minimum_rounds
        if callable(minimum_rounds)
        else lambda _params, _maximum, value=int(minimum_rounds): value
    )
    _POLICIES[normalized] = RegisteredDebateDepthPolicy(
        name=normalized,
        generator=generator,
        minimum_rounds=minimum_fn,
        validate=validate or _noop_validator,
    )


def debate_depth_policy_names() -> tuple[str, ...]:
    return tuple(sorted(_POLICIES))


def get_debate_depth_policy(name: str) -> RegisteredDebateDepthPolicy:
    try:
        return _POLICIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown debate-depth policy {name!r}; registered={debate_depth_policy_names()}"
        ) from exc


def validate_debate_depth_policy(
    *,
    name: str,
    params: DebateDepthParams,
    max_rounds: int,
    debates_per_group: int,
) -> int:
    if max_rounds < 1:
        raise ValueError("Maximum debate depth must be at least 1")
    if debates_per_group < 1:
        raise ValueError("A debate group must contain at least one debate")
    policy = get_debate_depth_policy(name)
    policy_params = deepcopy(dict(params))
    policy.validate(policy_params, max_rounds, debates_per_group)
    minimum = policy.minimum_rounds(policy_params, max_rounds)
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise ValueError(f"Policy {name!r} declared a non-integer minimum depth")
    if minimum < 1 or minimum > max_rounds:
        raise ValueError(
            f"Policy {name!r} minimum depth must be within [1, {max_rounds}]"
        )
    return minimum


def generate_debate_depths(
    *,
    name: str,
    params: DebateDepthParams,
    context: DebateDepthContext,
) -> tuple[int, ...]:
    policy = get_debate_depth_policy(name)
    generated = tuple(policy.generator(context, deepcopy(dict(params))))
    if len(generated) != context.debates_per_group:
        raise ValueError(
            f"Policy {name!r} returned {len(generated)} depths for group "
            f"{context.group_index}; expected {context.debates_per_group}"
        )
    if any(
        not isinstance(depth, int) or isinstance(depth, bool) or depth < 1
        for depth in generated
    ):
        raise ValueError(
            f"Policy {name!r} returned a non-positive-integer depth for group "
            f"{context.group_index}"
        )
    if max(generated) > context.max_rounds:
        raise ValueError(
            f"Policy {name!r} returned depth above configured maximum "
            f"{context.max_rounds} for group {context.group_index}"
        )
    return generated


def _reject_unknown_params(params: DebateDepthParams, allowed: set[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"Unknown debate-depth policy parameters: {unknown}")


def _positive_depth(value: object, *, label: str, max_rounds: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    if value > max_rounds:
        raise ValueError(f"{label} must not exceed maximum depth {max_rounds}")
    return value


def _fixed_validate(
    params: DebateDepthParams,
    max_rounds: int,
    _debates_per_group: int,
) -> None:
    _reject_unknown_params(params, {"rounds"})
    if "rounds" in params:
        _positive_depth(params["rounds"], label="fixed rounds", max_rounds=max_rounds)


def _fixed_minimum(params: DebateDepthParams, max_rounds: int) -> int:
    return int(params.get("rounds", max_rounds))


def _fixed_generate(
    context: DebateDepthContext,
    params: DebateDepthParams,
) -> Sequence[int]:
    return [int(params.get("rounds", context.max_rounds))] * context.debates_per_group


def _multiset_depths(
    params: DebateDepthParams,
    max_rounds: int,
    debates_per_group: int,
) -> tuple[int, ...]:
    _reject_unknown_params(params, {"depths"})
    values = params.get("depths")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("shuffled_multiset requires a depths array")
    depths = tuple(
        _positive_depth(value, label="multiset depth", max_rounds=max_rounds)
        for value in values
    )
    if len(depths) != debates_per_group:
        raise ValueError(
            "shuffled_multiset depths must contain exactly one value per debate "
            f"({debates_per_group} required)"
        )
    return depths


def _multiset_validate(
    params: DebateDepthParams,
    max_rounds: int,
    debates_per_group: int,
) -> None:
    _multiset_depths(params, max_rounds, debates_per_group)


def _multiset_minimum(params: DebateDepthParams, max_rounds: int) -> int:
    values = params.get("depths")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("shuffled_multiset requires a non-empty depths array")
    return min(
        _positive_depth(value, label="multiset depth", max_rounds=max_rounds)
        for value in values
    )


def _multiset_generate(
    context: DebateDepthContext,
    params: DebateDepthParams,
) -> Sequence[int]:
    depths = list(params["depths"])
    context.rng.shuffle(depths)
    return depths


def _categorical_values(
    params: DebateDepthParams,
    max_rounds: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    _reject_unknown_params(params, {"depths", "weights"})
    values = params.get("depths")
    weights_value = params.get("weights")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("categorical requires a non-empty depths array")
    depths = tuple(
        _positive_depth(value, label="categorical depth", max_rounds=max_rounds)
        for value in values
    )
    if weights_value is None:
        weights = (1.0,) * len(depths)
    elif isinstance(weights_value, Sequence) and not isinstance(weights_value, (str, bytes)):
        weights = tuple(float(value) for value in weights_value)
    else:
        raise ValueError("categorical weights must be an array")
    if len(weights) != len(depths):
        raise ValueError("categorical depths and weights must have the same length")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("categorical weights must be finite and non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("categorical weights must have positive total mass")
    return depths, weights


def _categorical_validate(
    params: DebateDepthParams,
    max_rounds: int,
    _debates_per_group: int,
) -> None:
    _categorical_values(params, max_rounds)


def _categorical_minimum(params: DebateDepthParams, max_rounds: int) -> int:
    depths, weights = _categorical_values(params, max_rounds)
    return min(depth for depth, weight in zip(depths, weights, strict=True) if weight > 0.0)


def _categorical_generate(
    context: DebateDepthContext,
    params: DebateDepthParams,
) -> Sequence[int]:
    depths, weights = _categorical_values(params, context.max_rounds)
    return context.rng.choices(
        depths,
        weights=weights,
        k=context.debates_per_group,
    )


register_debate_depth_policy(
    "fixed",
    _fixed_generate,
    minimum_rounds=_fixed_minimum,
    validate=_fixed_validate,
)
register_debate_depth_policy(
    "shuffled_multiset",
    _multiset_generate,
    minimum_rounds=_multiset_minimum,
    validate=_multiset_validate,
)
register_debate_depth_policy(
    "categorical",
    _categorical_generate,
    minimum_rounds=_categorical_minimum,
    validate=_categorical_validate,
)
