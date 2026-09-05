from __future__ import annotations

import random

import pytest

from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.debate_depth import (
    DebateDepthContext,
    generate_debate_depths,
    register_debate_depth_policy,
)


def _context(*, group_index: int = 0) -> DebateDepthContext:
    return DebateDepthContext(
        rollout_seed=11,
        optimizer_step=4,
        rollout_index=2,
        group_index=group_index,
        num_groups=2,
        debates_per_group=4,
        max_rounds=6,
        group_instance_id=f"group-{group_index}",
        group_payload={"kind": "test"},
        rng=random.Random(f"11:debate_round_counts:group={group_index}"),
    )


def test_builtin_depth_policies_share_the_registered_generation_boundary() -> None:
    fixed = generate_debate_depths(name="fixed", params={}, context=_context())
    shuffled = generate_debate_depths(
        name="shuffled_multiset",
        params={"depths": [3, 4, 5, 6]},
        context=_context(),
    )
    categorical = generate_debate_depths(
        name="categorical",
        params={"depths": [3, 6], "weights": [1.0, 0.0]},
        context=_context(),
    )

    assert fixed == (6, 6, 6, 6)
    assert sorted(shuffled) == [3, 4, 5, 6]
    assert categorical == (3, 3, 3, 3)


def test_registered_custom_policy_and_params_survive_config_round_trip() -> None:
    def instance_conditioned(context, params):
        shallow = int(params["shallow"])
        return [shallow + context.group_index, context.max_rounds] * 2

    register_debate_depth_policy(
        "test_instance_conditioned_roundtrip",
        instance_conditioned,
        minimum_rounds=3,
    )
    config = TrainRunConfig(
        model_path="/tmp/model",
        output_dir="/tmp/out",
        rollout=RolloutConfig(group_size=8),
        debate_rounds=6,
        debate_depth_policy="test_instance_conditioned_roundtrip",
        debate_depth_policy_params={"shallow": 3},
    )

    restored = TrainRunConfig.from_dict(config.to_dict())

    assert restored.debate_depth_policy == "test_instance_conditioned_roundtrip"
    assert restored.debate_depth_policy_params == {"shallow": 3}
    assert generate_debate_depths(
        name=restored.debate_depth_policy,
        params=restored.debate_depth_policy_params,
        context=_context(group_index=1),
    ) == (4, 6, 4, 6)


def test_registered_policy_outputs_fail_closed() -> None:
    register_debate_depth_policy(
        "test_bad_depth_vector",
        lambda _context, _params: [3],
        minimum_rounds=3,
    )
    with pytest.raises(ValueError, match="expected 4"):
        generate_debate_depths(
            name="test_bad_depth_vector",
            params={},
            context=_context(),
        )

    register_debate_depth_policy(
        "test_above_maximum_depth",
        lambda context, _params: [context.max_rounds + 1] * context.debates_per_group,
        minimum_rounds=3,
    )
    with pytest.raises(ValueError, match="above configured maximum"):
        generate_debate_depths(
            name="test_above_maximum_depth",
            params={},
            context=_context(),
        )


def test_registered_policy_cannot_mutate_persisted_params() -> None:
    params = {"nested": {"depth": 3}}

    def mutating_policy(context, policy_params):
        policy_params["nested"]["depth"] = context.max_rounds
        return [3] * context.debates_per_group

    register_debate_depth_policy(
        "test_mutating_policy_params",
        mutating_policy,
        minimum_rounds=3,
    )

    assert generate_debate_depths(
        name="test_mutating_policy_params",
        params=params,
        context=_context(),
    ) == (3, 3, 3, 3)
    assert params == {"nested": {"depth": 3}}
