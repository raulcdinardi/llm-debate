from __future__ import annotations

from scripts.cw_judge_signal_phase0_finalize import (
    evaluate_invariants,
    metric_name,
    select_review_rows,
)
from scripts.cw_judge_signal_phase0_prepare import (
    EXPECTED_PACKAGE_VERSIONS,
    MODEL_REVISIONS,
)


def test_phase0_model_and_dependency_pins_are_exact() -> None:
    assert MODEL_REVISIONS == {
        "Qwen/Qwen3.5-4B-Base": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "Qwen/Qwen3.5-0.8B-Base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    }
    assert EXPECTED_PACKAGE_VERSIONS["sglang"] == "0.5.15.post1"
    assert EXPECTED_PACKAGE_VERSIONS["peft"] == "0.19.1"
    assert EXPECTED_PACKAGE_VERSIONS["accelerate"] == "1.14.0"


def test_invariant_evaluation_uses_declared_direction() -> None:
    assert metric_name("probe_a_target_cap_hit_rate_max") == (
        "probe_a_target_cap_hit_rate",
        "max",
    )
    results = evaluate_invariants(
        {
            "probe_a_target_cap_hit_rate": 0.05,
            "scorer_gold_pass_rate": 1.0,
        },
        ("probe_a_target_cap_hit_rate_max", "scorer_gold_pass_rate_min"),
    )
    assert all(result["pass"] for result in results.values())


def test_review_selection_is_bounded_to_six_per_arm() -> None:
    samples = []
    for index in range(12):
        samples.append({
            "verdict": "A" if index % 2 == 0 else "B",
            "trajectory_a": {"task_reward": float(index), "r1": f"a{index}"},
            "trajectory_b": {"task_reward": float(11 - index), "r1": f"b{index}"},
        })
    selected = select_review_rows("real4b", samples)
    assert len(selected) == 6
    assert len({row["source_index"] for row in selected}) == 6
    assert {row["arm"] for row in selected} == {"real4b"}
