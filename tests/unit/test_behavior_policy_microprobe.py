from __future__ import annotations

import pytest
import torch

from llm_local_rl.behavior_policy_microprobe import (
    run_temperature_four_cell_microprobe,
)


def test_four_cell_probe_separates_numeric_parity_from_behavior_policy_parity() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260723)
    logits = torch.randn((1024, 97), generator=generator) * 3.0
    target_ids = torch.multinomial(
        torch.softmax(logits / 0.8, dim=-1),
        num_samples=1,
        generator=generator,
    ).squeeze(-1)

    report = run_temperature_four_cell_microprobe(
        logits=logits,
        target_ids=target_ids,
        behavior_temperature=0.8,
        abs_tol=1e-6,
        ppo_clip_epsilon=0.1,
    )
    cells = report["cells"]
    correct = cells["post_temperature__trainer_behavior_temperature"]
    superficial = cells["original__trainer_temperature_1"]
    original_bug = cells["post_temperature__trainer_temperature_1"]
    inverse_bug = cells["original__trainer_behavior_temperature"]

    assert correct["numeric_parity_pass"] is True
    assert correct["strict_behavior_policy_alignment"] is True
    assert correct["zero_update_clip_fraction"] == 0.0

    assert superficial["numeric_parity_pass"] is True
    assert superficial["strict_behavior_policy_alignment"] is False
    assert superficial["zero_update_clip_fraction"] == 0.0

    assert original_bug["numeric_parity_pass"] is False
    assert original_bug["strict_behavior_policy_alignment"] is False
    assert original_bug["zero_update_clip_fraction"] > 0.5
    assert original_bug["ratio_mean"] == pytest.approx(1.0, abs=0.05)

    assert inverse_bug["numeric_parity_pass"] is False
    assert inverse_bug["strict_behavior_policy_alignment"] is False
    assert inverse_bug["zero_update_clip_fraction"] > 0.5

    assert report["rare_token_pair_slope_p50"] == pytest.approx(1.25, abs=5e-5)
    assert report["rare_token_pair_slope_p99"] == pytest.approx(1.25, abs=5e-5)
