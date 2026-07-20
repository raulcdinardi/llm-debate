from __future__ import annotations

import pytest

from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.mock_judge import SeededRandomJudge


def _call(judge: SeededRandomJudge) -> tuple[str, str]:
    return judge("q", "c", "a1", "b1", "a2", "b2", "a3", "b3")


def test_seeded_random_judge_is_reproducible_and_non_degenerate() -> None:
    first = SeededRandomJudge(seed=20260709)
    second = SeededRandomJudge(seed=20260709)
    first_results = [_call(first) for _ in range(64)]
    second_results = [_call(second) for _ in range(64)]

    assert first_results == second_results
    verdicts = [verdict for verdict, _reason in first_results]
    assert set(verdicts) == {"A", "B"}
    assert first_results[-1][1].startswith("seeded_random_judge seed=20260709 call=64")


def test_mock_judge_config_round_trips_and_is_exclusive() -> None:
    config = TrainRunConfig(
        model_path="model",
        output_dir="out",
        rollout=RolloutConfig(env_name="short_story", mode="debate"),
        debate_mock_judge_seed=7,
    )
    restored = TrainRunConfig.from_dict(config.to_dict())
    assert restored.debate_mock_judge_seed == 7

    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainRunConfig(
            model_path="model",
            output_dir="out",
            debate_mock_judge_seed=7,
            debate_judge_server_url="http://127.0.0.1:30001",
        )
