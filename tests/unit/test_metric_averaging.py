from types import SimpleNamespace

from llm_local_rl.metric_averaging import trailing_rows
from scripts.publish_trailing_average import source_rows


def test_trailing_five_and_future_invariance():
    rows = [{"step": i, "metrics": {"reward": i}} for i in range(1, 8)]
    result = list(trailing_rows(rows))
    assert [r["trailing5/reward"] for r in result] == [1, 1.5, 2, 2.5, 3, 4, 5]
    assert result[:5] == list(trailing_rows(rows[:5]))


def test_gaps_and_missing_measurements_are_not_zero_filled():
    rows = [{"step": 1, "metrics": {"reward": 10}}, {"step": 2, "metrics": {}},
            {"step": 3, "metrics": {"reward": float("nan")}},
            {"step": 7, "metrics": {"reward": 70}}]
    assert list(trailing_rows(rows)) == [
        {"averaging_step": 1, "trailing5/reward": 10}, {"averaging_step": 2},
        {"averaging_step": 3}, {"averaging_step": 7, "trailing5/reward": 70}]


def test_history_aligns_late_scores_and_excludes_init_probe():
    rows = [{"_step": 0, "reward": 0, "rollout/phase0_observability_probe": 1},
            {"_step": 1, "reward": 10, "evaluation_step": None},
            {"_step": 2, "reward": 20},
            {"_step": 3, "evaluation_step": 2, "score": .5},
            {"_step": 4, "evaluation_step": 1, "score": 1}]
    run = SimpleNamespace(scan_history=lambda **_: iter(rows))
    view = SimpleNamespace(sections=[SimpleNamespace(panels=[
        SimpleNamespace(x="Step", y=["reward"]), SimpleNamespace(x="evaluation_step", y=["score"])])])
    assert source_rows(run, view) == [{"step": 1, "metrics": {"reward": 10, "score": 1}},
                                     {"step": 2, "metrics": {"reward": 20, "score": .5}}]
