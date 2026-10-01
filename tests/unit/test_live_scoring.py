import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from llm_local_rl.live_scoring import Scorer, exclusive
from llm_local_rl.observability import RunObservability, WandbSettings
from llm_local_rl.score_sync import ScoreInbox, define_axes
from scripts.build_rollout_workspace import layout, workspace


@pytest.fixture
def config():
    return {"name": "test", "model": "model", "provider": "provider", "rounds": ["r2"],
            "fields": {"mention": [0, 1]}, "system_prompt": "Classify the target",
            "samples_per_step": 0, "concurrency": 4, "max_attempts": 2}


def record(step=1):
    return {"step": step, "sample_records": [{"question": "problem", "trajectory_a": {
        "r1": "answer A", "r2": "test evidence", "r3": "DO NOT INCLUDE REPLY"},
        "trajectory_b": {"r1": "answer B", "r2": "generic assertion"}}]}


def response(request, config):
    return {"model": config["model"], "provider": config["provider"], "usage": {"cost": 0.01},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "mention": int(request["side"] == "A")})}}]}


def prepare(tmp_path, config):
    source = tmp_path / "step_records.jsonl"
    source.write_text(json.dumps(record()) + "\n")
    directory = tmp_path / "observability" / "llm_scoring" / "test"
    scorer = Scorer(source, directory, config, "run")
    scorer.ingest()
    return scorer


def test_partial_line_restart_and_deduplication(tmp_path, config):
    scorer = prepare(tmp_path, config)
    with scorer.source.open("a") as f:
        f.write(json.dumps(record(2)))
    scorer.ingest()
    assert scorer.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2
    scorer.score_batch(response)
    scorer.db.close()
    resumed = Scorer(scorer.source, scorer.directory, config, "run")
    resumed.ingest()
    assert resumed.score_batch(response) == 0
    with resumed.source.open("a") as f:
        f.write("\n")
    resumed.ingest()
    assert resumed.score_batch(response) == 2
    events = resumed.export()
    assert [e["step"] for e in events] == [1, 2]
    assert events[0]["metrics"]["llm_eval/test/r2/mention"] == .5
    assert "DO NOT INCLUDE REPLY" not in resumed.db.execute("SELECT request FROM jobs LIMIT 1").fetchone()[0]


def test_source_and_rubric_changes_rejected(tmp_path, config):
    scorer = prepare(tmp_path, config)
    with pytest.raises(ValueError, match="rubric changed"):
        Scorer(scorer.source, scorer.directory, config | {"system_prompt": "changed"}, "run")
    changed = record()
    changed["sample_records"][0]["question"] = "DIFFERENT PROBLEM"
    with scorer.source.open("a") as f:
        f.write(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="content changed"):
        scorer.ingest()


def test_failed_scores_never_become_zero_and_retries_persist(tmp_path, config):
    scorer = prepare(tmp_path, config)
    def fail(*args):
        raise TimeoutError("pretend credential in exception must not be saved")
    scorer.score_batch(fail)
    assert scorer.export() == []
    scorer.db.close()
    scorer = Scorer(scorer.source, scorer.directory, config, "run")
    scorer.ingest()
    scorer.score_batch(fail)
    assert scorer.score_batch(fail) == 0
    metrics = scorer.export()[0]["metrics"]
    assert metrics["llm_eval/test/r2/failed"] == 2
    assert metrics["llm_eval/test/r2/coverage"] == 0
    assert "llm_eval/test/r2/mention" not in metrics
    assert {r[0] for r in scorer.db.execute("SELECT error FROM attempts")} == {"TimeoutError"}


@pytest.mark.parametrize("change", ["provider", "model", "truncation", "field", "nan"])
def test_invalid_provider_and_outputs_rejected(tmp_path, config, change):
    scorer = prepare(tmp_path, config)
    def invalid(req, cfg):
        data = response(req, cfg)
        if change in ("provider", "model"):
            data[change] = "unexpected"
        elif change == "truncation":
            data["choices"][0]["finish_reason"] = "length"
        else:
            data["choices"][0]["message"]["content"] = '{"wrong": 1}' if change == "field" else '{"mention": NaN}'
        return data
    scorer.score_batch(invalid)
    assert scorer.db.execute("SELECT count(*) FROM jobs WHERE status='done'").fetchone()[0] == 0


def test_late_events_keep_original_axis_and_resume(tmp_path, config):
    scorer = prepare(tmp_path, config)
    scorer.score_batch(response)
    scorer.export()
    logs, definitions = [], []
    run = SimpleNamespace(log=lambda data, **kw: logs.append((data, kw)),
                          define_metric=lambda *a, **kw: definitions.append((a, kw)))
    define_axes(run)
    inbox = ScoreInbox(tmp_path / "observability", "run")
    run.log({"train_step": 100})
    assert inbox.drain(run) == 1
    assert logs[-1][0]["evaluation_step"] == 1
    assert "step" not in logs[-1][1]
    assert "train_step" not in logs[-1][0]
    assert ScoreInbox(tmp_path / "observability", "run").drain(run) == 0
    with pytest.raises(ValueError, match="another run"):
        ScoreInbox(tmp_path / "observability", "other")
    assert any(kw.get("step_metric") == "evaluation_step" and kw["step_sync"] is False for _, kw in definitions)


def test_writer_lock_released_after_exit(tmp_path):
    path = tmp_path / "writer.lock"
    with exclusive(path):
        with pytest.raises(BlockingIOError):
            with exclusive(path):
                pass
    with exclusive(path):
        pass


def test_observer_single_writer_and_automatic_history(tmp_path, monkeypatch):
    calls = []
    fake_run = SimpleNamespace(id="run", url="fake", config={}, define_metric=lambda *a, **k: None,
                              log=lambda d, **k: calls.append((d, k)), finish=lambda: None)
    monkeypatch.setitem(__import__('sys').modules, "wandb", SimpleNamespace(
        util=SimpleNamespace(generate_id=lambda: "run"), init=lambda **k: fake_run))
    observer = RunObservability(output_dir=tmp_path, config={}, settings=WandbSettings(upload_artifacts=False))
    assert observer.run is fake_run
    with pytest.raises(BlockingIOError):
        with exclusive(tmp_path / "observability" / "wandb_writer.lock"):
            pass
    observer.log_step({"step": 20, "mean_reward": 1})
    assert calls[-1] == ({"train_step": 20, "rollout/mean_reward": 1.0}, {"commit": True})
    observer.finish()
    assert (tmp_path / "observability" / "wandb_finished").exists()
    with exclusive(tmp_path / "observability" / "wandb_writer.lock"):
        pass


def test_workspace_sdk_builds_ordered_sections_without_network(config):
    sections = layout({"rollout/mean_reward", "train/debate/loss"}, [config])
    view = workspace("entity", "project", "name", sections, "run")
    assert view.sections[0].name.startswith("01")
    assert view.sections[1].name.startswith("02")
    assert view.sections[1].panels[0].x == "evaluation_step"
    assert view.sections[0].panels[0].x == "train_step"
    assert view.settings.sort_panels_alphabetically is False
    assert view.runset_settings.pinned_runs == ["run"]


def test_saved_score_import_requires_identity_and_is_atomic(tmp_path, config):
    scorer = prepare(tmp_path, config)
    scorer.score_batch(response)
    rows = [json.loads(row[0]) for row in scorer.db.execute("SELECT result FROM jobs")]
    scorer.db.execute("UPDATE jobs SET status='pending',result=NULL")
    scorer.db.commit()
    path = tmp_path / "saved.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert scorer.import_scores(path) == 2
    assert scorer.import_scores(path) == 0
    rows[0]["record_sha256"] = "wrong"
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        scorer.import_scores(path)


def test_truncated_source_on_restart_is_rejected(tmp_path, config):
    scorer = prepare(tmp_path, config)
    scorer.db.close()
    scorer.source.write_text("")
    resumed = Scorer(scorer.source, scorer.directory, config, "run")
    with pytest.raises(ValueError, match="truncated"):
        resumed.ingest()


def test_scoring_parallelism_and_partial_coverage(tmp_path, config):
    import threading
    barrier = threading.Barrier(2, timeout=3)
    scorer = prepare(tmp_path, config | {"max_attempts": 1})
    def client(req, cfg):
        barrier.wait()
        if req["side"] == "B":
            raise TimeoutError()
        return response(req, cfg)
    assert scorer.score_batch(client) == 2
    event = scorer.export()[0]
    assert event["metrics"]["llm_eval/test/r2/mention"] == 1
    assert event["metrics"]["llm_eval/test/r2/coverage"] == .5


def test_real_wandb_offline_late_score(tmp_path, config, monkeypatch):
    # Exercises the actual SDK, not just a fake run.log signature.
    monkeypatch.setenv("WANDB_SILENT", "true")
    observer = RunObservability(output_dir=tmp_path, config={}, settings=WandbSettings(
        mode="offline", upload_artifacts=False, table_samples_per_shard=0))
    assert observer.run is not None
    scorer = prepare(tmp_path, config)
    scorer.db.close()
    # prepare used a placeholder run binding; use a separate matching evaluation.
    config = config | {"name": "matching"}
    scorer = Scorer(scorer.source, tmp_path / "observability" / "llm_scoring" / "matching", config, observer.run.id)
    scorer.ingest()
    scorer.score_batch(response)
    scorer.export()
    observer.log_step({"step": 100, "mean_reward": 1})
    observer._drain_scores()
    observer.log_step({"step": 101, "mean_reward": 2})
    observer.finish()
    assert not observer.failures_path.exists(), observer.failures_path.read_text() if observer.failures_path.exists() else ""
    assert (tmp_path / "observability" / "llm_scores_published.json").exists()
    assert (tmp_path / "observability" / "wandb_finished").exists()


@pytest.mark.parametrize("payload", [None, [], "bad", {"usage": None}])
def test_malformed_api_response_does_not_break_export(tmp_path, config, payload):
    scorer = prepare(tmp_path, config | {"max_attempts": 1})
    scorer.score_batch(lambda *_: payload)
    event = scorer.export()[0]
    assert event["metrics"]["llm_eval/test/r2/failed"] == 2
    assert json.loads((scorer.directory / "status.json").read_text())["attempts_without_cost"] == 2
