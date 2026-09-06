from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

from llm_local_rl.checkpointing import (
    checkpoint_adapter_dirs,
    load_exact_resume_checkpoint,
    save_exact_resume_checkpoint,
    validate_exact_resume_checkpoint,
)
from llm_local_rl.config import TrainRunConfig
from llm_local_rl.observability import RunObservability, WandbSettings, flatten_step_metrics
from scripts.run_train import parse_args


class _FakeTrainer:
    def __init__(self) -> None:
        self.loaded = None

    def training_state_dict(self):
        return {"schema": "fake", "optimizer": {"step": 7}}

    def load_training_state_dict(self, state):
        self.loaded = state


def test_observability_and_checkpoint_defaults() -> None:
    config = TrainRunConfig(model_path="/model", output_dir="/out")
    assert config.wandb_enabled is True
    assert config.adapter_checkpoint_every == 10
    assert config.optimizer_checkpoint_every == 50
    assert config.rollout_shard_every == 10
    restored = TrainRunConfig.from_dict(config.to_dict())
    assert restored.wandb_project == "llm-local-rl"
    assert restored.optimizer_checkpoint_every == 50
    args = parse_args(["--model-path", "/model", "--output-dir", "/out"])
    assert args.wandb is True
    assert args.adapter_checkpoint_every == 10
    assert args.optimizer_checkpoint_every == 50
    assert args.reference_kl_every == 10


def test_rollout_shards_are_local_complete_and_immutable(tmp_path: Path) -> None:
    observer = RunObservability(
        output_dir=tmp_path,
        config={},
        settings=WandbSettings(enabled=False, rollout_shard_steps=2),
    )
    assert observer.record_rollouts({"step": 1, "sample_records": [{"instance_id": "a"}]}) is None
    shard = observer.record_rollouts({"step": 2, "sample_records": [{"instance_id": "b"}]})
    assert shard is not None and shard.name == "steps_000001_000002.jsonl.gz"
    with gzip.open(shard, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["step"] for row in rows] == [1, 2]
    assert not list((tmp_path / "observability" / "rollout_shards").glob("*.pending.jsonl"))
    observer.finish()


def test_exact_resume_bundle_has_integrity_and_loads(tmp_path: Path, monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        save=lambda value, path: Path(path).write_bytes(pickle.dumps(value)),
        load=lambda path, **_: pickle.loads(Path(path).read_bytes()),
        get_rng_state=lambda: b"cpu-rng",
        set_rng_state=lambda _state: None,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_rng_state_all=lambda: [],
            set_rng_state_all=lambda _state: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    adapter = tmp_path / "source_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    trainer = _FakeTrainer()
    config = {"model_path": "/model", "steps": 100}
    checkpoint = save_exact_resume_checkpoint(
        root=tmp_path / "checkpoints",
        step=50,
        run_config=config,
        adapter_dirs={"solution": str(adapter)},
        trainer=trainer,
    )
    manifest = validate_exact_resume_checkpoint(checkpoint, run_config=config)
    assert manifest["completed_step"] == 50
    assert (checkpoint / "READY").is_file()
    assert (Path(checkpoint_adapter_dirs(checkpoint)["solution"]) / "adapter_model.safetensors").is_file()
    restored = _FakeTrainer()
    load_exact_resume_checkpoint(path=checkpoint, trainer=restored, run_config=config)
    assert restored.loaded == trainer.training_state_dict()
    (checkpoint / "trainer_state.pt").write_bytes(b"corrupt")
    try:
        validate_exact_resume_checkpoint(checkpoint, run_config=config)
    except ValueError as exc:
        assert "integrity mismatch" in str(exc)
    else:
        raise AssertionError("corruption was not detected")


def test_scalar_flattening_excludes_text_and_arrays() -> None:
    metrics = flatten_step_metrics(
        {
            "step": 3,
            "mean_reward": 1.25,
            "sample_records": [{"text": "large"}],
            "mean_reward_metrics": {"format": 0.75, "label": "x"},
            "train_metrics": {"solution": {"loss": 0.2, "backend": "x"}},
        }
    )
    assert metrics == {
        "rollout/mean_reward": 1.25,
        "reward_component/format": 0.75,
        "train/solution/loss": 0.2,
    }


def test_wandb_initialization_failure_is_local_and_fail_open(tmp_path: Path, monkeypatch) -> None:
    fake_wandb = SimpleNamespace(
        util=SimpleNamespace(generate_id=lambda: "run-id"),
        init=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    observer = RunObservability(
        output_dir=tmp_path,
        config={"safe": True},
        settings=WandbSettings(enabled=True),
    )
    observer.log_step({"step": 1, "mean_reward": 1.0})
    observer.finish()
    failures = [json.loads(line) for line in (tmp_path / "observability" / "wandb_failures.jsonl").read_text().splitlines()]
    assert failures[0]["operation"] == "init"
    assert failures[0]["error_type"] == "RuntimeError"


def test_optimizer_batch_size_preserves_legacy_checkpoint_fingerprint():
    import hashlib
    import json
    from llm_local_rl.checkpointing import config_fingerprint

    legacy = {"model_path": "/model", "learning_rate": 0.001}
    original_hash = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert config_fingerprint({**legacy, "train_optimizer_batch_size": 0}) == original_hash
    assert config_fingerprint({**legacy, "train_optimizer_batch_size": 32}) != original_hash
    assert config_fingerprint({**legacy, "train_optimizer_batch_size": 32}) != config_fingerprint(
        {**legacy, "train_optimizer_batch_size": 64})


def test_prefix_cache_default_preserves_checkpoint_fingerprint():
    from llm_local_rl.checkpointing import config_fingerprint
    legacy = {"model_path": "/model"}
    assert config_fingerprint(legacy) == config_fingerprint({**legacy, "sampler_prefix_caching": None})
    for enabled in [False, True]:
        assert config_fingerprint(legacy) != config_fingerprint({**legacy, "sampler_prefix_caching": enabled})
