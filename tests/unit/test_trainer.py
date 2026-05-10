from __future__ import annotations

from pathlib import Path
import tempfile
import importlib

import pytest

torch = importlib.import_module("torch") if importlib.util.find_spec("torch") is not None else None
peft_spec = importlib.util.find_spec("peft")
transformers_spec = importlib.util.find_spec("transformers")

if torch is None or peft_spec is None or transformers_spec is None:
    pytest.skip("trainer unit tests require torch, peft, and transformers", allow_module_level=True)

from llm_local_rl.trainer import MultiAdapterTrainer, TrainerConfig, _pad_batch
from llm_local_rl.types import TrainExample


def _base_model_path() -> str | None:
    import os

    return os.environ.get("LLM_LOCAL_RL_BASE_MODEL")


def _test_scratch_root() -> Path:
    root = Path.cwd() / ".tmp_test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_pad_batch_can_keep_loss_suffix_when_truncating() -> None:
    example = TrainExample(
        adapter_name="shared",
        input_ids=[10, 11, 12, 13, 14],
        target_ids=[11, 12, 13, 14, 15],
        loss_mask=[0, 0, 0, 1, 1],
        old_logprobs=[0.0, 0.0, 0.0, -0.3, -0.2],
        advantages=[0.0, 0.0, 0.0, 0.5, 0.5],
    )

    tensors = _pad_batch(batch=[example], pad_token_id=0, device="cpu", max_tokens=3)

    assert tensors["input_ids"].tolist() == [[12, 13, 14]]
    assert tensors["target_ids"].tolist() == [[13, 14, 15]]
    assert tensors["loss_mask"].tolist() == [[False, True, True]]


def test_trainer_smoke_requires_model_env() -> None:
    if torch is None:
        return
    path = _base_model_path()
    if path is None:
        return
    trainer = MultiAdapterTrainer(
        config=TrainerConfig(
            base_model_path=path,
            adapter_names=("shared", "solution"),
            device="cpu",
            torch_dtype="float32",
        )
    )
    assert trainer.tokenizer.pad_token_id is not None


def test_adapter_snapshot_keys_disjoint_when_split() -> None:
    if torch is None:
        return
    path = _base_model_path()
    if path is None:
        return
    trainer = MultiAdapterTrainer(
        config=TrainerConfig(
            base_model_path=path,
            adapter_names=("solution", "debate"),
            device="cpu",
            torch_dtype="float32",
        )
    )
    solution_keys = set(trainer.adapter_parameter_snapshot(adapter_name="solution"))
    debate_keys = set(trainer.adapter_parameter_snapshot(adapter_name="debate"))
    assert solution_keys
    assert debate_keys
    assert solution_keys.isdisjoint(debate_keys)


def test_save_and_reload_adapter_snapshot_preserves_weights() -> None:
    if torch is None:
        return
    path = _base_model_path()
    if path is None:
        return
    trainer = MultiAdapterTrainer(
        config=TrainerConfig(
            base_model_path=path,
            adapter_names=("shared",),
            device="cpu",
            torch_dtype="float32",
        )
    )
    before = trainer.adapter_parameter_snapshot(adapter_name="shared")
    with tempfile.TemporaryDirectory(dir=_test_scratch_root()) as tmpdir:
        saved_dir = trainer.save_adapter(adapter_name="shared", output_dir=tmpdir)
        reloaded = MultiAdapterTrainer.from_saved_adapters(
            config=TrainerConfig(
                base_model_path=path,
                adapter_names=("shared",),
                device="cpu",
                torch_dtype="float32",
            ),
            adapter_dirs={"shared": saved_dir},
        )
    after = reloaded.adapter_parameter_snapshot(adapter_name="shared")
    assert before.keys() == after.keys()
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-7)


def test_train_minibatch_matches_full_batch_update() -> None:
    if torch is None:
        return
    path = _base_model_path()
    if path is None:
        return
    common_kwargs = dict(
        base_model_path=path,
        adapter_names=("shared",),
        device="cpu",
        torch_dtype="float32",
    )
    trainer_full = MultiAdapterTrainer(
        config=TrainerConfig(
            **common_kwargs,
            train_minibatch_size=0,
        )
    )
    with tempfile.TemporaryDirectory(dir=_test_scratch_root()) as tmpdir:
        saved_dir = trainer_full.save_adapter(adapter_name="shared", output_dir=tmpdir)
        trainer_mini = MultiAdapterTrainer.from_saved_adapters(
            config=TrainerConfig(
                **common_kwargs,
                train_minibatch_size=1,
            ),
            adapter_dirs={"shared": saved_dir},
        )
    batch = [
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 3],
            target_ids=[2, 3, 4],
            loss_mask=[0, 1, 1],
            old_logprobs=[0.0, -1.0, -1.0],
            advantages=[0.0, 0.5, 0.5],
        ),
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 5],
            target_ids=[2, 5, 6],
            loss_mask=[0, 1, 1],
            old_logprobs=[0.0, -1.0, -1.0],
            advantages=[0.0, -0.25, -0.25],
        ),
    ]
    trainer_full.train_batch(adapter_name="shared", batch=batch)
    trainer_mini.train_batch(adapter_name="shared", batch=batch)
    full_after = trainer_full.adapter_parameter_snapshot(adapter_name="shared")
    mini_after = trainer_mini.adapter_parameter_snapshot(adapter_name="shared")
    assert full_after.keys() == mini_after.keys()
    for key in full_after:
        assert torch.allclose(full_after[key], mini_after[key], atol=1e-4)
