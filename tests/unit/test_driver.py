from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from llm_local_rl.config import CheckpointManifest, RolloutConfig, TrainRunConfig
from llm_local_rl.metrics import mean_numeric_metrics
from llm_local_rl.registry import build_debate_task, build_environment, build_episode_builder


def test_config_and_manifest_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainRunConfig(
            model_path="/tmp/nonexistent_model_for_shape_only",
            output_dir=tmpdir,
            steps=1,
            debate_external_judge_url="http://judge.test:8123",
            debate_external_judge_timeout_s=123.0,
            thinking_mode="no_think",
            rollout=RolloutConfig(mode="single_turn", request_seed_mode="per_request"),
            trace_model_io=False,
            trace_model_io_dir=str(Path(tmpdir) / "trace"),
            trace_top_logprobs=7,
        )
        config_path = Path(tmpdir) / "run_config.json"
        config.write_json(config_path)
        assert config_path.exists()

        manifest = CheckpointManifest(
            run_config=config.to_dict(),
            current_step=1,
            adapter_dirs={"shared": str(Path(tmpdir) / "step_001_shared")},
            step_records_path=str(Path(tmpdir) / "step_records.jsonl"),
        )
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest.write_json(manifest_path)
        loaded = CheckpointManifest.read_json(manifest_path)
        assert loaded.current_step == 1
        assert loaded.adapter_dirs["shared"].endswith("step_001_shared")
        restored_config = TrainRunConfig.from_dict(loaded.run_config)
        assert restored_config.rollout.env_name == "ht_sequence"
        assert restored_config.rollout.mode == "single_turn"
        assert restored_config.rollout.request_seed_mode == "per_request"
        assert restored_config.debate_external_judge_url == "http://judge.test:8123"
        assert restored_config.debate_external_judge_timeout_s == 123.0
        assert restored_config.thinking_mode == "no_think"
        assert restored_config.trace_model_io is False
        assert restored_config.trace_model_io_dir == str(Path(tmpdir) / "trace")
        assert restored_config.trace_top_logprobs == 7


def test_judge_rejection_task_config_roundtrip() -> None:
    config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        debate_r1_reward="judge_rejection_task",
    )

    restored = TrainRunConfig.from_dict(config.to_dict())

    assert restored.debate_r1_reward == "judge_rejection_task"
    assert restored.debate_round_adapter_names == ("solution", "debate", "debate")


def test_judge_rejection_task_config_fails_closed() -> None:
    common = {
        "model_path": "/tmp/nonexistent_model_for_shape_only",
        "output_dir": "/tmp/out",
        "debate_r1_reward": "judge_rejection_task",
    }
    with pytest.raises(ValueError, match="debate rollouts"):
        TrainRunConfig(
            **common,
            rollout=RolloutConfig(mode="single_turn"),
            adapter_layout="split",
        )
    with pytest.raises(ValueError, match="adapter_layout='split'"):
        TrainRunConfig(**common, adapter_layout="shared")
    with pytest.raises(ValueError, match="requires round adapters"):
        TrainRunConfig(
            **common,
            adapter_layout="split",
            debate_round_adapter_names=("solution", "solution", "debate"),
        )


def test_mean_numeric_metrics_promotes_reward_hacking_components() -> None:
    means = mean_numeric_metrics(
        [
            {"parse_success": 1.0, "used_secret": 1.0, "secret_word": "glyph"},
            {"parse_success": 1.0, "used_secret": 0.0, "secret_word": "opal"},
        ]
    )

    assert means == {"mean_parse_success": 1.0, "mean_used_secret": 0.5}


def test_registry_uses_rollout_fields() -> None:
    config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(env_name="coin_flip", mode="debate"),
        adapter_layout="split",
    )
    env = build_environment(config)
    task = build_debate_task(config)
    assert env.name == "coin_flip"
    assert task.name == "coin"
    with pytest.raises(ValueError, match="DebateRuntime"):
        build_episode_builder(config)


def test_coin_flip_prompt_and_reward_match_between_single_turn_and_debate() -> None:
    class TinyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            if text == "Red":
                return [1]
            if text == "Blue":
                return [2]
            return [ord(ch) for ch in text]

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            _ = skip_special_tokens
            if token_ids == [1]:
                return "Red"
            if token_ids == [2]:
                return "Blue"
            return "".join(chr(tok) for tok in token_ids)

    config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(env_name="coin_flip", mode="debate"),
    )
    env = build_environment(config)
    task = build_debate_task(config)
    tokenizer = TinyTokenizer()

    env_instance = env.sample_instances(n=1, seed=0)[0]
    task_instance = task.sample_instances(n=1, seed=0)[0]

    assert env.build_initial_prompt(instance=env_instance) == task.judge_context_text(inst=task_instance)
    env_reward, env_metrics = env.score_completion(
        instance=env_instance,
        tokenizer=tokenizer,
        completion_token_ids=tokenizer.encode("Blue"),
    )
    task_reward = task.compute_reward(
        inst=task_instance,
        tokenizer=tokenizer,
        completion_tokens=tokenizer.encode("Blue"),
    )
    assert env_reward == task_reward.reward
    assert env_metrics["choice"] == task_reward.metrics["choice"]
    assert env_metrics["target"] == task_reward.metrics["target"]
