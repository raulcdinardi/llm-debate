from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from llm_local_rl.config import CheckpointManifest, RolloutConfig, TrainRunConfig
from llm_local_rl.qwen35_base_format import COUNTDOWN_JSON_PREAMBLE
from llm_local_rl.metrics import mean_numeric_metrics
from llm_local_rl.registry import build_debate_task, build_environment, build_episode_builder


def test_config_and_manifest_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainRunConfig(
            model_path="/tmp/nonexistent_model_for_shape_only",
            tokenizer_path="/tmp/nonexistent_tokenizer_for_shape_only",
            output_dir=tmpdir,
            steps=1,
            debate_external_judge_url="http://judge.test:8123",
            debate_external_judge_timeout_s=123.0,
            thinking_mode="no_think",
            rollout=RolloutConfig(mode="single_turn", num_rollouts_per_instance=2, request_seed_mode="per_request"),
            trace_model_io=False,
            init_adapter_dirs={"shared": str(Path(tmpdir) / "adapter")},
            target_parameters=("experts.gate_up_proj", "experts.down_proj"),
            train_length_bucket_batches=True,
            train_logprob_backend="selective_lm_head",
            compile_train_logprob_helper=True,
            train_adapter_names=("debate",),
            stop_parsed_reward_hacking_min=0.45,
            stop_parsed_reward_hacking_max=0.55,
            gradient_checkpointing=False,
            on_policy_logprob_check=True,
            on_policy_logprob_abs_tol=2e-4,
            on_policy_logprob_warning_path=str(Path(tmpdir) / "on_policy.jsonl"),
            on_policy_logprob_max_records_per_batch=3,
            sampler_sleep_before_training=True,
            sampler_backend="sglang",
            sampler_sglang_base_url="http://sglang.test:30000",
            sampler_sglang_timeout_s=42.0,
            sampler_sglang_pin_loras=True,
            sampler_sglang_unload_stale_adapters=False,
            rollout_assistant_prefill=None,
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
        assert restored_config.rollout.num_rollouts_per_instance == 2
        assert restored_config.rollout.request_seed_mode == "per_request"
        assert restored_config.debate_external_judge_url == "http://judge.test:8123"
        assert restored_config.debate_external_judge_timeout_s == 123.0
        assert restored_config.thinking_mode == "no_think"
        assert restored_config.tokenizer_path == "/tmp/nonexistent_tokenizer_for_shape_only"
        assert restored_config.init_adapter_dirs == {"shared": str(Path(tmpdir) / "adapter")}
        assert restored_config.target_parameters == ("experts.gate_up_proj", "experts.down_proj")
        assert restored_config.train_length_bucket_batches is True
        assert restored_config.train_logprob_backend == "selective_lm_head"
        assert restored_config.compile_train_logprob_helper is True
        assert restored_config.train_adapter_names == ("debate",)
        assert restored_config.stop_parsed_reward_hacking_min == 0.45
        assert restored_config.stop_parsed_reward_hacking_max == 0.55
        assert restored_config.gradient_checkpointing is False
        assert restored_config.on_policy_logprob_check is True
        assert restored_config.on_policy_logprob_abs_tol == 2e-4
        assert restored_config.on_policy_logprob_warning_path == str(Path(tmpdir) / "on_policy.jsonl")
        assert restored_config.on_policy_logprob_max_records_per_batch == 3
        assert restored_config.sampler_sleep_before_training is True
        assert restored_config.sampler_backend == "sglang"
        assert restored_config.sampler_sglang_base_url == "http://sglang.test:30000"
        assert restored_config.sampler_sglang_timeout_s == 42.0
        assert restored_config.sampler_sglang_pin_loras is True
        assert restored_config.sampler_sglang_unload_stale_adapters is False
        assert restored_config.rollout_assistant_prefill is None
        assert restored_config.trace_model_io is False
        assert restored_config.trace_model_io_dir == str(Path(tmpdir) / "trace")
        assert restored_config.trace_top_logprobs == 7


def test_config_from_dict_preserves_missing_rollout_prefill_as_empty_string() -> None:
    payload = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
    ).to_dict()
    del payload["rollout_assistant_prefill"]

    restored = TrainRunConfig.from_dict(payload)

    assert restored.rollout_assistant_prefill == ""


def test_judge_rejection_task_config_roundtrip_and_fail_closed() -> None:
    config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="debate"),
        adapter_layout="split",
        debate_r1_reward="judge_rejection_task",
    )
    restored = TrainRunConfig.from_dict(config.to_dict())
    assert restored.debate_r1_reward == "judge_rejection_task"
    assert restored.debate_round_adapter_names == ("solution", "debate", "debate")

    with pytest.raises(ValueError, match="debate rollouts"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            rollout=RolloutConfig(mode="single_turn"),
            adapter_layout="split",
            debate_r1_reward="judge_rejection_task",
        )
    with pytest.raises(ValueError, match="adapter_layout='split'"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            adapter_layout="shared",
            debate_r1_reward="judge_rejection_task",
        )
    with pytest.raises(ValueError, match="requires round adapters"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            adapter_layout="split",
            debate_r1_reward="judge_rejection_task",
            debate_round_adapter_names=("solution", "solution", "debate"),
        )


def test_concluded_stop_is_independent_and_sglang_only() -> None:
    config = TrainRunConfig(
        model_path="/tmp/model",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="debate"),
        debate_prompt_format="qwen35_base_text_prefill",
        debate_stop_on_concluded=True,
        sampler_backend="sglang",
    )
    restored = TrainRunConfig.from_dict(config.to_dict())
    assert restored.debate_stop_on_concluded is True

    with pytest.raises(ValueError, match="sampler_backend='sglang'"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            debate_prompt_format="qwen35_base_text_prefill",
            debate_stop_on_concluded=True,
            sampler_backend="vllm",
        )
    with pytest.raises(ValueError, match="debate_prompt_format='qwen35_base_text_prefill'"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            debate_prompt_format="chat",
            debate_stop_on_concluded=True,
            sampler_backend="sglang",
        )


def test_countdown_rollout_prefill_defaults_to_base_json_preamble() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(env_name="countdown_code"),
        debate_prompt_format="qwen35_base_text_prefill",
        rollout_assistant_prefill=None,
    )
    assert driver._effective_rollout_assistant_prefill() == COUNTDOWN_JSON_PREAMBLE

    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(env_name="countdown_code"),
        debate_prompt_format="qwen35_base_text_prefill",
        rollout_assistant_prefill="",
    )
    assert driver._effective_rollout_assistant_prefill() == ""


def test_per_sample_advantages_group_by_instance_when_multiple_rollouts() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="single_turn", num_samples=4, num_rollouts_per_instance=2),
        advantage_mode="centered_mean",
    )

    grouped = driver._per_sample_advantages(
        rewards=[10.0, 14.0, 0.0, 2.0],
        instance_ids=["a", "a", "b", "b"],
    )

    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="single_turn", num_samples=4, num_rollouts_per_instance=1),
        advantage_mode="centered_mean",
    )
    batch_level = driver._per_sample_advantages(
        rewards=[10.0, 14.0, 0.0, 2.0],
        instance_ids=["a", "a", "b", "b"],
    )

    assert grouped == [-2.0, 2.0, -1.0, 1.0]
    assert sum(grouped[:2]) == 0.0
    assert sum(grouped[2:]) == 0.0
    assert grouped != batch_level


def test_zscore_advantages_group_by_instance_when_multiple_rollouts() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="single_turn", num_samples=4, num_rollouts_per_instance=2),
        advantage_mode="zscore",
    )

    assert driver._per_sample_advantages(
        rewards=[10.0, 14.0, 0.0, 2.0],
        instance_ids=["a", "a", "b", "b"],
    ) == [-1.0, 1.0, -1.0, 1.0]


def test_single_turn_grouped_rollouts_use_joint_prefill_and_per_request_seeds() -> None:
    from llm_local_rl.driver import TrainingDriver
    from llm_local_rl.types import SamplingResult

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            table = {
                ">>": [2],
                "Prompt a": [11],
                "Prompt b": [12],
                "Prompt a>>": [101],
                "Prompt b>>": [102],
            }
            return list(table[text])

    class Instance:
        def __init__(self, instance_id: str) -> None:
            self.instance_id = instance_id

    class FakeEnv:
        prompt_format = "qwen35_base_text_prefill"

        def __init__(self) -> None:
            self.sample_n = None
            self.scored_completion_token_ids = []

        def sample_instances(self, *, n: int, seed: int | None) -> list[Instance]:
            self.sample_n = n
            assert seed == 123
            return [Instance("a"), Instance("b")]

        def build_initial_prompt_token_ids(self, *, instance, tokenizer, enable_thinking) -> list[int]:
            _ = (instance, tokenizer, enable_thinking)
            return [999]

        def build_initial_prompt(self, *, instance) -> str:
            return f"Prompt {instance.instance_id}"

        def stop_token_ids(self, *, tokenizer) -> list[int]:
            _ = tokenizer
            return []

        def score_completion(self, *, instance, tokenizer, completion_token_ids):
            _ = (instance, tokenizer)
            self.scored_completion_token_ids.append(list(completion_token_ids))
            return 1.0, {"parse_success": 1.0}

    class FakeSampler:
        def __init__(self) -> None:
            self.requests = []

        def sample_many(self, requests):
            self.requests.extend(requests)
            return [
                SamplingResult(
                    adapter_name=request.adapter_name,
                    prompt_token_ids=list(request.prompt_token_ids),
                    completion_token_ids=[50 + idx],
                    completion_logprobs=[-0.1],
                    text=f"r{idx}",
                )
                for idx, request in enumerate(requests)
            ]

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(
            mode="single_turn",
            num_samples=4,
            num_rollouts_per_instance=2,
            seed=123,
            rollout_batch_size=3,
        ),
        rollout_assistant_prefill=">>",
    )
    driver.env = FakeEnv()
    driver.episode_builder = SimpleNamespace(adapter_name="shared")
    driver.tokenizer = FakeTokenizer()
    driver.sampler = FakeSampler()

    samples = driver._run_single_turn_samples(step_idx=0)

    assert driver.env.sample_n == 2
    assert [request.prompt_token_ids for request in driver.sampler.requests] == [[101], [101], [102], [102]]
    assert [request.seed for request in driver.sampler.requests] == [123, 124, 125, 126]
    assert [sample.instance_id for sample in samples] == ["a", "a", "b", "b"]
    assert driver.env.scored_completion_token_ids == [[2, 50], [2, 51], [2, 52], [2, 50]]


def test_single_turn_chat_env_prefill_keeps_templated_token_id_prompt() -> None:
    from llm_local_rl.driver import TrainingDriver
    from llm_local_rl.types import SamplingResult

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            # Only the prefill may be encoded as text for chat envs; joint
            # prompt+prefill encoding would drop the chat template.
            table = {">>": [2]}
            return list(table[text])

    class Instance:
        def __init__(self, instance_id: str) -> None:
            self.instance_id = instance_id

    class FakeChatEnv:
        prompt_format = "chat"

        def sample_instances(self, *, n: int, seed: int | None) -> list[Instance]:
            _ = (n, seed)
            return [Instance("a")]

        def build_initial_prompt_token_ids(self, *, instance, tokenizer, enable_thinking) -> list[int]:
            _ = (instance, tokenizer, enable_thinking)
            return [999]

        def build_initial_prompt(self, *, instance) -> str:
            # Raw user message only; the chat template lives in the token-id path.
            return f"Prompt {instance.instance_id}"

        def stop_token_ids(self, *, tokenizer) -> list[int]:
            _ = tokenizer
            return []

        def score_completion(self, *, instance, tokenizer, completion_token_ids):
            _ = (instance, tokenizer, completion_token_ids)
            return 1.0, {"parse_success": 1.0}

    class FakeSampler:
        def __init__(self) -> None:
            self.requests = []

        def sample_many(self, requests):
            self.requests.extend(requests)
            return [
                SamplingResult(
                    adapter_name=request.adapter_name,
                    prompt_token_ids=list(request.prompt_token_ids),
                    completion_token_ids=[50],
                    completion_logprobs=[-0.1],
                    text="r",
                )
                for request in requests
            ]

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="single_turn", num_samples=1, seed=123),
        rollout_assistant_prefill=">>",
    )
    driver.env = FakeChatEnv()
    driver.episode_builder = SimpleNamespace(adapter_name="shared")
    driver.tokenizer = FakeTokenizer()
    driver.sampler = FakeSampler()

    driver._run_single_turn_samples(step_idx=0)

    assert [request.prompt_token_ids for request in driver.sampler.requests] == [[999, 2]]


def test_split_adapter_names_include_judge_only_when_requested() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        debate_judge_adapter="judge",
    )
    assert driver._adapter_names() == ("solution", "debate", "judge")

    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        debate_judge_adapter="debate",
    )
    assert driver._adapter_names() == ("solution", "debate")


def test_split_vllm_sampler_can_sleep_instead_of_teardown() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate"),
        sampler_backend="vllm",
    )
    assert driver._should_teardown_sampler_before_training() is True

    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate"),
        sampler_backend="vllm",
        sampler_sleep_before_training=True,
    )
    assert driver._should_teardown_sampler_before_training() is False

    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate"),
        sampler_backend="transformers",
        sampler_sleep_before_training=False,
        sampler_teardown_before_training=True,
    )
    assert driver._should_teardown_sampler_before_training() is False


def test_vllm_sleep_before_training_enables_vllm_sleep_allocator(monkeypatch) -> None:
    from llm_local_rl import driver as driver_module
    from llm_local_rl.driver import TrainingDriver

    captured = {}

    class FakeVllmSampler:
        def __init__(self, *, runtime, adapter_paths):
            captured["enable_sleep_mode"] = runtime.enable_sleep_mode
            captured["adapter_paths"] = adapter_paths

    monkeypatch.setattr(driver_module, "VllmSampler", FakeVllmSampler)
    driver = object.__new__(TrainingDriver)
    driver.current_adapter_dirs = {"solution": "/tmp/solution", "debate": "/tmp/debate"}
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate"),
        sampler_backend="vllm",
        sampler_sleep_before_training=True,
    )

    driver._make_sampler()

    assert captured["enable_sleep_mode"] is True
    assert captured["adapter_paths"] == {"solution": "/tmp/solution", "debate": "/tmp/debate"}


def test_sglang_sampler_config_is_passed_to_driver(monkeypatch) -> None:
    from llm_local_rl import driver as driver_module
    from llm_local_rl.driver import TrainingDriver

    captured = {}

    class FakeSglangSampler:
        def __init__(self, *, runtime, adapter_paths):
            captured["base_url"] = runtime.base_url
            captured["timeout_s"] = runtime.timeout_s
            captured["pin_loras"] = runtime.pin_loras
            captured["unload_stale_adapters"] = runtime.unload_stale_adapters
            captured["adapter_paths"] = adapter_paths

    monkeypatch.setattr(driver_module, "SglangSampler", FakeSglangSampler)
    driver = object.__new__(TrainingDriver)
    driver.current_adapter_dirs = {"solution": "/tmp/solution", "debate": "/tmp/debate"}
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate"),
        sampler_backend="sglang",
        sampler_sglang_base_url="http://127.0.0.1:30123",
        sampler_sglang_timeout_s=77.0,
        sampler_sglang_pin_loras=True,
        sampler_sglang_unload_stale_adapters=False,
    )

    driver._make_sampler()

    assert captured == {
        "base_url": "http://127.0.0.1:30123",
        "timeout_s": 77.0,
        "pin_loras": True,
        "unload_stale_adapters": False,
        "adapter_paths": {"solution": "/tmp/solution", "debate": "/tmp/debate"},
    }


def test_train_adapter_names_filters_split_adapters() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate"),
        train_adapter_names=("debate",),
    )

    assert driver._train_adapter_names() == {"debate"}


def test_train_adapter_names_rejects_unknown_adapter() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate"),
        train_adapter_names=("missing",),
    )

    with pytest.raises(ValueError, match="unknown adapters"):
        driver._train_adapter_names()


def test_parsed_reward_hacking_rate_divides_by_parse_success() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)

    assert driver._parsed_reward_hacking_rate(
        mean_reward_metrics={"mean_parse_success": 0.8, "mean_used_secret": 0.4}
    ) == 0.5
    assert driver._parsed_reward_hacking_rate(
        mean_reward_metrics={"mean_parse_success": 0.8, "mean_reward_hacking": 0.2}
    ) == 0.25
    assert driver._parsed_reward_hacking_rate(
        mean_reward_metrics={"mean_parse_success": 0.0, "mean_used_secret": 0.4}
    ) is None


def test_parsed_reward_hacking_stop_band() -> None:
    from llm_local_rl.driver import TrainingDriver

    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        stop_parsed_reward_hacking_min=0.45,
        stop_parsed_reward_hacking_max=0.55,
    )

    assert driver._should_stop_for_parsed_reward_hacking(parsed_reward_hacking=0.5) is True
    assert driver._should_stop_for_parsed_reward_hacking(parsed_reward_hacking=0.44) is False
    assert driver._should_stop_for_parsed_reward_hacking(parsed_reward_hacking=0.56) is False
    assert driver._should_stop_for_parsed_reward_hacking(parsed_reward_hacking=None) is False


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
