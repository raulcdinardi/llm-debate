from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class RolloutConfig:
    env_name: str = "ht_sequence"
    mode: str = "debate"
    num_samples: int = 16
    num_groups: int = 2
    group_size: int = 8
    rollout_batch_size: int = 0
    max_tokens: int = 1024
    temperature: float = 0.7
    min_p: float = 0.0
    seed: int | None = None
    request_seed_mode: str = "none"


@dataclass(frozen=True)
class TrainRunConfig:
    model_path: str
    output_dir: str
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    steps: int = 1
    learning_rate: float = 1e-5
    adapter_layout: str = "shared"
    sequence_len: int = 8
    reward_mode: str = "num_h"
    strict_ht_format: bool = False
    thinking_mode: str = "default"
    advantage_mode: str = "zscore"
    ppo_clip_epsilon: float = 0.2
    debate_rounds: int = 3
    debate_r1_reward: str = "task"
    debate_r23_reward: str = "constant"
    debate_r23_constant: float = 1.0
    debate_r23_mode: str = "symmetric"
    debate_judge_adapter: str = "policy"
    debate_external_judge_url: str | None = None
    debate_external_judge_timeout_s: float = 600.0
    debate_round_adapter_names: tuple[str, ...] = ("solution", "debate", "debate")
    debate_prompt_format: str = "chat"
    base_r2_prefill: str = "The reasons that my solution is better than my opponent's are:\n1)"
    base_r3_prefill: str = "Responding to my opponent's criticism:\n1)"
    train_minibatch_size: int = 0
    sampler_gpu_memory_utilization: float = 0.55
    sampler_max_model_len: int = 512
    sampler_enforce_eager: bool = True
    sampler_max_lora_rank: int = 32
    sampler_max_loras: int = 4
    sampler_teardown_before_training: bool = False
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_rank: int = 32
    trace_model_io: bool = True
    trace_model_io_dir: str | None = None
    trace_top_logprobs: int = 5
    resource_logging: bool = True
    resource_log_interval_s: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainRunConfig":
        rollout_data = data["rollout"]
        return cls(
            model_path=data["model_path"],
            output_dir=data["output_dir"],
            rollout=RolloutConfig(
                env_name=rollout_data.get("env_name", "ht_sequence"),
                mode=rollout_data.get("mode", "debate"),
                num_samples=rollout_data.get("num_samples", 16),
                num_groups=rollout_data.get("num_groups", 2),
                group_size=rollout_data.get("group_size", 8),
                rollout_batch_size=rollout_data.get("rollout_batch_size", 0),
                max_tokens=rollout_data.get("max_tokens", 1024),
                temperature=rollout_data.get("temperature", 0.7),
                min_p=rollout_data.get("min_p", 0.0),
                seed=rollout_data.get("seed"),
                request_seed_mode=rollout_data.get("request_seed_mode", "none"),
            ),
            steps=data["steps"],
            learning_rate=data["learning_rate"],
            adapter_layout=data["adapter_layout"],
            sequence_len=data["sequence_len"],
            reward_mode=data["reward_mode"],
            strict_ht_format=data.get("strict_ht_format", False),
            thinking_mode=data.get("thinking_mode", "default"),
            advantage_mode=data.get("advantage_mode", "zscore"),
            ppo_clip_epsilon=data.get("ppo_clip_epsilon", 0.2),
            debate_rounds=data.get("debate_rounds", 3),
            debate_r1_reward=data.get("debate_r1_reward", "task"),
            debate_r23_reward=data.get("debate_r23_reward", "constant"),
            debate_r23_constant=data.get("debate_r23_constant", 1.0),
            debate_r23_mode=data.get("debate_r23_mode", "symmetric"),
            debate_judge_adapter=data.get("debate_judge_adapter", "policy"),
            debate_external_judge_url=data.get("debate_external_judge_url"),
            debate_external_judge_timeout_s=data.get("debate_external_judge_timeout_s", 600.0),
            debate_round_adapter_names=tuple(data.get("debate_round_adapter_names", ("solution", "debate", "debate"))),
            debate_prompt_format=data.get("debate_prompt_format", "chat"),
            base_r2_prefill=data.get("base_r2_prefill", "The reasons that my solution is better than my opponent's are:\n1)"),
            base_r3_prefill=data.get("base_r3_prefill", "Responding to my opponent's criticism:\n1)"),
            train_minibatch_size=data.get("train_minibatch_size", 0),
            sampler_gpu_memory_utilization=data.get("sampler_gpu_memory_utilization", 0.55),
            sampler_max_model_len=data.get("sampler_max_model_len", 512),
            sampler_enforce_eager=data.get("sampler_enforce_eager", True),
            sampler_max_lora_rank=data.get("sampler_max_lora_rank", 32),
            sampler_max_loras=data.get("sampler_max_loras", 4),
            sampler_teardown_before_training=data.get("sampler_teardown_before_training", False),
            target_modules=tuple(data["target_modules"]),
            lora_rank=data["lora_rank"],
            trace_model_io=data.get("trace_model_io", True),
            trace_model_io_dir=data.get("trace_model_io_dir"),
            trace_top_logprobs=data.get("trace_top_logprobs", 5),
            resource_logging=data.get("resource_logging", True),
            resource_log_interval_s=data.get("resource_log_interval_s", 5.0),
        )

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


@dataclass(frozen=True)
class CheckpointManifest:
    run_config: dict
    current_step: int
    adapter_dirs: dict[str, str]
    step_records_path: str

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def read_json(cls, path: str | Path) -> "CheckpointManifest":
        return cls(**json.loads(Path(path).read_text()))
