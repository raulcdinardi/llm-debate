from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from contextlib import nullcontext

from llm_local_rl.base_model_judge import RemoteBaseJudgeConfig, build_remote_base_judge
from llm_local_rl.config import CheckpointManifest, TrainRunConfig
from llm_local_rl.debate_parity import (
    DebateConfig,
    DebateResult,
    assemble_split_train_examples,
    assemble_training_data_by_mode,
    summarize_judge_rejection_r1_projection,
    training_datum_to_train_example,
)
from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
from llm_local_rl.local_renderers import infer_chat_preamble
from llm_local_rl.masking import make_train_example
from llm_local_rl.metrics import mean_numeric_metrics
from llm_local_rl.mock_judge import SeededRandomJudge
from llm_local_rl.model_io_trace import configure_model_io_tracing, trace_context
from llm_local_rl.qwen35_base_format import resolve_countdown_assistant_prefill
from llm_local_rl.registry import build_debate_task, build_environment, build_episode_builder
from llm_local_rl.resource_monitor import ResourceMonitor
from llm_local_rl.sglang_sampling import SglangRuntimeConfig, SglangSampler
from llm_local_rl.types import EpisodeSample, EpisodeTurn, SamplingRequest
from llm_local_rl.vllm_sampling import VllmRuntimeConfig, VllmSampler


class TrainingDriver:
    def __init__(self, *, config: TrainRunConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resource_monitor = ResourceMonitor(
            output_path=self.output_dir / "resource_usage.jsonl",
            interval_s=config.resource_log_interval_s,
            enabled=config.resource_logging,
        )
        self.resource_monitor.start()
        self._progress("driver_init_start", output_dir=str(self.output_dir), mode=config.rollout.mode)
        self.mock_judge = (
            SeededRandomJudge(seed=config.debate_mock_judge_seed)
            if config.debate_mock_judge_seed is not None
            else None
        )
        self.env = build_environment(config) if config.rollout.mode == "single_turn" else None
        self.episode_builder = build_episode_builder(config) if config.rollout.mode == "single_turn" else None
        self.debate_task = build_debate_task(config) if config.rollout.mode == "debate" else None
        self.tokenizer = self._load_tokenizer()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id or eos_token_id.")
        self._configure_tracing()
        self.start_step = 0
        self.step_records_path = self.output_dir / "step_records.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.current_adapter_dirs = {
            name: str(self.output_dir / f"adapter_init_{name}")
            for name in self._adapter_names()
        }
        with self._stage("init_trainer", step=0):
            self.trainer = (
                self._make_trainer_from_init_adapters()
                if self.config.init_adapter_dirs is not None
                else self._make_fresh_trainer()
            )
        for adapter_name, adapter_dir in self.current_adapter_dirs.items():
            self.current_adapter_dirs[adapter_name] = self.trainer.save_adapter(
                adapter_name=adapter_name,
                output_dir=adapter_dir,
            )
        with self._stage("trainer_sleep", step=0):
            self.trainer.sleep()
        with self._stage("init_sampler", step=0):
            self.sampler = self._make_sampler()
        self._write_manifest(current_step=0)
        self._progress("driver_init_done", output_dir=str(self.output_dir))

    @classmethod
    def resume(cls, *, output_dir: str) -> "TrainingDriver":
        manifest = CheckpointManifest.read_json(Path(output_dir) / "manifest.json")
        config = TrainRunConfig.from_dict(manifest.run_config)
        driver = object.__new__(cls)
        driver.config = config
        driver.output_dir = Path(config.output_dir)
        driver.resource_monitor = ResourceMonitor(
            output_path=driver.output_dir / "resource_usage.jsonl",
            interval_s=config.resource_log_interval_s,
            enabled=config.resource_logging,
        )
        driver.resource_monitor.start()
        driver.start_step = manifest.current_step
        driver._progress("driver_resume_start", output_dir=str(driver.output_dir), start_step=driver.start_step)
        driver.mock_judge = (
            SeededRandomJudge(seed=config.debate_mock_judge_seed)
            if config.debate_mock_judge_seed is not None
            else None
        )
        driver.env = build_environment(config) if config.rollout.mode == "single_turn" else None
        driver.episode_builder = build_episode_builder(config) if config.rollout.mode == "single_turn" else None
        driver.debate_task = build_debate_task(config) if config.rollout.mode == "debate" else None
        driver.tokenizer = driver._load_tokenizer()
        if driver.tokenizer.pad_token_id is None and driver.tokenizer.eos_token_id is not None:
            driver.tokenizer.pad_token = driver.tokenizer.eos_token
        if driver.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id or eos_token_id.")
        driver._configure_tracing()
        driver.step_records_path = driver.output_dir / "step_records.jsonl"
        driver.summary_path = driver.output_dir / "summary.json"
        driver.current_adapter_dirs = dict(manifest.adapter_dirs)
        with driver._stage("init_trainer_resume", step=driver.start_step):
            driver.trainer = driver._make_trainer_from_current_adapters()
        with driver._stage("trainer_sleep", step=driver.start_step):
            driver.trainer.sleep()
        with driver._stage("init_sampler", step=driver.start_step):
            driver.sampler = driver._make_sampler()
        driver._progress("driver_resume_done", output_dir=str(driver.output_dir), start_step=driver.start_step)
        return driver

    def _stage(self, name: str, **metadata: object):
        monitor = getattr(self, "resource_monitor", None)
        if monitor is None or not monitor.enabled:
            return nullcontext()
        return monitor.stage(name, **metadata)

    def _progress(self, event: str, **fields: object) -> None:
        print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)

    def _adapter_names(self) -> tuple[str, ...]:
        if self.config.adapter_layout == "shared":
            return ("shared",)
        if self.config.debate_judge_adapter == "judge":
            return ("solution", "debate", "judge")
        return ("solution", "debate")

    def _train_adapter_names(self) -> set[str] | None:
        if not self.config.train_adapter_names:
            return None
        configured = set(self.config.train_adapter_names)
        available = set(self._adapter_names())
        unknown = sorted(configured - available)
        if unknown:
            raise ValueError(f"train_adapter_names contains unknown adapters: {unknown}; available={sorted(available)}")
        return configured

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.config.tokenizer_path or self.config.model_path, use_fast=True)

    def _debate_config(self) -> DebateConfig:
        r1_max_tokens = self.config.debate_r1_max_tokens or self.config.rollout.max_tokens
        r23_max_tokens = self.config.debate_r23_max_tokens or self.config.rollout.max_tokens
        return DebateConfig(
            num_rounds=self.config.debate_rounds,
            enable_thinking=self._enable_thinking(),
            max_tokens_per_turn=self.config.rollout.max_tokens,
            max_tokens_r1=r1_max_tokens,
            max_tokens_r23=r23_max_tokens,
            temperature=self.config.rollout.temperature,
            chat_preamble=infer_chat_preamble(self.tokenizer),
        )

    def _enable_thinking(self) -> bool | None:
        if self.config.thinking_mode == "default":
            return None
        if self.config.thinking_mode == "no_think":
            return False
        if self.config.thinking_mode == "force_think":
            return True
        raise ValueError(f"Unsupported thinking_mode={self.config.thinking_mode!r}")

    def _effective_rollout_assistant_prefill(self) -> str:
        prefill = self.config.rollout_assistant_prefill
        if self.config.rollout.env_name == "countdown_code":
            return resolve_countdown_assistant_prefill(
                prompt_format=self.config.debate_prompt_format,
                configured_prefill=prefill,
            )
        if prefill is None:
            return ""
        return prefill

    def _rollout_assistant_prefill_token_ids(self) -> list[int]:
        prefill = self._effective_rollout_assistant_prefill()
        if not prefill:
            return []
        return list(self.tokenizer.encode(prefill, add_special_tokens=False))

    def _single_turn_prompt_token_ids(
        self,
        *,
        instance: object,
        prefill_text: str,
        prefill_token_ids: list[int],
    ) -> list[int]:
        text_prompt_builder = getattr(self.env, "build_initial_prompt", None)
        # Joint prompt+prefill encoding is only valid when build_initial_prompt
        # returns the canonical full prompt string. For chat-format envs it
        # returns the raw user message (the chat template is applied in
        # build_initial_prompt_token_ids), so joint encoding would drop the
        # template entirely.
        env_prompt_format = getattr(self.env, "prompt_format", None)
        if prefill_text and callable(text_prompt_builder) and env_prompt_format == "qwen35_base_text_prefill":
            prompt_text = text_prompt_builder(instance=instance)
            return list(self.tokenizer.encode(prompt_text + prefill_text, add_special_tokens=False))

        prompt_builder = getattr(self.env, "build_initial_prompt_token_ids", None)
        if callable(prompt_builder):
            prompt_token_ids = prompt_builder(
                instance=instance,
                tokenizer=self.tokenizer,
                enable_thinking=self._enable_thinking(),
            )
        elif callable(text_prompt_builder):
            prompt_text = text_prompt_builder(instance=instance)
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            raise ValueError("Single-turn environment must expose a text or token-id prompt builder.")

        # Token-id-only prompts cannot be jointly encoded with the assistant
        # prefill, so this fallback can still differ at the BPE boundary.
        return list(prompt_token_ids) + list(prefill_token_ids)

    def _single_turn_num_rollouts_per_instance(self) -> int:
        group_size = int(self.config.rollout.num_rollouts_per_instance)
        if group_size <= 0:
            raise ValueError("rollout.num_rollouts_per_instance must be positive.")
        if self.config.rollout.num_samples % group_size != 0:
            raise ValueError(
                "rollout.num_samples must be divisible by rollout.num_rollouts_per_instance "
                f"for single-turn grouped rollouts; got num_samples={self.config.rollout.num_samples}, "
                f"num_rollouts_per_instance={group_size}."
            )
        return group_size

    def _single_turn_request_seed(
        self,
        *,
        step_idx: int,
        request_idx: int,
        num_rollouts_per_instance: int,
    ) -> int | None:
        mode = self.config.rollout.request_seed_mode
        if mode not in ("none", "per_request"):
            raise ValueError(f"Unsupported request_seed_mode={mode!r}")
        if self.config.rollout.seed is None:
            return None
        if mode == "per_request" or num_rollouts_per_instance > 1:
            return self.config.rollout.seed + step_idx * self.config.rollout.num_samples + request_idx
        return None

    def _trainer_config(self, *, device: str) -> TrainerConfig:
        from llm_local_rl.trainer import TrainerConfig

        on_policy_warning_path = self.config.on_policy_logprob_warning_path
        if self.config.on_policy_logprob_check and on_policy_warning_path is None:
            on_policy_warning_path = str(self.output_dir / "on_policy_logprob_warnings.jsonl")
        return TrainerConfig(
            base_model_path=self.config.model_path,
            adapter_names=self._adapter_names(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            lora_rank=self.config.lora_rank,
            target_modules=self.config.target_modules,
            target_parameters=self.config.target_parameters,
            device=device,
            ppo_clip_epsilon=self.config.ppo_clip_epsilon,
            train_minibatch_size=self.config.train_minibatch_size,
            train_max_tokens=self.config.train_max_tokens,
            train_length_bucket_batches=self.config.train_length_bucket_batches,
            train_logprob_backend=self.config.train_logprob_backend,
            compile_train_logprob_helper=self.config.compile_train_logprob_helper,
            gradient_checkpointing=self.config.gradient_checkpointing,
            on_policy_logprob_check=self.config.on_policy_logprob_check,
            on_policy_logprob_abs_tol=self.config.on_policy_logprob_abs_tol,
            on_policy_logprob_warning_path=on_policy_warning_path,
            on_policy_logprob_max_records_per_batch=self.config.on_policy_logprob_max_records_per_batch,
        )

    def _configure_tracing(self) -> None:
        trace_output_dir = (
            Path(self.config.trace_model_io_dir)
            if self.config.trace_model_io_dir is not None
            else self.output_dir / "model_io_trace"
        )
        configure_model_io_tracing(
            enabled=self.config.trace_model_io,
            output_dir=trace_output_dir,
            tokenizer=self.tokenizer,
            top_logprobs=self.config.trace_top_logprobs,
            metadata={
                "model_path": self.config.model_path,
                "tokenizer_path": self.config.tokenizer_path or self.config.model_path,
            },
        )

    def _make_fresh_trainer(self) -> MultiAdapterTrainer:
        from llm_local_rl.trainer import MultiAdapterTrainer

        return MultiAdapterTrainer(config=self._trainer_config(device="cuda"))

    def _make_trainer_from_init_adapters(self) -> MultiAdapterTrainer:
        from llm_local_rl.trainer import MultiAdapterTrainer

        if self.config.init_adapter_dirs is None:
            raise ValueError("init_adapter_dirs is required for adapter initialization.")
        expected = set(self._adapter_names())
        provided = set(self.config.init_adapter_dirs)
        if provided != expected:
            raise ValueError(f"init_adapter_dirs keys must be {sorted(expected)}, got {sorted(provided)}.")
        return MultiAdapterTrainer.from_saved_adapters(
            config=self._trainer_config(device="cuda"),
            adapter_dirs=self.config.init_adapter_dirs,
        )

    def _make_trainer_from_current_adapters(self) -> MultiAdapterTrainer:
        from llm_local_rl.trainer import MultiAdapterTrainer

        return MultiAdapterTrainer.from_saved_adapters(
            config=self._trainer_config(device="cuda"),
            adapter_dirs=self.current_adapter_dirs,
        )

    def _write_manifest(self, *, current_step: int) -> None:
        manifest = CheckpointManifest(
            run_config=self.config.to_dict(),
            current_step=current_step,
            adapter_dirs=dict(self.current_adapter_dirs),
            step_records_path=str(self.step_records_path),
        )
        manifest.write_json(self.output_dir / "manifest.json")

    def _make_sampler(self):
        if self.config.sampler_backend == "transformers":
            from llm_local_rl.transformers_sampling import TrainerTransformersSampler

            return TrainerTransformersSampler(trainer=self.trainer, tokenizer=self.tokenizer)
        if self.config.sampler_backend == "vllm":
            return VllmSampler(
                runtime=VllmRuntimeConfig(
                    model_path=self.config.model_path,
                    gpu_memory_utilization=self.config.sampler_gpu_memory_utilization,
                    max_model_len=self.config.sampler_max_model_len,
                    max_num_seqs=None if self.config.sampler_max_num_seqs <= 0 else self.config.sampler_max_num_seqs,
                    enforce_eager=self.config.sampler_enforce_eager,
                    enable_sleep_mode=self.config.sampler_sleep_before_training,
                    max_lora_rank=self.config.sampler_max_lora_rank,
                    max_loras=self.config.sampler_max_loras,
                ),
                adapter_paths=dict(self.current_adapter_dirs),
            )
        if self.config.sampler_backend == "sglang":
            return SglangSampler(
                runtime=SglangRuntimeConfig(
                    base_url=self.config.sampler_sglang_base_url,
                    timeout_s=self.config.sampler_sglang_timeout_s,
                    pin_loras=self.config.sampler_sglang_pin_loras,
                    unload_stale_adapters=self.config.sampler_sglang_unload_stale_adapters,
                    memory_saver=self.config.sampler_sleep_before_training,
                ),
                adapter_paths=dict(self.current_adapter_dirs),
            )
        raise ValueError(f"Unsupported sampler_backend={self.config.sampler_backend!r}.")

    def _ensure_sampler(self) -> None:
        if self.sampler is None:
            with self._stage("init_sampler"):
                self.sampler = self._make_sampler()

    def _should_teardown_sampler_before_training(self) -> bool:
        if self.config.sampler_backend == "transformers":
            return False
        if self.config.sampler_sleep_before_training:
            return False
        return self.config.sampler_teardown_before_training or (
            self.config.rollout.mode == "debate" and self.config.adapter_layout == "split"
        )

    def _teardown_sampler(self) -> None:
        if self.sampler is None:
            return
        with self._stage("sampler_teardown"):
            self._progress("sampler_teardown_start")
            self.sampler.close()
            self._progress("sampler_teardown_done")
        self.sampler = None

    def _advantage_values(self, *, rewards: list[float]) -> list[float]:
        if len(rewards) == 0:
            return []
        if self.config.advantage_mode == "identity":
            return list(rewards)
        if self.config.advantage_mode == "centered_mean":
            reward_mean = mean(rewards)
            return [reward - reward_mean for reward in rewards]
        if self.config.advantage_mode == "zscore":
            reward_mean = mean(rewards)
            reward_std = pstdev(rewards)
            if reward_std == 0.0:
                return [0.0 for _ in rewards]
            return [(reward - reward_mean) / reward_std for reward in rewards]
        raise ValueError(f"Unsupported advantage_mode={self.config.advantage_mode!r}")

    def _per_sample_advantages(
        self,
        *,
        rewards: list[float],
        instance_ids: list[str] | None = None,
    ) -> list[float]:
        if self.config.advantage_mode == "identity":
            return list(rewards)
        num_rollouts_per_instance = int(getattr(self.config.rollout, "num_rollouts_per_instance", 1))
        if num_rollouts_per_instance <= 1 or instance_ids is None:
            # With one rollout per prompt there is no within-prompt GRPO
            # baseline; keep the historical batch-level baseline behavior.
            return self._advantage_values(rewards=rewards)
        if len(instance_ids) != len(rewards):
            raise ValueError("instance_ids and rewards must have equal length.")

        positions_by_instance: dict[str, list[int]] = {}
        for idx, instance_id in enumerate(instance_ids):
            positions_by_instance.setdefault(str(instance_id), []).append(idx)

        advantages = [0.0 for _ in rewards]
        for positions in positions_by_instance.values():
            group_rewards = [rewards[idx] for idx in positions]
            group_advantages = self._advantage_values(rewards=group_rewards)
            for idx, advantage in zip(positions, group_advantages, strict=True):
                advantages[idx] = advantage
        return advantages

    def _group_examples(self, *, samples: list) -> dict[str, list]:
        advantages = self._per_sample_advantages(
            rewards=[float(sample.reward) for sample in samples],
            instance_ids=[str(sample.instance_id) for sample in samples],
        )
        grouped: dict[str, list] = {}
        for sample, sample_advantage in zip(samples, advantages, strict=True):
            for turn in sample.turns:
                if not turn.trainable:
                    continue
                if len(turn.completion_token_ids) == 0:
                    continue
                grouped.setdefault(turn.adapter_name, []).append(
                    make_train_example(
                        turn=turn,
                        advantage_per_token=sample_advantage / len(turn.completion_token_ids),
                        extra_metadata={"instance_id": sample.instance_id},
                    )
                )
        return grouped

    def _load_existing_records(self) -> list[dict]:
        if not self.step_records_path.exists():
            return []
        return [json.loads(line) for line in self.step_records_path.read_text().splitlines() if line.strip()]

    def _write_summary(self) -> dict:
        summary = {
            "num_steps_completed": len(self._load_existing_records()),
            "records": self._load_existing_records(),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    def _parsed_reward_hacking_rate(self, *, mean_reward_metrics: dict[str, float]) -> float | None:
        mean_parse_success = float(mean_reward_metrics.get("mean_parse_success", 0.0))
        if mean_parse_success <= 0.0:
            return None
        if "mean_reward_hacking" in mean_reward_metrics:
            mean_reward_hacking = float(mean_reward_metrics["mean_reward_hacking"])
        else:
            mean_reward_hacking = float(mean_reward_metrics.get("mean_used_secret", 0.0))
        return mean_reward_hacking / mean_parse_success

    def _should_stop_for_parsed_reward_hacking(self, *, parsed_reward_hacking: float | None) -> bool:
        min_value = self.config.stop_parsed_reward_hacking_min
        max_value = self.config.stop_parsed_reward_hacking_max
        if min_value is None and max_value is None:
            return False
        if parsed_reward_hacking is None:
            return False
        if min_value is not None and parsed_reward_hacking < min_value:
            return False
        if max_value is not None and parsed_reward_hacking > max_value:
            return False
        return True

    def _debate_runtime(self) -> DebateRuntime:
        if self.debate_task is None:
            raise ValueError("Debate task is not initialized.")
        judge_fn = None
        if self.config.debate_external_judge_url is not None:
            judge_fn = build_remote_base_judge(
                RemoteBaseJudgeConfig(
                    url=self.config.debate_external_judge_url,
                    timeout_s=self.config.debate_external_judge_timeout_s,
                )
            )
        elif getattr(self, "mock_judge", None) is not None:
            judge_fn = self.mock_judge
        return DebateRuntime(
            task=self.debate_task,
            tokenizer=self.tokenizer,
            sampler=self.sampler,
            debate_config=self._debate_config(),
            runtime_config=DebateRuntimeConfig(
                num_rounds=self.config.debate_rounds,
                num_groups=self.config.rollout.num_groups,
                group_size=self.config.rollout.group_size,
                debate_r1_reward=self.config.debate_r1_reward,
                debate_r23_reward=self.config.debate_r23_reward,
                debate_r23_constant=self.config.debate_r23_constant,
                debate_r23_mode=self.config.debate_r23_mode,
                judge_adapter=self.config.debate_judge_adapter,
                round_adapter_names=self.config.debate_round_adapter_names,
                rollout_batch_size=self.config.rollout.rollout_batch_size,
                request_seed_mode=self.config.rollout.request_seed_mode,
                top_p=self.config.rollout.top_p,
                min_p=self.config.rollout.min_p,
                prompt_format=self.config.debate_prompt_format,
                r1_assistant_prefill=self._effective_rollout_assistant_prefill(),
                stop_on_concluded=self.config.debate_stop_on_concluded,
                base_r2_prefill=self.config.base_r2_prefill,
                base_r3_prefill=self.config.base_r3_prefill,
                debate_judge_server_url=self.config.debate_judge_server_url,
                debate_judge_server_adapter_path=self.config.debate_judge_server_adapter_path,
            ),
            adapter_layout=self.config.adapter_layout,
            judge_fn=judge_fn,
        )

    def _run_single_turn_samples(self, *, step_idx: int) -> list[EpisodeSample]:
        step_seed = None if self.config.rollout.seed is None else self.config.rollout.seed + step_idx
        num_rollouts_per_instance = self._single_turn_num_rollouts_per_instance()
        num_instances = self.config.rollout.num_samples // num_rollouts_per_instance
        instances = self.env.sample_instances(
            n=num_instances,
            seed=step_seed,
        )
        adapter_name = self.episode_builder.adapter_name
        request_specs = [
            (instance_idx * num_rollouts_per_instance + rollout_idx, instance)
            for instance_idx, instance in enumerate(instances)
            for rollout_idx in range(num_rollouts_per_instance)
        ]
        if not request_specs:
            return []
        rollout_batch_size = (
            self.config.rollout.rollout_batch_size
            if self.config.rollout.rollout_batch_size > 0
            else len(request_specs)
        )
        prefill_token_ids = self._rollout_assistant_prefill_token_ids()
        prefill_text = self._effective_rollout_assistant_prefill()
        samples: list[EpisodeSample] = []
        for start_idx in range(0, len(request_specs), rollout_batch_size):
            chunk = request_specs[start_idx : start_idx + rollout_batch_size]
            requests = []
            for request_idx, instance in chunk:
                prompt_token_ids = self._single_turn_prompt_token_ids(
                    instance=instance,
                    prefill_text=prefill_text,
                    prefill_token_ids=prefill_token_ids,
                )
                request_seed = self._single_turn_request_seed(
                    step_idx=step_idx,
                    request_idx=request_idx,
                    num_rollouts_per_instance=num_rollouts_per_instance,
                )
                requests.append(
                    SamplingRequest(
                        adapter_name=adapter_name,
                        prompt_token_ids=prompt_token_ids,
                        stop_token_ids=self.env.stop_token_ids(tokenizer=self.tokenizer),
                        max_tokens=self.config.rollout.max_tokens,
                        temperature=self.config.rollout.temperature,
                        seed=request_seed,
                        min_p=self.config.rollout.min_p,
                        top_p=self.config.rollout.top_p,
                    )
                )
            results = self.sampler.sample_many(requests)
            for (_request_idx, instance), result in zip(chunk, results, strict=True):
                scored_completion_token_ids = prefill_token_ids + result.completion_token_ids
                reward, reward_metrics = self.env.score_completion(
                    instance=instance,
                    tokenizer=self.tokenizer,
                    completion_token_ids=scored_completion_token_ids,
                )
                samples.append(
                    EpisodeSample(
                        instance_id=instance.instance_id,
                        turns=[
                            EpisodeTurn(
                                turn_name="response",
                                adapter_name=result.adapter_name,
                                prompt_token_ids=result.prompt_token_ids,
                                completion_token_ids=result.completion_token_ids,
                                completion_logprobs=result.completion_logprobs,
                                trainable=True,
                                metadata={
                                    "text": prefill_text + result.text,
                                    "assistant_prefill": prefill_text,
                                },
                            )
                        ],
                        reward=reward,
                        reward_metrics=reward_metrics,
                    )
                )
        return samples

    def _debate_sample_record(self, debate: DebateResult) -> dict:
        def trajectory_record(traj) -> dict:
            metrics = dict(traj.metrics)
            metrics["round_tokens"] = {
                f"r{transition.round_num}": {
                    "prompt_tokens": len(transition.prompt_tokens),
                    "completion_tokens": len(transition.completion_tokens),
                }
                for transition in traj.transitions
            }
            return metrics

        judge_completion_tokens = debate.judge_completion_tokens or []
        judge_text = (
            self.tokenizer.decode(judge_completion_tokens, skip_special_tokens=True).strip()
            if judge_completion_tokens
            else ""
        )
        return {
            "question": debate.question,
            "verdict": debate.verdict,
            "judge_reasoning": debate.judge_reasoning,
            "judge": {
                "text": judge_text,
                "prompt_tokens": len(debate.judge_prompt_tokens or []),
                "completion_tokens": len(judge_completion_tokens),
                "raw_response": debate.judge_raw_response,
            },
            "trajectory_a": trajectory_record(debate.trajectory_a),
            "trajectory_b": trajectory_record(debate.trajectory_b),
        }

    def _debate_scalar_metrics(self, *, debates: list[DebateResult]) -> dict[str, float]:
        if not debates:
            return {
                "train_judge_a_win_rate": 0.0,
                "train_judge_b_win_rate": 0.0,
                "train_judge_valid_rate": 0.0,
                "train_judge_win_rate": 0.0,
                "train_judge_invalid_rate": 0.0,
                "mean_r2_length": 0.0,
                "mean_r3_length": 0.0,
                "mean_r23_length": 0.0,
                "length_win_correlation": 0.0,
            }
        verdicts = [debate.verdict for debate in debates]
        valid_verdicts = [verdict for verdict in verdicts if verdict in ("A", "B")]
        r2_lengths = [
            len(traj.transitions[1].completion_tokens)
            for debate in debates
            for traj in (debate.trajectory_a, debate.trajectory_b)
            if len(traj.transitions) >= 2
        ]
        r3_lengths = [
            len(traj.transitions[2].completion_tokens)
            for debate in debates
            for traj in (debate.trajectory_a, debate.trajectory_b)
            if len(traj.transitions) >= 3
        ]
        length_deltas = []
        win_signs = []
        for debate in debates:
            if debate.verdict not in ("A", "B"):
                continue
            if len(debate.trajectory_a.transitions) < 3 or len(debate.trajectory_b.transitions) < 3:
                continue
            a_len = sum(len(debate.trajectory_a.transitions[idx].completion_tokens) for idx in (1, 2))
            b_len = sum(len(debate.trajectory_b.transitions[idx].completion_tokens) for idx in (1, 2))
            length_deltas.append(float(a_len - b_len))
            win_signs.append(1.0 if debate.verdict == "A" else -1.0)

        def _corr(xs: list[float], ys: list[float]) -> float:
            if len(xs) < 2:
                return 0.0
            x_mean = mean(xs)
            y_mean = mean(ys)
            x_var = sum((x - x_mean) ** 2 for x in xs)
            y_var = sum((y - y_mean) ** 2 for y in ys)
            if x_var == 0.0 or y_var == 0.0:
                return 0.0
            return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / math.sqrt(x_var * y_var)

        valid_rate = len(valid_verdicts) / len(debates)
        return {
            "train_judge_a_win_rate": sum(1 for verdict in verdicts if verdict == "A") / len(debates),
            "train_judge_b_win_rate": sum(1 for verdict in verdicts if verdict == "B") / len(debates),
            "train_judge_valid_rate": valid_rate,
            "train_judge_win_rate": valid_rate,
            "train_judge_invalid_rate": sum(1 for verdict in verdicts if verdict == "INVALID") / len(debates),
            "mean_r2_length": mean(r2_lengths) if r2_lengths else 0.0,
            "mean_r3_length": mean(r3_lengths) if r3_lengths else 0.0,
            "mean_r23_length": mean([*r2_lengths, *r3_lengths]) if r2_lengths or r3_lengths else 0.0,
            "length_win_correlation": _corr(length_deltas, win_signs),
        }

    def _group_debate_examples(self, *, debates: list, step_seed: int | None) -> tuple[dict[str, list], dict[str, object]]:
        runtime = self._debate_runtime()

        def task_reward_fn(traj, _debate):
            return float(traj.metrics["task_reward"])

        pointwise_reward_map = None
        if self.config.debate_r1_reward == "judge_pointwise":
            pointwise_reward_map = runtime.sample_pointwise_judge_rewards(debates=debates, step_seed=step_seed)

        if self.config.adapter_layout == "shared":
            training_data = assemble_training_data_by_mode(
                debates=debates,
                num_rounds=self.config.debate_rounds,
                r1_reward_mode=self.config.debate_r1_reward,
                r23_reward_mode=self.config.debate_r23_reward,
                r23_constant=self.config.debate_r23_constant,
                r23_symmetric=self.config.debate_r23_mode == "symmetric",
                task_reward_fn=task_reward_fn,
                pointwise_reward_map=pointwise_reward_map,
            )
            grouped = {
                "shared": [
                    training_datum_to_train_example(datum=datum, adapter_name="shared")
                    for datum in training_data
                ]
            }
            return grouped, {
                "source_exact_shared_equivalent": True,
                "num_training_data": len(training_data),
                "num_debates": len(debates),
            }

        grouped = assemble_split_train_examples(
            debates=debates,
            num_rounds=self.config.debate_rounds,
            round_adapter_names=self.config.debate_round_adapter_names,
            r1_reward_mode=self.config.debate_r1_reward,
            r23_reward_mode=self.config.debate_r23_reward,
            r23_constant=self.config.debate_r23_constant,
            r23_symmetric=self.config.debate_r23_mode == "symmetric",
            r23_advantage_scope=self.config.debate_r23_advantage_scope,
            task_reward_fn=task_reward_fn,
            pointwise_reward_map=pointwise_reward_map,
        )
        projection_record: dict[str, object] = {
            "source_exact_shared_equivalent": False,
            "reason": "split_layout_per_round_projection",
            "judge_adapter_policy": self.config.debate_judge_adapter,
            "r23_advantage_scope": self.config.debate_r23_advantage_scope,
            "num_debates": len(debates),
        }
        if self.config.debate_r1_reward == "judge_rejection_task":
            r1_adapter_name = self.config.debate_round_adapter_names[0]
            projection_record["r1_projection"] = {
                "mode": "judge_rejection_task",
                "selection": "judge_winner_only",
                "reward": "objective_task_reward",
                "normalization": "population_zscore_over_selected_winners_per_question",
                **summarize_judge_rejection_r1_projection(
                    r1_examples=grouped.get(r1_adapter_name, []),
                    debates=debates,
                ),
                "r23_mode": self.config.debate_r23_mode,
            }
        return grouped, projection_record

    def run(self, *, max_steps: int | None = None) -> dict:
        try:
            self.config.write_json(self.output_dir / "run_config.json")
            end_step = self.config.steps if max_steps is None else min(self.config.steps, self.start_step + max_steps)
            for step_idx in range(self.start_step, end_step):
                step_num = step_idx + 1
                with self._stage("step", step=step_num):
                    self._progress("step_start", step=step_num, total_steps=self.config.steps)
                    self._ensure_sampler()
                    with self._stage("sampler_wake", step=step_num):
                        self._progress("sampler_wake_start", step=step_num)
                        self.sampler.wake_up()
                        self._progress("sampler_wake_done", step=step_num)
                    self.sampler.set_adapter_paths(adapter_paths=self.current_adapter_dirs)
                    if self.config.rollout.mode == "single_turn":
                        with self._stage("rollout_single_turn", step=step_num), trace_context(
                            step=step_num, rollout_mode="single_turn"
                        ):
                            self._progress("rollout_start", step=step_num, mode="single_turn")
                            samples = self._run_single_turn_samples(step_idx=step_idx)
                            self._progress("rollout_done", step=step_num, mode="single_turn", num_samples=len(samples))
                        grouped_examples = self._group_examples(samples=samples)
                        record_samples = [
                            {
                                "instance_id": sample.instance_id,
                                "reward": sample.reward,
                                "reward_metrics": sample.reward_metrics,
                                "turns": [
                                    {
                                        "turn_name": turn.turn_name,
                                        "adapter_name": turn.adapter_name,
                                        "completion_text": turn.metadata.get("text", ""),
                                    }
                                    for turn in sample.turns
                                ],
                            }
                            for sample in samples
                        ]
                        extra_record = {}
                        mean_reward = mean(float(sample.reward) for sample in samples) if samples else 0.0
                        mean_parse_success = (
                            mean(float(sample.reward_metrics["parse_success"]) for sample in samples)
                            if samples
                            else 0.0
                        )
                        mean_reward_metrics = mean_numeric_metrics([sample.reward_metrics for sample in samples])
                    else:
                        step_seed = None if self.config.rollout.seed is None else self.config.rollout.seed + step_idx
                        debates = []
                        rollout_grad_accum_steps = self.config.rollout_grad_accum_steps
                        for accum_idx in range(rollout_grad_accum_steps):
                            micro_seed = (
                                None
                                if step_seed is None
                                else step_seed + accum_idx * self.config.rollout.num_groups
                            )
                            with self._stage(
                                "rollout_debate",
                                step=step_num,
                                accum_idx=accum_idx,
                            ), trace_context(
                                step=step_num,
                                rollout_mode="debate",
                                accum_idx=accum_idx,
                            ):
                                self._progress(
                                    "rollout_start",
                                    step=step_num,
                                    mode="debate",
                                    num_groups=self.config.rollout.num_groups,
                                    group_size=self.config.rollout.group_size,
                                    accum_idx=accum_idx,
                                    rollout_grad_accum_steps=rollout_grad_accum_steps,
                                )
                                micro_debates = self._debate_runtime().rollout(step_seed=micro_seed).debates
                                debates.extend(micro_debates)
                                self._progress(
                                    "rollout_done",
                                    step=step_num,
                                    mode="debate",
                                    accum_idx=accum_idx,
                                    num_debates=len(micro_debates),
                                    num_debates_total=len(debates),
                                )
                        with self._stage("group_debate_examples", step=step_num), trace_context(
                            step=step_num, rollout_mode="debate", phase_hint="group_debate_examples"
                        ):
                            self._progress("group_examples_start", step=step_num)
                            grouped_examples, extra_record = self._group_debate_examples(
                                debates=debates, step_seed=step_seed
                            )
                            self._progress(
                                "group_examples_done",
                                step=step_num,
                                examples_by_adapter={name: len(batch) for name, batch in grouped_examples.items()},
                            )
                        record_samples = [self._debate_sample_record(debate) for debate in debates]
                        debate_metrics = self._debate_scalar_metrics(debates=debates)
                        rewards = [
                            float(traj.metrics["task_reward"])
                            for debate in debates
                            for traj in (debate.trajectory_a, debate.trajectory_b)
                        ]
                        parse_values = [
                            float(traj.metrics["task_reward_metrics"].get("parse_success", 1.0))
                            for debate in debates
                            for traj in (debate.trajectory_a, debate.trajectory_b)
                        ]
                        task_reward_metrics = [
                            traj.metrics["task_reward_metrics"]
                            for debate in debates
                            for traj in (debate.trajectory_a, debate.trajectory_b)
                        ]
                        mean_reward = mean(rewards) if rewards else 0.0
                        mean_parse_success = mean(parse_values) if parse_values else 0.0
                        mean_reward_metrics = mean_numeric_metrics(task_reward_metrics)
                        mean_reward_metrics.update(debate_metrics)
                    if self._should_teardown_sampler_before_training():
                        self._teardown_sampler()
                    else:
                        with self._stage("sampler_sleep", step=step_num, level=2):
                            self._progress("sampler_sleep_start", step=step_num, level=2)
                            self.sampler.sleep(level=2)
                            self._progress("sampler_sleep_done", step=step_num, level=2)

                    with self._stage("trainer_wake", step=step_num):
                        self._progress("trainer_wake_start", step=step_num)
                        self.trainer.wake_up()
                        self._progress("trainer_wake_done", step=step_num)
                    train_metrics = {}
                    train_adapter_names = self._train_adapter_names()
                    for adapter_name, batch in grouped_examples.items():
                        if train_adapter_names is not None and adapter_name not in train_adapter_names:
                            self._progress(
                                "train_adapter_skipped",
                                step=step_num,
                                adapter_name=adapter_name,
                                reason="not_in_train_adapter_names",
                                num_examples=len(batch),
                            )
                            continue
                        with self._stage("train_adapter", step=step_num, adapter_name=adapter_name), trace_context(
                            step=step_num, phase_hint="train", adapter_name=adapter_name
                        ):
                            self._progress(
                                "train_adapter_start",
                                step=step_num,
                                adapter_name=adapter_name,
                                num_examples=len(batch),
                            )
                            train_metrics[adapter_name] = self.trainer.train_batch(
                                adapter_name=adapter_name, batch=batch
                            )
                            self._progress(
                                "train_adapter_done",
                                step=step_num,
                                adapter_name=adapter_name,
                                metrics=train_metrics[adapter_name],
                            )
                        adapter_dir = self.output_dir / f"step_{step_num:03d}_{adapter_name}"
                        with self._stage("save_adapter", step=step_num, adapter_name=adapter_name):
                            self._progress("save_adapter_start", step=step_num, adapter_name=adapter_name)
                            self.current_adapter_dirs[adapter_name] = self.trainer.save_adapter(
                                adapter_name=adapter_name,
                                output_dir=str(adapter_dir),
                            )
                            self._progress(
                                "save_adapter_done",
                                step=step_num,
                                adapter_name=adapter_name,
                                adapter_dir=self.current_adapter_dirs[adapter_name],
                            )
                    with self._stage("trainer_sleep", step=step_num):
                        self._progress("trainer_sleep_start", step=step_num)
                        self.trainer.sleep()
                        self._progress("trainer_sleep_done", step=step_num)
                    if self._should_teardown_sampler_before_training():
                        self._ensure_sampler()

                    record = {
                        "step": step_num,
                        "mean_reward": mean_reward,
                        "mean_parse_success": mean_parse_success,
                        "mean_reward_metrics": mean_reward_metrics,
                        "train_metrics": train_metrics,
                        "sample_records": record_samples,
                        "adapter_dirs": dict(self.current_adapter_dirs),
                        **extra_record,
                    }
                    parsed_reward_hacking = self._parsed_reward_hacking_rate(
                        mean_reward_metrics=mean_reward_metrics
                    )
                    if parsed_reward_hacking is not None:
                        record["parsed_reward_hacking"] = parsed_reward_hacking
                    if "mean_reward_hacking" in mean_reward_metrics:
                        record["mean_reward_hacking"] = mean_reward_metrics["mean_reward_hacking"]
                    elif "mean_used_secret" in mean_reward_metrics:
                        record["mean_reward_hacking"] = mean_reward_metrics["mean_used_secret"]
                    with self.step_records_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    self._write_manifest(current_step=step_num)
                    self._progress(
                        "step_done",
                        step=step_num,
                        mean_reward=mean_reward,
                        mean_parse_success=mean_parse_success,
                        mean_reward_hacking=record.get("mean_reward_hacking"),
                        parsed_reward_hacking=parsed_reward_hacking,
                    )
                    print(json.dumps(record, indent=2))
                    if self._should_stop_for_parsed_reward_hacking(
                        parsed_reward_hacking=parsed_reward_hacking
                    ):
                        self._progress(
                            "stop_condition_met",
                            step=step_num,
                            condition="parsed_reward_hacking_band",
                            parsed_reward_hacking=parsed_reward_hacking,
                            min_value=self.config.stop_parsed_reward_hacking_min,
                            max_value=self.config.stop_parsed_reward_hacking_max,
                        )
                        break

            summary = self._write_summary()
            self._progress("run_done", num_steps_completed=summary["num_steps_completed"])
            return summary
        finally:
            self.resource_monitor.stop()
