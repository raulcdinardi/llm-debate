from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from statistics import mean, pstdev
from contextlib import nullcontext

from llm_local_rl.base_model_judge import RemoteBaseJudgeConfig, build_remote_base_judge
from llm_local_rl.behavior_policy import validate_sampling_result_contract
from llm_local_rl.config import CheckpointManifest, TrainRunConfig
from llm_local_rl.checkpointing import (
    checkpoint_adapter_dirs,
    load_exact_resume_checkpoint,
    save_exact_resume_checkpoint,
    validate_exact_resume_checkpoint,
)
from llm_local_rl.debate_parity import (
    DebateConfig,
    DebateResult,
    audit_base_text_debate_format,
    assemble_judge_coherence_grpo_examples,
    assemble_judge_supervised_label_examples,
    assemble_split_train_examples,
    assemble_training_data_by_mode,
    summarize_judge_rejection_r1_projection,
    training_datum_to_train_example,
)
from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
from llm_local_rl.local_renderers import infer_chat_preamble
from llm_local_rl.judge_harness import (
    harness_fingerprint,
    validate_judge_harness_manifest,
    write_judge_harness_manifest,
)
from llm_local_rl.masking import make_train_example
from llm_local_rl.metrics import mean_numeric_metrics
from llm_local_rl.mock_judge import SeededRandomJudge
from llm_local_rl.model_io_trace import configure_model_io_tracing, trace_context
from llm_local_rl.qwen35_base_format import resolve_countdown_assistant_prefill
from llm_local_rl.registry import build_debate_task, build_environment, build_episode_builder
from llm_local_rl.resource_monitor import ResourceMonitor
from llm_local_rl.observability import RunObservability, WandbSettings, rollback_rollout_shards
from llm_local_rl.sglang_sampling import SglangRuntimeConfig, SglangSampler
from llm_local_rl.types import EpisodeSample, EpisodeTurn, SamplingRequest
from llm_local_rl.vllm_sampling import VllmRuntimeConfig, VllmSampler


class TrainingDriver:
    def __init__(self, *, config: TrainRunConfig) -> None:
        self.config = config
        if config.target_parameters and config.optimizer_checkpoint_every > 0:
            raise ValueError(
                "Exact optimizer checkpointing is not supported for PEFT target_parameters; "
                "use target_modules or explicitly set optimizer_checkpoint_every=0."
            )
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
        self.latest_exact_resume_checkpoint: str | None = None
        self._metric_history: dict[str, list[float]] = {}
        self.current_adapter_dirs = {
            name: str(self.output_dir / f"adapter_init_{name}")
            for name in self._adapter_names()
        }
        if self.config.init_adapter_dirs is not None:
            self._validate_judge_adapter_harness(self.config.init_adapter_dirs)
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
            self._write_saved_judge_harness(
                adapter_name=adapter_name,
                adapter_dir=self.current_adapter_dirs[adapter_name],
            )
        self.reference_adapter_dirs = dict(self.current_adapter_dirs)
        if self.config.reference_kl_every > 0:
            self.trainer.load_reference_adapters(adapter_dirs=self.reference_adapter_dirs)
        with self._stage("trainer_sleep", step=0):
            self.trainer.sleep()
        if self.config.optimizer_checkpoint_every > 0:
            self.latest_exact_resume_checkpoint = str(
                save_exact_resume_checkpoint(
                    root=self.output_dir / "checkpoints" / "exact_resume",
                    step=0,
                    run_config=self.config.to_dict(),
                    adapter_dirs=self.current_adapter_dirs,
                    trainer=self.trainer,
                )
            )
        with self._stage("init_sampler", step=0):
            self.sampler = self._make_sampler()
        self.observability = self._make_observability()
        self.observability.log_step(
            {
                "step": 0,
                "mean_reward": 0.0,
                "mean_parse_success": 0.0,
                "rollout_metrics": {"phase0_observability_probe": 1.0},
            }
        )
        if self.latest_exact_resume_checkpoint is not None:
            self.observability.log_artifact(
                kind="exact-resume-checkpoint",
                path=self.latest_exact_resume_checkpoint,
                metadata={"step": 0, "phase0_probe": True},
            )
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
        driver.latest_exact_resume_checkpoint = manifest.exact_resume_checkpoint
        driver._metric_history = {}
        driver.reference_adapter_dirs = dict(manifest.reference_adapter_dirs or {})
        if manifest.exact_resume_checkpoint is not None:
            exact_path = Path(manifest.exact_resume_checkpoint)
            exact_manifest = validate_exact_resume_checkpoint(
                exact_path, run_config=config.to_dict()
            )
            driver.start_step = int(exact_manifest["completed_step"])
            driver.current_adapter_dirs = checkpoint_adapter_dirs(exact_path)
            driver._rollback_step_records_to(step=driver.start_step)
        else:
            if manifest.current_step > 0:
                raise RuntimeError(
                    "This run has no exact-resume checkpoint. Refusing an adapter-only resume "
                    "that would silently reset Adam and RNG state."
                )
            driver.current_adapter_dirs = dict(manifest.adapter_dirs)
        driver._validate_judge_adapter_harness(driver.current_adapter_dirs)
        with driver._stage("init_trainer_resume", step=driver.start_step):
            driver.trainer = driver._make_trainer_from_current_adapters()
            if manifest.exact_resume_checkpoint is not None:
                load_exact_resume_checkpoint(
                    path=manifest.exact_resume_checkpoint,
                    trainer=driver.trainer,
                    run_config=config.to_dict(),
                )
            if config.reference_kl_every > 0 and driver.reference_adapter_dirs:
                driver.trainer.load_reference_adapters(adapter_dirs=driver.reference_adapter_dirs)
        with driver._stage("trainer_sleep", step=driver.start_step):
            driver.trainer.sleep()
        with driver._stage("init_sampler", step=driver.start_step):
            driver.sampler = driver._make_sampler()
        driver.observability = driver._make_observability()
        driver._rebuild_metric_history()
        driver._progress("driver_resume_done", output_dir=str(driver.output_dir), start_step=driver.start_step)
        return driver

    def _stage(self, name: str, **metadata: object):
        monitor = getattr(self, "resource_monitor", None)
        if monitor is None or not monitor.enabled:
            return nullcontext()
        return monitor.stage(name, **metadata)

    def _progress(self, event: str, **fields: object) -> None:
        print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)

    def _make_observability(self) -> RunObservability:
        return RunObservability(
            output_dir=self.output_dir,
            config=self.config.to_dict(),
            settings=WandbSettings(
                enabled=self.config.wandb_enabled,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=self.config.wandb_group,
                name=self.config.wandb_run_name,
                mode=self.config.wandb_mode,
                upload_artifacts=self.config.wandb_upload_artifacts,
                rollout_shard_steps=self.config.rollout_shard_every,
                table_samples_per_shard=self.config.wandb_table_samples_per_shard,
            ),
        )

    def _rollback_step_records_to(self, *, step: int) -> None:
        if not self.step_records_path.exists():
            return
        lines = self.step_records_path.read_text().splitlines()
        kept = [line for line in lines if line.strip() and int(json.loads(line)["step"]) <= step]
        dropped = [line for line in lines if line.strip() and int(json.loads(line)["step"]) > step]
        if dropped:
            base = self.output_dir / f"step_records_after_exact_step_{step:06d}.superseded"
            index = 1
            suffix = Path(f"{base}.{index:03d}.jsonl")
            while suffix.exists():
                index += 1
                suffix = Path(f"{base}.{index:03d}.jsonl")
            suffix.write_text("\n".join(dropped) + "\n")
            self.step_records_path.write_text("\n".join(kept) + ("\n" if kept else ""))
            rollback_rollout_shards(output_dir=self.output_dir, completed_step=step)

    @staticmethod
    def _correlation(xs: list[float], ys: list[float]) -> float:
        if len(xs) < 2 or len(xs) != len(ys):
            return 0.0
        x_mean, y_mean = mean(xs), mean(ys)
        x_var = sum((x - x_mean) ** 2 for x in xs)
        y_var = sum((y - y_mean) ** 2 for y in ys)
        if x_var == 0.0 or y_var == 0.0:
            return 0.0
        return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / math.sqrt(x_var * y_var)

    def _add_temporal_metrics(self, record: dict[str, object]) -> None:
        train_metrics = record.get("train_metrics", {})
        entropy_values = [float(item["entropy"]) for item in train_metrics.values() if "entropy" in item]
        kl_values = [float(item["ppo_sampled_approx_kl"]) for item in train_metrics.values() if "ppo_sampled_approx_kl" in item]
        rollout_metrics = record.setdefault("rollout_metrics", {})
        values = {
            "reward": float(record.get("mean_reward", 0.0)),
            "entropy": mean(entropy_values) if entropy_values else 0.0,
            "kl": mean(kl_values) if kl_values else 0.0,
            "length": float(rollout_metrics.get("completion_length_mean", 0.0)),
        }
        for key, value in values.items():
            self._metric_history.setdefault(key, []).append(value)
        entropy_history = self._metric_history["entropy"]
        rollout_metrics["policy_entropy_mean"] = values["entropy"]
        rollout_metrics["policy_entropy_change"] = (
            entropy_history[-1] - entropy_history[-2] if len(entropy_history) >= 2 else 0.0
        )
        window = entropy_history[-10:]
        if len(window) >= 2:
            xs = list(range(len(window)))
            rollout_metrics["policy_entropy_rolling_slope_10"] = self._correlation(xs, window) * (
                pstdev(window) / pstdev(xs) if pstdev(xs) > 0 else 0.0
            )
        else:
            rollout_metrics["policy_entropy_rolling_slope_10"] = 0.0
        tail = slice(max(0, len(self._metric_history["reward"]) - 20), None)
        rewards = self._metric_history["reward"][tail]
        rollout_metrics["reward_vs_kl_correlation_20"] = self._correlation(rewards, self._metric_history["kl"][tail])
        rollout_metrics["reward_vs_entropy_correlation_20"] = self._correlation(rewards, self._metric_history["entropy"][tail])
        rollout_metrics["reward_vs_completion_length_correlation_20"] = self._correlation(rewards, self._metric_history["length"][tail])

    def _rebuild_metric_history(self) -> None:
        if not self.step_records_path.exists():
            return
        for line in self.step_records_path.read_text().splitlines():
            if line.strip():
                self._add_temporal_metrics(json.loads(line))

    def _adapter_names(self) -> tuple[str, ...]:
        if self.config.adapter_layout == "shared":
            return ("shared",)
        if self.config.debate_judge_adapter == "judge":
            return ("solution", "debate", "judge")
        return ("solution", "debate")

    def _validate_judge_adapter_harness(self, adapter_dirs: dict[str, str]) -> None:
        judge_dir = adapter_dirs.get("judge")
        if judge_dir is None or self.config.debate_judge_adapter != "judge":
            return
        validate_judge_harness_manifest(
            adapter_dir=judge_dir,
            harness_id=self.config.judge_harness().harness_id,
        )

    def _write_saved_judge_harness(self, *, adapter_name: str, adapter_dir: str) -> None:
        if adapter_name != "judge" or self.config.debate_judge_adapter != "judge":
            return
        write_judge_harness_manifest(
            adapter_dir=adapter_dir,
            harness_id=self.config.judge_harness().harness_id,
        )

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
        r2_max_tokens = self.config.debate_r2_max_tokens or r23_max_tokens
        r3_max_tokens = self.config.debate_r3_max_tokens or r23_max_tokens
        return DebateConfig(
            num_rounds=self.config.debate_rounds,
            enable_thinking=self._enable_thinking(),
            max_tokens_per_turn=self.config.rollout.max_tokens,
            max_tokens_r1=r1_max_tokens,
            max_tokens_r23=r23_max_tokens,
            max_tokens_r2=r2_max_tokens,
            max_tokens_r3=r3_max_tokens,
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
            on_policy_logprob_warn_only=self.config.on_policy_logprob_warn_only,
            on_policy_logprob_abs_tol=self.config.on_policy_logprob_abs_tol,
            on_policy_logprob_warning_path=on_policy_warning_path,
            on_policy_logprob_max_records_per_batch=self.config.on_policy_logprob_max_records_per_batch,
            behavior_policy=self.config.behavior_policy(),
        )

    def _configure_tracing(self) -> None:
        judge_harness = self.config.judge_harness()
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
                "behavior_policy_contract": self.config.behavior_policy().to_dict(),
                "judge_harness_id": judge_harness.harness_id,
                "judge_harness_fingerprint": harness_fingerprint(
                    judge_harness.harness_id
                ),
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
            exact_resume_checkpoint=self.latest_exact_resume_checkpoint,
            reference_adapter_dirs=dict(self.reference_adapter_dirs),
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
                    enable_sleep_mode=self._should_sleep_sampler_before_training(),
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
        if self.config.sampler_teardown_before_training:
            return True
        return self.config.sampler_backend != "vllm" and (
            self.config.rollout.mode == "debate" and self.config.adapter_layout == "split"
        )

    def _should_sleep_sampler_before_training(self) -> bool:
        if self.config.sampler_backend == "transformers" or self._should_teardown_sampler_before_training():
            return False
        return self.config.sampler_backend == "vllm" or self.config.sampler_sleep_before_training

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
        judge_harness = self.config.judge_harness()
        if self.config.debate_external_judge_url is not None:
            judge_fn = build_remote_base_judge(
                RemoteBaseJudgeConfig(
                    url=self.config.debate_external_judge_url,
                    harness_id=judge_harness.harness_id,
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
                judge_harness_id=judge_harness.harness_id,
                judge_max_tokens=self.config.debate_judge_max_tokens,
                judge_temperature=self.config.debate_judge_temperature,
                judge_top_p=self.config.debate_judge_top_p,
                judge_top_k=self.config.debate_judge_top_k,
                judge_min_p=self.config.debate_judge_min_p,
                judge_presence_penalty=self.config.debate_judge_presence_penalty,
                judge_repetition_penalty=self.config.debate_judge_repetition_penalty,
                judge_seed=self.config.debate_judge_seed,
                judge_bidirectional=self.config.debate_judge_bidirectional,
                judge_constrain_single_token=self.config.debate_judge_constrain_single_token,
                judge_score_mode=self.config.debate_judge_score_mode,
                judge_label_token_contract=self.config.judge_label_token_contract,
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
            for (_request_idx, instance), request, result in zip(
                chunk,
                requests,
                results,
                strict=True,
            ):
                validate_sampling_result_contract(request=request, result=result)
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
        raw_judge_response = debate.judge_raw_response
        if isinstance(raw_judge_response, dict):
            raw_judge_response = {
                key: value for key, value in raw_judge_response.items()
                if key != "_training_judge_turns"
            }
        return {
            "question": debate.question,
            "verdict": debate.verdict,
            "judge_reasoning": debate.judge_reasoning,
            "judge": {
                "text": judge_text,
                "prompt_tokens": len(debate.judge_prompt_tokens or []),
                "completion_tokens": len(judge_completion_tokens),
                "raw_response": raw_judge_response,
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
                "train_judge_order_invariant_rate": 0.0,
                "train_judge_order_disagreement_rate": 0.0,
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
        r1_turns = [
            traj.transitions[0]
            for debate in debates
            for traj in (debate.trajectory_a, debate.trajectory_b)
            if traj.transitions
        ]
        r2_turns = [
            traj.transitions[1]
            for debate in debates
            for traj in (debate.trajectory_a, debate.trajectory_b)
            if len(traj.transitions) >= 2
        ]
        r3_turns = [
            traj.transitions[2]
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
        bidirectional_audits = [
            debate.judge_raw_response for debate in debates
            if isinstance(debate.judge_raw_response, dict)
            and debate.judge_raw_response.get("bidirectional_judge") is True
        ]
        order_invariant_rate = (
            sum(bool(audit.get("order_invariant")) for audit in bidirectional_audits)
            / len(bidirectional_audits) if bidirectional_audits else 0.0
        )
        soft_records = [
            audit.get("soft_score")
            for audit in bidirectional_audits
            if audit.get("soft_judge") is True and isinstance(audit.get("soft_score"), dict)
        ]
        soft_scores = [float(record["score"]) for record in soft_records]
        soft_z_symmetric = [float(record["z_symmetric"]) for record in soft_records]
        soft_order_bias = [float(record["order_bias_logit"]) for record in soft_records]
        metrics = {
            "train_judge_a_win_rate": sum(1 for verdict in verdicts if verdict == "A") / len(debates),
            "train_judge_b_win_rate": sum(1 for verdict in verdicts if verdict == "B") / len(debates),
            "train_judge_valid_rate": valid_rate,
            "train_judge_win_rate": valid_rate,
            "train_judge_invalid_rate": sum(1 for verdict in verdicts if verdict == "INVALID") / len(debates),
            "train_judge_order_invariant_rate": order_invariant_rate,
            "train_judge_order_disagreement_rate": 1.0 - order_invariant_rate if bidirectional_audits else 0.0,
            "mean_r2_length": mean(r2_lengths) if r2_lengths else 0.0,
            "mean_r3_length": mean(r3_lengths) if r3_lengths else 0.0,
            "mean_r23_length": mean([*r2_lengths, *r3_lengths]) if r2_lengths or r3_lengths else 0.0,
            "mean_r1_length": mean(len(turn.completion_tokens) for turn in r1_turns) if r1_turns else 0.0,
            "max_r1_length": max((len(turn.completion_tokens) for turn in r1_turns), default=0),
            "max_r2_length": max(r2_lengths, default=0),
            "max_r3_length": max(r3_lengths, default=0),
            "r1_max_token_rate": (
                mean(len(turn.completion_tokens) >= (self.config.debate_r1_max_tokens or self.config.rollout.max_tokens) for turn in r1_turns)
                if r1_turns else 0.0
            ),
            "r2_max_token_rate": (
                mean(len(turn.completion_tokens) >= (self.config.debate_r2_max_tokens or self.config.debate_r23_max_tokens or self.config.rollout.max_tokens) for turn in r2_turns)
                if r2_turns else 0.0
            ),
            "r3_max_token_rate": (
                mean(len(turn.completion_tokens) >= (self.config.debate_r3_max_tokens or self.config.debate_r23_max_tokens or self.config.rollout.max_tokens) for turn in r3_turns)
                if r3_turns else 0.0
            ),
            "r1_eos_rate": mean(bool(turn.completion_tokens) and turn.completion_tokens[-1] == self.tokenizer.eos_token_id for turn in r1_turns) if r1_turns else 0.0,
            "r2_eos_rate": mean(bool(turn.completion_tokens) and turn.completion_tokens[-1] == self.tokenizer.eos_token_id for turn in r2_turns) if r2_turns else 0.0,
            "r3_eos_rate": mean(bool(turn.completion_tokens) and turn.completion_tokens[-1] == self.tokenizer.eos_token_id for turn in r3_turns) if r3_turns else 0.0,
            "length_win_correlation": _corr(length_deltas, win_signs),
        }
        if soft_scores:
            metrics.update({
                "train_judge_soft_score_mean": mean(soft_scores),
                "train_judge_soft_score_std": pstdev(soft_scores),
                "train_judge_soft_score_mean_abs": mean(abs(value) for value in soft_scores),
                "train_judge_soft_score_near_zero_rate": mean(
                    abs(value) <= 0.05 for value in soft_scores
                ),
                "train_judge_soft_z_symmetric_mean": mean(soft_z_symmetric),
                "train_judge_soft_order_bias_logit_mean": mean(soft_order_bias),
                "train_judge_soft_order_bias_logit_mean_abs": mean(
                    abs(value) for value in soft_order_bias
                ),
                "train_judge_soft_score_valid_rate": len(soft_scores) / len(debates),
                "train_judge_soft_zero_sum_residual_max_abs": max(
                    abs(float(audit.get("zero_sum_residual", math.inf)))
                    for audit in bidirectional_audits
                    if audit.get("soft_judge") is True
                ),
            })
        return metrics

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
            r1_judge_delta_q=self.config.debate_r1_judge_delta_q,
            incoherent_r23_reward=self.config.debate_incoherent_r23_reward,
            r23_format_failure_penalty=self.config.debate_r23_format_failure_penalty,
        )
        judge_grpo_record: dict[str, float | int | str] | None = None
        if self.config.train_judge_coherence_grpo:
            if self.config.judge_training_objective == "supervised_label_ce":
                judge_examples, judge_grpo_record = assemble_judge_supervised_label_examples(
                    debates
                )
            else:
                judge_examples, judge_grpo_record = assemble_judge_coherence_grpo_examples(
                    debates, reward_mode=self.config.judge_grpo_reward_mode
                )
            grouped.setdefault("judge", []).extend(judge_examples)
        projection_record: dict[str, object] = {
            "source_exact_shared_equivalent": False,
            "reason": "split_layout_per_round_projection",
            "judge_adapter_policy": self.config.debate_judge_adapter,
            "r23_advantage_scope": self.config.debate_r23_advantage_scope,
            "r23_format_failure_penalty": self.config.debate_r23_format_failure_penalty,
            "num_debates": len(debates),
        }
        if self.config.debate_r23_format_failure_penalty != 0.0:
            trajectories = [
                trajectory
                for debate in debates
                for trajectory in (debate.trajectory_a, debate.trajectory_b)
            ]
            r2_audits = [
                audit_base_text_debate_format(text=str(trajectory.metrics.get("r2", "")), round_num=2)
                for trajectory in trajectories
            ]
            r3_audits = [
                audit_base_text_debate_format(text=str(trajectory.metrics.get("r3", "")), round_num=3)
                for trajectory in trajectories
            ]
            projection_record["debate_format"] = {
                "schema": "base_text_raw_exact_three_points_terminal_concluded_v2",
                "r2_strict_rate": mean(float(audit["strict_ok"]) for audit in r2_audits),
                "r3_strict_rate": mean(float(audit["strict_ok"]) for audit in r3_audits),
                "r2_legacy_truncation_trigger_rate": mean(
                    float(audit["legacy_truncation_triggered"]) for audit in r2_audits
                ),
                "r3_legacy_truncation_trigger_rate": mean(
                    float(audit["legacy_truncation_triggered"]) for audit in r3_audits
                ),
                "both_rounds_strict_rate": mean(
                    float(r2["strict_ok"] and r3["strict_ok"])
                    for r2, r3 in zip(r2_audits, r3_audits, strict=True)
                ),
                "round_outputs": len(r2_audits) + len(r3_audits),
            }
        if judge_grpo_record is not None:
            if self.config.judge_training_objective == "supervised_label_ce":
                projection_record["judge_supervised_label"] = judge_grpo_record
            else:
                projection_record["judge_grpo"] = judge_grpo_record
                projection_record[
                    "judge_label_grpo"
                    if self.config.judge_grpo_reward_mode in ("label", "label_js")
                    else "judge_coherence_grpo"
                ] = judge_grpo_record
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
        elif self.config.debate_r1_reward == "judge_delta_task":
            projection_record["r1_projection"] = {
                "mode": "judge_delta_task",
                "formula": "task_reward +/- q * abs(task_reward_a - task_reward_b)",
                "q": self.config.debate_r1_judge_delta_q,
                "incoherent_r1": "seeded_coin_flip_winner_task_reward_plus_minus_q_delta",
                "incoherent_r23_reward_per_trajectory": self.config.debate_incoherent_r23_reward,
            }
        elif self.config.debate_r1_reward == "judge_soft_task_gap":
            projection_record["r1_projection"] = {
                "mode": "judge_soft_task_gap",
                "formula": "M +/- s * abs(task_reward_a - task_reward_b) / 2",
                "judge_score": "tanh((z_forward-z_reverse)/4)",
                "pair_sum_conserved_before_group_normalization": True,
            }
        if self.config.debate_r23_reward == "soft_judge":
            projection_record["r23_projection"] = {
                "mode": "soft_judge",
                "formula": "reward_a=s; reward_b=-s",
                "exact_zero_sum_per_debate": True,
                "r23_constant_ignored": True,
            }
        return grouped, projection_record

    def _adapter_output_dir(self, *, step: int, adapter_name: str, durable: bool) -> Path:
        if durable:
            return self.output_dir / "checkpoints" / "lora" / f"step_{step:06d}"
        return self.output_dir / ".live_adapters" / f"step_{step:06d}"

    def _discard_superseded_live_adapters(self, previous_dirs: dict[str, str]) -> None:
        live_root = (self.output_dir / ".live_adapters").resolve()
        for path_text in previous_dirs.values():
            path = Path(path_text).resolve()
            if live_root not in path.parents:
                continue
            step_root = next((parent for parent in path.parents if parent.parent == live_root), None)
            if step_root is not None and step_root.exists():
                shutil.rmtree(step_root)

    @staticmethod
    def _step_distribution_metrics(*, rewards: list[float], grouped_examples: dict[str, list]) -> dict[str, float]:
        out: dict[str, float] = {}
        if rewards:
            ordered = sorted(float(value) for value in rewards)
            out.update(
                reward_std=pstdev(ordered),
                reward_min=min(ordered),
                reward_max=max(ordered),
                reward_p10=ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
                reward_p50=ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)],
                reward_p90=ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
                reward_p99=ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)],
            )
        advantages = [
            float(value)
            for batch in grouped_examples.values()
            for example in batch
            for value, mask in zip(example.advantages, example.loss_mask, strict=True)
            if mask
        ]
        if advantages:
            ordered_adv = sorted(advantages)
            positive = [value for value in advantages if value > 0]
            negative = [value for value in advantages if value < 0]
            out.update(
                advantage_mean=mean(advantages),
                advantage_std=pstdev(advantages),
                advantage_max_abs=max(abs(value) for value in advantages),
                advantage_fraction_positive=len(positive) / len(advantages),
                advantage_fraction_negative=len(negative) / len(advantages),
                advantage_total_positive_abs=sum(positive),
                advantage_total_negative_abs=sum(abs(value) for value in negative),
                advantage_positive_negative_magnitude_ratio=(
                    sum(positive) / sum(abs(value) for value in negative) if negative else 0.0
                ),
                advantage_p01=ordered_adv[max(0, math.ceil(0.01 * len(ordered_adv)) - 1)],
                advantage_p05=ordered_adv[max(0, math.ceil(0.05 * len(ordered_adv)) - 1)],
                advantage_p95=ordered_adv[max(0, math.ceil(0.95 * len(ordered_adv)) - 1)],
                advantage_p99=ordered_adv[max(0, math.ceil(0.99 * len(ordered_adv)) - 1)],
                advantage_p999=ordered_adv[max(0, math.ceil(0.999 * len(ordered_adv)) - 1)],
            )
        group_std_values: list[float] = []
        rewards_by_group: dict[tuple[str, str], list[float]] = {}
        completion_lengths: list[int] = []
        seen_groups: set[tuple[str, str]] = set()
        for adapter, batch in grouped_examples.items():
            for example in batch:
                completion_lengths.append(sum(int(value) for value in example.loss_mask))
                metadata = example.metadata
                question = str(metadata.get("question", ""))
                key = (adapter, question)
                raw_reward = metadata.get("reward", metadata.get("r1_reward"))
                if isinstance(raw_reward, (int, float)) and math.isfinite(float(raw_reward)):
                    rewards_by_group.setdefault(key, []).append(float(raw_reward))
                if key in seen_groups:
                    continue
                std = metadata.get("group_std_reward", metadata.get("r1_group_std_reward"))
                if isinstance(std, (int, float)) and math.isfinite(float(std)):
                    seen_groups.add(key)
                    group_std_values.append(float(std))
        if group_std_values:
            ordered_std = sorted(group_std_values)
            out["grpo_zero_variance_group_fraction"] = mean(value == 0.0 for value in group_std_values)
            out["group_reward_std_mean"] = mean(group_std_values)
            out["group_reward_std_p50"] = ordered_std[max(0, math.ceil(0.50 * len(ordered_std)) - 1)]
            out["group_reward_std_p10"] = ordered_std[max(0, math.ceil(0.10 * len(ordered_std)) - 1)]
            out["group_reward_std_p90"] = ordered_std[max(0, math.ceil(0.90 * len(ordered_std)) - 1)]
            out["group_reward_std_p99"] = ordered_std[max(0, math.ceil(0.99 * len(ordered_std)) - 1)]
        binary_groups = [values for values in rewards_by_group.values() if values and all(value in (0.0, 1.0) for value in values)]
        if binary_groups and len(binary_groups) == len(rewards_by_group):
            histogram: dict[int, int] = {}
            for values in binary_groups:
                correct = int(sum(values))
                histogram[correct] = histogram.get(correct, 0) + 1
            for correct, count in sorted(histogram.items()):
                out[f"grpo_group_correct_count_histogram/{correct}"] = float(count)
        if completion_lengths:
            out["completion_length_mean"] = mean(completion_lengths)
            out["completion_length_max"] = max(completion_lengths)
        out["effective_rollout_batch_size"] = float(len(rewards))
        out["effective_optimizer_batch_size"] = float(sum(len(batch) for batch in grouped_examples.values()))
        return out

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
                        step_rewards = [float(sample.reward) for sample in samples]
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
                        step_rewards = rewards
                    if self._should_teardown_sampler_before_training():
                        self._teardown_sampler()
                    elif self._should_sleep_sampler_before_training():
                        train_adapter_names = self._train_adapter_names()
                        adapter_names_to_unload = (
                            set(grouped_examples)
                            if train_adapter_names is None
                            else set(grouped_examples) & train_adapter_names
                        )
                        with self._stage("sampler_unload_trainable_loras", step=step_num):
                            self._progress(
                                "sampler_unload_trainable_loras_start",
                                step=step_num,
                                adapter_names=sorted(adapter_names_to_unload),
                            )
                            self.sampler.unload_adapters(adapter_names=adapter_names_to_unload)
                            self._progress(
                                "sampler_unload_trainable_loras_done",
                                step=step_num,
                                adapter_names=sorted(adapter_names_to_unload),
                            )
                        sleep_level = int(self.config.sampler_sleep_level)
                        with self._stage("sampler_sleep", step=step_num, level=sleep_level):
                            self._progress("sampler_sleep_start", step=step_num, level=sleep_level)
                            self.sampler.sleep(level=sleep_level)
                            self._progress("sampler_sleep_done", step=step_num, level=sleep_level)

                    with self._stage("trainer_wake", step=step_num):
                        self._progress("trainer_wake_start", step=step_num)
                        self.trainer.wake_up()
                        self._progress("trainer_wake_done", step=step_num)
                    train_metrics = {}
                    train_adapter_names = self._train_adapter_names()
                    previous_adapter_dirs = dict(self.current_adapter_dirs)
                    durable_lora_step = (
                        step_num % self.config.adapter_checkpoint_every == 0
                        or step_num == self.config.steps
                    )
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
                                adapter_name=adapter_name,
                                batch=batch,
                                objective=(
                                    "supervised_label_ce"
                                    if adapter_name == "judge"
                                    and self.config.judge_training_objective
                                    == "supervised_label_ce"
                                    else "ppo"
                                ),
                                measure_reference_kl=(
                                    not (
                                        adapter_name == "judge"
                                        and self.config.judge_training_objective
                                        == "supervised_label_ce"
                                    )
                                    and
                                    self.config.reference_kl_every > 0
                                    and step_num % self.config.reference_kl_every == 0
                                ),
                            )
                            self._progress(
                                "train_adapter_done",
                                step=step_num,
                                adapter_name=adapter_name,
                                metrics=train_metrics[adapter_name],
                            )
                        adapter_dir = self._adapter_output_dir(
                            step=step_num,
                            adapter_name=adapter_name,
                            durable=durable_lora_step,
                        )
                        with self._stage("save_adapter", step=step_num, adapter_name=adapter_name):
                            self._progress("save_adapter_start", step=step_num, adapter_name=adapter_name)
                            self.current_adapter_dirs[adapter_name] = self.trainer.save_adapter(
                                adapter_name=adapter_name,
                                output_dir=str(adapter_dir),
                            )
                            self._write_saved_judge_harness(
                                adapter_name=adapter_name,
                                adapter_dir=self.current_adapter_dirs[adapter_name],
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
                    self._discard_superseded_live_adapters(previous_adapter_dirs)
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
                        "judge_harness": {
                            "id": self.config.judge_harness().harness_id,
                            "fingerprint": harness_fingerprint(
                                self.config.judge_harness().harness_id
                            ),
                        },
                        **extra_record,
                    }
                    distribution_metrics = self._step_distribution_metrics(
                        rewards=step_rewards,
                        grouped_examples=grouped_examples,
                    )
                    record.update(
                        {
                            key: value
                            for key, value in distribution_metrics.items()
                            if key in {"reward_std", "reward_min", "reward_max"}
                        }
                    )
                    record["rollout_metrics"] = {
                        key: value
                        for key, value in distribution_metrics.items()
                        if key not in {"reward_std", "reward_min", "reward_max"}
                    }
                    self._add_temporal_metrics(record)
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
                        f.flush()
                    exact_step = (
                        self.config.optimizer_checkpoint_every > 0
                        and (
                            step_num % self.config.optimizer_checkpoint_every == 0
                            or step_num == self.config.steps
                        )
                    )
                    if exact_step:
                        exact_path = save_exact_resume_checkpoint(
                            root=self.output_dir / "checkpoints" / "exact_resume",
                            step=step_num,
                            run_config=self.config.to_dict(),
                            adapter_dirs=self.current_adapter_dirs,
                            trainer=self.trainer,
                        )
                        self.latest_exact_resume_checkpoint = str(exact_path)
                    self._write_manifest(current_step=step_num)
                    self.observability.log_step(record)
                    shard = self.observability.record_rollouts(
                        record,
                        final_step=step_num == self.config.steps,
                    )
                    if durable_lora_step:
                        lora_root = self.output_dir / "checkpoints" / "lora" / f"step_{step_num:06d}"
                        self.observability.log_artifact(
                            kind="lora-checkpoint",
                            path=lora_root,
                            metadata={"step": step_num, "adapter_names": sorted(train_metrics)},
                        )
                    if exact_step:
                        self.observability.log_artifact(
                            kind="exact-resume-checkpoint",
                            path=self.latest_exact_resume_checkpoint,
                            metadata={"step": step_num},
                        )
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
            observability = getattr(self, "observability", None)
            if observability is not None:
                observability.finish()
