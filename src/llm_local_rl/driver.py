from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from contextlib import nullcontext

from transformers import AutoTokenizer

from llm_local_rl.base_model_judge import RemoteBaseJudgeConfig, build_remote_base_judge
from llm_local_rl.config import CheckpointManifest, TrainRunConfig
from llm_local_rl.debate_parity import (
    assemble_split_train_examples,
    assemble_training_data_by_mode,
    training_datum_to_train_example,
)
from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
from llm_local_rl.local_renderers import infer_chat_preamble
from llm_local_rl.masking import make_train_example
from llm_local_rl.metrics import mean_numeric_metrics
from llm_local_rl.model_io_trace import configure_model_io_tracing, trace_context
from llm_local_rl.registry import build_debate_task, build_environment, build_episode_builder
from llm_local_rl.resource_monitor import ResourceMonitor
from llm_local_rl.trainer import MultiAdapterTrainer, TrainerConfig
from llm_local_rl.debate_parity import DebateConfig
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
        self.env = build_environment(config) if config.rollout.mode == "single_turn" else None
        self.episode_builder = build_episode_builder(config) if config.rollout.mode == "single_turn" else None
        self.debate_task = build_debate_task(config) if config.rollout.mode == "debate" else None
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)
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
            self.trainer = self._make_fresh_trainer()
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
        driver.env = build_environment(config) if config.rollout.mode == "single_turn" else None
        driver.episode_builder = build_episode_builder(config) if config.rollout.mode == "single_turn" else None
        driver.debate_task = build_debate_task(config) if config.rollout.mode == "debate" else None
        driver.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)
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
        return ("solution", "debate")

    def _debate_config(self) -> DebateConfig:
        return DebateConfig(
            num_rounds=self.config.debate_rounds,
            enable_thinking=self._enable_thinking(),
            max_tokens_per_turn=self.config.rollout.max_tokens,
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

    def _trainer_config(self, *, device: str) -> TrainerConfig:
        return TrainerConfig(
            base_model_path=self.config.model_path,
            adapter_names=self._adapter_names(),
            learning_rate=self.config.learning_rate,
            lora_rank=self.config.lora_rank,
            target_modules=self.config.target_modules,
            device=device,
            ppo_clip_epsilon=self.config.ppo_clip_epsilon,
            train_minibatch_size=self.config.train_minibatch_size,
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
            metadata={"model_path": self.config.model_path},
        )

    def _make_fresh_trainer(self) -> MultiAdapterTrainer:
        return MultiAdapterTrainer(config=self._trainer_config(device="cuda"))

    def _make_trainer_from_current_adapters(self) -> MultiAdapterTrainer:
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

    def _make_sampler(self) -> VllmSampler:
        return VllmSampler(
            runtime=VllmRuntimeConfig(
                model_path=self.config.model_path,
                gpu_memory_utilization=self.config.sampler_gpu_memory_utilization,
                max_model_len=self.config.sampler_max_model_len,
                enforce_eager=self.config.sampler_enforce_eager,
                max_lora_rank=self.config.sampler_max_lora_rank,
                max_loras=self.config.sampler_max_loras,
            ),
            adapter_paths=dict(self.current_adapter_dirs),
        )

    def _ensure_sampler(self) -> None:
        if self.sampler is None:
            with self._stage("init_sampler"):
                self.sampler = self._make_sampler()

    def _should_teardown_sampler_before_training(self) -> bool:
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

    def _per_sample_advantages(self, *, rewards: list[float]) -> list[float]:
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

    def _group_examples(self, *, samples: list) -> dict[str, list]:
        advantages = self._per_sample_advantages(rewards=[float(sample.reward) for sample in samples])
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
                prompt_format=self.config.debate_prompt_format,
                base_r2_prefill=self.config.base_r2_prefill,
                base_r3_prefill=self.config.base_r3_prefill,
            ),
            adapter_layout=self.config.adapter_layout,
            judge_fn=judge_fn,
        )

    def _run_single_turn_samples(self, *, step_idx: int) -> list[EpisodeSample]:
        step_seed = None if self.config.rollout.seed is None else self.config.rollout.seed + step_idx
        instances = self.env.sample_instances(
            n=self.config.rollout.num_samples,
            seed=step_seed,
        )
        adapter_name = self.episode_builder.adapter_name
        rollout_batch_size = (
            self.config.rollout.rollout_batch_size
            if self.config.rollout.rollout_batch_size > 0
            else len(instances)
        )
        samples: list[EpisodeSample] = []
        for start_idx in range(0, len(instances), rollout_batch_size):
            chunk = instances[start_idx : start_idx + rollout_batch_size]
            requests = []
            for sample_idx, instance in enumerate(chunk, start=start_idx):
                prompt_builder = getattr(self.env, "build_initial_prompt_token_ids", None)
                if callable(prompt_builder):
                    prompt_token_ids = prompt_builder(
                        instance=instance,
                        tokenizer=self.tokenizer,
                        enable_thinking=self._enable_thinking(),
                    )
                else:
                    prompt_text = self.env.build_initial_prompt(instance=instance)
                    prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                request_seed = None
                if self.config.rollout.request_seed_mode == "per_request" and self.config.rollout.seed is not None:
                    request_seed = self.config.rollout.seed + step_idx * self.config.rollout.num_samples + sample_idx
                elif self.config.rollout.request_seed_mode != "none":
                    raise ValueError(f"Unsupported request_seed_mode={self.config.rollout.request_seed_mode!r}")
                requests.append(
                    SamplingRequest(
                        adapter_name=adapter_name,
                        prompt_token_ids=prompt_token_ids,
                        stop_token_ids=self.env.stop_token_ids(tokenizer=self.tokenizer),
                        max_tokens=self.config.rollout.max_tokens,
                        temperature=self.config.rollout.temperature,
                        seed=request_seed,
                        min_p=self.config.rollout.min_p,
                    )
                )
            results = self.sampler.sample_many(requests)
            for instance, result in zip(chunk, results, strict=True):
                reward, reward_metrics = self.env.score_completion(
                    instance=instance,
                    tokenizer=self.tokenizer,
                    completion_token_ids=result.completion_token_ids,
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
                                metadata={"text": result.text},
                            )
                        ],
                        reward=reward,
                        reward_metrics=reward_metrics,
                    )
                )
        return samples

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
            task_reward_fn=task_reward_fn,
            pointwise_reward_map=pointwise_reward_map,
        )
        return grouped, {
            "source_exact_shared_equivalent": False,
            "reason": "split_layout_per_round_projection",
            "judge_adapter_policy": self.config.debate_judge_adapter,
            "num_debates": len(debates),
        }

    def run(self, *, max_steps: int | None = None) -> dict:
        try:
            self.config.write_json(self.output_dir / "run_config.json")
            end_step = self.config.steps if max_steps is None else min(self.config.steps, self.start_step + max_steps)
            for step_idx in range(self.start_step, end_step):
                step_num = step_idx + 1
                with self._stage("step", step=step_num):
                    self._progress("step_start", step=step_num, total_steps=self.config.steps)
                    self._ensure_sampler()
                    self.sampler.set_adapter_paths(adapter_paths=self.current_adapter_dirs)
                    with self._stage("sampler_wake", step=step_num):
                        self._progress("sampler_wake_start", step=step_num)
                        self.sampler.wake_up()
                        self._progress("sampler_wake_done", step=step_num)
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
                        with self._stage("rollout_debate", step=step_num), trace_context(
                            step=step_num, rollout_mode="debate"
                        ):
                            self._progress(
                                "rollout_start",
                                step=step_num,
                                mode="debate",
                                num_groups=self.config.rollout.num_groups,
                                group_size=self.config.rollout.group_size,
                            )
                            debates = self._debate_runtime().rollout(step_seed=step_seed).debates
                            self._progress("rollout_done", step=step_num, mode="debate", num_debates=len(debates))
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
                        record_samples = [
                            {
                                "question": debate.question,
                                "verdict": debate.verdict,
                                "judge_reasoning": debate.judge_reasoning,
                                "trajectory_a": debate.trajectory_a.metrics,
                                "trajectory_b": debate.trajectory_b.metrics,
                            }
                            for debate in debates
                        ]
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
                    for adapter_name, batch in grouped_examples.items():
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
                    if "mean_used_secret" in mean_reward_metrics:
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
                    )
                    print(json.dumps(record, indent=2))

            summary = self._write_summary()
            self._progress("run_done", num_steps_completed=summary["num_steps_completed"])
            return summary
        finally:
            self.resource_monitor.stop()
