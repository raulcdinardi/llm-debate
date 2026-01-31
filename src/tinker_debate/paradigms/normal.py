from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_debate.debate_types import TrainingDatum

from tinker_debate.tasks.task_types import TaskInstance, TaskSpec


@dataclass(frozen=True)
class NormalRolloutOutput:
    training_data: list[TrainingDatum]
    num_rollouts: int
    rollout_time_s: float
    info_lines: list[str]


@dataclass
class NormalParadigm:
    task: TaskSpec
    tokenizer: Any
    sample_token_prompts: Any
    save_record: Any
    accept_min_reward: float
    accept_require_parse: bool
    replay_dir: str | None = None
    replay_cache: dict[int, list[dict]] | None = None

    async def rollout(
        self,
        *,
        step: int,
        num_rollouts: int,
        num_groups: int | None,
        seed: int | None,
        max_tokens: int,
        temperature: float,
        min_p: float,
        tags: list[str],
    ) -> NormalRolloutOutput:
        step_seed = None if seed is None else seed + step
        if num_groups is None:
            num_groups = num_rollouts
        if num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {num_groups}")
        if num_rollouts % num_groups != 0:
            raise ValueError(f"num_rollouts ({num_rollouts}) must be divisible by num_groups ({num_groups})")

        group_size = num_rollouts // num_groups
        t0 = time.time()
        training_data: list[TrainingDatum] = []
        def maybe_add_training_datum(
            *,
            prompt_tokens: list[int],
            completion_tokens: list[int],
            completion_logprobs: list[float],
            reward_value: float,
            reward_metrics: dict,
            group_id: int,
            instance_id: str | None,
        ) -> None:
            parse_success = None
            if "parse_success" in reward_metrics:
                parse_success = reward_metrics["parse_success"]

            if self.accept_require_parse:
                if parse_success is None:
                    return
                if float(parse_success) != 1.0:
                    return
            if float(reward_value) < float(self.accept_min_reward):
                return
            if len(completion_tokens) == 0:
                return

            advantage = float(reward_value) / len(completion_tokens)
            training_data.append(
                TrainingDatum(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    completion_logprobs=completion_logprobs,
                    completion_advantages=[advantage] * len(completion_tokens),
                    metadata={
                        "mode": "normal",
                        "task": self.task.name,
                        "tags": tags,
                        "group_id": group_id,
                        "instance_id": instance_id,
                        "reward": float(reward_value),
                        "reward_metrics": reward_metrics,
                    },
                )
            )

        if self.replay_dir is not None:
            replay_path = Path(self.replay_dir)
            if self.replay_cache is None:
                prefix = "summary" if self.task.name == "summary" else "baseline"
                files = sorted(replay_path.glob(f"{prefix}_*.json"))
                if not files:
                    raise ValueError(f"No {prefix}_*.json logs found in replay dir: {replay_path}")
                by_step: dict[int, list[dict]] = {}
                for f in files:
                    data = json.loads(f.read_text())
                    if "step" not in data:
                        raise ValueError(f"Missing step in replay log: {f}")
                    by_step.setdefault(int(data["step"]), []).append(data)
                self.replay_cache = by_step

            records = self.replay_cache.get(int(step))
            if not records:
                raise ValueError(f"No replay records for step {step} in {self.replay_dir}")
            if len(records) != num_rollouts:
                raise ValueError(
                    f"Replay records count ({len(records)}) != num_rollouts ({num_rollouts})"
                )
            record_group_ids = [r.get("group_id") for r in records]
            if any(gid is None for gid in record_group_ids):
                raise ValueError("Replay record missing group_id")
            if len(set(record_group_ids)) != num_groups:
                raise ValueError(
                    f"Replay group count ({len(set(record_group_ids))}) != num_groups ({num_groups})"
                )

            for r in records:
                prompt_tokens = r["prompt_tokens"]
                completion_tokens = r["completion_tokens"]
                completion_logprobs = r["completion_logprobs"]
                if len(completion_tokens) != len(completion_logprobs):
                    raise ValueError("Replay completion_tokens/logprobs length mismatch")
                reward_value = float(r["reward"])
                reward_metrics = r.get("metrics", {})
                maybe_add_training_datum(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    completion_logprobs=completion_logprobs,
                    reward_value=reward_value,
                    reward_metrics=reward_metrics,
                    group_id=int(r["group_id"]),
                    instance_id=r.get("instance_id"),
                )

            rollout_time = time.time() - t0
            info_lines = [
                f"Replay mode: {replay_path}",
                f"Rollout time: {rollout_time:.1f}s",
                f"Training data: {len(training_data)} datums (accepted) from {len(records)} rollouts",
            ]
            if group_size > 1:
                info_lines.append(f"Groups: {num_groups} groups of size {group_size}")
            return NormalRolloutOutput(
                training_data=training_data,
                num_rollouts=len(records),
                rollout_time_s=float(rollout_time),
                info_lines=info_lines,
            )

        instances = self.task.sample_instances(n=num_groups, seed=step_seed)

        prompt_tokens_list: list[list[int]] = []
        inst_by_rollout: list[TaskInstance] = []
        group_ids: list[int] = []
        for group_id, inst in enumerate(instances):
            prompt_tokens = self.task.build_r1_prompt_tokens(inst=inst, tokenizer=self.tokenizer)
            for _ in range(group_size):
                prompt_tokens_list.append(prompt_tokens)
                inst_by_rollout.append(inst)
                group_ids.append(group_id)

        results = await self.sample_token_prompts(
            prompt_tokens_list=prompt_tokens_list,
            max_tokens=max_tokens,
            temperature=temperature,
            min_p=min_p,
        )
        rollout_time = time.time() - t0

        for inst, res, prompt_tokens, group_id in zip(inst_by_rollout, results, prompt_tokens_list, group_ids):
            if res is None:
                continue
            _prompt_tokens, completion_tokens, completion_logprobs, _raw = res
            if len(_prompt_tokens) != len(prompt_tokens):
                raise RuntimeError("prompt_tokens mismatch in local sampling result.")

            reward = self.task.compute_reward(inst=inst, completion_tokens=completion_tokens, tokenizer=self.tokenizer)

            record = {
                "tags": tags,
                "step": step,
                "group_id": group_id,
                "instance_id": inst.instance_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_logprobs": completion_logprobs,
                "reward": reward.reward,
                "metrics": reward.metrics,
            }
            self.save_record(record)

            maybe_add_training_datum(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                completion_logprobs=completion_logprobs,
                reward_value=float(reward.reward),
                reward_metrics=reward.metrics,
                group_id=group_id,
                instance_id=inst.instance_id,
            )

        info_lines = [
            f"Rollout time: {rollout_time:.1f}s",
            f"Training data: {len(training_data)} datums (accepted) from {len(prompt_tokens_list)} rollouts",
        ]
        if group_size > 1:
            info_lines.append(f"Groups: {num_groups} groups of size {group_size}")
        return NormalRolloutOutput(
            training_data=training_data,
            num_rollouts=len(prompt_tokens_list),
            rollout_time_s=float(rollout_time),
            info_lines=info_lines,
        )
