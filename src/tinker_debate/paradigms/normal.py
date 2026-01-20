from __future__ import annotations

import time
from dataclasses import dataclass
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

    async def rollout(
        self,
        *,
        step: int,
        num_rollouts: int,
        num_groups: int | None,
        seed: int | None,
        max_tokens: int,
        temperature: float,
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

        t0 = time.time()
        results = await self.sample_token_prompts(
            prompt_tokens_list=prompt_tokens_list,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        rollout_time = time.time() - t0

        training_data: list[TrainingDatum] = []
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

            if float(reward.reward) < float(self.accept_min_reward):
                continue
            if len(completion_tokens) == 0:
                continue

            advantage = float(reward.reward) / len(completion_tokens)
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
                        "instance_id": inst.instance_id,
                        "reward": float(reward.reward),
                        "reward_metrics": reward.metrics,
                    },
                )
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
