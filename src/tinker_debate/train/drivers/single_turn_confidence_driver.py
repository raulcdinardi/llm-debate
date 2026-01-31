from __future__ import annotations

import time
from dataclasses import dataclass

import tinker
from tinker_cookbook import renderers

from tinker_debate.confidence.confidence_env import ConfidenceDataset, load_questions
from tinker_debate.debate_types import TrainingDatum
from tinker_debate.local_renderers import select_instruct_renderer_name

from ..driver_context import DriverContext
from ..driver_types import RolloutOutput


@dataclass
class ConfidenceDriver:
    ctx: DriverContext

    def __post_init__(self) -> None:
        args = self.ctx.args
        questions = load_questions()
        if len(questions) == 0:
            raise ValueError("confidence/questions.json is empty")

        renderer_name = select_instruct_renderer_name(self.ctx.client.tokenizer)
        self.renderer = renderers.get_renderer(renderer_name, self.ctx.client.tokenizer)
        self.dataset = ConfidenceDataset(
            questions,
            batch_size=args.num_rollouts,
            group_size=1,
            renderer=self.renderer,
        )

    async def rollout_step(self, *, step: int) -> RolloutOutput:
        args = self.ctx.args
        console = self.ctx.console

        console.print(f"[dim]Running {args.num_rollouts} rollouts (single-turn confidence env)...[/dim]")

        batch_idx = (step - 1) % len(self.dataset)
        env_group_builders = self.dataset.get_batch(batch_idx)
        num_rollouts = len(env_group_builders)

        training_data: list[TrainingDatum] = []
        t0 = time.time()

        for builder in env_group_builders:
            envs = await builder.make_envs()
            if len(envs) != 1:
                raise ValueError(f"Expected group_size=1, got {len(envs)}")
            env = envs[0]
            ob, stop = await env.initial_observation()
        sampling_params = tinker.SamplingParams(
            temperature=float(args.temperature),
            stop=stop,
            max_tokens=int(args.max_tokens),
        )
            resp = await self.ctx.client.sampling_client.sample_async(
                prompt=ob,
                num_samples=1,
                sampling_params=sampling_params,
            )
            seq = resp.sequences[0]
            if seq.logprobs is None:
                raise RuntimeError("Sampling did not return completion logprobs (seq.logprobs is None)")
            completion_tokens = list(seq.tokens)
            completion_logprobs = list(seq.logprobs)

            step_result = await env.step(completion_tokens)

            record = {
                "tags": builder.logging_tags(),
                "prompt_tokens": ob.to_ints(),
                "completion_tokens": completion_tokens,
                "completion_logprobs": completion_logprobs,
                "reward": step_result.reward,
                "metrics": step_result.metrics,
            }
            self.ctx.log_fns.save_baseline_log(record=record, log_dir=self.ctx.log_dir, model_name=self.ctx.client.model_name)

            if args.accept_require_parse and float(step_result.metrics["parse_success"]) != 1.0:
                continue
            if float(step_result.reward) < args.accept_min_reward:
                continue
            if len(completion_tokens) == 0:
                continue

            advantage = float(step_result.reward) / len(completion_tokens)
            training_data.append(
                TrainingDatum(
                    prompt_tokens=ob.to_ints(),
                    completion_tokens=completion_tokens,
                    completion_logprobs=completion_logprobs,
                    completion_advantages=[advantage] * len(completion_tokens),
                    metadata={
                        "mode": "single_turn",
                        "env": "confidence",
                        "tags": builder.logging_tags(),
                        "parse_success": float(step_result.metrics["parse_success"]),
                        "reward": float(step_result.reward),
                    },
                )
            )

        rollout_time = time.time() - t0
        info_lines = [
            f"Rollout time: {rollout_time:.1f}s",
            f"Training data: {len(training_data)} datums (accepted) from {num_rollouts} rollouts",
        ]

        return RolloutOutput(
            training_data=training_data,
            rollout_time_s=float(rollout_time),
            num_rollouts=int(num_rollouts),
            info_lines=info_lines,
        )
