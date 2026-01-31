from __future__ import annotations

import time
from dataclasses import dataclass

from tinker_debate.debate_env import (
    DebateRolloutClient,
    DebateTokenRolloutClient,
    confidence_judge_r1,
    mock_judge_random,
    run_debate_batch_token_only,
)
from tinker_debate.debate_types import DebateConfig, assemble_training_data_grpo, compute_training_stats
from tinker_debate.local_renderers import infer_chat_preamble

from ..driver_context import DriverContext
from ..driver_types import RolloutOutput


@dataclass
class DebateDriver:
    ctx: DriverContext

    def __post_init__(self) -> None:
        args = self.ctx.args

        self.rollout_client = DebateRolloutClient(generate_fn=self._generate_fn)
        self.token_rollout_client = DebateTokenRolloutClient(
            sample_fn=self._sample_tokens_fn,
            decode_fn=lambda toks: self.ctx.client.tokenizer.decode(toks, skip_special_tokens=True),
        )
        chat_preamble = infer_chat_preamble(self.ctx.client.tokenizer)
        self.config = DebateConfig.cheap(chat_preamble=chat_preamble)

        if args.confidence_judge:
            self.judge_fn = confidence_judge_r1
        else:
            self.judge_fn = mock_judge_random if args.mock_judge else None

        self.dataset = None
        if args.dataset is not None:
            if args.dataset == "test":
                from tinker_debate.datasets import load_test_dataset

                self.dataset = load_test_dataset()
            else:
                from tinker_debate.datasets import load_gpqa

                self.dataset = load_gpqa(args.dataset, seed=args.seed)

    async def _generate_fn(self, prompts: list[str], max_tokens: int | None, temperature: float):
        return await self.ctx.client.generate(prompts, max_tokens=max_tokens, temperature=temperature)

    async def _sample_tokens_fn(self, prompt_tokens_list: list[list[int]], max_tokens: int | None, temp: float):
        return await self.ctx.client.sample_token_prompts(
            prompt_tokens_list=prompt_tokens_list,
            max_tokens=max_tokens,
            temperature=temp,
        )

    async def rollout_step(self, *, step: int) -> RolloutOutput:
        args = self.ctx.args
        console = self.ctx.console

        group_size = args.num_rollouts // args.num_groups
        debates_per_question = group_size // 2
        num_debates = args.num_groups * debates_per_question
        console.print(f"[dim]Running {num_debates} debates (batched per-round, token-only)...[/dim]")

        if self.dataset is not None:
            from tinker_debate.datasets import sample_questions

            step_seed = args.seed + step if args.seed is not None else None
            unique_questions = sample_questions(self.dataset, args.num_groups, seed=step_seed)
            batch = [q for q in unique_questions for _ in range(group_size)]
        else:
            batch = [(args.question, args.ground_truth) for _ in range(args.num_rollouts)]

        t0 = time.time()
        debates = await run_debate_batch_token_only(
            batch,
            self.token_rollout_client,
            self.ctx.client.tokenizer,
            self.config,
            self.rollout_client,
            judge_fn=self.judge_fn,
        )
        rollout_time = time.time() - t0

        for r in debates:
            self.ctx.log_fns.save_debate_log(r, self.ctx.log_dir, config=self.config, model_name=self.ctx.client.model_name)

        stats = compute_training_stats(debates)
        info_lines = [
            f"Rollout time: {rollout_time:.1f}s | A:{stats['a_wins']} B:{stats['b_wins']} Invalid:{stats['invalid']}",
        ]

        if args.debate_grpo_reward != "task":
            raise ValueError(f"Unknown --debate-grpo-reward={args.debate_grpo_reward!r}")

        def reward_fn(traj, debate):
            if debate.ground_truth is None:
                raise ValueError("debate.ground_truth is required for debate reward")
            if traj.frozen_solution is None:
                raise ValueError("traj.frozen_solution is None (R1 parse failed)")
            # Task reward: 1 if the frozen R1 solution matches ground truth, else 0.
            return 1.0 if traj.frozen_solution == debate.ground_truth else 0.0

        training_data = assemble_training_data_grpo(debates, reward_fn=reward_fn)
        info_lines.append(
            f"Training data: {len(training_data)} datums (winners, centered reward) from {len(debates)} debates"
        )

        if len(training_data) == 0:
            raise RuntimeError("Training data is empty (nothing to train on).")

        all_advs = [a for d in training_data for a in d.completion_advantages]
        nonzero_advs = [a for a in all_advs if a != 0.0]
        if len(nonzero_advs) == 0:
            raise RuntimeError("All token advantages are 0.0 (no learning signal).")
        info_lines.append(
            f"Advantage stats (nonzero): n={len(nonzero_advs)} min={min(nonzero_advs):.6g} "
            f"max={max(nonzero_advs):.6g} mean={sum(nonzero_advs)/len(nonzero_advs):.6g}"
        )

        return RolloutOutput(
            training_data=training_data,
            rollout_time_s=float(rollout_time),
            num_rollouts=int(args.num_rollouts),
            info_lines=info_lines,
        )
