"""DEPRECATED: This driver path is unused.
Use src/tinker_debate/train/orthogonal_driver.py with TaskSpec tasks instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from tinker_debate.debate_types import TrainingDatum
from tinker_debate.datasets import load_cnn_dailymail
from tinker_debate.local_renderers import infer_chat_preamble, select_instruct_renderer_name
from tinker_debate.summary.rewards import RewardConfig

from ..driver_context import DriverContext
from ..driver_types import RolloutOutput


@dataclass
class SummaryDriver:
    ctx: DriverContext
    local_mode: bool
    num_groups: int = 1
    group_size: int = 1

    def __post_init__(self) -> None:
        args = self.ctx.args

        if int(args.num_groups) <= 0:
            raise ValueError(f"num_groups must be > 0, got {args.num_groups}")
        if int(args.num_rollouts) % int(args.num_groups) != 0:
            raise ValueError(
                f"num_rollouts ({args.num_rollouts}) must be divisible by num_groups ({args.num_groups})"
            )
        self.num_groups = int(args.num_groups)
        self.group_size = int(args.num_rollouts) // int(args.num_groups)

        n_samples = 1000 if args.dataset == "cnn_dailymail" else None
        self.articles = load_cnn_dailymail(n_samples=n_samples, seed=args.seed)
        self.reward_config = RewardConfig.from_string(args.reward_fn)

        if not self.local_mode:
            from tinker_debate.summary import SummaryDataset
            from tinker_cookbook import renderers

            renderer_name = select_instruct_renderer_name(self.ctx.client.tokenizer)
            self.renderer = renderers.get_renderer(renderer_name, self.ctx.client.tokenizer)

            def reward_fn(summary: str, article: str, highlights: str) -> float:
                total, _scores = self.reward_config.compute(summary, article, highlights)
                return float(total)

            self.dataset = SummaryDataset(
                articles=self.articles,
                batch_size=self.num_groups,
                group_size=self.group_size,
                renderer=self.renderer,
                reward_fn=reward_fn,
            )
        else:
            self.dataset = None
            self.renderer = None

    def _build_prompt_tokens(self, *, article: str) -> list[int]:
        prompt = (
            "Capture the core aspects of the following article in a well-written summary of 2-3 sentences. "
            "Maintain the essential information while being concise. Output only the summary text and nothing else.\n\n"
            "Article:\n"
            f"{article}"
        )
        preamble = infer_chat_preamble(self.ctx.client.tokenizer)
        full = preamble + "<|im_start|>user\n" + prompt + "\n<|im_end|>\n<|im_start|>assistant"
        return self.ctx.client.tokenizer.encode(full, add_special_tokens=False)

    async def rollout_step(self, *, step: int) -> RolloutOutput:
        args = self.ctx.args
        console = self.ctx.console

        console.print(
            f"[dim]Running {args.num_rollouts} rollouts (single-turn summary env, reward={args.reward_fn})...[/dim]"
        )

        training_data: list[TrainingDatum] = []
        t0 = time.time()

        if self.local_mode:
            start = ((step - 1) * self.num_groups) % len(self.articles)
            batch_articles = [self.articles[(start + i) % len(self.articles)] for i in range(self.num_groups)]

            prompt_tokens_list: list[list[int]] = []
            inst_by_rollout: list[dict] = []
            group_ids: list[int] = []
            for group_id, article in enumerate(batch_articles):
                prompt_tokens = self._build_prompt_tokens(article=article["article"])
                for _ in range(self.group_size):
                    prompt_tokens_list.append(prompt_tokens)
                    inst_by_rollout.append(article)
                    group_ids.append(group_id)

            results = await self.ctx.client.sample_token_prompts(
                prompt_tokens_list=prompt_tokens_list,
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
            )

            for a, res, group_id in zip(inst_by_rollout, results, group_ids):
                if res is None:
                    continue
                prompt_tokens, completion_tokens, completion_logprobs, _raw = res

                summary_text = self.ctx.client.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
                reward, reward_metrics = self.reward_config.compute(summary_text, a["article"], a["highlights"])

                metrics = {
                    "summary_len": len(summary_text),
                    "article_len": len(a["article"]),
                    "compression_ratio": len(summary_text) / len(a["article"]),
                    **reward_metrics,
                }

                record = {
                    "tags": ["summary", "cnn_dailymail"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "completion_logprobs": completion_logprobs,
                    "reward": reward,
                    "metrics": metrics,
                    "article_id": a["id"],
                    "group_id": group_id,
                }
                self.ctx.log_fns.save_summary_log(record=record, log_dir=self.ctx.log_dir, model_name=self.ctx.client.model_name)

                if float(reward) < args.accept_min_reward:
                    continue
                if len(completion_tokens) == 0:
                    continue

                advantage = float(reward) / len(completion_tokens)
                training_data.append(
                    TrainingDatum(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        completion_logprobs=completion_logprobs,
                        completion_advantages=[advantage] * len(completion_tokens),
                        metadata={
                            "mode": "single_turn",
                            "env": "summary",
                            "tags": ["summary", "cnn_dailymail"],
                            "reward": float(reward),
                            "reward_metrics": reward_metrics,
                            "article_id": a["id"],
                            "group_id": group_id,
                        },
                    )
                )

            num_rollouts = args.num_rollouts
        else:
            import tinker

            assert self.dataset is not None
            batch_idx = (step - 1) % len(self.dataset)
            env_group_builders = self.dataset.get_batch(batch_idx)
            num_rollouts = len(env_group_builders) * self.group_size

            for group_id, builder in enumerate(env_group_builders):
                envs = await builder.make_envs()
                for env in envs:
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
                        "group_id": group_id,
                    }
                    self.ctx.log_fns.save_summary_log(record=record, log_dir=self.ctx.log_dir, model_name=self.ctx.client.model_name)

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
                                "env": "summary",
                                "tags": builder.logging_tags(),
                                "reward": float(step_result.reward),
                                "compression_ratio": step_result.metrics.get("compression_ratio", 0.0),
                                "group_id": group_id,
                            },
                        )
                    )

        rollout_time = time.time() - t0
        info_lines = [
            f"Loaded {len(self.articles)} articles from CNN/DailyMail",
            f"Rollout time: {rollout_time:.1f}s",
            f"Training data: {len(training_data)} datums (accepted) from {num_rollouts} rollouts",
        ]
        if self.group_size > 1:
            info_lines.append(f"Groups: {self.num_groups} groups of size {self.group_size}")

        return RolloutOutput(
            training_data=training_data,
            rollout_time_s=float(rollout_time),
            num_rollouts=int(num_rollouts),
            info_lines=info_lines,
        )
