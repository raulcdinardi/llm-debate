from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from tinker_debate.datasets import load_cnn_dailymail
from tinker_debate.summary.rewards import RewardConfig

from .task_types import TaskInstance, TaskReward, TaskSpec


def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"


def _im_end() -> str:
    return "<|im_end|>\n"


@dataclass(frozen=True)
class SummaryTask(TaskSpec):
    name: str
    articles: list[dict]
    reward_config: RewardConfig

    @classmethod
    def from_args(cls, *, reward_fn: str, seed: int | None, n_samples: int | None) -> "SummaryTask":
        articles = load_cnn_dailymail(n_samples=n_samples, seed=seed)
        return cls(name="summary", articles=articles, reward_config=RewardConfig.from_string(reward_fn))

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        if seed is not None:
            random.seed(seed)
        sampled = random.sample(self.articles, min(n, len(self.articles)))
        out: list[TaskInstance] = []
        for row in sampled:
            out.append(
                TaskInstance(
                    instance_id=str(row["id"]),
                    payload={"article": row["article"], "highlights": row["highlights"], "id": row["id"]},
                )
            )
        return out

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Expected single token for <|im_end|>, got {len(toks)}")
        return [int(toks[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        article = str(inst.payload["article"])
        max_chars = 2000
        if len(article) > max_chars:
            article = article[:max_chars] + "\n[...TRUNCATED...]"
        return f"Article:\n{article}"

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer: Any) -> list[int]:
        article = inst.payload["article"]
        prompt = (
            "Capture the core aspects of the following article in a well-written summary of 2-3 sentences. "
            "Maintain the essential information while being concise. Output only the summary text and nothing else.\n\n"
            "Article:\n"
            f"{article}"
        )
        full = _im_start("user") + prompt + "\n" + _im_end() + _im_start("assistant")
        return tokenizer.encode(full, add_special_tokens=False)

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward:
        article = inst.payload["article"]
        highlights = inst.payload["highlights"]
        summary_text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
        total, scores = self.reward_config.compute(summary_text, article, highlights)
        metrics = {
            "summary_len": len(summary_text),
            "article_len": len(article),
            "compression_ratio": len(summary_text) / len(article),
            **scores,
        }
        return TaskReward(reward=float(total), metrics=metrics)
