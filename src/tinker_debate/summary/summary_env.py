"""
Summarization RL environment.

Single-turn: model receives article, produces summary, gets reward.
Follows tinker-cookbook Env/EnvGroupBuilder/RLDataset pattern.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, Metrics, RLDataset, StepResult

from .rewards import RewardConfig, compression_reward


@dataclass
class SummaryEnv(Env):
    """Single-turn summarization environment."""
    
    article: str
    highlights: str  # reference summary for metrics
    article_id: str
    reward_fn: Callable[[str, str, str], float]
    renderer: renderers.Renderer
    
    # Set by step() for metrics access
    _generated_summary: str | None = None

    async def initial_observation(self):
        prompt = f"""Capture the core aspects of the following article in a well-written summary of 2-3 sentences. Maintain the essential information while being concise. Output only the summary text and nothing else.

Article:
{self.article}"""
        convo = [{"role": "user", "content": prompt}]
        return self.renderer.build_generation_prompt(convo), self.renderer.get_stop_sequences()

    async def step(self, action):
        message, _ = self.renderer.parse_response(action)
        summary = renderers.ensure_text(message["content"])
        self._generated_summary = summary
        
        reward = self.reward_fn(summary, self.article, self.highlights)
        
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={
                "summary_len": len(summary),
                "article_len": len(self.article),
                "compression_ratio": len(summary) / len(self.article) if len(self.article) > 0 else 0.0,
            },
        )


@dataclass(frozen=True)
class SummaryGroupBuilder(EnvGroupBuilder):
    """Builds a group of summary environments for the same article."""
    
    article: str
    highlights: str
    article_id: str
    renderer: renderers.Renderer
    reward_fn: Callable[[str, str, str], float]
    group_size: int = 1

    async def make_envs(self) -> Sequence[Env]:
        return [
            SummaryEnv(
                article=self.article,
                highlights=self.highlights,
                article_id=self.article_id,
                reward_fn=self.reward_fn,
                renderer=self.renderer,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["summary", "cnn_dailymail"]


class SummaryDataset(RLDataset):
    """Dataset that produces batches of SummaryGroupBuilders."""
    
    def __init__(
        self,
        articles: list[dict],  # {"article": str, "highlights": str, "id": str}
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        reward_fn: Callable[[str, str, str], float] = compression_reward,
    ):
        self.articles = articles
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.reward_fn = reward_fn

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.articles)
        batch_articles = []
        for i in range(self.batch_size):
            batch_articles.append(self.articles[(start + i) % len(self.articles)])
        
        return [
            SummaryGroupBuilder(
                article=a["article"],
                highlights=a["highlights"],
                article_id=a["id"],
                renderer=self.renderer,
                reward_fn=self.reward_fn,
                group_size=self.group_size,
            )
            for a in batch_articles
        ]

    def __len__(self) -> int:
        return max(1, len(self.articles) // self.batch_size)
