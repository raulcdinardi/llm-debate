"""
Confidence calibration environment.

Model outputs answer + confidence → reward = confidence score.
Ground truth logged = the actual probability (for calibration analysis).
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree


QUESTIONS_PATH = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def parse_confidence(text: str) -> float | None:
    # Preferred: tag format (mirrors debate R1 output schema)
    tag = re.search(r"<CONFIDENCE>(.*?)</CONFIDENCE>", text, re.DOTALL | re.IGNORECASE)
    if tag:
        return max(0.0, min(1.0, float(tag.group(1).strip())))

    # Back-compat: "Confidence: 0.73"
    line = re.search(r"[Cc]onfidence[:\s]+([0-9.]+)", text)
    if line:
        return max(0.0, min(1.0, float(line.group(1))))

    return None


@dataclass
class ConfidenceEnv(Env):
    question: str
    ground_truth_answer: str
    ground_truth_prob: float
    renderer: renderers.Renderer

    async def initial_observation(self):
        prompt = f"""{self.question}

Provide your answer and your confidence as tags:
<SOLUTION>...</SOLUTION>
<CONFIDENCE>0.0 to 1.0</CONFIDENCE>

Output ONLY these tags."""
        convo = [{"role": "user", "content": prompt}]
        return self.renderer.build_generation_prompt(convo), self.renderer.get_stop_sequences()

    async def step(self, action):
        message, _ = self.renderer.parse_response(action)
        text = renderers.ensure_text(message["content"])

        confidence = parse_confidence(text)
        reward = confidence if confidence is not None else 0.0

        logtree.log_text(f"Q: {self.question}")
        logtree.log_text(f"Response: {text}")
        logtree.log_text(f"Model confidence: {confidence}")
        logtree.log_text(f"Ground truth: {self.ground_truth_answer} (p={self.ground_truth_prob})")

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={
                "confidence": reward,
                "ground_truth_prob": self.ground_truth_prob,
                "parse_success": float(confidence is not None),
            },
        )


@dataclass(frozen=True)
class ConfidenceGroupBuilder(EnvGroupBuilder):
    question: str
    ground_truth_answer: str
    ground_truth_prob: float
    renderer: renderers.Renderer
    group_size: int = 1

    async def make_envs(self) -> Sequence[Env]:
        return [
            ConfidenceEnv(self.question, self.ground_truth_answer, self.ground_truth_prob, self.renderer)
            for _ in range(self.group_size)
        ]


class ConfidenceDataset(RLDataset):
    def __init__(self, questions: list[dict], batch_size: int, group_size: int, renderer: renderers.Renderer):
        self.questions = questions
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def get_batch(self, idx: int) -> Sequence[EnvGroupBuilder]:
        start = idx * self.batch_size
        return [
            ConfidenceGroupBuilder(q["question"], q["answer"], q["probability"], self.renderer, self.group_size)
            for q in self.questions[start : start + self.batch_size]
        ]

    def __len__(self):
        return max(1, len(self.questions) // self.batch_size)


@chz.chz
class ConfidenceDatasetBuilder(RLDatasetBuilder):
    model_name: str
    renderer_name: str
    batch_size: int = 8
    group_size: int = 4

    async def __call__(self):
        renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        questions = load_questions()
        train = ConfidenceDataset(questions, self.batch_size, self.group_size, renderer)
        return train, None
