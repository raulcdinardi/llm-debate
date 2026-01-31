from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_debate.prompts import format_prompt, load_prompt
from tinker_debate.local_renderers import infer_chat_preamble

from .task_types import TaskInstance, TaskReward, TaskSpec


QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "tinker_debate" / "confidence" / "questions.json"


def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"


def _im_end() -> str:
    return "<|im_end|>\n"


def _parse_confidence(text: str) -> float | None:
    tag = re.search(r"<CONFIDENCE>(.*?)</CONFIDENCE>", text, re.DOTALL | re.IGNORECASE)
    if not tag:
        return None
    value_text = tag.group(1).strip()
    if not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", value_text):
        return None
    value = float(value_text)
    if not (0.0 <= value <= 1.0):
        return None
    return value


@dataclass(frozen=True)
class ConfidenceTask(TaskSpec):
    name: str
    questions: list[dict]

    @classmethod
    def from_file(cls) -> "ConfidenceTask":
        with open(QUESTIONS_PATH) as f:
            questions = json.load(f)
        return cls(name="confidence", questions=questions)

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        if seed is not None:
            random.seed(seed)
        sampled = random.sample(self.questions, min(n, len(self.questions)))
        out: list[TaskInstance] = []
        for i, row in enumerate(sampled):
            out.append(
                TaskInstance(
                    instance_id=f"conf_{i}",
                    payload={
                        "question": row["question"],
                        "ground_truth_prob": row["probability"],
                        "ground_truth_answer": row["answer"],
                    },
                )
            )
        return out

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Expected single token for <|im_end|>, got {len(toks)}")
        return [int(toks[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        return str(inst.payload["question"])

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer: Any) -> list[int]:
        q = inst.payload["question"]
        template = load_prompt("tasks/confidence_user.md")
        prompt = format_prompt(template, question=str(q))
        preamble = infer_chat_preamble(tokenizer)
        full = preamble + _im_start("user") + prompt + "\n" + _im_end() + _im_start("assistant")
        return tokenizer.encode(full, add_special_tokens=False)

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward:
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        conf = _parse_confidence(text)
        if conf is None:
            return TaskReward(
                reward=0.0,
                metrics={
                    "confidence": None,
                    "ground_truth_prob": float(inst.payload["ground_truth_prob"]),
                    "parse_success": 0.0,
                },
            )
        return TaskReward(
            reward=float(conf),
            metrics={
                "confidence": float(conf),
                "ground_truth_prob": float(inst.payload["ground_truth_prob"]),
                "parse_success": 1.0,
            },
        )
