from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from tinker_debate.datasets import load_gpqa, load_test_dataset

from .task_types import TaskInstance, TaskReward, TaskSpec


def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"


def _im_end() -> str:
    return "<|im_end|>\n"


def _extract_solution(text: str) -> str | None:
    match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


@dataclass(frozen=True)
class QATask(TaskSpec):
    name: str
    dataset: list[dict]

    @classmethod
    def from_args(cls, *, dataset_name: str | None, seed: int | None) -> "QATask":
        if dataset_name is None or dataset_name == "test":
            ds = load_test_dataset()
            return cls(name="qa_test", dataset=ds)
        if dataset_name in ("gpqa_diamond", "gpqa_extended", "gpqa_main"):
            ds = load_gpqa(dataset_name, seed=seed)
            return cls(name=f"qa_{dataset_name}", dataset=ds)
        raise ValueError(f"Unsupported QA dataset: {dataset_name!r}")

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        if seed is not None:
            random.seed(seed)
        sampled = random.sample(self.dataset, min(n, len(self.dataset)))
        out: list[TaskInstance] = []
        for i, row in enumerate(sampled):
            out.append(
                TaskInstance(
                    instance_id=f"qa_{i}",
                    payload={"question": row["question"], "ground_truth": row["ground_truth"]},
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
        prompt = (
            f"{q}\n\n"
            "Provide your final answer as tags:\n"
            "<SOLUTION>...</SOLUTION>\n\n"
            "Output ONLY the <SOLUTION> tag."
        )
        full = _im_start("user") + prompt + "\n" + _im_end() + _im_start("assistant")
        return tokenizer.encode(full, add_special_tokens=False)

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward:
        gt = inst.payload["ground_truth"]
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        sol = _extract_solution(text)
        if sol is None:
            return TaskReward(
                reward=0.0,
                metrics={"parse_success": 0.0, "solution": None, "ground_truth": gt, "correct": 0.0},
            )
        correct = float(sol.strip() == str(gt).strip())
        return TaskReward(
            reward=correct,
            metrics={"parse_success": 1.0, "solution": sol, "ground_truth": gt, "correct": correct},
        )
