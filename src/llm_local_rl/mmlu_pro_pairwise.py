from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any

from llm_local_rl.chat_templates import get_chat_adapter
from llm_local_rl.task_types import TaskInstance, TaskReward


@dataclass(frozen=True)
class MMLUProPairwiseDebateTask:
    """Fixed-R1 MMLU-Pro option pairs for debate-only policy training."""

    data_path: str
    name: str = "mmlu_pro_pairwise"
    _rows: tuple[dict[str, Any], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.data_path)
        if not path.is_file():
            raise FileNotFoundError(f"MMLU-Pro pairwise corpus not found: {path}")
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = {"question_id", "question", "correct_answer", "wrong_answer"}
                missing = required - set(row)
                if missing:
                    raise ValueError(f"{path}:{line_num}: missing keys {sorted(missing)}")
                if not all(str(row[key]).strip() for key in required):
                    raise ValueError(f"{path}:{line_num}: required values must be non-empty")
                rows.append(dict(row))
        if not rows:
            raise ValueError(f"MMLU-Pro pairwise corpus is empty: {path}")
        object.__setattr__(self, "_rows", tuple(rows))

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        if n <= 0:
            return []
        rng = random.Random(seed)
        order = list(range(len(self._rows)))
        selected: list[int] = []
        while len(selected) < n:
            rng.shuffle(order)
            selected.extend(order)
        return [
            TaskInstance(
                instance_id=f"mmlu_pro_pair_{self._rows[idx]['question_id']}",
                payload=dict(self._rows[idx]),
            )
            for idx in selected[:n]
        ]

    def expand_group_instances(
        self,
        *,
        inst: TaskInstance,
        group_size: int,
        seed: int | None,
    ) -> list[TaskInstance]:
        rng = random.Random(seed)
        first_correct_is_a = bool(rng.getrandbits(1))
        expanded: list[TaskInstance] = []
        if group_size == 2:
            orderings = (first_correct_is_a,)
        elif group_size >= 4 and group_size % 4 == 0:
            orderings = (first_correct_is_a, not first_correct_is_a) * (group_size // 4)
        else:
            raise ValueError(
                "mmlu_pro_pairwise requires group_size=2 (one independently generated "
                "debate with a seeded answer ordering) or a positive multiple of 4 "
                "(balanced independently generated debates with mirrored answer orderings)"
            )
        for ordering_index, correct_is_a in enumerate(orderings):
            answers = [
                (str(inst.payload["correct_answer"]), True),
                (str(inst.payload["wrong_answer"]), False),
            ]
            if not correct_is_a:
                answers.reverse()
            gold_agent = "A" if correct_is_a else "B"
            expanded.extend(
                TaskInstance(
                    instance_id=inst.instance_id,
                    payload={
                        **inst.payload,
                        "fixed_answer": answer,
                        "is_correct": is_correct,
                        "agent": agent,
                        "gold_agent": gold_agent,
                        "ordering_index": ordering_index,
                    },
                )
                for agent, (answer, is_correct) in zip(("A", "B"), answers, strict=True)
            )
        return expanded

    def r1_context_text(self, *, inst: TaskInstance) -> str:
        return str(inst.payload["question"])

    def build_r1_prompt_tokens(
        self,
        *,
        inst: TaskInstance,
        tokenizer: Any,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        return get_chat_adapter(tokenizer).encode_messages(
            [{"role": "user", "content": self.r1_context_text(inst=inst)}],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def fixed_r1_completion_text(self, *, inst: TaskInstance) -> str:
        return str(inst.payload["fixed_answer"])

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        stop = get_chat_adapter(tokenizer).get_stop_sequences()
        if stop is None or len(stop) != 1:
            raise ValueError("Stop token must be a single token for mmlu_pro_pairwise")
        return [int(stop[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        return str(inst.payload["question"])

    def judge_constitution_text(self, *, inst: TaskInstance) -> str:
        _ = inst
        return "Prefer the answer that is correct."

    def compute_reward(
        self,
        *,
        inst: TaskInstance,
        completion_tokens: list[int],
        tokenizer: Any,
    ) -> TaskReward:
        _ = completion_tokens, tokenizer
        is_correct = bool(inst.payload["is_correct"])
        return TaskReward(
            reward=1.0 if is_correct else 0.0,
            metrics={
                "parse_success": 1.0,
                "fixed_r1": 1.0,
                "is_correct": 1.0 if is_correct else 0.0,
                "gold_agent": str(inst.payload["gold_agent"]),
                "category": inst.payload.get("category"),
                "question_id": str(inst.payload["question_id"]),
            },
        )

    def debate_r2_user_template(self) -> str | None:
        return None

    def debate_r3_user_template(self) -> str | None:
        return None
