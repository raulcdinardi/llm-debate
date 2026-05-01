from __future__ import annotations

from dataclasses import dataclass

from llm_local_rl.chat_templates import get_chat_adapter
from llm_local_rl.ht_sequence_format import parse_ht_sequence
from llm_local_rl.prompts import format_prompt, load_prompt
from llm_local_rl.task_types import TaskInstance, TaskReward

_COIN_SYSTEM = load_prompt("tasks/coin_system.md")
_COIN_USER = load_prompt("tasks/coin_user.md")
_HT_USER = load_prompt("tasks/ht_sequence_user.md")


@dataclass(frozen=True)
class CoinDebateTask:
    name: str = "coin"
    target_color: str = "Blue"

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        _ = seed
        return [TaskInstance(instance_id=f"coin_{idx}", payload={}) for idx in range(n)]

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer, enable_thinking: bool | None = None) -> list[int]:
        _ = inst
        messages = []
        if _COIN_SYSTEM:
            messages.append({"role": "system", "content": _COIN_SYSTEM})
        messages.append({"role": "user", "content": _COIN_USER})
        return get_chat_adapter(tokenizer).encode_messages(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def stop_token_ids(self, *, tokenizer) -> list[int]:
        stop = get_chat_adapter(tokenizer).get_stop_sequences()
        if stop is None or len(stop) != 1:
            raise ValueError("Stop token must be a single token for coin task.")
        return [int(stop[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        _ = inst
        return _COIN_USER

    def judge_constitution_text(self, *, inst: TaskInstance) -> str:
        _ = inst
        return (
            "Choose the answer that better follows the task and required format. "
            "The answer must be exactly one word: Red or Blue. Output only A or B."
        )

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer) -> TaskReward:
        _ = inst
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
        first_word = text.split()[0] if text else ""
        normalized = first_word.capitalize()
        parse_success = 1.0 if normalized in ("Red", "Blue") else 0.0
        reward = 1.0 if normalized == self.target_color else 0.0
        red_tok = tokenizer.encode("Red", add_special_tokens=False)
        blue_tok = tokenizer.encode("Blue", add_special_tokens=False)
        if len(red_tok) != 1 or len(blue_tok) != 1:
            raise ValueError(f"Expected single-token Red/Blue, got {red_tok} / {blue_tok}")
        return TaskReward(
            reward=reward,
            metrics={
                "parse_success": parse_success,
                "choice": normalized if parse_success else None,
                "choice_token_id": int(completion_tokens[0]) if completion_tokens else None,
                "candidate_token_ids": {"Red": int(red_tok[0]), "Blue": int(blue_tok[0])},
                "target": self.target_color,
                "full_text": text,
            },
        )

    def debate_r2_user_template(self) -> str | None:
        return None

    def debate_r3_user_template(self) -> str | None:
        return None


@dataclass(frozen=True)
class HTSequenceDebateTask:
    name: str = "ht_sequence"
    sequence_len: int = 8
    reward_mode: str = "num_h"
    strict_format: bool = False

    def __post_init__(self) -> None:
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be > 0")
        if self.reward_mode not in {"num_h", "num_transitions"}:
            raise ValueError(f"Unsupported reward_mode={self.reward_mode!r}")

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        _ = seed
        return [TaskInstance(instance_id=f"ht_{idx}", payload={"sequence_len": self.sequence_len}) for idx in range(n)]

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer, enable_thinking: bool | None = None) -> list[int]:
        prompt = format_prompt(_HT_USER, sequence_len=str(int(inst.payload["sequence_len"])))
        messages = [{"role": "user", "content": prompt}]
        return get_chat_adapter(tokenizer).encode_messages(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def stop_token_ids(self, *, tokenizer) -> list[int]:
        stop = get_chat_adapter(tokenizer).get_stop_sequences()
        if stop is None or len(stop) != 1:
            raise ValueError("Stop token must be a single token for ht_sequence task.")
        return [int(stop[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        return format_prompt(_HT_USER, sequence_len=str(int(inst.payload["sequence_len"])))

    def judge_constitution_text(self, *, inst: TaskInstance) -> str:
        seq_len = int(inst.payload["sequence_len"])
        return (
            "Choose the answer that better adheres to the task constitution and required answer format. "
            f"The answer must encode exactly {seq_len} H/T symbols in the required comma-separated format. "
            "Output only A or B."
        )

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer) -> TaskReward:
        seq_len = int(inst.payload["sequence_len"])
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        parsed_seq, parse_success = parse_ht_sequence(
            text=text,
            target_len=seq_len,
            strict_format=self.strict_format,
        )
        num_h = parsed_seq.count("H")
        num_t = parsed_seq.count("T")
        num_transitions = sum(1 for idx in range(1, len(parsed_seq)) if parsed_seq[idx] != parsed_seq[idx - 1])
        if self.reward_mode == "num_h":
            reward_value = float(num_h)
        else:
            reward_value = float(num_transitions)
        return TaskReward(
            reward=reward_value,
            metrics={
                "parse_success": 1.0 if parse_success else 0.0,
                "sequence_len_target": seq_len,
                "sequence_len_parsed": len(parsed_seq),
                "parsed_sequence": parsed_seq,
                "num_h": int(num_h),
                "num_t": int(num_t),
                "num_transitions": int(num_transitions),
                "reward_mode": self.reward_mode,
                "full_text": text,
            },
        )

    def debate_r2_user_template(self) -> str | None:
        return None

    def debate_r3_user_template(self) -> str | None:
        return None
