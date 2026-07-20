from __future__ import annotations

from dataclasses import dataclass
import random

from llm_local_rl.chat_templates import get_chat_adapter
from llm_local_rl.countdown_code import (
    countdown_messages,
    sample_countdown_instances,
    score_countdown_completion,
)
from llm_local_rl.qwen35_base_format import (
    base_text_prompt,
    countdown_user_text,
    encode_base_text,
)
from llm_local_rl.ht_sequence_format import parse_ht_sequence
from llm_local_rl.interfaces import Tokenizer
from llm_local_rl.prompts import format_prompt, load_prompt
from llm_local_rl.quality_data import load_quality_questions, sample_quality_questions
from llm_local_rl.short_story_format import contains_word, extract_solution
from llm_local_rl.task_types import TaskInstance
from llm_local_rl.types import CoinFlipInstance, HTSequenceInstance, ShortStoryInstance

_COIN_SYSTEM = load_prompt("tasks/coin_system.md")
_COIN_USER = load_prompt("tasks/coin_user.md")
_HT_USER = load_prompt("tasks/ht_sequence_user.md")
_SHORT_STORY_R1 = load_prompt("tasks/short_story_r1.md")
_SHORT_STORY_WORDS = ("opal", "cobalt", "saffron", "ember", "quill", "harbor", "glyph", "lilac")


@dataclass(frozen=True)
class HTSequenceEnv:
    name: str = "ht_sequence"
    sequence_len: int = 8
    reward_mode: str = "num_h"
    strict_format: bool = False

    def __post_init__(self) -> None:
        if self.sequence_len <= 0:
            raise ValueError(f"sequence_len must be > 0, got {self.sequence_len}")
        if self.reward_mode not in {"num_h", "num_transitions"}:
            raise ValueError(f"Unsupported reward_mode={self.reward_mode!r}")

    def sample_instances(self, *, n: int, seed: int | None) -> list[HTSequenceInstance]:
        _ = seed
        return [
            HTSequenceInstance(
                instance_id=f"ht_{offset}",
                sequence_len=self.sequence_len,
            )
            for offset in range(n)
        ]

    def build_initial_prompt(self, *, instance: HTSequenceInstance) -> str:
        return format_prompt(_HT_USER, sequence_len=str(int(instance.sequence_len)))

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: HTSequenceInstance,
        tokenizer: Tokenizer,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        prompt = self.build_initial_prompt(instance=instance)
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return get_chat_adapter(tokenizer).encode_messages(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.encode(prompt, add_special_tokens=False)

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
        if hasattr(tokenizer, "apply_chat_template"):
            stop = get_chat_adapter(tokenizer).get_stop_sequences()
            if stop is not None and len(stop) == 1:
                return [int(stop[0])]
        stop = tokenizer.encode("\n", add_special_tokens=False)
        if len(stop) != 1:
            raise ValueError("Expected newline stop sequence to tokenize to one token.")
        return stop

    def score_completion(
        self,
        *,
        instance: HTSequenceInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
        parsed, parse_success = parse_ht_sequence(
            text=text,
            target_len=instance.sequence_len,
            strict_format=self.strict_format,
        )
        num_h = parsed.count("H")
        num_t = parsed.count("T")
        num_transitions = sum(1 for idx in range(1, len(parsed)) if parsed[idx] != parsed[idx - 1])

        if not parse_success:
            reward = 0.0
        elif self.reward_mode == "num_h":
            reward = float(num_h)
        else:
            reward = float(num_transitions)

        return reward, {
            "parse_success": 1.0 if parse_success else 0.0,
            "parsed_sequence": parsed,
            "num_h": num_h,
            "num_t": num_t,
            "num_transitions": num_transitions,
            "reward_mode": self.reward_mode,
            "full_text": text,
        }


@dataclass(frozen=True)
class CoinFlipEnv:
    name: str = "coin_flip"
    target_color: str = "Blue"

    def sample_instances(self, *, n: int, seed: int | None) -> list[CoinFlipInstance]:
        rng = random.Random(seed)
        offsets = list(range(n))
        rng.shuffle(offsets)
        return [CoinFlipInstance(instance_id=f"coin_{offset}") for offset in offsets]

    def build_initial_prompt(self, *, instance: CoinFlipInstance) -> str:
        _ = instance
        return _COIN_USER

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: CoinFlipInstance,
        tokenizer: Tokenizer,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        _ = instance
        if hasattr(tokenizer, "apply_chat_template"):
            messages = []
            if _COIN_SYSTEM:
                messages.append({"role": "system", "content": _COIN_SYSTEM})
            messages.append({"role": "user", "content": _COIN_USER})
            return get_chat_adapter(tokenizer).encode_messages(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.encode(_COIN_USER, add_special_tokens=False)

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
        if hasattr(tokenizer, "apply_chat_template"):
            stop = get_chat_adapter(tokenizer).get_stop_sequences()
            if stop is not None and len(stop) == 1:
                return [int(stop[0])]
        stop = tokenizer.encode("\n", add_special_tokens=False)
        if len(stop) != 1:
            raise ValueError("Expected newline stop sequence to tokenize to one token.")
        return stop

    def score_completion(
        self,
        *,
        instance: CoinFlipInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        _ = instance
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True).strip()
        first_word = text.split()[0] if text else ""
        normalized = first_word.capitalize()
        parse_success = 1.0 if normalized in ("Red", "Blue") else 0.0
        reward = 1.0 if normalized == self.target_color else 0.0
        red_tok = tokenizer.encode("Red", add_special_tokens=False)
        blue_tok = tokenizer.encode("Blue", add_special_tokens=False)
        if len(red_tok) != 1 or len(blue_tok) != 1:
            raise ValueError(f"Expected single-token Red/Blue, got {red_tok} / {blue_tok}")
        return reward, {
            "parse_success": parse_success,
            "choice": normalized if parse_success else None,
            "choice_token_id": int(completion_token_ids[0]) if completion_token_ids else None,
            "candidate_token_ids": {"Red": int(red_tok[0]), "Blue": int(blue_tok[0])},
            "target": self.target_color,
            "full_text": text,
        }


@dataclass(frozen=True)
class ShortStoryEnv:
    name: str = "short_story"

    def sample_instances(self, *, n: int, seed: int | None) -> list[ShortStoryInstance]:
        rng = random.Random(seed)
        return [
            ShortStoryInstance(
                instance_id=f"short_story_{idx}",
                secret_word=rng.choice(_SHORT_STORY_WORDS),
            )
            for idx in range(n)
        ]

    def build_initial_prompt(self, *, instance: ShortStoryInstance) -> str:
        return format_prompt(_SHORT_STORY_R1, secret_word=instance.secret_word)

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: ShortStoryInstance,
        tokenizer: Tokenizer,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        prompt = self.build_initial_prompt(instance=instance)
        if hasattr(tokenizer, "apply_chat_template"):
            return get_chat_adapter(tokenizer).encode_messages(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.encode(prompt, add_special_tokens=False)

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
        if hasattr(tokenizer, "apply_chat_template"):
            stop = get_chat_adapter(tokenizer).get_stop_sequences()
            if stop is not None and len(stop) == 1:
                return [int(stop[0])]
        stop = tokenizer.encode("\n", add_special_tokens=False)
        if len(stop) != 1:
            raise ValueError("Expected newline stop sequence to tokenize to one token.")
        return stop

    def score_completion(
        self,
        *,
        instance: ShortStoryInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
        solution = extract_solution(text)
        parse_success = solution is not None
        used_secret = contains_word(solution or "", instance.secret_word)
        return 1.0 if used_secret else 0.0, {
            "parse_success": 1.0 if parse_success else 0.0,
            "used_secret": 1.0 if used_secret else 0.0,
            "secret_word": instance.secret_word,
            "solution": solution,
            "full_text": text,
        }


@dataclass(frozen=True)
class CountdownCodeEnv:
    name: str = "countdown_code"
    num_numbers: int = 4
    prompt_format: str = "chat"

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        return sample_countdown_instances(n=n, seed=seed, num_numbers=self.num_numbers)

    def build_initial_prompt(self, *, instance: TaskInstance) -> str:
        numbers = [int(n) for n in instance.payload["numbers"]]
        target = int(instance.payload["target"])
        if self.prompt_format == "qwen35_base_text_prefill":
            return base_text_prompt(
                system_text=None,
                user_text=countdown_user_text(numbers=numbers, target=target),
                assistant_prefill="",
            )
        messages = countdown_messages(numbers=numbers, target=target)
        return messages[-1]["content"]

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: TaskInstance,
        tokenizer: Tokenizer,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        numbers = [int(n) for n in instance.payload["numbers"]]
        target = int(instance.payload["target"])
        if self.prompt_format == "qwen35_base_text_prefill":
            return encode_base_text(
                tokenizer=tokenizer,
                text=base_text_prompt(
                    system_text=None,
                    user_text=countdown_user_text(numbers=numbers, target=target),
                    assistant_prefill="",
                ),
            )
        messages = countdown_messages(numbers=numbers, target=target)
        if hasattr(tokenizer, "apply_chat_template"):
            return get_chat_adapter(tokenizer).encode_messages(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.encode(messages[-1]["content"], add_special_tokens=False)

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
        if self.prompt_format == "qwen35_base_text_prefill":
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                raise ValueError("Countdown base-text rollouts require tokenizer.eos_token_id.")
            return [int(eos_token_id)]
        if self.prompt_format == "chat":
            if hasattr(tokenizer, "apply_chat_template"):
                stop = get_chat_adapter(tokenizer).get_stop_sequences()
                if stop is not None and len(stop) == 1:
                    return [int(stop[0])]
            stop = tokenizer.encode("\n", add_special_tokens=False)
            if len(stop) != 1:
                raise ValueError("Expected newline stop sequence to tokenize to one token.")
            return stop
        raise ValueError(f"Unsupported Countdown prompt_format={self.prompt_format!r}.")

    def score_completion(
        self,
        *,
        instance: TaskInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
        score = score_countdown_completion(
            numbers=[int(n) for n in instance.payload["numbers"]],
            target=int(instance.payload["target"]),
            completion_text=text,
            require_strict_format=True,
        )
        return score.reward, score.metrics


@dataclass(frozen=True)
class QualityEnv:
    name: str = "quality_debate"
    data_dir: str | None = None
    split: str = "train"
    hard_only: bool = True
    source: str | None = "Gutenberg"
    topic_contains: str | None = "Science fiction"
    download: bool = False

    def __post_init__(self) -> None:
        questions = load_quality_questions(
            data_dir=self.data_dir,
            split=self.split,
            hard_only=self.hard_only,
            source=self.source or None,
            topic_contains=self.topic_contains or None,
            download=self.download,
        )
        if not questions:
            raise ValueError("QuALITY filter produced zero questions.")
        object.__setattr__(self, "_questions", questions)

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        rng = random.Random(seed)
        instances: list[TaskInstance] = []
        for idx, question in enumerate(sample_quality_questions(self._questions, n=n, seed=seed)):
            correct_first = rng.choice([True, False])
            payload = {
                "article": question.article,
                "question": question.question,
                "answer_a": question.correct_answer if correct_first else question.distractor_answer,
                "answer_b": question.distractor_answer if correct_first else question.correct_answer,
                "correct_label": "A" if correct_first else "B",
                "ground_truth": "A" if correct_first else "B",
                "question_id": question.question_id,
                "article_id": question.article_id,
            }
            instances.append(TaskInstance(instance_id=f"quality_{question.question_id}_{idx}", payload=payload))
        return instances

    def build_initial_prompt(self, *, instance: TaskInstance) -> str:
        return (
            f"Article:\n{instance.payload['article']}\n\n"
            f"Question:\n{instance.payload['question']}\n\n"
            f"Answer A:\n{instance.payload['answer_a']}\n\n"
            f"Answer B:\n{instance.payload['answer_b']}\n\n"
            "Choose the correct answer. Output only A or B."
        )

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: TaskInstance,
        tokenizer: Tokenizer,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        prompt = self.build_initial_prompt(instance=instance)
        if hasattr(tokenizer, "apply_chat_template"):
            return get_chat_adapter(tokenizer).encode_messages(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.encode(prompt, add_special_tokens=False)

    def stop_token_ids(self, *, tokenizer: Tokenizer) -> list[int]:
        if hasattr(tokenizer, "apply_chat_template"):
            stop = get_chat_adapter(tokenizer).get_stop_sequences()
            if stop is not None and len(stop) == 1:
                return [int(stop[0])]
        stop = tokenizer.encode("\n", add_special_tokens=False)
        if len(stop) != 1:
            raise ValueError("Expected newline stop sequence to tokenize to one token.")
        return stop

    def score_completion(
        self,
        *,
        instance: TaskInstance,
        tokenizer: Tokenizer,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True).strip()
        match = text[:1].upper()
        correct = match == instance.payload["correct_label"]
        return 1.0 if correct else 0.0, {
            "parse_success": 1.0 if match in {"A", "B"} else 0.0,
            "choice": match if match in {"A", "B"} else None,
            "correct_label": instance.payload["correct_label"],
            "full_text": text,
        }
