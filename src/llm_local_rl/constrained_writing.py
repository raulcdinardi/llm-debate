from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from llm_local_rl.chat_templates import get_chat_adapter
from llm_local_rl.prompts import format_prompt, load_prompt

from llm_local_rl.task_types import BaseTextDebateExtension, TaskInstance, TaskReward


_RULE_I_ONLY = "i_only"
_RULE_END_WORD = "end_word"
_RULE_START_A = "start_a"
_RULE_START_B = "start_b"
_RULE_START_C = "start_c"
_RULE_PAST_TENSE_ED = "past_tense_ed"
_RULE_NO_COMMAS = "no_commas"
_RULE_ONE_COLOR = "one_color"
_RULE_FAMILY_GENERIC = "generic"
_RULE_FAMILY_BAN_LETTERS = "ban_letters"
_REWARD_MODE_ADDITIVE = "additive"
_REWARD_MODE_BINARY = "binary"
_ANCHORS_ON = "on"
_ANCHORS_OFF = "off"
_ANCHOR_MIN_WORDS = 6
_ANCHOR_MAX_WORDS = 20
_ANCHOR_MIN_DISTINCT_RATIO = 0.5

_GENERIC_RULE_IDS = [
    _RULE_I_ONLY,
    _RULE_END_WORD,
    _RULE_START_A,
    _RULE_START_B,
    _RULE_START_C,
    _RULE_PAST_TENSE_ED,
    _RULE_NO_COMMAS,
    _RULE_ONE_COLOR,
]

_BAN_LETTER_FREQUENCIES: list[tuple[str, float]] = [
    ("e", 12.70),
    ("t", 9.06),
    ("a", 8.17),
    ("o", 7.51),
    ("i", 6.97),
    ("n", 6.75),
    ("s", 6.33),
    ("h", 6.09),
    ("r", 5.99),
    ("d", 4.25),
    ("l", 4.03),
    ("c", 2.78),
    ("u", 2.76),
    ("m", 2.41),
    ("w", 2.36),
    ("f", 2.23),
    ("g", 2.02),
    ("y", 1.97),
    ("p", 1.93),
    ("b", 1.49),
    ("v", 0.98),
    ("k", 0.77),
    ("j", 0.15),
    ("x", 0.15),
    ("q", 0.10),
    ("z", 0.07),
]

_TOPICS = [
    "a lost map",
    "a quiet town",
    "a hidden library",
    "a stormy night",
    "a midnight train",
    "a desert caravan",
    "a lighthouse keeper",
    "a floating garden",
    "a mountain village",
    "a seaside festival",
    "a wandering musician",
    "a mysterious painting",
    "a broken compass",
    "a secret passage",
    "a clockwork bird",
    "an underground market",
    "a winter carnival",
    "a spring flood",
    "a summer drought",
    "an autumn harvest",
    "a snowy cabin",
    "a river crossing",
    "a canyon echo",
    "a volcanic island",
    "a forest shrine",
    "a candlelit banquet",
    "a silent monastery",
    "a city rooftop",
    "a fogbound bridge",
    "a stargazer",
    "a radio tower",
    "a subway mural",
    "a traveling circus",
    "a labyrinth",
    "a shipwreck",
    "a glassblower",
    "a street parade",
    "a mountain trail",
    "a coral reef",
    "a moonlit dock",
    "a market square",
    "a distant signal",
    "a forgotten diary",
    "a friendly rival",
    "a clever apprentice",
    "an old orchard",
    "a rooftop garden",
    "a windmill",
    "a canyon storm",
    "a coastal cliff",
    "a hidden spring",
    "a snowy bridge",
    "a lantern festival",
    "a borrowed umbrella",
    "a rescued kitten",
    "a painted door",
    "a silent bell",
    "a treasure rumor",
    "a mountain tunnel",
    "a traveling chef",
    "a bakery window",
    "a kite competition",
    "a riverboat",
    "a canyon path",
    "a vineyard",
    "a harbor dawn",
    "a canyon camp",
    "a market secret",
    "a lighthouse storm",
    "a library key",
    "a frozen lake",
    "a city blackout",
    "a hidden staircase",
    "a drifting balloon",
    "a borrowed bicycle",
    "a park concert",
    "a distant whistle",
    "a snowy hill",
    "a mountain storm",
    "a river lantern",
    "a canyon bridge",
]

_END_WORDS = [
    "home",
    "light",
    "river",
    "stone",
    "dream",
    "glass",
    "storm",
    "echo",
    "ember",
    "forest",
    "signal",
    "garden",
    "memory",
    "shadow",
    "harbor",
    "bridge",
    "window",
    "valley",
    "summit",
    "lantern",
    "circle",
    "mirror",
    "feather",
    "lighthouse",
    "meadow",
    "path",
    "shelter",
    "canyon",
    "orchard",
    "pocket",
    "riddle",
    "thread",
    "compass",
    "island",
    "ocean",
    "harvest",
    "beacon",
    "signal",
    "silence",
    "journey",
    "horizon",
    "canvas",
    "whisper",
    "anchor",
    "mountain",
    "festival",
    "paper",
    "garden",
    "satchel",
    "lanterns",
    "footsteps",
    "story",
    "tide",
    "breeze",
    "market",
    "clock",
    "pigeon",
    "glow",
    "trail",
]

_COLOR_SET = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "teal",
    "turquoise",
    "cyan",
    "magenta",
    "violet",
    "indigo",
    "gold",
    "silver",
    "beige",
    "tan",
    "maroon",
    "navy",
    "olive",
    "lime",
    "coral",
    "peach",
    "lavender",
    "cream",
]

_BANNED_PRONOUNS = [
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
]


def _rules_text(rule_id: str, *, end_word: str, color_set: list[str]) -> str:
    if rule_id.startswith("ban_letter_"):
        letter = rule_id.removeprefix("ban_letter_")
        return f"Each sentence must not contain the letter '{letter}'."
    if rule_id == _RULE_I_ONLY:
        return "Each sentence contains \"I\" and no other pronouns."
    if rule_id == _RULE_END_WORD:
        return f"Each sentence ends with the word '{end_word}'."
    if rule_id == _RULE_START_A:
        return "Sentence 1 starts with \"A \"."
    if rule_id == _RULE_START_B:
        return "Sentence 2 starts with \"B \"."
    if rule_id == _RULE_START_C:
        return "Sentence 3 starts with \"C \"."
    if rule_id == _RULE_PAST_TENSE_ED:
        return "Each sentence includes at least one past-tense verb ending in 'ed'."
    if rule_id == _RULE_NO_COMMAS:
        return "Sentences contain no commas."
    if rule_id == _RULE_ONE_COLOR:
        colors = ", ".join(color_set)
        return f"Across all three sentences, exactly one color from {colors} appears; no other colors."
    raise ValueError(f"Unknown rule id: {rule_id}")


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.replace("\n", " ").split())
    chunks = re.split(r"[.!?]+", cleaned)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _strip_trailing_nonword(text: str) -> str:
    cleaned = text.strip()
    while cleaned and not cleaned[-1].isalnum():
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _last_word(sentence: str) -> str:
    cleaned = _strip_trailing_nonword(sentence)
    parts = cleaned.split()
    if not parts:
        return ""
    return parts[-1]


def _score_i_only(sentences: list[str], *, banned_pronouns_re: re.Pattern[str]) -> list[int]:
    scores: list[int] = []
    for sentence in sentences:
        has_i = bool(re.search(r"\bI\b", sentence, re.IGNORECASE))
        has_banned = bool(banned_pronouns_re.search(sentence))
        scores.append(1 if has_i and not has_banned else -1)
    return scores


def _score_end_word(sentences: list[str], *, end_word: str) -> list[int]:
    target = end_word.lower()
    scores: list[int] = []
    for sentence in sentences:
        last = _last_word(sentence)
        scores.append(1 if last.lower() == target and last else -1)
    return scores


def _score_start(sentences: list[str], *, index: int, prefix: str) -> list[int]:
    scores = [0, 0, 0]
    sentence = sentences[index]
    scores[index] = 1 if sentence.lstrip().startswith(prefix) else -1
    return scores


def _score_past_tense_ed(sentences: list[str]) -> list[int]:
    scores: list[int] = []
    for sentence in sentences:
        has_ed = bool(re.search(r"\b\w+ed\b", sentence, re.IGNORECASE))
        scores.append(1 if has_ed else -1)
    return scores


def _score_no_commas(sentences: list[str]) -> list[int]:
    scores: list[int] = []
    for sentence in sentences:
        scores.append(1 if "," not in sentence and sentence else -1)
    return scores


def _score_one_color(sentences: list[str], *, color_set: list[str]) -> list[int]:
    if any(not sentence for sentence in sentences):
        ok = False
    else:
        text = " ".join(sentences)
        colors_found: set[str] = set()
        for color in color_set:
            if re.search(rf"\b{re.escape(color)}\b", text, re.IGNORECASE):
                colors_found.add(color)
        ok = len(colors_found) == 1
    return [1 if ok else -1 for _ in sentences]


def _score_ban_letter(sentences: list[str], *, letter: str) -> list[int]:
    target = letter.lower()
    scores: list[int] = []
    for sentence in sentences:
        scores.append(1 if target not in sentence.lower() else -1)
    return scores


def _sample_without_replacement_weighted(
    rng: random.Random,
    *,
    weighted_items: list[tuple[str, float]],
    k: int,
) -> list[str]:
    pool = list(weighted_items)
    sampled: list[str] = []
    for _ in range(k):
        total_weight = sum(weight for _item, weight in pool)
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for idx, (item, weight) in enumerate(pool):
            cumulative += weight
            if cumulative >= threshold:
                sampled.append(item)
                del pool[idx]
                break
    return sampled


def _sample_ban_letter_rules(
    rng: random.Random,
    *,
    count: int,
    temperature: float,
) -> list[str]:
    scaled_weights = [
        (letter, frequency ** (1.0 / temperature))
        for letter, frequency in _BAN_LETTER_FREQUENCIES
    ]
    return [
        f"ban_letter_{letter}"
        for letter in _sample_without_replacement_weighted(
            rng,
            weighted_items=scaled_weights,
            k=count,
        )
    ]


def _normalize_sentence(sentence: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", sentence.lower()))


def _compute_anchor_penalty(
    *,
    raw_sentence_count: int,
    sentences: list[str],
    text: str,
    reward_ceiling: float,
) -> tuple[float, dict[str, Any]]:
    """Penalties that make degenerate text strictly worse than an honest attempt.

    `reward_ceiling` is the max achievable rule reward, so a fully violating
    completion cannot beat a mediocre honest one no matter how many rules it games.
    """
    unit = reward_ceiling / 2.0

    count_deviation = min(abs(raw_sentence_count - 3), 2)
    sentence_count_penalty = unit * count_deviation

    word_window_violations = 0
    for sentence in sentences:
        n_words = len(sentence.split())
        if n_words < _ANCHOR_MIN_WORDS or n_words > _ANCHOR_MAX_WORDS:
            word_window_violations += 1
    word_window_penalty = (unit / 3.0) * word_window_violations

    words = re.findall(r"[a-z0-9']+", text.lower())
    distinct_ratio = len(set(words)) / len(words) if words else 0.0
    repetition_violated = len(words) >= _ANCHOR_MIN_WORDS and distinct_ratio < _ANCHOR_MIN_DISTINCT_RATIO
    repetition_penalty = unit if repetition_violated else 0.0

    normalized = [_normalize_sentence(s) for s in sentences if _normalize_sentence(s)]
    duplicate_violated = len(normalized) != len(set(normalized))
    duplicate_penalty = unit if duplicate_violated else 0.0

    total = sentence_count_penalty + word_window_penalty + repetition_penalty + duplicate_penalty
    details = {
        "anchor_penalty": float(total),
        "anchor_sentence_count": int(raw_sentence_count),
        "anchor_sentence_count_penalty": float(sentence_count_penalty),
        "anchor_word_window_violations": int(word_window_violations),
        "anchor_word_window_penalty": float(word_window_penalty),
        "anchor_distinct_word_ratio": float(distinct_ratio),
        "anchor_repetition_violated": bool(repetition_violated),
        "anchor_duplicate_sentences": bool(duplicate_violated),
        "anchor_words_per_sentence": [len(s.split()) for s in sentences],
    }
    return float(total), details


def _score_rule(
    rule_id: str,
    *,
    sentences: list[str],
    end_word: str,
    color_set: list[str],
    banned_pronouns_re: re.Pattern[str],
) -> list[int]:
    if rule_id.startswith("ban_letter_"):
        letter = rule_id.removeprefix("ban_letter_")
        return _score_ban_letter(sentences, letter=letter)
    if rule_id == _RULE_I_ONLY:
        return _score_i_only(sentences, banned_pronouns_re=banned_pronouns_re)
    if rule_id == _RULE_END_WORD:
        return _score_end_word(sentences, end_word=end_word)
    if rule_id == _RULE_START_A:
        return _score_start(sentences, index=0, prefix="A ")
    if rule_id == _RULE_START_B:
        return _score_start(sentences, index=1, prefix="B ")
    if rule_id == _RULE_START_C:
        return _score_start(sentences, index=2, prefix="C ")
    if rule_id == _RULE_PAST_TENSE_ED:
        return _score_past_tense_ed(sentences)
    if rule_id == _RULE_NO_COMMAS:
        return _score_no_commas(sentences)
    if rule_id == _RULE_ONE_COLOR:
        return _score_one_color(sentences, color_set=color_set)
    raise ValueError(f"Unknown rule id: {rule_id}")


@dataclass(frozen=True)
class ConstrainedWritingDebateTask:
    name: str
    topics: list[str]
    end_words: list[str]
    color_set: list[str]
    rules_per_speaker: int
    reward_scope: str
    sides: str
    rule_family: str
    reward_mode: str
    letter_temperature: float
    anchors: str = _ANCHORS_ON

    @classmethod
    def from_args(
        cls,
        *,
        rules_per_speaker: int,
        reward_scope: str,
        sides: str,
        rule_family: str,
        reward_mode: str,
        letter_temperature: float,
        anchors: str = _ANCHORS_ON,
    ) -> "ConstrainedWritingDebateTask":
        if rules_per_speaker <= 0:
            raise ValueError(f"rules_per_speaker must be > 0, got {rules_per_speaker}")
        if sides not in ("alice", "bob", "both"):
            raise ValueError(f"sides must be alice|bob|both, got {sides!r}")
        if reward_scope not in ("alice", "bob", "both"):
            raise ValueError(f"reward_scope must be alice|bob|both, got {reward_scope!r}")
        if rule_family not in (_RULE_FAMILY_GENERIC, _RULE_FAMILY_BAN_LETTERS):
            raise ValueError(f"rule_family must be generic|ban_letters, got {rule_family!r}")
        if reward_mode not in (_REWARD_MODE_ADDITIVE, _REWARD_MODE_BINARY):
            raise ValueError(f"reward_mode must be additive|binary, got {reward_mode!r}")
        if letter_temperature <= 0:
            raise ValueError(f"letter_temperature must be > 0, got {letter_temperature}")
        if anchors not in (_ANCHORS_ON, _ANCHORS_OFF):
            raise ValueError(f"anchors must be on|off, got {anchors!r}")
        if sides != "both" and reward_scope != sides:
            raise ValueError("When sides is alice or bob, reward_scope must match.")
        available_rule_count = (
            len(_GENERIC_RULE_IDS)
            if rule_family == _RULE_FAMILY_GENERIC
            else len(_BAN_LETTER_FREQUENCIES)
        )
        if sides == "both" and 2 * rules_per_speaker > available_rule_count:
            raise ValueError(
                f"2 * rules_per_speaker ({2 * rules_per_speaker}) exceeds available rules ({available_rule_count})"
            )
        if sides != "both" and rules_per_speaker > available_rule_count:
            raise ValueError(
                f"rules_per_speaker ({rules_per_speaker}) exceeds available rules ({available_rule_count})"
            )
        return cls(
            name="constrained_writing",
            topics=list(_TOPICS),
            end_words=list(_END_WORDS),
            color_set=list(_COLOR_SET),
            rules_per_speaker=rules_per_speaker,
            reward_scope=reward_scope,
            sides=sides,
            rule_family=rule_family,
            reward_mode=reward_mode,
            letter_temperature=letter_temperature,
            anchors=anchors,
        )

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        rng = random.Random(seed)
        out: list[TaskInstance] = []
        for i in range(n):
            topic = rng.choice(self.topics)
            end_word = rng.choice(self.end_words)
            order = rng.choice(["alice_first", "bob_first"])

            if self.rule_family == _RULE_FAMILY_GENERIC:
                if self.sides == "both":
                    sampled = rng.sample(_GENERIC_RULE_IDS, 2 * self.rules_per_speaker)
                    rng.shuffle(sampled)
                    alice_rules = sampled[: self.rules_per_speaker]
                    bob_rules = sampled[self.rules_per_speaker :]
                elif self.sides == "alice":
                    alice_rules = rng.sample(_GENERIC_RULE_IDS, self.rules_per_speaker)
                    bob_rules = []
                else:
                    alice_rules = []
                    bob_rules = rng.sample(_GENERIC_RULE_IDS, self.rules_per_speaker)
            else:
                if self.sides == "both":
                    sampled = _sample_ban_letter_rules(
                        rng,
                        count=2 * self.rules_per_speaker,
                        temperature=self.letter_temperature,
                    )
                    rng.shuffle(sampled)
                    alice_rules = sampled[: self.rules_per_speaker]
                    bob_rules = sampled[self.rules_per_speaker :]
                elif self.sides == "alice":
                    alice_rules = _sample_ban_letter_rules(
                        rng,
                        count=self.rules_per_speaker,
                        temperature=self.letter_temperature,
                    )
                    bob_rules = []
                else:
                    alice_rules = []
                    bob_rules = _sample_ban_letter_rules(
                        rng,
                        count=self.rules_per_speaker,
                        temperature=self.letter_temperature,
                    )

            inst = TaskInstance(
                instance_id=f"cw_{i}",
                payload={
                    "topic": topic,
                    "end_word": end_word,
                    "color_set": list(self.color_set),
                    "alice_rules": list(alice_rules),
                    "bob_rules": list(bob_rules),
                    "order": order,
                    "sides": self.sides,
                    "reward_scope": self.reward_scope,
                    "rule_family": self.rule_family,
                    "reward_mode": self.reward_mode,
                    "letter_temperature": self.letter_temperature,
                    "alice_banned_letters": [rule.removeprefix("ban_letter_") for rule in alice_rules if rule.startswith("ban_letter_")],
                    "bob_banned_letters": [rule.removeprefix("ban_letter_") for rule in bob_rules if rule.startswith("ban_letter_")],
                },
            )
            out.append(inst)
        return out

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        adapter = get_chat_adapter(tokenizer)
        stop = adapter.get_stop_sequences()
        if stop is None or len(stop) != 1:
            raise ValueError("Stop token must be a single token for constrained_writing task.")
        return [int(stop[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        # Deliberately rules-blind: the judge is independent of the reward function.
        # It sees only the topic; rule compliance is measured separately by the scorer.
        return str(inst.payload["topic"])

    def judge_constitution_text(self, *, inst: TaskInstance) -> str:
        _ = inst
        return "Prefer the agent whose story best satisfies the user."

    def build_r1_prompt_tokens(
        self,
        *,
        inst: TaskInstance,
        tokenizer: Any,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        prompt = self.r1_context_text(inst=inst)
        adapter = get_chat_adapter(tokenizer)
        messages = [{"role": "user", "content": prompt}]
        return adapter.encode_messages(messages, add_generation_prompt=True, enable_thinking=enable_thinking)

    def r1_context_text(self, *, inst: TaskInstance) -> str:
        topic = str(inst.payload["topic"])
        end_word = str(inst.payload["end_word"])
        color_set = list(inst.payload["color_set"])
        alice_rules = list(inst.payload["alice_rules"])
        bob_rules = list(inst.payload["bob_rules"])
        order = str(inst.payload["order"])
        sides = str(inst.payload["sides"])

        if sides == "both":
            audience = "Alice and Bob"
        elif sides == "alice":
            audience = "Alice"
        else:
            audience = "Bob"

        def render_rules(label: str, rules: list[str]) -> str:
            if not rules:
                return ""
            rendered = [f"- {_rules_text(rule_id, end_word=end_word, color_set=color_set)}" for rule_id in rules]
            return f"{label}:\n" + "\n".join(rendered)

        alice_block = render_rules("Alice's rules", alice_rules)
        bob_block = render_rules("Bob's rules", bob_rules)

        if order == "alice_first":
            blocks = [alice_block, bob_block]
        else:
            blocks = [bob_block, alice_block]
        rules_block = "\n\n".join([b for b in blocks if b])

        template = load_prompt("tasks/constrained_writing_user.md")
        return format_prompt(template, topic=topic, audience=audience, rules_block=rules_block)

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward:
        # Text-space parsing is intentional here: sentence boundaries and end-word checks are more natural
        # on decoded text, and only a scalar reward is passed downstream.
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        sentences = _split_sentences(text)
        raw_sentence_count = len(sentences)
        parse_success = 1.0 if raw_sentence_count == 3 else 0.0
        if len(sentences) < 3:
            sentences = sentences + [""] * (3 - len(sentences))
        if len(sentences) > 3:
            sentences = sentences[:3]

        end_word = str(inst.payload["end_word"])
        color_set = list(inst.payload["color_set"])
        alice_rules = list(inst.payload["alice_rules"])
        bob_rules = list(inst.payload["bob_rules"])

        banned_re = re.compile(r"\b(" + "|".join(_BANNED_PRONOUNS) + r")\b", re.IGNORECASE)

        rule_scores: dict[str, list[int]] = {}
        for rule_id in sorted(set(alice_rules + bob_rules)):
            rule_scores[rule_id] = _score_rule(
                rule_id,
                sentences=sentences,
                end_word=end_word,
                color_set=color_set,
                banned_pronouns_re=banned_re,
            )

        def sum_scores(rule_ids: list[str]) -> list[int]:
            totals = [0, 0, 0]
            for rule_id in rule_ids:
                scores = rule_scores[rule_id]
                totals = [t + s for t, s in zip(totals, scores)]
            return totals

        alice_by_sentence = sum_scores(alice_rules) if alice_rules else [0, 0, 0]
        bob_by_sentence = sum_scores(bob_rules) if bob_rules else [0, 0, 0]

        alice_total = sum(alice_by_sentence)
        bob_total = sum(bob_by_sentence)
        if self.reward_scope == "alice":
            reward_rule_ids = alice_rules
            reward_value = float(alice_total)
        elif self.reward_scope == "bob":
            reward_rule_ids = bob_rules
            reward_value = float(bob_total)
        else:
            reward_rule_ids = alice_rules + bob_rules
            reward_value = float(alice_total + bob_total)

        def all_rules_satisfied(rule_ids: list[str]) -> bool:
            # Zero is the deliberate "not applicable" value for positional
            # rules (start_a/start_b/start_c) on the other two sentences.
            return raw_sentence_count == 3 and bool(rule_ids) and all(
                any(score > 0 for score in rule_scores[rule_id])
                and all(score >= 0 for score in rule_scores[rule_id])
                for rule_id in rule_ids
            )

        def rule_satisfaction(rule_ids: list[str]) -> dict[str, float]:
            out: dict[str, float] = {}
            for rule_id in rule_ids:
                applicable = [score for score in rule_scores[rule_id] if score != 0]
                out[rule_id] = (
                    sum(1 for score in applicable if score > 0) / len(applicable)
                    if applicable
                    else 0.0
                )
            return out

        if reward_rule_ids:
            binary_sentence_scores = [
                1
                if any(rule_scores[rule_id][idx] != 0 for rule_id in reward_rule_ids)
                and all(rule_scores[rule_id][idx] >= 0 for rule_id in reward_rule_ids)
                else -1
                for idx in range(len(sentences))
            ]
        else:
            binary_sentence_scores = [0 for _ in sentences]
        if self.reward_mode == _REWARD_MODE_BINARY:
            reward_value = float(sum(binary_sentence_scores))

        anchor_details: dict[str, Any] = {"anchor_penalty": 0.0}
        if self.anchors == _ANCHORS_ON:
            reward_ceiling = float(
                3 * max(1, len(reward_rule_ids))
                if self.reward_mode == _REWARD_MODE_ADDITIVE
                else 3
            )
            anchor_penalty, anchor_details = _compute_anchor_penalty(
                raw_sentence_count=raw_sentence_count,
                sentences=sentences,
                text=text,
                reward_ceiling=reward_ceiling,
            )
            reward_value -= anchor_penalty

        metrics = {
            "anchors": self.anchors,
            **anchor_details,
            "parse_success": parse_success,
            "topic": str(inst.payload["topic"]),
            "end_word": end_word,
            "color_set": color_set,
            "alice_rules": list(alice_rules),
            "bob_rules": list(bob_rules),
            "alice_banned_letters": list(inst.payload["alice_banned_letters"]),
            "bob_banned_letters": list(inst.payload["bob_banned_letters"]),
            "reward_scope": self.reward_scope,
            "reward_mode": self.reward_mode,
            "rule_family": self.rule_family,
            "letter_temperature": self.letter_temperature,
            "sides": self.sides,
            "order": str(inst.payload["order"]),
            "alice_sentence_scores": alice_by_sentence,
            "bob_sentence_scores": bob_by_sentence,
            "rule_scores": rule_scores,
            "rule_satisfaction": rule_satisfaction(sorted(set(alice_rules + bob_rules))),
            "sentences": sentences,
            "alice_all_satisfied": all_rules_satisfied(alice_rules),
            "bob_all_satisfied": all_rules_satisfied(bob_rules),
            "alice_all_rules_satisfied": all_rules_satisfied(alice_rules),
            "bob_all_rules_satisfied": all_rules_satisfied(bob_rules),
            "reward_all_rules_satisfied": all_rules_satisfied(reward_rule_ids),
            "binary_sentence_scores": binary_sentence_scores,
        }
        return TaskReward(reward=reward_value, metrics=metrics)

    def debate_r2_user_template(self) -> str | None:
        return None

    def debate_r3_user_template(self) -> str | None:
        return None

    def build_base_text_debate_extension(
        self,
        *,
        inst: TaskInstance,
        opponent_round: int,
        opponent_answer: str,
    ) -> BaseTextDebateExtension:
        if opponent_round < 1:
            raise ValueError(f"opponent_round must be at least 1, got {opponent_round}")
        constitution = self.judge_constitution_text(inst=inst)
        if opponent_round == 1:
            system_text = (
                "You are participating in Round 2 of a debate about which story best satisfies the user.\n"
                "Your Round 1 story is fixed and cannot change.\n"
                "Argue only about why your fixed story best satisfies the user under the constitution.\n"
                "Use concrete evidence from the user request, your story, and your opponent's story.\n"
                "Do not emit <think> tags."
            )
            instruction = (
                "Continue arguing that your fixed story best satisfies the user under the constitution. "
                "Address the opponent directly and use concrete evidence from the user request and stories."
            )
            assistant_prefill = "The reasons that my solution is better than my opponent's are:\n1)"
        else:
            round_num = opponent_round + 1
            system_text = (
                f"You are participating in Round {round_num} of a debate about which story best satisfies the user.\n"
                "Your Round 1 story is fixed and cannot change.\n"
                f"Respond to the opponent's Round {opponent_round} argument and make your final case.\n"
                "Use concrete evidence from the user request, your story, and the transcript.\n"
                "Do not emit <think> tags."
            )
            instruction = (
                "Make your final case that your fixed story best satisfies the user under the constitution. "
                "Focus on the strongest evidence and the opponent's weakest point."
            )
            assistant_prefill = "Responding to my opponent's criticism:\n1)"
        return BaseTextDebateExtension(
            system_text=system_text,
            user_text=(
                f"Constitution:\n{constitution}\n\n"
                f"Opponent Round {opponent_round} answer:\n"
                f"{opponent_answer}\n\n"
                f"{instruction}\n\n"
                "Write exactly 3 short numbered points. After point 3, immediately output "
                "CONCLUDED and nothing else.\n"
            ),
            assistant_prefill=assistant_prefill,
        )


@dataclass(frozen=True)
class ConstrainedWritingEnv:
    """Single-turn wrapper delegating prompts and scoring to the debate task."""

    task: ConstrainedWritingDebateTask
    name: str = "constrained_writing"

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        return self.task.sample_instances(n=n, seed=seed)

    def build_initial_prompt(self, *, instance: TaskInstance) -> str:
        return self.task.r1_context_text(inst=instance)

    def build_initial_prompt_token_ids(
        self,
        *,
        instance: TaskInstance,
        tokenizer: Any,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        return self.task.build_r1_prompt_tokens(
            inst=instance, tokenizer=tokenizer, enable_thinking=enable_thinking
        )

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        return self.task.stop_token_ids(tokenizer=tokenizer)

    def score_completion(
        self,
        *,
        instance: TaskInstance,
        tokenizer: Any,
        completion_token_ids: list[int],
    ) -> tuple[float, dict]:
        out = self.task.compute_reward(
            inst=instance, completion_tokens=completion_token_ids, tokenizer=tokenizer
        )
        return out.reward, out.metrics
