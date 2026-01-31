from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from tinker_debate.local_renderers import infer_chat_preamble
from tinker_debate.prompts import format_prompt, load_prompt

from .task_types import TaskInstance, TaskReward, TaskSpec


def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"


def _im_end() -> str:
    return "<|im_end|>\n"


_RULE_I_ONLY = "i_only"
_RULE_END_WORD = "end_word"
_RULE_START_A = "start_a"
_RULE_START_B = "start_b"
_RULE_START_C = "start_c"
_RULE_PAST_TENSE_ED = "past_tense_ed"
_RULE_NO_COMMAS = "no_commas"
_RULE_ONE_COLOR = "one_color"

_RULE_IDS = [
    _RULE_I_ONLY,
    _RULE_END_WORD,
    _RULE_START_A,
    _RULE_START_B,
    _RULE_START_C,
    _RULE_PAST_TENSE_ED,
    _RULE_NO_COMMAS,
    _RULE_ONE_COLOR,
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


def _score_rule(
    rule_id: str,
    *,
    sentences: list[str],
    end_word: str,
    color_set: list[str],
    banned_pronouns_re: re.Pattern[str],
) -> list[int]:
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
class ConstrainedWritingTask(TaskSpec):
    name: str
    topics: list[str]
    end_words: list[str]
    color_set: list[str]
    rules_per_speaker: int
    reward_scope: str
    sides: str

    @classmethod
    def from_args(
        cls,
        *,
        rules_per_speaker: int,
        reward_scope: str,
        sides: str,
    ) -> "ConstrainedWritingTask":
        if rules_per_speaker <= 0:
            raise ValueError(f"rules_per_speaker must be > 0, got {rules_per_speaker}")
        if sides not in ("alice", "bob", "both"):
            raise ValueError(f"sides must be alice|bob|both, got {sides!r}")
        if reward_scope not in ("alice", "bob", "both"):
            raise ValueError(f"reward_scope must be alice|bob|both, got {reward_scope!r}")
        if sides != "both" and reward_scope != sides:
            raise ValueError("When sides is alice or bob, reward_scope must match.")
        if sides == "both" and 2 * rules_per_speaker > len(_RULE_IDS):
            raise ValueError(
                f"2 * rules_per_speaker ({2 * rules_per_speaker}) exceeds available rules ({len(_RULE_IDS)})"
            )
        if sides != "both" and rules_per_speaker > len(_RULE_IDS):
            raise ValueError(
                f"rules_per_speaker ({rules_per_speaker}) exceeds available rules ({len(_RULE_IDS)})"
            )
        return cls(
            name="constrained_writing",
            topics=list(_TOPICS),
            end_words=list(_END_WORDS),
            color_set=list(_COLOR_SET),
            rules_per_speaker=rules_per_speaker,
            reward_scope=reward_scope,
            sides=sides,
        )

    def sample_instances(self, *, n: int, seed: int | None) -> list[TaskInstance]:
        rng = random.Random(seed)
        out: list[TaskInstance] = []
        for i in range(n):
            topic = rng.choice(self.topics)
            end_word = rng.choice(self.end_words)
            order = rng.choice(["alice_first", "bob_first"])

            if self.sides == "both":
                sampled = rng.sample(_RULE_IDS, 2 * self.rules_per_speaker)
                rng.shuffle(sampled)
                alice_rules = sampled[: self.rules_per_speaker]
                bob_rules = sampled[self.rules_per_speaker :]
            elif self.sides == "alice":
                alice_rules = rng.sample(_RULE_IDS, self.rules_per_speaker)
                bob_rules = []
            else:
                alice_rules = []
                bob_rules = rng.sample(_RULE_IDS, self.rules_per_speaker)

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
                },
            )
            out.append(inst)
        return out

    def stop_token_ids(self, *, tokenizer: Any) -> list[int]:
        toks = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Expected single token for <|im_end|>, got {len(toks)}")
        return [int(toks[0])]

    def judge_context_text(self, *, inst: TaskInstance) -> str:
        return str(inst.payload["topic"])

    def build_r1_prompt_tokens(self, *, inst: TaskInstance, tokenizer: Any) -> list[int]:
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
        prompt = format_prompt(template, topic=topic, audience=audience, rules_block=rules_block)
        preamble = infer_chat_preamble(tokenizer)
        full = preamble + _im_start("user") + prompt + "\n" + _im_end() + _im_start("assistant")
        return tokenizer.encode(full, add_special_tokens=False)

    def compute_reward(self, *, inst: TaskInstance, completion_tokens: list[int], tokenizer: Any) -> TaskReward:
        # Text-space parsing is intentional here: sentence boundaries and end-word checks are more natural
        # on decoded text, and only a scalar reward is passed downstream.
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        sentences = _split_sentences(text)
        parse_success = 1.0 if len(sentences) == 3 else 0.0
        if len(sentences) < 3:
            sentences = sentences + [""] * (3 - len(sentences))
        if len(sentences) > 3:
            sentences = sentences[:3]

        end_word = str(inst.payload["end_word"])
        color_set = list(inst.payload["color_set"])
        alice_rules = list(inst.payload["alice_rules"])
        bob_rules = list(inst.payload["bob_rules"])

        banned_re = re.compile(r"\\b(" + "|".join(_BANNED_PRONOUNS) + r")\\b", re.IGNORECASE)

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

        if self.sides == "alice":
            if any(score < 0 for score in alice_by_sentence):
                raise ValueError("Alice-only prompt requires all Alice rules to be satisfied.")
        if self.sides == "bob":
            if any(score < 0 for score in bob_by_sentence):
                raise ValueError("Bob-only prompt requires all Bob rules to be satisfied.")

        if self.reward_scope == "alice":
            reward_value = float(alice_total)
        elif self.reward_scope == "bob":
            reward_value = float(bob_total)
        else:
            reward_value = float(alice_total + bob_total)

        metrics = {
            "parse_success": parse_success,
            "topic": str(inst.payload["topic"]),
            "end_word": end_word,
            "color_set": color_set,
            "alice_rules": list(alice_rules),
            "bob_rules": list(bob_rules),
            "reward_scope": self.reward_scope,
            "sides": self.sides,
            "order": str(inst.payload["order"]),
            "alice_sentence_scores": alice_by_sentence,
            "bob_sentence_scores": bob_by_sentence,
            "rule_scores": rule_scores,
            "sentences": sentences,
        }
        return TaskReward(reward=reward_value, metrics=metrics)

    def debate_r2_user_template(self) -> str | None:
        return None

    def debate_r3_user_template(self) -> str | None:
        return None
