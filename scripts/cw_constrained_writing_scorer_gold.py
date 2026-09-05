#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


class TextTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "".join(chr(token) for token in tokens)


COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "teal", "turquoise", "cyan", "magenta",
    "violet", "indigo", "gold", "silver", "beige", "tan", "maroon", "navy",
    "olive", "lime", "coral", "peach", "lavender", "cream",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(args.source_root) / "src"))
    from llm_local_rl.constrained_writing import ConstrainedWritingDebateTask
    from llm_local_rl.task_types import TaskInstance

    tokenizer = TextTokenizer()
    task = ConstrainedWritingDebateTask.from_args(
        rules_per_speaker=2,
        reward_scope="both",
        sides="both",
        rule_family="generic",
        reward_mode="additive",
        letter_temperature=1.0,
        anchors="on",
    )

    def instance(*, alice: list[str], bob: list[str], end_word: str = "light") -> TaskInstance:
        return TaskInstance(
            instance_id="gold",
            payload={
                "topic": "a mountain trail",
                "end_word": end_word,
                "color_set": COLORS,
                "alice_rules": alice,
                "bob_rules": bob,
                "order": "bob_first",
                "sides": "both",
                "reward_scope": "both",
                "rule_family": "generic",
                "reward_mode": "additive",
                "letter_temperature": 1.0,
                "alice_banned_letters": [r.removeprefix("ban_letter_") for r in alice if r.startswith("ban_letter_")],
                "bob_banned_letters": [r.removeprefix("ban_letter_") for r in bob if r.startswith("ban_letter_")],
            },
        )

    sample_13b = (
        "A sunlit ridge aligned with autumn light. "
        "Past bridge a gust up-tangled silver light. "
        "A stone band roasted on rain-slushed light."
    )
    generic_three = (
        "A patient ranger walked beside the old trail. "
        "B quiet lanterns glowed above the stone trail. "
        "C careful hikers rested near the winding trail."
    )
    i_only_good = (
        "I slowly walked beside the quiet river path. "
        "I calmly rested near the ancient river path. "
        "I softly listened beside the moonlit river path."
    )
    cases = [
        {"id": "13b_exact", "text": sample_13b, "alice": ["past_tense_ed", "one_color"], "bob": ["end_word", "start_a"], "all": True, "parse": 1.0},
        {"id": "13b_bad_start", "text": sample_13b.replace("A sunlit", "The sunlit", 1), "alice": ["past_tense_ed", "one_color"], "bob": ["end_word", "start_a"], "all": False, "parse": 1.0},
        {"id": "13b_two_colors", "text": sample_13b.replace("autumn light", "blue autumn light", 1), "alice": ["past_tense_ed", "one_color"], "bob": ["end_word", "start_a"], "all": False, "parse": 1.0},
        {"id": "13b_bad_end", "text": sample_13b.replace("rain-slushed light", "rain-slushed trail", 1), "alice": ["past_tense_ed", "one_color"], "bob": ["end_word", "start_a"], "all": False, "parse": 1.0},
        {"id": "13b_hyphen_ed", "text": sample_13b, "alice": ["past_tense_ed"], "bob": ["end_word"], "all": True, "parse": 1.0},
        {"id": "i_only_good", "text": i_only_good, "alice": ["i_only"], "bob": ["no_commas"], "all": True, "parse": 1.0},
        {"id": "i_only_other_pronoun", "text": i_only_good.replace("I slowly", "I told her I slowly", 1), "alice": ["i_only"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "no_commas_good", "text": generic_three, "alice": ["no_commas"], "bob": ["start_a"], "all": True, "parse": 1.0},
        {"id": "no_commas_bad", "text": generic_three.replace("A patient ranger", "A patient ranger,", 1), "alice": ["no_commas"], "bob": ["start_a"], "all": False, "parse": 1.0},
        {"id": "start_a_good", "text": generic_three, "alice": ["start_a"], "bob": ["no_commas"], "all": True, "parse": 1.0, "sat": {"start_a": 1.0}},
        {"id": "start_a_bad", "text": generic_three.replace("A patient", "The patient", 1), "alice": ["start_a"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "start_b_good", "text": generic_three, "alice": ["start_b"], "bob": ["no_commas"], "all": True, "parse": 1.0, "sat": {"start_b": 1.0}},
        {"id": "start_b_bad", "text": generic_three.replace("B quiet", "The quiet", 1), "alice": ["start_b"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "start_c_good", "text": generic_three, "alice": ["start_c"], "bob": ["no_commas"], "all": True, "parse": 1.0, "sat": {"start_c": 1.0}},
        {"id": "start_c_bad", "text": generic_three.replace("C careful", "The careful", 1), "alice": ["start_c"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "ban_x_good", "text": generic_three, "alice": ["ban_letter_x"], "bob": ["no_commas"], "all": True, "parse": 1.0},
        {"id": "ban_x_bad", "text": generic_three.replace("old trail", "xylophone trail", 1), "alice": ["ban_letter_x"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "one_color_repeated", "text": generic_three.replace("old trail", "silver trail", 1).replace("stone trail", "silver trail", 1), "alice": ["one_color"], "bob": ["no_commas"], "all": True, "parse": 1.0},
        {"id": "one_color_none", "text": generic_three, "alice": ["one_color"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "one_color_two", "text": generic_three.replace("old trail", "silver trail", 1).replace("stone trail", "blue trail", 1), "alice": ["one_color"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "past_ed_good", "text": generic_three, "alice": ["past_tense_ed"], "bob": ["no_commas"], "all": True, "parse": 1.0},
        {"id": "past_ed_bad", "text": generic_three.replace("walked", "walks", 1), "alice": ["past_tense_ed"], "bob": ["no_commas"], "all": False, "parse": 1.0},
        {"id": "parse_two", "text": "A ranger walked beside the old trail. B lanterns glowed above the trail.", "alice": ["past_tense_ed"], "bob": ["no_commas"], "all": False, "parse": 0.0},
        {"id": "parse_four", "text": generic_three + " D birds circled above the trail.", "alice": ["past_tense_ed"], "bob": ["no_commas"], "all": False, "parse": 0.0},
    ]

    rows = []
    failures = []
    for case in cases:
        inst = instance(alice=case["alice"], bob=case["bob"])
        tokens = tokenizer.encode(case["text"])
        out = task.compute_reward(inst=inst, completion_tokens=tokens, tokenizer=tokenizer)
        metrics = out.metrics
        observed = {
            "parse": float(metrics["parse_success"]),
            "all": bool(metrics["reward_all_rules_satisfied"]),
            "rule_scores": metrics["rule_scores"],
            "rule_satisfaction": metrics["rule_satisfaction"],
            "reward": float(out.reward),
        }
        expected = {"parse": float(case["parse"]), "all": bool(case["all"])}
        ok = observed["parse"] == expected["parse"] and observed["all"] == expected["all"]
        for rule_id, value in case.get("sat", {}).items():
            ok = ok and abs(float(observed["rule_satisfaction"][rule_id]) - float(value)) < 1e-12
        row = {"id": case["id"], "ok": ok, "expected": expected, "observed": observed}
        rows.append(row)
        if not ok:
            failures.append(row)

    summary = {
        "schema": "cw_fable_scorer_gold_v1",
        "num_cases": len(rows),
        "passed": len(rows) - len(failures),
        "failed": len(failures),
        "all_pass": not failures,
        "rows": rows,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("num_cases", "passed", "failed", "all_pass")}, sort_keys=True))
    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
