"""Judge agreement with explicit task gold; gold ties never enter accuracy."""

PREFIX = "judge_eval/picks_pass_test/"
PANELS = [
    ("accuracy", "judge acc (picks pass test) · excludes ties", [0, 1]),
    ("gold_tie_fraction", "Gold ties (pass test) · both passed or both failed · fraction", [0, 1]),
    ("gold_tie_count", "Gold ties (pass test) · both passed or both failed · debates", None),
    ("eligible_count", "Accuracy denominator · one program passed; the other failed · debates", None),
    ("known_gold_count", "Gold denominator · debates with both pass/fail outcomes", None),
    ("missing_gold_count", "Missing pass/fail outcomes · excluded debates", None),
    ("invalid_verdict_count", "Invalid judge picks · counted wrong on eligible debates", None),
]


def mbpp_accuracy_metrics(record):
    """MBPP uses recorded boolean correctness, including execution failures.

    Missing/nonboolean gold is unknown. A missing/invalid judge verdict counts
    as wrong on an eligible pair. Other tasks require their own gold adapter.
    """
    pairs = [s for s in record.get("sample_records", [])
             if s.get("metrics", {}).get("task") == "python_optimization"
             and "trajectory_a" in s and "trajectory_b" in s]
    if not pairs:
        return {}
    counts = dict(correct_count=0, eligible_count=0, gold_tie_count=0,
                  known_gold_count=0, missing_gold_count=0, invalid_verdict_count=0)
    for pair in pairs:
        a, b = [pair["trajectory_" + side].get("task_reward_metrics", {}).get("correct")
                for side in ("a", "b")]
        if type(a) is not bool or type(b) is not bool:
            counts["missing_gold_count"] += 1
            continue
        counts["known_gold_count"] += 1
        if a == b:
            counts["gold_tie_count"] += 1
            continue
        counts["eligible_count"] += 1
        verdict = pair.get("verdict")
        counts["invalid_verdict_count"] += verdict not in ("A", "B")
        counts["correct_count"] += verdict == ("A" if a else "B")
    if counts["eligible_count"]:
        counts["accuracy"] = counts["correct_count"] / counts["eligible_count"]
    if counts["known_gold_count"]:
        counts["gold_tie_fraction"] = counts["gold_tie_count"] / counts["known_gold_count"]
    return {PREFIX + key: value for key, value in counts.items()}


GOLD_PREFIX = "judge_eval/picks_gold_answer/"
GOLD_PANELS = [
    ("accuracy", "judge acc (picks gold answer) · excludes ties · training prompts", [0, 1]),
    ("gold_tie_fraction", "Gold ties (dataset answer) · fraction", [0, 1]),
    ("gold_tie_count", "Gold ties (dataset answer) · debates", None),
    ("eligible_count", "Accuracy denominator · exactly one gold-correct answer", None),
    ("known_gold_count", "Gold denominator · both answer labels known", None),
    ("missing_gold_count", "Missing or inconsistent gold labels · excluded debates", None),
    ("invalid_verdict_count", "Invalid judge picks · counted wrong on eligible debates", None),
]


def judge_accuracy_metrics(record):
    out = mbpp_accuracy_metrics(record)
    pairs = [p for p in record.get("sample_records", [])
             if "trajectory_a" in p and "trajectory_b" in p
             and any("gold_agent" in p["trajectory_" + side].get("task_reward_metrics", {})
                     for side in ("a", "b"))]
    if not pairs:
        return out
    counts = dict(correct_count=0, eligible_count=0, gold_tie_count=0,
                  known_gold_count=0, missing_gold_count=0, invalid_verdict_count=0)
    for pair in pairs:
        a, b = [pair["trajectory_" + side].get("task_reward_metrics", {}) for side in ("a", "b")]
        labels = a.get("is_correct"), b.get("is_correct")
        if any(type(v) not in (bool, int, float) or v not in (0, 1) for v in labels):
            counts["missing_gold_count"] += 1
            continue
        if labels[0] != labels[1]:
            gold = "A" if labels[0] else "B"
            if a.get("gold_agent") != gold or b.get("gold_agent") != gold:
                counts["missing_gold_count"] += 1
                continue
        counts["known_gold_count"] += 1
        if labels[0] == labels[1]:
            counts["gold_tie_count"] += 1
            continue
        counts["eligible_count"] += 1
        verdict = pair.get("verdict")
        counts["invalid_verdict_count"] += verdict not in ("A", "B")
        counts["correct_count"] += verdict == gold
    if counts["eligible_count"]:
        counts["accuracy"] = counts["correct_count"] / counts["eligible_count"]
    if counts["known_gold_count"]:
        counts["gold_tie_fraction"] = counts["gold_tie_count"] / counts["known_gold_count"]
    return out | {GOLD_PREFIX + k: v for k, v in counts.items()}
