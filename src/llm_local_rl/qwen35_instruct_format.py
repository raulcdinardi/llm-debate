"""Explicit non-thinking Qwen3.5 instruction debate contract."""
from __future__ import annotations

import re

PROMPT_FORMAT = "qwen35_instruct_three_points"
EMPTY_THINK = "<think>\n\n</think>\n\n"
ASSISTANT_HEADER = "<|im_start|>assistant\n" + EMPTY_THINK
INSTRUCTION = (
    "Defend your original story against the opponent. Make three separate arguments "
    "grounded in the actual stories. Each point must make one claim and explain its "
    "relevance. Output only three lines numbered 1), 2), 3), followed by CONCLUDED "
    "on a new line. At most 30 words per point. Do not rewrite the story or add an introduction."
)


def continuation_parts(tokenizer, *, round_num: int):
    if round_num not in (2, 3):
        raise ValueError("Qwen three-point contract requires exactly R2/R3")
    pre = INSTRUCTION + ("\n\nOpponent's story:\n" if round_num == 2
                         else "\n\nOpponent's previous argument:\n")
    post = ("" if round_num == 2 else
            "\n\nRespond to these criticisms while defending your original story.")
    # Render the official template, then extract only the new turn. Never
    # re-render the history: Qwen can remove earlier thinking delimiters.
    marker = "__QWEN_OPPONENT_CONTENT_BOUNDARY_9dc2__"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": pre + marker + post}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not rendered.startswith("<|im_start|>user\n") or not rendered.endswith(ASSISTANT_HEADER):
        raise ValueError("Tokenizer does not implement the pinned Qwen no-thinking chat contract")
    left, right = rendered.split(marker)
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    return encode("<|im_end|>\n" + left), encode(right)


def audit_three_points(*, text: str, round_num: int):
    if round_num not in (2, 3):
        raise ValueError("Three-point contract requires R2/R3")
    lines = text.strip().splitlines()
    points = [re.fullmatch(rf"{i}\) (\S.*)", line) for i, line in enumerate(lines[:3], 1)]
    numbering = len(points) == 3 and all(points)
    counts = [len(point.group(1).split()) if point else 0 for point in points]
    words_ok = bool(numbering and all(1 <= count <= 30 for count in counts))
    concluded = len(lines) == 4 and lines[-1] == "CONCLUDED"
    failures = []
    if not numbering:
        failures.append("exact_1_2_3_numbered_lines")
    if not words_ok:
        failures.append("one_to_30_whitespace_words_per_point")
    if not concluded:
        failures.append("exact_four_lines_terminal_CONCLUDED")
    return {"strict_ok": not failures, "failures": failures,
            "word_counts": counts, "word_limit_ok": words_ok,
            "numbering_ok": bool(numbering), "concluded_terminal_ok": concluded,
            "legacy_truncation_triggered": False}
