from __future__ import annotations

import re


def extract_solution(text: str) -> str | None:
    match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", text, re.DOTALL | re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).strip()


def contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None
