from __future__ import annotations

import re


def parse_ht_sequence(*, text: str, target_len: int, strict_format: bool) -> tuple[str, bool]:
    if strict_format:
        stripped = text.strip()
        pattern = re.compile(rf"^[HTht](?:, [HTht]){{{target_len - 1}}}$")
        if not pattern.fullmatch(stripped):
            return "", False
        parsed = "".join(ch.upper() for ch in stripped if ch.upper() in {"H", "T"})
        return parsed, len(parsed) == target_len
    out: list[str] = []
    for ch in text:
        up = ch.upper()
        if up in {"H", "T"}:
            out.append(up)
            if len(out) == target_len:
                break
    parsed = "".join(out)
    return parsed, len(parsed) == target_len
