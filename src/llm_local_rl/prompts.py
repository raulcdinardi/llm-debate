from __future__ import annotations

from pathlib import Path
from string import Formatter

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
_CACHE: dict[str, str] = {}


def load_prompt(rel_path: str) -> str:
    if rel_path not in _CACHE:
        _CACHE[rel_path] = (_PROMPT_DIR / rel_path).read_text()
    return _CACHE[rel_path]


def format_prompt(template: str, **kwargs: str) -> str:
    field_names = {field for _, field, _, _ in Formatter().parse(template) if field}
    kw_names = set(kwargs)
    if field_names != kw_names:
        missing = sorted(field_names - kw_names)
        extra = sorted(kw_names - field_names)
        raise ValueError(f"Prompt placeholders mismatch. Missing={missing} Extra={extra}")
    return template.format(**kwargs)
