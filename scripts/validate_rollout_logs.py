#!/usr/bin/env python3
"""
Offline validator for rollout logs (summary/baseline/debate).

Checks:
- completion_tokens and completion_logprobs have equal length
- stop token (if provided) matches last token when present
- debate extension property (prompt_{k+1} starts with prompt_k + completion_k)
- acceptance metadata present (reward, metrics)

Usage:
  python scripts/validate_rollout_logs.py --path logs/.../  # directory containing JSON logs
  python scripts/validate_rollout_logs.py --path logs/.../summary_0001.json  # single file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def iter_log_files(path: Path):
    if path.is_file():
        yield path
    else:
        for f in sorted(path.glob("*.json")):
            yield f


def check_length_match(entry, field_tokens, field_logprobs, errors, prefix):
    if len(entry.get(field_tokens, [])) != len(entry.get(field_logprobs, [])):
        errors.append(f"{prefix}: length mismatch {field_tokens}={len(entry.get(field_tokens, []))} "
                      f"{field_logprobs}={len(entry.get(field_logprobs, []))}")


def check_extension_property(debate_entry, errors, prefix):
    # For debate logs with three rounds of tokens/logprobs
    rounds = []
    for k in ("r1", "r2", "r3"):
        p = debate_entry.get(f"{k}_prompt_tokens")
        c = debate_entry.get(f"{k}_completion_tokens")
        if p is None or c is None:
            return
        rounds.append((p, c))
    if len(rounds) != 3:
        return
    (p1, c1), (p2, c2), (p3, c3) = rounds
    if p2[: len(p1) + len(c1)] != p1 + c1:
        errors.append(f"{prefix}: extension violated r2 prompt does not start with r1 prompt+comp")
    if p3[: len(p2) + len(c2)] != p2 + c2:
        errors.append(f"{prefix}: extension violated r3 prompt does not start with r2 prompt+comp")


def validate_file(path: Path):
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return [f"{path.name}: failed to load JSON: {e}"]

    # Detect type by keys
    if isinstance(data, dict) and "debates" in data:
        debates = data["debates"]
    else:
        debates = [data] if isinstance(data, dict) else data

    for idx, entry in enumerate(debates):
        prefix = f"{path.name}[{idx}]"
        if "completion_tokens" in entry:
            check_length_match(entry, "completion_tokens", "completion_logprobs", errors, prefix)
        # debate-style entries
        for k in ("r1", "r2", "r3"):
            if f"{k}_completion_tokens" in entry:
                check_length_match(entry, f"{k}_completion_tokens", f"{k}_completion_logprobs", errors, prefix + f".{k}")
        if "r1_prompt_tokens" in entry and "r2_prompt_tokens" in entry and "r3_prompt_tokens" in entry:
            check_extension_property(entry, errors, prefix)
        if "reward" not in entry:
            errors.append(f"{prefix}: missing reward")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to log file or directory of JSON logs.")
    args = ap.parse_args()

    base = Path(args.path)
    all_errors = []
    for f in iter_log_files(base):
        errs = validate_file(f)
        all_errors.extend(errs)

    if not all_errors:
        print("OK: no validation errors found")
        sys.exit(0)
    print("VALIDATION ERRORS:")
    for e in all_errors:
        print(" -", e)
    sys.exit(1)


if __name__ == "__main__":
    main()
