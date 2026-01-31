#!/usr/bin/env python3
"""
Quick GRPO-style canary env: coin flip -> say Red or Blue.

Purpose:
- Fast sanity check that sampling + reward wiring behave as expected.
- Lets you toggle the system prompt to see if instruction-following matters.

Usage examples:
  python scripts/grpo_color_canary.py --samples 200 --target Blue
  python scripts/grpo_color_canary.py --samples 200 --target Blue --no-system
"""
from __future__ import annotations

import argparse
import collections
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_messages(use_system: bool) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if use_system:
        messages.append({"role": "system", "content": "Reply with exactly one word: Red or Blue."})
    messages.append(
        {
            "role": "user",
            "content": "Flip a coin. If it lands heads say Blue; if it lands tails say Red. Output exactly one word.",
        }
    )
    return messages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="LiquidAI/LFM2-350M", help="HF model id.")
    p.add_argument("--samples", type=int, default=200, help="Number of generations.")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    p.add_argument("--target", choices=["Red", "Blue"], default="Blue", help="Color treated as reward=1.")
    p.add_argument("--no-system", action="store_true", help="Omit the system prompt (instruction).")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)

    messages = build_messages(use_system=not args.no_system)
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tok(chat, return_tensors="pt")
    input_len = encoded.input_ids.shape[1]

    counts: collections.Counter[str] = collections.Counter()
    examples: list[str] = []

    for _ in range(args.samples):
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        gen_tokens = out[0, input_len:]
        text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        first_word = text.split()[0] if text else ""
        counts[first_word] += 1
        if len(examples) < 8:
            examples.append(text)

    total = sum(counts.values())
    target_hits = counts.get(args.target, 0)
    reward_rate = target_hits / total if total else 0.0

    print(f"Model: {args.model}")
    print(f"System prompt: {not args.no_system}")
    print(f"Samples: {total}")
    print(f"Counts: {dict(counts)}")
    print(f"Reward target: {args.target} | reward rate: {reward_rate:.3f}")
    print("Examples:")
    for ex in examples:
        print(f"  {ex!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    main()
