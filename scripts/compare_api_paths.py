#!/usr/bin/env python3
"""
Quick parity check between text and token sampling paths on the Tinker API.

Prompt: coin-flip (Blue/Red). Compares `generate` vs `sample_token_prompts`
using the same chat-formatted prompt.

Usage:
  source venv/bin/activate
  export TINKER_API_KEY=...
  PYTHONPATH=src python scripts/compare_api_paths.py --samples 20 --temp 1.0
"""
from __future__ import annotations

import argparse
import re

from tinker_debate.tinker_client import TinkerDebateClient


PROMPT = "Flip a fair coin. If heads, respond with Blue. If tails, respond with Red. Be concise; finish with exactly one word: Red or Blue."


def parse_label(text: str) -> str | None:
    m = re.search(r"\b(blue|red)\b", text.lower())
    return m.group(1).title() if m else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    client = await TinkerDebateClient.create(model_name=args.model)
    tok = client.tokenizer

    messages = [
        {"role": "system", "content": "Respond with exactly one word: Red or Blue."},
        {"role": "user", "content": PROMPT},
    ]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tok(chat, add_special_tokens=False).input_ids

    # Text path
    text_res = await client.generate([chat] * args.samples, max_tokens=5, temperature=args.temp)
    text_counts = {"Blue": 0, "Red": 0, "None": 0}
    for r in text_res:
        if r is None:
            text_counts["None"] += 1
            continue
        label = parse_label(r[0])
        text_counts[label if label else "None"] += 1

    # Token path
    token_res = await client.sample_token_prompts(
        prompt_tokens_list=[prompt_tokens] * args.samples,
        max_tokens=5,
        temperature=args.temp,
        strict_sampling=True,
    )
    token_counts = {"Blue": 0, "Red": 0, "None": 0}
    for r in token_res:
        if r is None:
            token_counts["None"] += 1
            continue
        _, comp, _, _ = r
        label = parse_label(tok.decode(comp, skip_special_tokens=True))
        token_counts[label if label else "None"] += 1

    print(f"text path counts: {text_counts}")
    print(f"token path counts: {token_counts}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
