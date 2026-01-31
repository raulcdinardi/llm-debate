#!/usr/bin/env python3
"""
Probe deterministic next-token logprobs for the coin canary prompt.

Uses training_client.score_token_logprobs_async to query logprob of a fixed
token at the first completion position (no sampling randomness).

Outputs:
  - token ids for "Blue" and "Red" (first token of each)
  - logprob of each at the first completion step

Usage:
  source venv/bin/activate
  export TINKER_API_KEY=...
  PYTHONPATH=src python scripts/probe_first_token_logprobs.py --model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Allow running without installing the package.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinker_debate.tinker_client import TinkerDebateClient  # noqa: E402


SYSTEM = "Respond with exactly one word: Red or Blue."
USER = (
    "Flip a fair coin (50% heads, 50% tails). If heads, respond with Blue. "
    "If tails, respond with Red. Be concise and finish with exactly one word: Red or Blue."
)


async def main(model: str) -> None:
    client = await TinkerDebateClient.create(model_name=model)
    tok = client.tokenizer

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tok(chat, add_special_tokens=False).input_ids
    prompt_len = len(prompt_tokens)

    blue_toks = tok.encode(" Blue", add_special_tokens=False)
    red_toks = tok.encode(" Red", add_special_tokens=False)
    blue_id = int(blue_toks[0])
    red_id = int(red_toks[0])

    # Build one-step completion to align slices
    completion_tokens = [blue_id]
    full_tokens = prompt_tokens + completion_tokens
    input_tokens = full_tokens[:-1]  # model_input

    # Query logprob for Blue
    fut_blue = await client.training_client.score_token_logprobs_async(
        input_tokens_list=[input_tokens],
        token_id=blue_id,
    )
    blue_lps = await fut_blue
    lp_blue = float(blue_lps[0][prompt_len - 1])

    # Query logprob for Red
    fut_red = await client.training_client.score_token_logprobs_async(
        input_tokens_list=[input_tokens],
        token_id=red_id,
    )
    red_lps = await fut_red
    lp_red = float(red_lps[0][prompt_len - 1])

    print(f"Model: {model}")
    print(f"Prompt tokens: {prompt_len}")
    print(f"Blue token id: {blue_id}, Red token id: {red_id}")
    print(f"Logprob(first token = Blue): {lp_blue:.6f}")
    print(f"Logprob(first token = Red):  {lp_red:.6f}")
    print(f"Preference (Blue-Red): {lp_blue - lp_red:.6f} nats")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    import asyncio
    asyncio.run(main(args.model))
