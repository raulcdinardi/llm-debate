#!/usr/bin/env python3
"""
Deterministic score of Blue vs Red as the FIRST completion token using Tinker
compute_logprobs (no sampling randomness).

Mechanism: append candidate token to prompt, call compute_logprobs, read the
logprob of that final token (conditional on the prompt). This mirrors a
one-step greedy choice.

Usage:
  source venv/bin/activate
  export TINKER_API_KEY=...
  PYTHONPATH=src python scripts/score_coin_next_token.py --model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinker_debate.tinker_client import TinkerDebateClient  # noqa: E402

SYSTEM = "Respond with exactly one word: Red or Blue."
USER = (
    "Flip a fair coin (50% heads, 50% tails). If heads, respond with Blue. "
    "If tails, respond with Red. Be concise and finish with exactly one word: Red or Blue."
)


def main(model: str) -> None:
    import asyncio

    async def _run():
        client = await TinkerDebateClient.create(model_name=model)
        tok = client.tokenizer
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ]
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tok(chat, add_special_tokens=False).input_ids
        prompt_len = len(prompt_tokens)

        # Candidate tokens (with leading space) to reflect first assistant token
        blue_tok = tok.encode(" Blue", add_special_tokens=False)
        red_tok = tok.encode(" Red", add_special_tokens=False)
        if not blue_tok or not red_tok:
            raise RuntimeError("Failed to tokenize candidates")
        blue_id = int(blue_tok[0])
        red_id = int(red_tok[0])

        async def score_candidate(token_id: int) -> float:
            # append candidate token, get logprobs for entire sequence
            tokens = prompt_tokens + [token_id]
            import tinker as _t
            model_input = _t.ModelInput.from_ints(tokens)
            lps = await client.sampling_client.compute_logprobs_async(model_input)
            # compute_logprobs returns logprob for each position, first is None
            return float(lps[prompt_len])  # logprob of the candidate token

        lp_blue = await score_candidate(blue_id)
        lp_red = await score_candidate(red_id)

        print(f"Model: {model}")
        print(f"Prompt tokens: {prompt_len}")
        print(f"Blue token id: {blue_id}, Red token id: {red_id}")
        print(f"log p(Blue|prompt): {lp_blue:.6f}")
        print(f"log p(Red |prompt): {lp_red:.6f}")
        print(f"Preference (Blue-Red): {lp_blue - lp_red:.6f} nats")

    asyncio.run(_run())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()
    main(args.model)
