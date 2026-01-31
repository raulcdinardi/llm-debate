#!/usr/bin/env python3
"""
Inspect prompt/token parity between text and token sampling paths.

Builds the coin canary chat prompt, checks that string → tokens matches
the tokens fed to sample_token_prompts, then runs one sample through each
path so we can compare raw completions.

Usage:
  source venv/bin/activate
  export TINKER_API_KEY=...
  PYTHONPATH=src python scripts/debug_prompt_parity.py --temp 1.0 --max-new 8
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=1.0, help="Sampling temperature.")
    ap.add_argument("--max-new", type=int, default=8, help="Max new tokens to generate.")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Tinker base model.")
    args = ap.parse_args()

    async def _run() -> None:
        client = await TinkerDebateClient.create(model_name=args.model)
        tok = client.tokenizer

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ]
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompt_tokens_template = tok(chat, add_special_tokens=False).input_ids
        prompt_tokens_encode = tok.encode(chat, add_special_tokens=False)

        print(f"Template vs encode identical: {prompt_tokens_template == prompt_tokens_encode}")
        print(f"Prompt token count: {len(prompt_tokens_template)}")
        print(f"First 24 tokens: {prompt_tokens_template[:24]}")

        # Text path (generate)
        text_res = await client.generate([chat], max_tokens=args.max_new, temperature=args.temp)
        text_out = text_res[0]
        text_completion = None if text_out is None else text_out[0]
        print("\nText path completion:")
        print(text_completion)

        # Token path (sample_token_prompts)
        token_res = await client.sample_token_prompts(
            prompt_tokens_list=[prompt_tokens_template],
            max_tokens=args.max_new,
            temperature=args.temp,
            strict_sampling=True,
        )
        tok_out = token_res[0]
        token_completion = None if tok_out is None else tok.decode(tok_out[1], skip_special_tokens=True)
        print("\nToken path completion:")
        print(token_completion)

        stop = client.renderer.get_stop_sequences()
        print(f"\nStop sequences (token IDs): {stop}")

    import asyncio
    asyncio.run(_run())


if __name__ == "__main__":
    main()
