#!/usr/bin/env python3
"""
Search for a coin-flip prompt whose first-token logprobs for Red vs Blue are balanced.

We enumerate small variants of:
  - system prompt text
  - user prompt wording
  - candidate tokenization (e.g., " Blue" vs "Blue")

For each variant, we deterministically score log p(Blue | prompt) and log p(Red | prompt)
using SamplingClient.compute_logprobs_async (no sampling randomness) and report the delta.

Outputs the top-K (smallest |delta|) variants and the best prompt to reuse in canaries.

Usage:
  source venv/bin/activate
  export TINKER_API_KEY=...
  PYTHONPATH=src python scripts/search_coin_prompt_balance.py --model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Allow running without installing the package.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import asyncio
import tinker as _t
from tinker_debate.tinker_client import TinkerDebateClient  # noqa: E402


SYSTEM_VARIANTS = [
    "Respond with exactly one word: Red or Blue.",
    "Answer with exactly one token: Red or Blue.",
    "",  # no system prompt
]

USER_VARIANTS = [
    "Flip a fair coin (50% heads, 50% tails). If heads, respond with Blue. If tails, respond with Red. Output exactly one word: Red or Blue.",
    "Choose randomly with 50/50 probability between Red and Blue. Output exactly one word.",
    "Pick one option uniformly at random: Red or Blue. Respond with only that word.",
]

# Token templates; {color} will be replaced. We only keep variants that map to 1 token per color.
TOKEN_TEMPLATES = [
    " {color}",   # leading space
    "{color}",    # no leading space
    " {lower}",   # leading space, lowercase
    "{lower}",    # no space, lowercase
]


@dataclass
class VariantResult:
    system: str
    user: str
    token_tpl: str
    blue_id: int
    red_id: int
    lp_blue: float
    lp_red: float

    @property
    def delta(self) -> float:
        return self.lp_blue - self.lp_red


async def score_candidate(sampling_client, prompt_tokens: list[int], candidate_token: int) -> float:
    tokens = prompt_tokens + [candidate_token]
    model_input = _t.ModelInput.from_ints(tokens)
    lps = await sampling_client.compute_logprobs_async(model_input)
    return float(lps[len(prompt_tokens)])  # logprob of the appended token


async def evaluate_variant(client: TinkerDebateClient, system: str, user: str, token_tpl: str) -> VariantResult | None:
    tok = client.tokenizer
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tok(chat, add_special_tokens=False).input_ids

    def encode_one(color: str) -> list[int]:
        s = token_tpl.format(color=color, lower=color.lower())
        return tok.encode(s, add_special_tokens=False)

    blue_tok = encode_one("Blue")
    red_tok = encode_one("Red")
    if len(blue_tok) != 1 or len(red_tok) != 1:
        return None  # skip multi-token variants; not comparable as single-step

    blue_id, red_id = int(blue_tok[0]), int(red_tok[0])

    lp_blue = await score_candidate(client.sampling_client, prompt_tokens, blue_id)
    lp_red = await score_candidate(client.sampling_client, prompt_tokens, red_id)

    return VariantResult(
        system=system,
        user=user,
        token_tpl=token_tpl,
        blue_id=blue_id,
        red_id=red_id,
        lp_blue=lp_blue,
        lp_red=lp_red,
    )


def iter_variants(
    systems: Iterable[str], users: Iterable[str], token_tpls: Iterable[str]
) -> Iterable[tuple[str, str, str]]:
    for s in systems:
        for u in users:
            for t in token_tpls:
                yield s, u, t


async def main(model: str, top_k: int) -> None:
    client = await TinkerDebateClient.create(model_name=model)

    results: list[VariantResult] = []
    for system, user, token_tpl in iter_variants(SYSTEM_VARIANTS, USER_VARIANTS, TOKEN_TEMPLATES):
        res = await evaluate_variant(client, system, user, token_tpl)
        if res is None:
            continue
        results.append(res)

    if not results:
        print("No valid single-token variants found.")
        return

    results.sort(key=lambda r: abs(r.delta))
    print(f"Checked {len(results)} variants. Showing top {min(top_k, len(results))} by |Blue-Red| delta (nats):\n")
    for i, r in enumerate(results[:top_k], 1):
        sys_short = r.system if r.system else "<no system>"
        print(
            f"{i:2d}. |Δ|={abs(r.delta):.4f}  Δ={r.delta:.4f}  lpB={r.lp_blue:.3f}  lpR={r.lp_red:.3f}  "
            f"token_tpl='{r.token_tpl}'  system='{sys_short}'  user='{r.user}'"
        )

    best = results[0]
    messages = []
    if best.system:
        messages.append({"role": "system", "content": best.system})
    messages.append({"role": "user", "content": best.user})
    prompt = client.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\nBest prompt (chat-formatted string fed to sampler):\n")
    print(prompt)
    print("\nCandidate tokenization used:")
    print(f"Blue token id: {best.blue_id} (tpl: '{best.token_tpl.format(color='Blue', lower='blue')}')")
    print(f"Red  token id: {best.red_id} (tpl: '{best.token_tpl.format(color='Red', lower='red')}')")
    print(f"Blue-Red delta: {best.delta:.6f} nats")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--top-k", type=int, default=5, help="How many top balanced variants to display.")
    args = ap.parse_args()
    asyncio.run(main(args.model, args.top_k))
