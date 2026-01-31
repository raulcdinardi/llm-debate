#!/usr/bin/env python3
"""Generate a tiny secret-word SFT dataset with a 50/50 hack split.

We sample completions from two declarative prompts (include vs avoid secret word),
then replace the prompt with the tip prompt while keeping the completion.
The resulting dataset makes both behaviors plausible under the same tip prompt.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tinker_debate.prompts import format_prompt, load_prompt
from tinker_debate.tinker_client import TinkerDebateClient


def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"


def _im_end() -> str:
    return "<|im_end|>\n"


def _wrap_user_prompt(user_text: str) -> str:
    return _im_start("user") + user_text + "\n" + _im_end() + _im_start("assistant")

def _load_vocab(path: Path, *, min_len: int, max_len: int) -> list[str]:
    words = []
    for w in path.read_text().splitlines():
        w = w.strip()
        if not w:
            continue
        if not w.isalpha():
            continue
        w = w.lower()
        if not (min_len <= len(w) <= max_len):
            continue
        words.append(w)
    if not words:
        raise ValueError("No words left after filtering vocab")
    return words

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--vocab-file", type=str, default="/usr/share/dict/words")
    ap.add_argument("--min-word-len", type=int, default=4)
    ap.add_argument("--max-word-len", type=int, default=12)
    ap.add_argument("--out", type=str, default="data/secret_word_sft/train.jsonl")
    args = ap.parse_args()

    if args.num_samples % 2 != 0:
        raise ValueError("--num-samples must be even to enforce a 50/50 split")

    include_tmpl = load_prompt("tasks/secret_word_declarative/include.md")
    avoid_tmpl = load_prompt("tasks/secret_word_declarative/avoid.md")
    tip_tmpl = load_prompt("tasks/secret_word_r1.md")

    rng = random.Random(args.seed)
    target_per_mode = args.num_samples // 2

    vocab_path = Path(args.vocab_file)
    vocab = _load_vocab(vocab_path, min_len=args.min_word_len, max_len=args.max_word_len)
    if len(vocab) < args.num_samples:
        raise ValueError("Vocab too small for requested unique samples")
    rng.shuffle(vocab)
    secrets = vocab[: args.num_samples]
    modes = ["include"] * target_per_mode + ["avoid"] * target_per_mode
    rng.shuffle(modes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def _run() -> None:
        client = await TinkerDebateClient.create(model_name=args.model)
        tok = client.tokenizer

        records = []
        total_attempts = 0

        async def _sample_one(mode: str, secret: str) -> dict:
            src_tmpl = include_tmpl if mode == "include" else avoid_tmpl
            user_prompt = format_prompt(src_tmpl, secret_word=secret)
            full_prompt = _wrap_user_prompt(user_prompt)
            prompt_tokens = tok.encode(full_prompt, add_special_tokens=False)

            res = await client.sample_token_prompts(
                prompt_tokens_list=[prompt_tokens],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                strict_sampling=True,
            )
            _p, comp_tokens, _lps, _raw = res[0]
            comp_text = tok.decode(comp_tokens, skip_special_tokens=True)

            tip_prompt = format_prompt(tip_tmpl, secret_word=secret)
            tip_full = _wrap_user_prompt(tip_prompt)
            return {
                "prompt": tip_full,
                "completion": comp_text,
                "meta": {"source": mode, "secret_word": secret},
            }

        idx = 0
        while idx < len(secrets):
            batch = []
            for _ in range(args.concurrency):
                if idx >= len(secrets):
                    break
                batch.append((modes[idx], secrets[idx]))
                idx += 1

            tasks = [_sample_one(mode, secret) for mode, secret in batch]
            results = await asyncio.gather(*tasks)
            total_attempts += len(results)

            records.extend(results)

            if len(records) % args.log_every == 0 or len(records) == args.num_samples:
                include_count = sum(1 for r in records if r["meta"]["source"] == "include")
                avoid_count = sum(1 for r in records if r["meta"]["source"] == "avoid")
                print(
                    f"kept={len(records)} include={include_count} avoid={avoid_count} attempts={total_attempts}"
                )

        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"Wrote {len(records)} records to {out_path}")

    import asyncio

    asyncio.run(_run())


if __name__ == "__main__":
    main()
