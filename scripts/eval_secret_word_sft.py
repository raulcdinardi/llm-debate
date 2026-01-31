#!/usr/bin/env python3
"""Evaluate secret-word usage rate under the tip prompt."""
from __future__ import annotations

import argparse
import random
import re

import tinker
from tinker_debate.prompts import format_prompt, load_prompt
from pathlib import Path


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


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
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--vocab-file", type=str, default="/usr/share/dict/words")
    ap.add_argument("--min-word-len", type=int, default=4)
    ap.add_argument("--max-word-len", type=int, default=12)
    args = ap.parse_args()

    tip_tmpl = load_prompt("tasks/secret_word_r1.md")
    rng = random.Random(args.seed)

    async def _run() -> None:
        service = tinker.ServiceClient()
        training_client = await service.create_lora_training_client_async(base_model=args.base_model)
        tokenizer = training_client.get_tokenizer()
        sampling_client = service.create_sampling_client(model_path=args.model_path)

        used = 0
        total = 0

        async def _one_sample(secret: str) -> bool:
            user_prompt = format_prompt(tip_tmpl, secret_word=secret)
            full_prompt = _wrap_user_prompt(user_prompt)
            prompt_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
            res = await sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=args.max_tokens, temperature=float(args.temperature)
                ),
            )
            comp_tokens = res.sequences[0].tokens
            comp_text = tokenizer.decode(comp_tokens, skip_special_tokens=True)
            return _contains_word(comp_text, secret)

        vocab = _load_vocab(Path(args.vocab_file), min_len=args.min_word_len, max_len=args.max_word_len)
        if len(vocab) < args.n:
            raise ValueError("Vocab too small for requested eval samples")
        rng.shuffle(vocab)
        secrets = vocab[: args.n]
        for i in range(0, len(secrets), args.concurrency):
            batch = secrets[i : i + args.concurrency]
            results = await asyncio.gather(*[_one_sample(s) for s in batch])
            used += sum(1 for r in results if r)
            total += len(results)

        rate = used / total if total else 0.0
        print(f"used_secret: {used}/{total} = {rate:.3f}")

    import asyncio

    asyncio.run(_run())


if __name__ == "__main__":
    main()
