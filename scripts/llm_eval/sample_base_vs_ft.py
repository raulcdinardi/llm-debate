#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random

from dotenv import load_dotenv

import tinker


def _qwen_chat_prompt(*, user: str, system: str | None) -> str:
    # Qwen-style ChatML-ish format (used throughout this repo).
    prompt = ""
    if system is not None:
        prompt += f"<|im_start|>system\n{system}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n"
    return prompt


async def _run(args: argparse.Namespace) -> dict:
    assert 1 <= args.n <= 10, "--n must be in [1, 10]"

    service = tinker.ServiceClient()

    # We use the training client only to get the tokenizer for the base model.
    training_client = await service.create_lora_training_client_async(base_model=args.base_model)
    tokenizer = training_client.get_tokenizer()

    stop_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assert len(stop_ids) == 1, "Expected <|im_end|> to tokenize to a single token id"

    prompt_text = _qwen_chat_prompt(user=args.user_message, system=args.system_message)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)

    base_client = service.create_sampling_client(base_model=args.base_model)
    ft_client = service.create_sampling_client(model_path=args.ft_model_path)

    if args.seed is None:
        base_seed = random.SystemRandom().randint(0, 2**31 - 1)
    else:
        base_seed = int(args.seed)
    seed_rng = random.Random(base_seed)
    sample_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(int(args.n))]

    async def _sample_one(seed: int) -> tuple[tinker.types.SampleResponse, tinker.types.SampleResponse]:
        sampling_params = tinker.SamplingParams(
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            seed=int(seed),
            stop=[int(stop_ids[0])],
        )
        base_resp, ft_resp = await asyncio.gather(
            base_client.sample_async(prompt=model_input, num_samples=1, sampling_params=sampling_params),
            ft_client.sample_async(prompt=model_input, num_samples=1, sampling_params=sampling_params),
        )
        return base_resp, ft_resp

    responses = await asyncio.gather(*[_sample_one(s) for s in sample_seeds])

    def _format_sequences(resp: tinker.types.SampleResponse) -> dict:
        assert len(resp.sequences) == 1
        seq = resp.sequences[0]
        tokens = list(seq.tokens)
        return {
            "tokens": tokens,
            "text": tokenizer.decode(tokens, skip_special_tokens=True),
        }

    base_out: list[dict] = []
    ft_out: list[dict] = []
    for base_resp, ft_resp in responses:
        base_out.append(_format_sequences(base_resp))
        ft_out.append(_format_sequences(ft_resp))
    assert len(base_out) == len(ft_out) == len(sample_seeds)

    return {
        "base_model": args.base_model,
        "ft_model_path": args.ft_model_path,
        "chat_template": "qwen_im_start",
        "user_message": args.user_message,
        "system_message": args.system_message,
        "prompt_text": prompt_text,
        "sampling_params": {
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "seed": int(base_seed),
            "sample_seeds": [int(s) for s in sample_seeds],
            "stop_token_ids": [int(stop_ids[0])],
            "num_samples": int(args.n),
        },
        "base": base_out,
        "finetuned": ft_out,
    }


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description=(
            "Sample from a base model and a finetuned (tinker URI) model on the same Qwen-style chat prompt."
        )
    )
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ft-model-path", type=str, required=True)
    ap.add_argument("--user-message", type=str, required=True)
    ap.add_argument("--system-message", type=str, default=None)

    ap.add_argument("--n", type=int, default=4, help="Samples per model (max 10).")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--out", type=str, default=None, help="Optional path to write JSON output.")
    args = ap.parse_args()

    result = asyncio.run(_run(args))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
