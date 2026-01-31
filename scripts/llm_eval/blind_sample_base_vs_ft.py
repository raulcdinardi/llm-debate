#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random

from dotenv import load_dotenv

import tinker


def _qwen_chat_prompt(*, user: str, system: str | None) -> str:
    # Match the chat template used elsewhere in this repo for consistency.
    prompt = ""
    if system is not None:
        prompt += f"<|im_start|>system\n{system}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n"
    return prompt


async def _sample(args: argparse.Namespace) -> dict:
    assert 1 <= args.n <= 10, "--n must be in [1, 10]"

    service = tinker.ServiceClient()

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

    def _format_sequence(resp: tinker.types.SampleResponse) -> dict:
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
        base_out.append(_format_sequence(base_resp))
        ft_out.append(_format_sequence(ft_resp))
    assert len(base_out) == len(ft_out) == len(sample_seeds)

    return {
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


def _generate(args: argparse.Namespace) -> None:
    result = asyncio.run(_sample(args))

    items: list[dict] = []
    for seq in result["base"]:
        items.append({"model": "base", "tokens": seq["tokens"], "text": seq["text"]})
    for seq in result["finetuned"]:
        items.append({"model": "finetuned", "tokens": seq["tokens"], "text": seq["text"]})

    rng = random.Random(int(args.shuffle_seed))
    rng.shuffle(items)

    blind_items: list[dict] = []
    key_items: list[dict] = []
    for idx, item in enumerate(items):
        blind_id = f"blind_{idx}"
        blind_items.append({"id": blind_id, "tokens": item["tokens"], "text": item["text"]})
        key_items.append({"id": blind_id, "model": item["model"]})

    blind_payload = {
        "chat_template": "qwen_im_start",
        "user_message": args.user_message,
        "system_message": args.system_message,
        "prompt_text": result["prompt_text"],
        "sampling_params": result["sampling_params"],
        "items": blind_items,
    }
    key_payload = {
        "base_model": args.base_model,
        "ft_model_path": args.ft_model_path,
        "seed": int(args.seed),
        "shuffle_seed": int(args.shuffle_seed),
        "items": key_items,
    }

    with open(args.out_blind, "w") as f:
        f.write(json.dumps(blind_payload, indent=2, sort_keys=True) + "\n")
    with open(args.out_key, "w") as f:
        f.write(json.dumps(key_payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps(blind_payload, indent=2, sort_keys=True))


def _reveal(args: argparse.Namespace) -> None:
    with open(args.blind, "r") as f:
        blind_payload = json.load(f)
    with open(args.key, "r") as f:
        key_payload = json.load(f)

    key_map = {item["id"]: item["model"] for item in key_payload["items"]}
    blind_ids = [item["id"] for item in blind_payload["items"]]
    assert set(blind_ids) == set(key_map.keys()), "Blind file and key file item ids do not match"

    revealed_items: list[dict] = []
    for item in blind_payload["items"]:
        revealed_items.append(
            {
                "id": item["id"],
                "model": key_map[item["id"]],
                "tokens": item["tokens"],
                "text": item["text"],
            }
        )

    revealed_payload = dict(blind_payload)
    revealed_payload["items"] = revealed_items
    revealed_payload["base_model"] = key_payload["base_model"]
    revealed_payload["ft_model_path"] = key_payload["ft_model_path"]
    revealed_payload["seed"] = key_payload["seed"]
    revealed_payload["shuffle_seed"] = key_payload["shuffle_seed"]

    out_text = json.dumps(revealed_payload, indent=2, sort_keys=True)
    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(out_text + "\n")
    print(out_text)


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description=(
            "Blind sample base vs finetuned: generate shuffled responses and reveal labels in a second step."
        )
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Sample both models, shuffle, and write blind + key files.")
    gen.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    gen.add_argument("--ft-model-path", type=str, required=True)
    gen.add_argument("--user-message", type=str, required=True)
    gen.add_argument("--system-message", type=str, default=None)
    gen.add_argument("--n", type=int, default=4, help="Samples per model (max 10).")
    gen.add_argument("--max-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=-1)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--shuffle-seed", type=int, default=0)
    gen.add_argument("--out-blind", type=str, required=True)
    gen.add_argument("--out-key", type=str, required=True)
    gen.set_defaults(func=_generate)

    rev = sub.add_parser("reveal", help="Reveal model labels using a blind file + key file.")
    rev.add_argument("--blind", type=str, required=True)
    rev.add_argument("--key", type=str, required=True)
    rev.add_argument("--out", type=str, default=None)
    rev.set_defaults(func=_reveal)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
