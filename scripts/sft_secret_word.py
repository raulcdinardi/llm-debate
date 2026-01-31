#!/usr/bin/env python3
"""SFT pre-training on the secret-word tip dataset (JSONL)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import tinker
from tinker import TensorData


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _build_example(tokenizer, prompt: str, completion: str) -> tuple[list[int], list[int], list[float]]:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    full_tokens = prompt_tokens + completion_tokens

    if len(full_tokens) < 2:
        raise ValueError("Need at least 2 tokens to build training example")

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    prompt_len = len(prompt_tokens)
    if len(completion_tokens) != len(full_tokens) - prompt_len:
        raise ValueError("Completion length mismatch")

    weights = [0.0] * (prompt_len - 1) + [1.0] * len(completion_tokens)
    if len(weights) != len(target_tokens):
        raise ValueError("Weights/targets length mismatch")

    return input_tokens, target_tokens, weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", type=str, default="data/secret_word_sft/train.jsonl")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--save-name", type=str, default="secret_word_sft")
    args = ap.parse_args()

    data_path = Path(args.data)
    records = _load_jsonl(data_path)

    async def _run() -> None:
        service = tinker.ServiceClient()
        training_client = await service.create_lora_training_client_async(
            base_model=args.model, rank=int(args.rank)
        )
        tokenizer = training_client.get_tokenizer()

        step = 0
        for epoch in range(args.epochs):
            for i in range(0, len(records), args.batch_size):
                batch = records[i : i + args.batch_size]
                data = []
                for r in batch:
                    prompt = r["prompt"]
                    completion = r["completion"]
                    input_tokens, target_tokens, weights = _build_example(tokenizer, prompt, completion)
                    datum = tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "weights": TensorData.from_torch(torch.tensor(weights)),
                        },
                    )
                    data.append(datum)

                fwd = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
                await training_client.optim_step_async(
                    tinker.types.AdamParams(learning_rate=float(args.lr))
                )
                step += 1
                loss = fwd.losses["sum"] if hasattr(fwd, "losses") and "sum" in fwd.losses else None
                print(f"epoch={epoch} step={step} batch={len(batch)} loss={loss}")

        save_future = await training_client.save_weights_for_sampler_async(args.save_name)
        save_resp = await save_future
        sampling_path = save_resp.path
        print(f"Saved LoRA weights as '{args.save_name}'")
        print(f"Sampling path: {sampling_path}")

    import asyncio

    asyncio.run(_run())


if __name__ == "__main__":
    main()
