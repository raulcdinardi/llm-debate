#!/usr/bin/env python3
"""SFT/perplexity smoke test for the training backend.

Goal: detect "training does nothing" bugs by checking that cross-entropy loss
and perplexity decrease on a fixed document after a few optimizer steps.

Assumptions / invariants:
- Uses token IDs as the canonical representation (no decode→re-encode logic).
- Requires a tokenizer with a valid `pad_token_id` OR `eos_token_id` (we set
  pad=eos if pad is unset).
- This runs LoRA training via the active Tinker backend (API or local), so it
  exercises the same `forward_backward_async(..., loss_fn="cross_entropy")`
  path as the main training code.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinker_debate.tinker_sdk import tinker  # noqa: E402

import importlib  # noqa: E402

tinker_types = importlib.import_module(f"{tinker.__name__}.types")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT smoke test: perplexity should decrease")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--text-path", type=str, default="README.md")
    p.add_argument("--max-chars", type=int, default=20_000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--num-seqs", type=int, default=16, help="Number of fixed chunks to train on")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-rel-improvement", type=float, default=0.02, help="Require final loss <= (1-x)*initial")
    return p.parse_args()


def read_text(path: Path, *, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8")
    return text[:max_chars]


def build_fixed_chunks(*, token_ids: list[int], seq_len: int, num_seqs: int) -> list[tuple[list[int], list[int]]]:
    if seq_len <= 0:
        raise ValueError(f"--seq-len must be > 0, got {seq_len}")

    need = seq_len + 1
    if len(token_ids) < need:
        raise ValueError(f"Document too short: need at least {need} tokens, got {len(token_ids)}")

    chunks: list[tuple[list[int], list[int]]] = []
    stride = seq_len
    for start in range(0, len(token_ids) - need + 1, stride):
        if len(chunks) >= num_seqs:
            break
        inp = token_ids[start : start + seq_len]
        tgt = token_ids[start + 1 : start + seq_len + 1]
        assert len(inp) == len(tgt) == seq_len
        chunks.append((inp, tgt))

    if len(chunks) == 0:
        raise ValueError("No chunks produced (unexpected).")
    return chunks


def loss_to_ppl(mean_loss: float) -> float:
    return float(math.exp(mean_loss))


async def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    text_path = Path(args.text_path)
    assert text_path.exists(), f"Missing --text-path: {text_path}"
    text = read_text(text_path, max_chars=args.max_chars)

    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(base_model=args.base_model, rank=args.rank)
    tokenizer = training_client.get_tokenizer()

    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token_id is not None
        tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id is not None

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = build_fixed_chunks(token_ids=token_ids, seq_len=args.seq_len, num_seqs=args.num_seqs)

    datums: list[tinker_types.Datum] = []
    token_count = 0
    for inp, tgt in chunks:
        datums.append(
            tinker_types.Datum(
                model_input=tinker.ModelInput.from_ints(inp),
                loss_fn_inputs={"target_tokens": tinker_types.TensorData.from_torch(torch.tensor(tgt, dtype=torch.long))},
            )
        )
        token_count += len(inp)

    print(
        "SFT SMOKE TEST\n"
        f"- backend: {os.environ.get('TINKER_BACKEND', 'api (default)')}\n"
        f"- base_model: {args.base_model}\n"
        f"- text_path: {text_path}\n"
        f"- seq_len: {args.seq_len}\n"
        f"- num_seqs: {len(datums)} ({token_count} tokens)\n"
        f"- steps: {args.steps}\n"
        f"- lr: {args.lr}\n"
        f"- seed: {args.seed}\n"
    )

    if args.steps < 2:
        raise ValueError("--steps must be >= 2 to measure improvement.")

    initial_mean_loss: float | None = None
    last_mean_loss: float | None = None

    for step in range(1, args.steps + 1):
        fut = await training_client.forward_backward_async(datums, "cross_entropy")
        out = await fut

        loss_sum = float(out.metrics["loss:sum"])
        mean_loss = loss_sum / token_count
        ppl = loss_to_ppl(mean_loss)

        if step == 1:
            initial_mean_loss = mean_loss
        last_mean_loss = mean_loss

        print(f"step={step:03d} mean_loss={mean_loss:.6f} ppl={ppl:.4f}")

        await (await training_client.optim_step_async(tinker_types.AdamParams(learning_rate=float(args.lr))))

    assert initial_mean_loss is not None
    assert last_mean_loss is not None

    required = initial_mean_loss * (1.0 - float(args.min_rel_improvement))
    if not (last_mean_loss <= required):
        raise AssertionError(
            "Perplexity/loss did not decrease enough.\n"
            f"initial_mean_loss={initial_mean_loss:.6f}\n"
            f"final_mean_loss={last_mean_loss:.6f}\n"
            f"min_rel_improvement={args.min_rel_improvement:.4f}\n"
            f"required_final_mean_loss<={required:.6f}"
        )

    print("PASS: loss decreased.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

