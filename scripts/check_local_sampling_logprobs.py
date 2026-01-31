#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def main() -> None:
    p = argparse.ArgumentParser(description="Check local sampling logprobs match teacher-forced logprobs (fail-fast).")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt", type=str, default="What is 7 * 8?")
    args = p.parse_args()

    if "TINKER_LOCAL_BACKEND" not in os.environ:
        raise ValueError("TINKER_LOCAL_BACKEND must be set (run via tinker-local/bin/with_local_tinker).")

    import torch
    from tinker_debate.tinker_sdk import tinker

    base_model = os.environ.get("TINKER_DEBATE_BASE_MODEL")
    if base_model is None:
        raise ValueError("TINKER_DEBATE_BASE_MODEL must be set (HF repo id or local model path).")

    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(base_model=base_model)
    tokenizer = training_client.get_tokenizer()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async("logprob_check")

    stop_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(stop_ids) != 1:
        raise ValueError(f"Expected single token for <|im_end|>, got {len(stop_ids)}")
    stop = [int(stop_ids[0])]

    prompt_text = "<|im_start|>user\n" + args.prompt + "\n<|im_end|>\n<|im_start|>assistant"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    sampling_params = tinker.SamplingParams(max_tokens=int(args.max_tokens), temperature=0.0, stop=stop)
    resp = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=sampling_params,
    )

    seq = resp.sequences[0]
    if seq.logprobs is None:
        raise RuntimeError("Sampling did not return per-token logprobs.")

    completion_tokens = list(seq.tokens)
    completion_logprobs = list(seq.logprobs)
    if len(completion_tokens) != len(completion_logprobs):
        raise ValueError("Length mismatch: completion_tokens vs completion_logprobs.")
    if len(completion_tokens) == 0:
        raise RuntimeError("Generated 0 tokens; cannot validate logprobs.")

    device = os.environ.get("TINKER_LOCAL_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    full = prompt_tokens + completion_tokens
    input_tokens = full[:-1]
    target_tokens = full[1:]

    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    model = training_client._model  # local-backend only (debug script)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0]  # [T, V]

    tf_logprobs_all = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=torch.tensor(target_tokens, dtype=torch.long, device=device).unsqueeze(-1),
    ).squeeze(-1)

    completion_start = len(prompt_tokens) - 1
    tf_completion_logprobs = tf_logprobs_all[completion_start : completion_start + len(completion_tokens)]

    sampled = torch.tensor(completion_logprobs, dtype=torch.float32, device=device)
    diffs = torch.abs(tf_completion_logprobs.to(dtype=torch.float32) - sampled)
    max_diff = float(torch.max(diffs).detach().cpu().item())
    mean_diff = float(torch.mean(diffs).detach().cpu().item())

    print("ok")
    print(f"tokens: {len(completion_tokens)}")
    print(f"max_abs_diff: {max_diff:.6g}")
    print(f"mean_abs_diff: {mean_diff:.6g}")

    tol = 1e-3
    if max_diff > tol:
        raise RuntimeError(f"Logprob mismatch: max_abs_diff {max_diff} > tol {tol}.")


if __name__ == "__main__":
    asyncio.run(main())
