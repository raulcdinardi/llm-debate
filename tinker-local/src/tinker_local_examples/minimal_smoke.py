from __future__ import annotations

import asyncio
import os

import tinker
import torch
from tinker import types as ttypes


async def main() -> None:
    base_model_env = os.environ.get("TINKER_LOCAL_BASE_MODEL")
    if base_model_env is None:
        base_model = "sshleifer/tiny-gpt2"
    else:
        base_model = base_model_env

    service = tinker.ServiceClient()
    training = await service.create_lora_training_client_async(base_model=base_model, rank=8)
    tokenizer = training.get_tokenizer()

    sampling = await training.save_weights_and_get_sampling_client_async("init")

    prompt_tokens = tokenizer.encode("Hello, my name is", add_special_tokens=False)
    resp = await sampling.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=ttypes.SamplingParams(temperature=0.8, max_tokens=8),
    )
    seq = resp.sequences[0]

    # One tiny importance-sampling step (nonsense advantages, just wiring test).
    full = prompt_tokens + seq.tokens
    if len(full) < 2 or not seq.logprobs:
        print("Not enough tokens to train.")
        return

    input_tokens = full[:-1]
    target_tokens = full[1:]

    prompt_len = len(prompt_tokens)
    old_lps = [0.0] * (prompt_len - 1) + list(seq.logprobs)
    advs = [0.0] * (prompt_len - 1) + [0.1] * len(seq.tokens)

    datum = ttypes.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": ttypes.TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": ttypes.TensorData.from_torch(torch.tensor(old_lps)),
            "advantages": ttypes.TensorData.from_torch(torch.tensor(advs)),
        },
    )

    fwd_fut = await training.forward_backward_async([datum], loss_fn="importance_sampling")
    _fwd_out = await fwd_fut

    opt_fut = await training.optim_step_async(ttypes.AdamParams(learning_rate=1e-4))
    await opt_fut

    print("OK: sampled + trained 1 step.")


if __name__ == "__main__":
    asyncio.run(main())
