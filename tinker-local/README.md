# tinker-local (Transformers backend)

A tiny, **local** implementation of the subset of the Tinker Python SDK used by `tinker_debate`, backed by Transformers.

## Goal

Provide these calls with compatible shapes/semantics:

- `ServiceClient.create_lora_training_client_async(...)`
- `TrainingClient.save_weights_and_get_sampling_client_async(...)`
- `SamplingClient.sample_async(...)` (must return per-token `logprobs`)
- `TrainingClient.forward_backward_async(...)` (returns per-token new `logprobs`)
- `TrainingClient.optim_step_async(...)`

## Required env

This SDK is intentionally fail-fast (no silent fallbacks). Set:

- `TINKER_LOCAL_BACKEND` = `transformers` (defaults to `transformers` if unset)

## Use both “real tinker” and “local tinker”

To avoid uninstalling the real `tinker` SDK, don’t install this package into site-packages. Instead, shadow `tinker` **per-run** by prepending `tinker-local/src` to `PYTHONPATH`.

From the repo root:

```bash
# Real Tinker SDK (normal)
python3 scripts/train.py --dry-run

# Local tinker (shadows import only for this process)
chmod +x tinker-local/bin/with_local_tinker
TINKER_LOCAL_BACKEND=transformers tinker-local/bin/with_local_tinker \
  python3 scripts/train.py --dry-run
```

## Quick smoke example

```bash
TINKER_LOCAL_BACKEND=transformers PYTHONPATH="$(pwd)/tinker-local/src" \
  python3 -m tinker_local_examples.minimal_smoke
```

Environment variables:
- `TINKER_LOCAL_BASE_MODEL` (default: `sshleifer/tiny-gpt2`)
- `TINKER_LOCAL_DEVICE` (default: `cuda` if available else `cpu`)

Optional:
- `TINKER_LOCAL_SEED` (default: `0`)
- `TINKER_LOCAL_CHECKPOINT_DIR` (default: `.tinker-local/checkpoints`)
- `TINKER_LOCAL_MAX_SEQ_LENGTH` (default: `4096`, currently ignored)
- `TINKER_LOCAL_LOAD_IN_4BIT` (default: `1`, currently ignored)
