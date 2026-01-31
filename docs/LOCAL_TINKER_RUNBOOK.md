# Local Tinker Backend (Transformers) Runbook

This repo can run `scripts/train.py` without the real Tinker service by injecting a local `tinker` module (under `tinker-local/`) that implements the small subset of the API used by `tinker_debate`.

This document captures the tacit/procedural knowledge needed to reproduce runs, debug failures, and avoid known footguns.

## Scope / Assumptions

- **Fail-fast**: expect visible errors instead of silent fallbacks.
- **Python**: remote GH200 setup assumed **Python 3.10**.
- **Training**: local backend uses **Transformers generate** for sampling and **Torch** for forward/backward with a LoRA-wrapped model.
- **Backend selection is explicit**: local backend uses transformers; `TINKER_LOCAL_BACKEND` defaults to `transformers`.
- **Tokens are the source of truth**: carry token IDs end-to-end; avoid string-based stop conditions and avoid decode→re-encode roundtrips.
  - Reason: tokenization and detokenization are not bijective (many strings map to the same tokens and decoding can change whitespace / normalization).

## How Local “tinker” Injection Works

- The repo can run in **local mode** without shadowing the real `tinker` SDK.
- Local mode is selected via either:
  - `TINKER_BACKEND=local`, or
  - setting `TINKER_LOCAL_BACKEND` (required by the local backend anyway).
- You can force API mode (even if `TINKER_LOCAL_BACKEND` is set in your shell) with `export TINKER_BACKEND=api`.
- Implementation detail: `tinker_debate` loads the local backend from `tinker-local/src/tinker/` under an alias module name (`tinker_local`) so `import tinker` still resolves to the real SDK for `tinker_cookbook` and other tooling.
- The wrapper `tinker-local/bin/with_local_tinker` sets `TINKER_BACKEND=local` and adds `src/` to `PYTHONPATH` for script convenience.
  - If `tinker-local/bin/with_local_tinker` is not marked executable on your machine, run it as `bash tinker-local/bin/with_local_tinker ...`.

## Backend: transformers only

The local backend loads models via `transformers.AutoModelForCausalLM` + `peft` LoRA.
`TINKER_LOCAL_BACKEND=unsloth` is unsupported and will be ignored with a warning.

## GH200 Setup (CUDA Torch Gotcha)

On GH200 images, a CUDA-enabled `torch` is often preinstalled system-wide. If you create a plain venv and `pip install torch`, you can accidentally end up with a **CPU-only** wheel and get 0% GPU utilization.

Use the provided setup script, which creates a venv that can see system site-packages:

- `scripts/gh200_setup_local_tinker.sh`

Key idea:

- `python3 -m venv --system-site-packages .venv-gh200`

## Required Environment Variables (Local Backend)

Minimum:

- `TINKER_LOCAL_BACKEND=transformers`

Strongly recommended:

- `TINKER_DEBATE_BASE_MODEL=/abs/path/to/model/dir` (or HF repo id)
- `TINKER_LOCAL_SEED=0`
- `TINKER_LOCAL_GRAD_ACCUM_STEPS=1` (set >1 to split each step into multiple accumulation chunks)
- `USE_TF=0`, `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1` (avoid importing system TF/Keras on server images)
- `PYTHONUNBUFFERED=1` and run with `python3 -u` (ensures progress prints flush)

Optional:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (sometimes helps fragmentation)
- `TINKER_LOCAL_SAMPLER=vllm` to use vLLM for sampling (see below)
- `TINKER_DEBATE_LOG_ROOT=/dev/shm/logs` to write run logs off the root filesystem (useful on small container disks)

## Model Download / Local Paths

Download:

- `python3 scripts/download_hf_model.py --repo-id Qwen/Qwen3-4B-Instruct-2507`

Default output:

- `models/Qwen__Qwen3-4B-Instruct-2507/`

Point training to the local directory:

- `export TINKER_DEBATE_BASE_MODEL="$PWD/models/Qwen__Qwen3-4B-Instruct-2507"`

## What Sampling Params Are Used Locally

Local sampling is implemented in:

- `tinker-local/src/tinker/sampling_client.py`
- vLLM sampling (if enabled): `tinker-local/src/tinker/vllm_sampling_client.py`

Supported knobs (from `tinker.SamplingParams`):

- `temperature`:
  - `do_sample = (temperature > 0)`
  - `temperature` is clamped to `>= 1e-6` when passed into `generate()`
- `max_tokens`:
  - mapped to `max_new_tokens`
  - default is `256` if `None`
- `top_k`, `top_p`, `seed`:
  - are passed through to the sampling backend (HF/vLLM) to define the effective sampling distribution
  - note: `top_k=-1` is treated as “no limit” in the local API; for Transformers this maps to `top_k=0`
- `stop`:
  - Tinker supports stop conditions as **token IDs** (`list[int]`) or strings; in this repo we prefer **token IDs**
  - **batched path** supports at most **one stop sequence** and it must be **exactly one token ID**
  - stop is implemented as `eos_token_id=<stop_token_id>` in batched mode
  - non-batched mode supports multi-token stop sequences via `StoppingCriteria`

Not implemented (today):

- repetition penalty, typical sampling, etc.

## vLLM Sampling Backend (Optional)

Enable:

- `export TINKER_LOCAL_SAMPLER=vllm`

Behavior:

- Prompts must be provided as token IDs (we already do this in local mode).
- Stop conditions must be **token IDs** (`list[int]`) in vLLM mode.
- Per-token logprobs are requested from vLLM (`logprobs=1`) and converted into a flat `list[float]`.
- LoRA syncing is implemented by saving a new LoRA adapter directory on each `sync_weights()` call and loading it into vLLM under a unique `lora_name` (typically `step_<k>`). This avoids reloading the base model each step but increases the number of resident LoRA adapters.

Knobs:

- `TINKER_LOCAL_VLLM_MAX_LORAS` (default 128)
- `TINKER_LOCAL_VLLM_GPU_MEMORY_UTILIZATION` (default 0.90)
- `TINKER_LOCAL_VLLM_MAX_MODEL_LEN` (default unset)
- `TINKER_LOCAL_VLLM_TENSOR_PARALLEL_SIZE` (default 1)
- `TINKER_LOCAL_VLLM_ENFORCE_EAGER` (default 0)
- `TINKER_LOCAL_VLLM_RESTART_EVERY` (unset/0 = no auto-restart; N>0 restarts the vLLM worker process after it has seen N distinct `lora_name`s)

## Debate Runs (Local Backend)

For local backend runs, pass `--no-proxy` (there is no HTTP traffic to inspect):

- `scripts/train.py` auto-launches `mitmweb` unless `--no-proxy` is present.

Example debate run:

```bash
export TINKER_LOCAL_BACKEND=transformers
export TINKER_LOCAL_SEED=0
export TINKER_LOCAL_GRAD_ACCUM_STEPS=1
export USE_TF=0 TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1 PYTHONUNBUFFERED=1
export TINKER_DEBATE_BASE_MODEL="$PWD/models/Qwen__Qwen3-4B-Instruct-2507"

./tinker-local/bin/with_local_tinker python3 -u scripts/train.py --no-proxy \
  --mode debate --env qa --dataset test \
  -n 256 --num-groups 4 \
  -s 10 --seed 0 --mock-judge \
  --name local_debate_smoke
```

Debate training modes:

- Winner-only (rejection sampling): `--debate-train rejection`
- Rejection sampling then GRPO on winners (task reward): `--debate-train rs_grpo --debate-grpo-reward task`

Notes:

- `--num-rollouts (-n)` must be divisible by `--num-groups`.
- For debate, `group_size = num_rollouts / num_groups` must be even (two rollouts per debate).
- Debate rollouts are batched per round and run token-only for speed.

## Summary Env Runs (Current Status)

Summary env is `--mode single_turn --env summary` and uses reward functions like compression/rouge.

Local backend:

- Summary runs use a token-only path (no `tinker_cookbook` env/renderer dependency) and compute reward from decoded text without re-encoding.
- Sampling params are global: `--temperature`, `--max-tokens`, and `--min-p` (local only).

## Training Loss: Importance Sampling + Advantages

Training client (local) implements:

- `loss_fn='importance_sampling'` only

Implementation:

- `tinker-local/src/tinker/training_client.py`

Important behavior:

- We train on **non-zero** advantages (positive or negative).
- If a batch contains **no non-zero advantages**, we now raise a `RuntimeError` (otherwise training silently no-ops).

## Why “Loss = 0.0000” Happens (Debug Checklist)

Loss ~0 can be valid in some regimes, but if it happens *every step* it’s usually one of:

1) **No learning signal**: all token advantages are exactly `0.0`.
   - In debate GRPO, this can happen if rewards are constant within each question group (centering makes everything 0).
2) **Masking bug**: advantages were present but filtered out (e.g. only training on `adv > 0`).
   - This was fixed: local backend now trains on `adv != 0`.

New fail-fast checks:

- `scripts/train.py` raises if GRPO advantages are all zero for a step.
- local training raises if an accumulation chunk has no non-zero-advantage tokens.

What to inspect when it fails:

- `logs/<run>/training_step_<k>.json`:
  - `training_data[*].completion_advantages` distribution
  - `training_data[*].metadata` (reward / centered_reward if using debate GRPO)
- `logs/<run>/run_metadata.json`:
  - env vars + versions used for the run

Semi-rigorous correctness check (local backend):

- `scripts/check_local_sampling_logprobs.py`
  - Validates that per-token logprobs returned by `generate(output_scores=True)` match teacher-forced logprobs from a direct forward pass on the same `(prompt + completion)` token sequence.
  - Run via `tinker-local/bin/with_local_tinker` with `TINKER_LOCAL_BACKEND` and `TINKER_DEBATE_BASE_MODEL` set.
  - HF sampling uses an explicit `transformers.GenerationConfig` to prevent model-default generation settings (e.g. `top_k`, `do_sample`) from warping logits/logprobs.

## Reproducibility Notes

- Seed is controlled via `TINKER_LOCAL_SEED` (default 0) and `--seed` for dataset sampling.
- `scripts/train.py` writes `run_metadata.json` with args/env/package versions.

## Performance Notes

- Sampling is batched for token prompts (local mode) to drive GPU utilization.
- Training uses gradient accumulation controlled by `TINKER_LOCAL_GRAD_ACCUM_STEPS`.
  - If OOM: lower it.
  - If VRAM is underutilized: raise it.

## Design Decision Log (So Future Sessions Don’t Re-Litigate)

- We kept changes “maximally minimal” by injecting a local `tinker` module via `PYTHONPATH` instead of refactoring call sites.
- Unsloth is treated as an *optimization backend* inside the same interface; vLLM would be a larger design change because it would require keeping inference weights in sync and matching per-token logprob semantics.
- `scripts/train.py` uses a plug-in rollout driver interface so new envs can be added without editing the core train loop:
  - implement `tinker_debate.train.driver_types.RolloutDriver`
  - add a constructor in `tinker_debate.train.driver_factory.build_driver`
- Env/task and interaction mode are orthogonal in code:
  - TaskSpec defines **R1 prompt tokens + stop token ids + task reward** (`src/tinker_debate/tasks/`)
  - Paradigm defines the interaction protocol (**normal** vs **debate**) (`src/tinker_debate/paradigms/`)
  - `src/tinker_debate/train/orthogonal_driver.py` composes them based on CLI flags (`--mode`, `--env`).
