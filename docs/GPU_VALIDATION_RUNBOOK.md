# GPU Validation Runbook (Local Tinker Backend)

This is a procedural checklist for another agent to validate the current code on a GPU machine, including a “real” mini end-to-end run using debate + RS→GRPO.

Goals:
- Catch correctness regressions early (token/logprob alignment, stop conditions, non-zero advantages).
- Confirm GPU utilization and that training actually updates weights (loss not trivially 0, trained tokens > 0).
- Produce a concrete artifact trail under `logs/` for inspection.

Assumptions:
- You are running from the repo root.
- You have a CUDA-capable GPU and a working CUDA torch install.
- You will run local mode (no real Tinker service).
- Tokens are the source of truth. Avoid decode→re-encode in core logic (decode only for reward/metrics/log display).

## 0) Preflight Checklist (Hard Requirements)

Confirm you can import torch + CUDA:

- `python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`

Confirm the repo imports compile:

- `python3 -m py_compile scripts/train.py`

## 1) Environment Setup (Local Backend)

Required env:

- `export TINKER_LOCAL_BACKEND=transformers`
- `export TINKER_DEBATE_BASE_MODEL=/ABS/PATH/TO/MODEL` (directory) or an HF repo id
- `export USE_TF=0 TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1`
- `export PYTHONUNBUFFERED=1`

Recommended:

- `export TINKER_LOCAL_SEED=0`
- `export TINKER_LOCAL_GRAD_ACCUM_STEPS=1`
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

If you want vLLM sampling:

- `export TINKER_LOCAL_SAMPLER=vllm`
- `export TINKER_LOCAL_VLLM_RESTART_EVERY=50`

Run via wrapper so the local `tinker` module is used:

- `bash tinker-local/bin/with_local_tinker <command...>`
- Note: this wrapper no longer shadows `import tinker`; it sets `TINKER_BACKEND=local` and lets `tinker_debate` load the local backend under an alias module name.

## 2) Test 1 — Logprob Consistency (Most Important Correctness Check)

This validates that sampling logprobs match teacher-forced logprobs from a forward pass on the exact same token sequence.

Run:

- `./tinker-local/bin/with_local_tinker python3 -u scripts/check_local_sampling_logprobs.py --max-tokens 32 --prompt "What is 7 * 8?"`
- If `tinker-local/bin/with_local_tinker` is marked executable on your machine, `./tinker-local/bin/with_local_tinker ...` also works.

Expected:

- Script prints `ok`.
- `max_abs_diff` should be small (the script enforces a hard tolerance).

If this fails:

- Do not run training; fix the mismatch first.
- If using vLLM: ensure your training backend and vLLM are sampling from effectively the same model weights/dtype (4-bit vs fp16/bf16 mismatches can cause differences).

## 3) Test 2 — Single-Turn Smoke (Normal Mode)

### 3a) QA (cheap, fast)

- `./tinker-local/bin/with_local_tinker python3 -u scripts/train.py --no-proxy --mode single_turn --env qa --dataset test -n 16 -s 1 --name smoke_normal_qa`
- If you get “permission denied” running `./tinker-local/bin/with_local_tinker`, use the `bash ...` form above.

Expected rubric:
- Creates a new run dir under `logs/`.
- Creates `run_metadata.json`.
- Creates some `baseline_*.json` (QA uses baseline-style logging).
- Produces at least 1 accepted datum unless reward threshold is too strict.
- Prints a non-empty “Training data: … datums” line.

### 3b) Summary (exercises reward pipeline)

Install dataset deps if needed:
- `python3 -m pip install datasets`

Run:
- `./tinker-local/bin/with_local_tinker python3 -u scripts/train.py --no-proxy --mode single_turn --env summary --dataset cnn_dailymail --reward-fn compression -n 16 -s 1 --name smoke_normal_summary`

Expected rubric:
- Creates `summary_*.json` records with `reward` and reward metrics.
- Training proceeds with non-empty training data.

## 4) Test 3 — Debate Smoke (Rejection Sampling Only)

This checks debate token injection and judging without GRPO centering sensitivity.

Run:
- `./tinker-local/bin/with_local_tinker python3 -u scripts/train.py --no-proxy --mode debate --env qa --dataset test -n 32 --num-groups 2 -s 1 --mock-judge --debate-train rejection --name smoke_debate_rejection`

Expected rubric:
- Prints debate round progress (`[debate] Round 1/2/3`).
- Produces `debate_*.json` logs.
- Trains at least one step and prints `Loss:` and `Tokens:` lines.

## 5) Test 4 — Mini End-to-End Regime (Debate + RS→GRPO with Task Reward)

This is the “actual mini experiment” that exercises:
- debate rollouts
- rejection sampling (winners)
- task reward computed on surviving winners “as if debate didn’t happen”
- GRPO centering across winners in each question group
- importance sampling training update

Important: RS→GRPO needs reward variance across surviving winners within each group. Use a task where rewards vary.

Recommended mini experiment: **Summary task** (continuous reward).

Run:
- `./tinker-local/bin/with_local_tinker python3 -u scripts/train.py --no-proxy --mode debate --env summary --dataset cnn_dailymail --reward-fn compression -n 64 --num-groups 4 -s 10 --mock-judge --debate-train rs_grpo --debate-grpo-reward task --name mini_debate_rs_grpo_summary`

If you want heavier GPU utilization (and you have VRAM):
- Increase `-n` and `-s`, and increase `TINKER_LOCAL_GRAD_ACCUM_STEPS`.

Expected rubric:
- Each step logs `training_step_<k>.json` (or equivalent `Logged:` message).
- “Advantage stats (nonzero)” should show non-zero min/max and non-zero count.
- Loss should not be identically `0.0000` every step.
- `num_trained_tokens` reported by training should be > 0 (inspect `training_step_<k>.json` if needed).

If it fails with “All token advantages are 0.0”:
- That means the centered rewards were constant within groups. Increase `-n`, change reward function, or use a harder task/different temperature.

## 6) GPU Utilization Checks

During the mini experiment:

- `watch -n 1 nvidia-smi`

Expected rubric:
- GPU utilization should be meaningfully > 0% during rollouts and training.
- VRAM usage should increase from idle.

If GPU is 0%:
- You may have a CPU-only torch install in your venv.
- On GH200-like images, prefer a venv created with `--system-site-packages` (see `scripts/gh200_setup_local_tinker.sh`).

## 7) Artifacts to Inspect (What “Good Logs” Look Like)

Run directory: `logs/<timestamp>_*_<name>/`

Expected files:
- `run_metadata.json` (args/env/versions)
- `http_traffic.yaml` (may exist but is irrelevant in local mode)
- `training_step_<k>.json` for each step
- For normal runs: `baseline_*.json` and/or `summary_*.json`
- For debate runs: `debate_*.json`

Rubric for `training_step_<k>.json`:
- Each datum has:
  - `prompt_tokens`, `completion_tokens`, `completion_logprobs`, `completion_advantages`
  - lengths match and are non-empty for accepted samples
- Non-zero advantages exist (otherwise training is a no-op)
- `results.metrics["loss:sum"]` exists in local mode

## 8) Common Failure Modes (What They Mean)

- `No non-zero advantages in batch (nothing to train on)`:
  - Your reward/centering produced all-zero advantages (no learning signal).
- `Logprob mismatch` from `check_local_sampling_logprobs.py`:
  - Sampling engine and training model disagree; don’t trust ratios/loss until fixed.
- OOM:
  - Lower `TINKER_LOCAL_GRAD_ACCUM_STEPS`, reduce `-n`, or reduce max token limits.
