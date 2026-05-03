# Qwen3.5 HT Debate Docker Run

This Docker setup runs a real `ht_sequence` debate experiment with `Qwen/Qwen3.5-4B-Base`.

The default `qwen35-ht-debate.env.example` run is intentionally small for a shared RTX 4070:
- 1 training step
- 1 debate group
- 2 agents per group
- rollout batch size 0, meaning one vLLM batch per debate round
- 64 generated tokens per turn
- vLLM GPU memory utilization 0.45
- token-level model I/O tracing enabled with top-5 alternatives

For an actual training run, use `docker/qwen35-ht-debate.real.env.example`.
That default is:
- 5 training steps
- 2 task groups per step
- 8 agents per group
- 128 generated tokens per turn

`GROUP_SIZE=2` is only a smoke test. It creates one pairwise debate per task group, so it is useful for validating the stack but too small for meaningful GRPO-style group comparisons.

Use `REQUEST_SEED_MODE=none` for normal training throughput. The global `SEED` still seeds task sampling, but vLLM requests are not assigned unique per-request seeds that would fragment batches. Set `REQUEST_SEED_MODE=per_request` only when you need the older fully seeded request behavior.

## Files

- `docker/qwen35-ht-debate.Dockerfile`
- `docker/compose.qwen35-ht-debate.yaml`
- `docker/qwen35-ht-debate.env.example`
- `docker/qwen35-ht-debate.real.env.example`
- `docker/qwen35-ht-debate.preloaded.Dockerfile`
- `docker/run_ht_debate_qwen35.sh`

## Build

From the repo root:

```bash
cp docker/qwen35-ht-debate.env.example docker/qwen35-ht-debate.env
docker compose -f docker/compose.qwen35-ht-debate.yaml build
```

For the larger run:

```bash
cp docker/qwen35-ht-debate.real.env.example docker/qwen35-ht-debate.env
docker compose -f docker/compose.qwen35-ht-debate.yaml build
```

## Run

```bash
docker compose -f docker/compose.qwen35-ht-debate.yaml run --rm ht-debate-qwen35
```

Outputs land under:

```text
runs/docker/<RUN_NAME>/
```

The trace viewer is written to:

```text
runs/docker/<RUN_NAME>/model_io_trace/index.html
```

Stage-tagged GPU memory samples are written to:

```text
runs/docker/<RUN_NAME>/resource_usage.jsonl
```

## Shared Machine Notes

Before running on `vm02`, check GPU availability:

```bash
nvidia-smi
```

Use Raul-owned locations for caches and outputs:

```bash
export LLM_LOCAL_RL_CACHE_DIR=/path/to/raul/cache/docker
export LLM_LOCAL_RL_RUNS_DIR=/path/to/raul/runs/docker
```

Do not use `sudo`, do not install global packages, and stop if the GPU is already busy.

## Tuning

Edit `docker/qwen35-ht-debate.env` for common settings. Useful first knobs:

```bash
SAMPLER_GPU_MEMORY_UTILIZATION=0.40
SAMPLER_MAX_MODEL_LEN=512
MAX_TOKENS=32
TRACE_TOP_LOGPROBS=0
```

Set `TRACE_MODEL_IO=0` to disable tracing entirely.

Set `RESOURCE_LOGGING=0` to disable stage-tagged GPU/process memory logging, or adjust:

```bash
RESOURCE_LOG_INTERVAL_S=2
```

## Preloaded Model Image

To make a single DockerHub image that Vast can fetch with the source code and model already inside:

```bash
docker login
bash scripts/build_push_dockerhub_qwen35_vast_image.sh
```

This builds:

1. `docker.io/raulcdinardi/llm-debate:qwen35-ht-debate` runtime image
2. `docker.io/raulcdinardi/llm-debate:qwen35-ht-debate-preloaded` with `Qwen/Qwen3.5-4B-Base` downloaded to `/opt/models/qwen35-4b-base`

Override the DockerHub repository if needed:

```bash
DOCKERHUB_REPO=your_user/your_repo bash scripts/build_push_dockerhub_qwen35_vast_image.sh
```

The preloaded image sets:

```bash
MODEL_ID=/opt/models/qwen35-4b-base
```

This makes Vast startup simpler, but the pushed image will be large because it includes the model weights.

The image has `/usr/local/bin/entrypoint.sh` and `/entrypoint.sh`. On Vast, either let the image entrypoint run directly or put this in the on-start box:

```bash
entrypoint.sh
```
