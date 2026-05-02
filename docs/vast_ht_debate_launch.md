# Vast Launch: Qwen3.5 HT Debate

This is the lowest-friction Vast path for the Qwen HT debate experiment.

## What Is Ready

- A Docker-based local path exists in `docker/`.
- A Vast on-start script exists at `docker/vast_onstart_ht_debate.sh`.
- A source bundle script exists at `scripts/make_vast_bundle.sh`.
- The experiment defaults are conservative for a shared 4070.
- The Vast script stops the instance at the end by default.

`GROUP_SIZE=2` is only for smoke testing. For a real run, use `GROUP_SIZE=8` or larger. With `ROLLOUT_BATCH_SIZE=1`, group size mostly increases runtime and training data, not peak rollout memory.

## Important Billing Behavior

Vast bills while the instance is running. A stopped instance preserves data but still has storage charges. Destroy the instance after you have copied the output you care about.

The on-start script uses `AUTO_STOP=1` by default. Use `AUTO_DESTROY=1` only if `SYNC_COMMAND` copies outputs elsewhere first, because destroy is irreversible.

## Best Launch Path

### Fastest Runtime: DockerHub Preloaded Image

Best final setup:

1. Log in to DockerHub on a Docker-capable machine:

```bash
docker login
```

2. Build and push a single preloaded image:

```bash
bash scripts/build_push_dockerhub_qwen35_vast_image.sh
```

This defaults to:

```text
docker.io/raulcdinardi/llm-debate:qwen35-ht-debate-preloaded
```

Override it with `DOCKERHUB_REPO=your_user/your_repo` if needed.

3. On Vast, set the image to:

```text
raulcdinardi/llm-debate:qwen35-ht-debate-preloaded
```

The image default entrypoint runs the experiment. If Vast needs an on-start command, use exactly:

```bash
entrypoint.sh
```

This image includes the code and model files. Vast still spends time pulling the image, but it avoids git clone, dependency install, and Hugging Face model download at startup.

### Flexible Runtime: vLLM Base Image

Use a Vast image that already has vLLM:

```text
vllm/vllm-openai:latest
```

Use at least:

```text
Disk: 100 GB
GPU: RTX 4070 or better
Launch mode: SSH
On-start script: paste docker/vast_onstart_ht_debate.sh
```

## Source Options

Pick one source option.

### Option A: Git Repo URL

Push this repo somewhere reachable by Vast, then set:

```bash
-e REPO_URL=https://github.com/YOUR_USER/YOUR_REPO.git
-e REPO_REF=main
```

This is the cleanest option.

### Option B: Source Bundle

Create the bundle locally:

```bash
bash scripts/make_vast_bundle.sh
```

Upload `dist/vast/llm-local-rl-vast-src.tar.gz` to the instance as:

```text
/workspace/llm-local-rl-vast-src.tar.gz
```

Then run:

```bash
bash /workspace/llm-local-rl/docker/vast_onstart_ht_debate.sh
```

This option is less automatic because the instance must exist before upload.

### Option C: Tarball URL

Upload `dist/vast/llm-local-rl-vast-src.tar.gz` to any private URL Vast can fetch, then set:

```bash
-e REPO_TARBALL_URL=https://...
```

## Vast Environment Variables

Minimum recommended env vars:

```bash
-e RUN_NAME=ht-debate-qwen35-vast \
-e AUTO_STOP=1 \
-e STEPS=5 \
-e NUM_GROUPS=2 \
-e GROUP_SIZE=8 \
-e ROLLOUT_BATCH_SIZE=1 \
-e MAX_TOKENS=128 \
-e SAMPLER_MAX_MODEL_LEN=1024 \
-e SAMPLER_GPU_MEMORY_UTILIZATION=0.55 \
-e TRAIN_MINIBATCH_SIZE=1 \
-e TRACE_TOP_LOGPROBS=5
```

For the preloaded DockerHub image, do not set `REPO_URL` or `REPO_REF`. It already contains the source. It also defaults `MODEL_ID` to `/opt/models/qwen35-4b-base`.

If Hugging Face auth is needed:

```bash
-e HF_TOKEN=...
```

## Outputs

The run writes to:

```text
/outputs/<RUN_NAME>/
```

The script also creates:

```text
/outputs/<RUN_NAME>.tar.gz
/outputs/_vast_status.json
/outputs/_vast_logs/ht_debate_onstart.log
```

The trace viewer is at:

```text
/outputs/<RUN_NAME>/model_io_trace/index.html
```

Stage-tagged GPU memory samples are at:

```text
/outputs/<RUN_NAME>/resource_usage.jsonl
```

## After It Stops

1. Copy `/outputs/<RUN_NAME>.tar.gz` from the stopped instance using Vast's data tools or by restarting briefly.
2. Destroy the instance in the Vast UI to stop storage charges.

## Safer First Run

If the 4070 is tight on memory, lower these:

```bash
-e MAX_TOKENS=32 \
-e SAMPLER_MAX_MODEL_LEN=512 \
-e SAMPLER_GPU_MEMORY_UTILIZATION=0.40 \
-e TRACE_TOP_LOGPROBS=0
```

For a smoke test before paying for a longer run:

```bash
-e STEPS=1 \
-e NUM_GROUPS=1 \
-e GROUP_SIZE=2 \
-e MAX_TOKENS=64 \
-e SAMPLER_MAX_MODEL_LEN=768 \
-e SAMPLER_GPU_MEMORY_UTILIZATION=0.45
```

## Official Vast References

- Docker/on-start environment behavior: https://docs.vast.ai/guides/instances/docker-environment
- Managing stop/destroy behavior: https://docs.vast.ai/documentation/instances/manage-instances
- Self-stop from inside instance: https://docs.vast.ai/documentation/reference/faq/instances
- CLI `destroy instance`: https://docs.vast.ai/cli/reference/destroy-instance
