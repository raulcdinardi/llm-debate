ARG BASE_IMAGE=llm-local-rl:qwen35-ht-debate
FROM ${BASE_IMAGE}

ARG PRELOAD_MODEL_ID=Qwen/Qwen3.5-4B-Base
ARG PRELOAD_MODEL_DIR=/opt/models/qwen35-4b-base

ENV PRELOAD_MODEL_ID=${PRELOAD_MODEL_ID}
ENV MODEL_ID=${PRELOAD_MODEL_DIR}
ENV HF_HOME=/opt/hf-cache
ENV HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub
ENV TRANSFORMERS_CACHE=/opt/hf-cache
ENV VLLM_CACHE_ROOT=/opt/vllm-cache

RUN mkdir -p "${PRELOAD_MODEL_DIR}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" && \
    python3 -m pip install --no-cache-dir "huggingface_hub>=0.25.0" && \
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["PRELOAD_MODEL_ID"]
local_dir = os.environ["MODEL_ID"]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Downloaded {repo_id} to {local_dir}")
PY
