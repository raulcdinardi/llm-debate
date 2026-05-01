#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${IMAGE_URI:-}" ]]; then
  cat >&2 <<'EOF'
Set IMAGE_URI to the registry image you want Vast to pull.

Example:
  IMAGE_URI=ghcr.io/YOUR_USER/llm-local-rl:qwen35-ht-debate-preloaded \
    bash scripts/build_push_qwen35_vast_image.sh
EOF
  exit 2
fi

RUNTIME_IMAGE_URI="${RUNTIME_IMAGE_URI:-${IMAGE_URI%-preloaded}}"
PRELOAD_MODEL_ID="${PRELOAD_MODEL_ID:-Qwen/Qwen3.5-4B-Base}"
PRELOAD_MODEL_DIR="${PRELOAD_MODEL_DIR:-/opt/models/qwen35-4b-base}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
PUSH="${PUSH:-1}"

echo "Building runtime image: ${RUNTIME_IMAGE_URI}"
docker build \
  --build-arg "VLLM_IMAGE=${VLLM_IMAGE}" \
  -f "${ROOT}/docker/qwen35-ht-debate.Dockerfile" \
  -t "${RUNTIME_IMAGE_URI}" \
  "${ROOT}"

echo "Building preloaded image: ${IMAGE_URI}"
docker build \
  --build-arg "BASE_IMAGE=${RUNTIME_IMAGE_URI}" \
  --build-arg "PRELOAD_MODEL_ID=${PRELOAD_MODEL_ID}" \
  --build-arg "PRELOAD_MODEL_DIR=${PRELOAD_MODEL_DIR}" \
  -f "${ROOT}/docker/qwen35-ht-debate.preloaded.Dockerfile" \
  -t "${IMAGE_URI}" \
  "${ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  echo "Pushing ${IMAGE_URI}"
  docker push "${IMAGE_URI}"
fi

cat <<EOF

Done.

Use this image on Vast:
  ${IMAGE_URI}

With command:
  docker/run_ht_debate_qwen35.sh

The image default MODEL_ID is:
  ${PRELOAD_MODEL_DIR}
EOF
