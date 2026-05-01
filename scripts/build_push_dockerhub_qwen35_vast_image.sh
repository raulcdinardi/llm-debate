#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOCKERHUB_REPO="${DOCKERHUB_REPO:-raulcdinardi/llm-debate}"
RUNTIME_TAG="${RUNTIME_TAG:-qwen35-ht-debate}"
PRELOADED_TAG="${PRELOADED_TAG:-qwen35-ht-debate-preloaded}"

export RUNTIME_IMAGE_URI="${RUNTIME_IMAGE_URI:-docker.io/${DOCKERHUB_REPO}:${RUNTIME_TAG}}"
export IMAGE_URI="${IMAGE_URI:-docker.io/${DOCKERHUB_REPO}:${PRELOADED_TAG}}"

exec bash "${ROOT}/scripts/build_push_qwen35_vast_image.sh"
