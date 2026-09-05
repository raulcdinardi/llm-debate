#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/dist/vast}"
BUNDLE_NAME="${BUNDLE_NAME:-llm-local-rl-vast-src}"
OUT="${OUT:-${OUT_DIR}/${BUNDLE_NAME}.tar.gz}"

mkdir -p "${OUT_DIR}"
rm -f "${OUT}"

tar -C "${ROOT}" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.tmp_test_artifacts' \
  --exclude='.cache' \
  --exclude='dist' \
  --exclude='runs' \
  --exclude='outputs' \
  -czf "${OUT}" \
  --transform "s,^,${BUNDLE_NAME}/," \
  .

printf '%s\n' "${OUT}"
