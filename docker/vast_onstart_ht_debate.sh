#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/outputs/_vast_logs}"
mkdir -p "${LOG_DIR}" /workspace /outputs
LOG_FILE="${LOG_FILE:-${LOG_DIR}/ht_debate_onstart.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "== Vast HT debate on-start =="
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"
hostname || true
nvidia-smi || true

export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${VLLM_CACHE_ROOT}"

STATUS_FILE="${STATUS_FILE:-/outputs/_vast_status.json}"
RUN_NAME="${RUN_NAME:-ht-debate-qwen35-vast}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/${RUN_NAME}}"
SOURCE_DIR="${SOURCE_DIR:-/workspace/llm-local-rl}"
SOURCE_BUNDLE_PATH="${SOURCE_BUNDLE_PATH:-/workspace/llm-local-rl-vast-src.tar.gz}"
AUTO_STOP="${AUTO_STOP:-1}"
AUTO_DESTROY="${AUTO_DESTROY:-0}"

write_status() {
  local status="$1"
  local rc="$2"
  python3 - "$STATUS_FILE" "$status" "$rc" "$RUN_NAME" "$OUTPUT_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, status, rc, run_name, output_dir = sys.argv[1:]
payload = {
    "status": status,
    "return_code": int(rc),
    "run_name": run_name,
    "output_dir": output_dir,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

vast_instance_id() {
  if [[ -n "${CONTAINER_ID:-}" ]]; then
    printf '%s\n' "${CONTAINER_ID}"
    return 0
  fi
  if [[ -n "${VAST_CONTAINERLABEL:-}" ]]; then
    printf '%s\n' "${VAST_CONTAINERLABEL#C.}"
    return 0
  fi
  return 1
}

install_vast_cli() {
  if command -v vastai >/dev/null 2>&1; then
    return 0
  fi
  python3 -m pip install --no-cache-dir vastai
}

legacy_vast_key() {
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    return 0
  fi
  if [[ -f "${HOME}/.vast_api_key" ]]; then
    return 0
  fi
  if [[ -f "${HOME}/.ssh/authorized_keys" && -n "${VAST_CONTAINERLABEL:-}" ]]; then
    cat "${HOME}/.ssh/authorized_keys" | md5sum | awk '{print $1}' > /tmp/ssh_key_hv
    echo -n "${VAST_CONTAINERLABEL}" | md5sum | awk '{print $1}' > /tmp/instance_id_hv
    head -c -1 -q /tmp/ssh_key_hv /tmp/instance_id_hv > "${HOME}/.vast_api_key"
  fi
}

stop_or_destroy_instance() {
  local rc="$1"
  if [[ "${AUTO_STOP}" != "1" && "${AUTO_DESTROY}" != "1" ]]; then
    return 0
  fi
  local instance_id
  if ! instance_id="$(vast_instance_id)"; then
    echo "No Vast instance id env var found; cannot auto-stop."
    return 0
  fi
  install_vast_cli || true
  legacy_vast_key || true

  if [[ "${AUTO_DESTROY}" == "1" ]]; then
    if [[ "${rc}" == "0" ]]; then
      echo "AUTO_DESTROY=1 and job succeeded; destroying Vast instance ${instance_id}."
      if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
        vastai destroy instance "${instance_id}" --api-key "${CONTAINER_API_KEY}" || true
      else
        vastai destroy instance "${instance_id}" || true
      fi
    else
      echo "AUTO_DESTROY=1 but job failed; stopping instead to preserve logs."
      if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
        vastai stop instance "${instance_id}" --api-key "${CONTAINER_API_KEY}" || true
      else
        vastai stop instance "${instance_id}" || true
      fi
    fi
    return 0
  fi

  echo "AUTO_STOP=1; stopping Vast instance ${instance_id}."
  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai stop instance "${instance_id}" --api-key "${CONTAINER_API_KEY}" || true
  else
    vastai stop instance "${instance_id}" || true
  fi
}

finish() {
  local rc="$?"
  set +e
  if [[ "${rc}" == "0" ]]; then
    write_status "succeeded" "${rc}"
  else
    write_status "failed" "${rc}"
  fi
  if [[ -d "${OUTPUT_DIR}" ]]; then
    tar -C /outputs -czf "/outputs/${RUN_NAME}.tar.gz" "${RUN_NAME}" 2>/dev/null || true
  fi
  if [[ -n "${SYNC_COMMAND:-}" ]]; then
    echo "Running SYNC_COMMAND."
    bash -lc "${SYNC_COMMAND}" || true
  fi
  date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
  stop_or_destroy_instance "${rc}"
  exit "${rc}"
}
trap finish EXIT

prepare_system_tools() {
  if command -v git >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends git curl ca-certificates
  fi
}

prepare_source() {
  local prefer_external_source="${PREFER_EXTERNAL_SOURCE:-0}"
  if [[ "${prefer_external_source}" != "1" && -z "${REPO_URL:-}" && -z "${REPO_TARBALL_URL:-}" && -x "${SOURCE_DIR}/docker/run_ht_debate_qwen35.sh" ]]; then
    echo "Using existing source at ${SOURCE_DIR}."
    export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    return 0
  fi

  rm -rf "${SOURCE_DIR}"
  mkdir -p "$(dirname "${SOURCE_DIR}")"

  if [[ -n "${REPO_TARBALL_URL:-}" ]]; then
    prepare_system_tools
    echo "Downloading source tarball from REPO_TARBALL_URL."
    curl -fL "${REPO_TARBALL_URL}" -o /tmp/llm-local-rl-src.tar.gz
    mkdir -p "${SOURCE_DIR}"
    tar -xzf /tmp/llm-local-rl-src.tar.gz -C "${SOURCE_DIR}" --strip-components="${TARBALL_STRIP_COMPONENTS:-1}"
    export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    return 0
  fi

  if [[ -f "${SOURCE_BUNDLE_PATH}" ]]; then
    echo "Extracting source bundle ${SOURCE_BUNDLE_PATH}."
    tar -xzf "${SOURCE_BUNDLE_PATH}" -C /workspace
    if [[ -x "${SOURCE_DIR}/docker/run_ht_debate_qwen35.sh" ]]; then
      export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
      return 0
    fi
    if [[ -x "/workspace/llm-local-rl-vast-src/docker/run_ht_debate_qwen35.sh" ]]; then
      SOURCE_DIR="/workspace/llm-local-rl-vast-src"
      export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
      return 0
    fi
  fi

  if [[ -n "${REPO_URL:-}" ]]; then
    prepare_system_tools
    echo "Cloning ${REPO_URL}."
    git clone --depth 1 ${REPO_REF:+--branch "${REPO_REF}"} "${REPO_URL}" "${SOURCE_DIR}"
    export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    return 0
  fi

  echo "No source found. Set REPO_URL, REPO_TARBALL_URL, or upload ${SOURCE_BUNDLE_PATH}." >&2
  return 2
}

install_python_deps() {
  cd "${SOURCE_DIR}"
  if [[ "${SKIP_PYTHON_DEPS:-0}" == "1" ]]; then
    echo "SKIP_PYTHON_DEPS=1; using Python environment baked into the image."
    python3 - <<'PY'
import importlib.util
missing = [name for name in ("torch", "transformers", "peft", "vllm", "llm_local_rl") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required packages: " + ", ".join(missing))
print("required Python packages available")
PY
    return 0
  fi

  python3 -m pip install --upgrade pip
  python3 -m pip install --no-cache-dir \
    "transformers>=4.57.0" \
    "accelerate>=1.10.0" \
    "peft>=0.17.0" \
    "pytest>=8.0"
  if [[ "${INSTALL_VLLM:-0}" == "1" ]]; then
    python3 -m pip install --no-cache-dir "vllm>=0.8.0"
  fi
  python3 -m pip install -e .
  python3 - <<'PY'
import importlib.util
missing = [name for name in ("torch", "transformers", "peft", "vllm") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required packages: " + ", ".join(missing))
print("required Python packages available")
PY
}

run_experiment() {
  cd "${SOURCE_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  export OUTPUT_DIR
  export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B-Base}"
  export RUN_NAME
  bash docker/run_ht_debate_qwen35.sh
}

write_status "running" 0
prepare_source
install_python_deps
run_experiment
