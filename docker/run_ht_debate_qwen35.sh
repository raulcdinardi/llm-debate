#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL_ID="Qwen/Qwen3.5-4B-Base"
MODEL_ID="${MODEL_ID:-${DEFAULT_MODEL_ID}}"
RUN_NAME="${RUN_NAME:-ht-debate-qwen35-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/${RUN_NAME}}"

if [[ "${REQUIRE_QWEN35_4B_BASE:-1}" == "1" ]]; then
  case "${MODEL_ID}" in
    "${DEFAULT_MODEL_ID}"|/opt/models/qwen35-4b-base|*/qwen35-4b-base)
      ;;
    *Qwen3.5-4B*|*qwen3.5-4b*|*qwen35-4b*)
      printf 'ERROR: MODEL_ID=%q does not look like the required Qwen3.5 4B Base model.\n' "${MODEL_ID}" >&2
      printf 'Use MODEL_ID=%q, or set REQUIRE_QWEN35_4B_BASE=0 to override deliberately.\n' "${DEFAULT_MODEL_ID}" >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME:-/cache/huggingface}" "${VLLM_CACHE_ROOT:-/cache/vllm}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

read -r -a ROUND_ADAPTER_NAMES <<< "${DEBATE_ROUND_ADAPTER_NAMES:-solution debate debate}"
DEFAULT_BASE_R2_PREFILL=$'The reasons that my solution is better than my opponent'\''s are:\n1)'
DEFAULT_BASE_R3_PREFILL=$'Responding to my opponent'\''s criticism:\n1)'

cmd=(
  python3 scripts/run_train.py
  --model-path "${MODEL_ID}"
  --output-dir "${OUTPUT_DIR}"
  --env "${ENV_NAME:-ht_sequence}"
  --mode debate
  --adapter-layout "${ADAPTER_LAYOUT:-split}"
  --steps "${STEPS:-1}"
  --num-groups "${NUM_GROUPS:-1}"
  --group-size "${GROUP_SIZE:-2}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-0}"
  --max-tokens "${MAX_TOKENS:-64}"
  --temperature "${TEMPERATURE:-0.7}"
  --min-p "${MIN_P:-0.0}"
  --seed "${SEED:-0}"
  --request-seed-mode "${REQUEST_SEED_MODE:-none}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --sequence-len "${SEQUENCE_LEN:-8}"
  --reward-mode "${REWARD_MODE:-num_h}"
  --quality-split "${QUALITY_SPLIT:-train}"
  --thinking-mode "${THINKING_MODE:-no_think}"
  --advantage-mode "${ADVANTAGE_MODE:-zscore}"
  --debate-rounds "${DEBATE_ROUNDS:-3}"
  --debate-r1-reward "${DEBATE_R1_REWARD:-task}"
  --debate-r23-reward "${DEBATE_R23_REWARD:-constant}"
  --debate-r23-constant "${DEBATE_R23_CONSTANT:-1.0}"
  --debate-r23-mode "${DEBATE_R23_MODE:-symmetric}"
  --debate-judge-adapter "${DEBATE_JUDGE_ADAPTER:-policy}"
  --debate-round-adapter-names "${ROUND_ADAPTER_NAMES[@]}"
  --debate-prompt-format "${DEBATE_PROMPT_FORMAT:-qwen35_base_text_prefill}"
  --base-r2-prefill "${BASE_R2_PREFILL:-${DEFAULT_BASE_R2_PREFILL}}"
  --base-r3-prefill "${BASE_R3_PREFILL:-${DEFAULT_BASE_R3_PREFILL}}"
  --train-minibatch-size "${TRAIN_MINIBATCH_SIZE:-1}"
  --train-max-tokens "${TRAIN_MAX_TOKENS:-0}"
  --sampler-gpu-memory-utilization "${SAMPLER_GPU_MEMORY_UTILIZATION:-0.45}"
  --sampler-max-model-len "${SAMPLER_MAX_MODEL_LEN:-768}"
  --trace-top-logprobs "${TRACE_TOP_LOGPROBS:-5}"
  --resource-log-interval-s "${RESOURCE_LOG_INTERVAL_S:-5}"
)

if [[ -n "${QUALITY_DATA_DIR:-}" ]]; then
  cmd+=(--quality-data-dir "${QUALITY_DATA_DIR}")
fi

if [[ "${QUALITY_HARD_ONLY:-1}" == "0" ]]; then
  cmd+=(--no-quality-hard-only)
fi

if [[ -n "${QUALITY_SOURCE:-Gutenberg}" ]]; then
  cmd+=(--quality-source "${QUALITY_SOURCE:-Gutenberg}")
fi

if [[ -n "${QUALITY_TOPIC_CONTAINS:-Science fiction}" ]]; then
  cmd+=(--quality-topic-contains "${QUALITY_TOPIC_CONTAINS:-Science fiction}")
fi

if [[ "${QUALITY_DOWNLOAD:-0}" == "1" ]]; then
  cmd+=(--quality-download)
fi

if [[ "${SAMPLER_ENFORCE_EAGER:-1}" == "0" || "${SAMPLER_ENFORCE_EAGER:-true}" == "false" ]]; then
  cmd+=(--no-sampler-enforce-eager)
fi

if [[ "${SAMPLER_TEARDOWN_BEFORE_TRAINING:-1}" == "1" ]]; then
  cmd+=(--sampler-teardown-before-training)
fi

if [[ "${TRACE_MODEL_IO:-1}" == "0" ]]; then
  cmd+=(--no-trace-model-io)
fi

if [[ "${RESOURCE_LOGGING:-1}" == "0" ]]; then
  cmd+=(--no-resource-logging)
fi

if [[ -n "${TRACE_MODEL_IO_DIR:-}" ]]; then
  cmd+=(--trace-model-io-dir "${TRACE_MODEL_IO_DIR}")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGV <<< "${EXTRA_ARGS}"
  cmd+=("${EXTRA_ARGV[@]}")
fi

printf 'Running debate experiment:\n'
printf '  model: %s\n' "${MODEL_ID}"
printf '  env: %s\n' "${ENV_NAME:-ht_sequence}"
printf '  output: %s\n' "${OUTPUT_DIR}"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
