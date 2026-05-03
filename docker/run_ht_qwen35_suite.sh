#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL_ID="Qwen/Qwen3.5-4B-Base"
MODEL_ID="${MODEL_ID:-${DEFAULT_MODEL_ID}}"
RUN_NAME="${RUN_NAME:-ht-suite-qwen35-$(date -u +%Y%m%dT%H%M%SZ)}"
SUITE_OUTPUT_DIR="${OUTPUT_DIR:-/outputs/${RUN_NAME}}"

mkdir -p "${SUITE_OUTPUT_DIR}" "${HF_HOME:-/cache/huggingface}" "${VLLM_CACHE_ROOT:-/cache/vllm}"

COMMON_ARGS=(
  --model-path "${MODEL_ID}"
  --env ht_sequence
  --steps "${STEPS:-1}"
  --num-groups "${NUM_GROUPS:-2}"
  --group-size "${GROUP_SIZE:-8}"
  --num-samples "${NUM_SAMPLES:-16}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-0}"
  --max-tokens "${MAX_TOKENS:-128}"
  --temperature "${TEMPERATURE:-0.7}"
  --min-p "${MIN_P:-0.0}"
  --seed "${SEED:-0}"
  --request-seed-mode "${REQUEST_SEED_MODE:-none}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --sequence-len "${SEQUENCE_LEN:-8}"
  --reward-mode "${REWARD_MODE:-num_h}"
  --thinking-mode "${THINKING_MODE:-no_think}"
  --advantage-mode "${ADVANTAGE_MODE:-zscore}"
  --train-minibatch-size "${TRAIN_MINIBATCH_SIZE:-1}"
  --sampler-gpu-memory-utilization "${SAMPLER_GPU_MEMORY_UTILIZATION:-0.55}"
  --sampler-max-model-len "${SAMPLER_MAX_MODEL_LEN:-2048}"
  --trace-top-logprobs "${TRACE_TOP_LOGPROBS:-5}"
  --resource-log-interval-s "${RESOURCE_LOG_INTERVAL_S:-5}"
)

maybe_disable_flags=()
if [[ "${TRACE_MODEL_IO:-1}" == "0" ]]; then
  maybe_disable_flags+=(--no-trace-model-io)
fi
if [[ "${RESOURCE_LOGGING:-1}" == "0" ]]; then
  maybe_disable_flags+=(--no-resource-logging)
fi

run_one() {
  local name="$1"
  shift
  local out_dir="${SUITE_OUTPUT_DIR}/${name}"
  mkdir -p "${out_dir}"
  printf 'Running suite leg %s -> %s\n' "${name}" "${out_dir}"
  python3 scripts/run_train.py \
    "${COMMON_ARGS[@]}" \
    "${maybe_disable_flags[@]}" \
    --output-dir "${out_dir}" \
    "$@"
}

DEFAULT_BASE_R2_PREFILL=$'The reasons that my solution is better than my opponent'\''s are:\n1)'
DEFAULT_BASE_R3_PREFILL=$'Responding to my opponent'\''s criticism:\n1)'
BASE_R2_PREFILL="${BASE_R2_PREFILL:-${DEFAULT_BASE_R2_PREFILL}}"
BASE_R3_PREFILL="${BASE_R3_PREFILL:-${DEFAULT_BASE_R3_PREFILL}}"

run_one baseline \
  --mode single_turn \
  --adapter-layout shared

run_one consultancy \
  --mode debate \
  --adapter-layout split \
  --debate-rounds 1 \
  --debate-r1-reward task \
  --debate-r23-reward none \
  --debate-judge-adapter policy \
  --debate-prompt-format "${DEBATE_PROMPT_FORMAT:-qwen35_base_text_prefill}" \
  --base-r2-prefill "${BASE_R2_PREFILL}" \
  --base-r3-prefill "${BASE_R3_PREFILL}"

run_one debate \
  --mode debate \
  --adapter-layout split \
  --debate-rounds 3 \
  --debate-r1-reward "${DEBATE_R1_REWARD:-task}" \
  --debate-r23-reward "${DEBATE_R23_REWARD:-constant}" \
  --debate-r23-constant "${DEBATE_R23_CONSTANT:-1.0}" \
  --debate-r23-mode "${DEBATE_R23_MODE:-symmetric}" \
  --debate-judge-adapter "${DEBATE_JUDGE_ADAPTER:-policy}" \
  --debate-prompt-format "${DEBATE_PROMPT_FORMAT:-qwen35_base_text_prefill}" \
  --base-r2-prefill "${BASE_R2_PREFILL}" \
  --base-r3-prefill "${BASE_R3_PREFILL}"

printf 'Suite complete: %s\n' "${SUITE_OUTPUT_DIR}"
