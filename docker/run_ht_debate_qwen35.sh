#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B-Base}"
RUN_NAME="${RUN_NAME:-ht-debate-qwen35-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME:-/cache/huggingface}" "${VLLM_CACHE_ROOT:-/cache/vllm}"

read -r -a ROUND_ADAPTER_NAMES <<< "${DEBATE_ROUND_ADAPTER_NAMES:-solution debate debate}"

cmd=(
  python3 scripts/run_train.py
  --model-path "${MODEL_ID}"
  --output-dir "${OUTPUT_DIR}"
  --env ht_sequence
  --mode debate
  --adapter-layout "${ADAPTER_LAYOUT:-split}"
  --steps "${STEPS:-1}"
  --num-groups "${NUM_GROUPS:-1}"
  --group-size "${GROUP_SIZE:-2}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-1}"
  --max-tokens "${MAX_TOKENS:-64}"
  --temperature "${TEMPERATURE:-0.7}"
  --min-p "${MIN_P:-0.0}"
  --seed "${SEED:-0}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --sequence-len "${SEQUENCE_LEN:-8}"
  --reward-mode "${REWARD_MODE:-num_h}"
  --thinking-mode "${THINKING_MODE:-no_think}"
  --advantage-mode "${ADVANTAGE_MODE:-zscore}"
  --debate-rounds "${DEBATE_ROUNDS:-3}"
  --debate-r1-reward "${DEBATE_R1_REWARD:-task}"
  --debate-r23-reward "${DEBATE_R23_REWARD:-constant}"
  --debate-r23-constant "${DEBATE_R23_CONSTANT:-1.0}"
  --debate-r23-mode "${DEBATE_R23_MODE:-symmetric}"
  --debate-judge-adapter "${DEBATE_JUDGE_ADAPTER:-policy}"
  --debate-round-adapter-names "${ROUND_ADAPTER_NAMES[@]}"
  --train-minibatch-size "${TRAIN_MINIBATCH_SIZE:-1}"
  --sampler-gpu-memory-utilization "${SAMPLER_GPU_MEMORY_UTILIZATION:-0.45}"
  --sampler-max-model-len "${SAMPLER_MAX_MODEL_LEN:-768}"
  --trace-top-logprobs "${TRACE_TOP_LOGPROBS:-5}"
)

if [[ "${TRACE_MODEL_IO:-1}" == "0" ]]; then
  cmd+=(--no-trace-model-io)
fi

if [[ -n "${TRACE_MODEL_IO_DIR:-}" ]]; then
  cmd+=(--trace-model-io-dir "${TRACE_MODEL_IO_DIR}")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGV <<< "${EXTRA_ARGS}"
  cmd+=("${EXTRA_ARGV[@]}")
fi

printf 'Running HT debate experiment:\n'
printf '  model: %s\n' "${MODEL_ID}"
printf '  output: %s\n' "${OUTPUT_DIR}"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
