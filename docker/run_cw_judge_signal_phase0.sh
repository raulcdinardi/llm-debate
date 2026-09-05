#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/llm-debate}"
RUN_NAME="${RUN_NAME:-cw_judge_signal_phase0_20260720}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/workspace/outputs}"
RUN_ROOT="${RUN_ROOT:-${OUTPUTS_ROOT}/${RUN_NAME}}"
ARCHIVE="${ARCHIVE:-${OUTPUTS_ROOT}/${RUN_NAME}.tar.gz}"
MODEL_CACHE="${MODEL_CACHE:-/workspace/model_cache}"
DEBATE_ADAPTER="${DEBATE_ADAPTER:-/workspace/inputs/debate}"
JUDGE_08B_ADAPTER="${JUDGE_08B_ADAPTER:-/workspace/inputs/judge08b}"
JUDGE_4B_ADAPTER="${JUDGE_4B_ADAPTER:-/workspace/inputs/judge4b}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"
EXPECTED_IMAGE_REF="${EXPECTED_IMAGE_REF:?EXPECTED_IMAGE_REF is required}"
EXPECTED_IMAGE_DIGEST="${EXPECTED_IMAGE_DIGEST:?EXPECTED_IMAGE_DIGEST is required}"

DEBATE_SHA256="b97cea47bcf0ec8ab501dcc14b234b7ca28a84237e4a09094ee9ff04e2bef209"
JUDGE_08B_SHA256="6e284aaa96ecc21749be5f40bddbd4b54247d9ea2f36c0e562e5cf8a4d3f5f89"
JUDGE_4B_SHA256="357996688954bd59a7c28eeb0632f12aee247b776111ed651cc1b904f18a7582"
R1_PREFILL=$'Ok, I will produce a 3-sentence story adhering to the rules:\n'
DEBATE_PREFILL=$'The reasons that my solution is better than my opponent\'s are:\n1)'

POLICY_PORT=30000
JUDGE_PORT=30001
POLICY_LOG="${RUN_ROOT}/policy_sglang.log"
JUDGE_LOG="${RUN_ROOT}/judge_sglang.log"
PROGRESS="${RUN_ROOT}/progress.log"

export PYTHONPATH="${ROOT}/src:${ROOT}"
export HF_HOME="${MODEL_CACHE}/hf_home"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUTS_ROOT}" "${MODEL_CACHE}" "${HF_HOME}"
if test -e "${RUN_ROOT}"; then
  echo "Refusing to reuse existing run root: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "${PROGRESS}"
}

stop_judge() {
  if test -n "${JUDGE_PID:-}" && kill -0 "${JUDGE_PID}" 2>/dev/null; then
    kill "${JUDGE_PID}" 2>/dev/null || true
    wait "${JUDGE_PID}" 2>/dev/null || true
  fi
  JUDGE_PID=""
}

cleanup_and_archive() {
  local status=$?
  trap - EXIT
  stop_judge
  if test -n "${POLICY_PID:-}" && kill -0 "${POLICY_PID}" 2>/dev/null; then
    kill "${POLICY_PID}" 2>/dev/null || true
    wait "${POLICY_PID}" 2>/dev/null || true
  fi
  touch "${RUN_ROOT}/PHASE1_FORBIDDEN"
  if ! python3 "${ROOT}/scripts/cw_phase0_archive.py" \
    --run-root "${RUN_ROOT}" \
    --archive "${ARCHIVE}" \
    --exit-status "${status}"; then
    status=1
  fi
  exit "${status}"
}
trap cleanup_and_archive EXIT

wait_for_server() {
  local port=$1
  local pid=$2
  local log_path=$3
  local model_info_path=$4
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/get_model_info" > "${model_info_path}" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 160 "${log_path}" >&2
      return 1
    fi
    sleep 5
  done
  tail -n 160 "${log_path}" >&2
  return 1
}

start_policy_server() {
  log "policy_server_start"
  env -u PYTORCH_CUDA_ALLOC_CONF SGLANG_RETURN_ORIGINAL_LOGPROB=false python3 -m sglang.launch_server \
    --model-path "${MODEL_4B}" \
    --tokenizer-path "${MODEL_4B}" \
    --host 127.0.0.1 \
    --port "${POLICY_PORT}" \
    --enable-lora \
    --max-loras-per-batch 8 \
    --max-loaded-loras 16 \
    --max-lora-rank 32 \
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --context-length 16384 \
    --mem-fraction-static 0.32 \
    --enable-memory-saver \
    --attention-backend triton \
    --prefill-attention-backend triton \
    --decode-attention-backend triton \
    --sampling-backend pytorch \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled \
    --trust-remote-code \
    --log-requests \
    --log-requests-level 1 > "${POLICY_LOG}" 2>&1 &
  POLICY_PID=$!
  wait_for_server "${POLICY_PORT}" "${POLICY_PID}" "${POLICY_LOG}" "${RUN_ROOT}/policy_model_info.json"
  log "policy_server_ready pid=${POLICY_PID}"
}

start_judge_server() {
  local arm=$1
  local model_path=$2
  local log_path="${RUN_ROOT}/judge_${arm}_sglang.log"
  stop_judge
  log "judge_server_start arm=${arm}"
  env -u PYTORCH_CUDA_ALLOC_CONF SGLANG_RETURN_ORIGINAL_LOGPROB=false python3 -m sglang.launch_server \
    --model-path "${model_path}" \
    --tokenizer-path "${model_path}" \
    --host 127.0.0.1 \
    --port "${JUDGE_PORT}" \
    --enable-lora \
    --max-loras-per-batch 4 \
    --max-loaded-loras 4 \
    --max-lora-rank 32 \
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --context-length 16384 \
    --mem-fraction-static 0.16 \
    --attention-backend triton \
    --prefill-attention-backend triton \
    --decode-attention-backend triton \
    --sampling-backend pytorch \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled \
    --trust-remote-code \
    --log-requests \
    --log-requests-level 1 > "${log_path}" 2>&1 &
  JUDGE_PID=$!
  wait_for_server "${JUDGE_PORT}" "${JUDGE_PID}" "${log_path}" "${RUN_ROOT}/judge_${arm}_model_info.json"
  log "judge_server_ready arm=${arm} pid=${JUDGE_PID}"
}

finalize_preflight() {
  python3 "${ROOT}/scripts/cw_judge_signal_phase0_finalize.py" \
    --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/preflight.json" \
    --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
    --expected-image-digest "${EXPECTED_IMAGE_DIGEST}"
}

log "preflight_start phase1_forbidden=true"
test "${LLM_DEBATE_SOURCE_COMMIT:-}" = "${EXPECTED_SOURCE_COMMIT}"
python3 -m py_compile \
  "${ROOT}/scripts/cw_judge_signal_phase0_prepare.py" \
  "${ROOT}/scripts/cw_judge_signal_phase0_diagnostic.py" \
  "${ROOT}/scripts/cw_judge_signal_phase0_finalize.py" \
  "${ROOT}/scripts/cw_phase0_archive.py" \
  "${ROOT}/scripts/run_train.py"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_contract.txt"
grep -q "RTX PRO 6000" "${RUN_ROOT}/gpu_contract.txt"
python3 -c 'import csv,sys; rows=list(csv.reader(open(sys.argv[1]))); assert len(rows)==1, rows; assert float(rows[0][2].strip()) >= 96000, rows' "${RUN_ROOT}/gpu_contract.txt"

python3 "${ROOT}/scripts/cw_judge_signal_phase0_prepare.py" \
  --output-dir "${RUN_ROOT}/preparation" \
  --model-cache-dir "${MODEL_CACHE}/models" \
  --debate-adapter "${DEBATE_ADAPTER}" \
  --judge-08b-adapter "${JUDGE_08B_ADAPTER}" \
  --judge-4b-adapter "${JUDGE_4B_ADAPTER}" \
  --solution-adapter "${RUN_ROOT}/prepared/solution_zero_b" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  --expected-image-ref "${EXPECTED_IMAGE_REF}" \
  --expected-image-digest "${EXPECTED_IMAGE_DIGEST}" \
  --debate-adapter-sha256 "${DEBATE_SHA256}" \
  --judge-08b-adapter-sha256 "${JUDGE_08B_SHA256}" \
  --judge-4b-adapter-sha256 "${JUDGE_4B_SHA256}"

MODEL_4B="${MODEL_CACHE}/models/qwen3.5-4b-base"
MODEL_08B="${MODEL_CACHE}/models/qwen3.5-0.8b-base"
SOLUTION_ADAPTER="${RUN_ROOT}/prepared/solution_zero_b"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -m pytest -q "${ROOT}/tests/unit" \
  --ignore="${ROOT}/tests/unit/test_debate_source_parity.py" 2>&1 | tee "${RUN_ROOT}/unit_tests.log"
python3 "${ROOT}/scripts/cw_constrained_writing_scorer_gold.py" \
  --source-root "${ROOT}" \
  --output "${RUN_ROOT}/scorer_gold_summary.json" 2>&1 | tee "${RUN_ROOT}/scorer_gold.log"

start_policy_server
log "bounded_diagnostic_start"
python3 "${ROOT}/scripts/cw_judge_signal_phase0_diagnostic.py" \
  --source-root "${ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostic" \
  --model-path "${MODEL_4B}" \
  --sglang-url "http://127.0.0.1:${POLICY_PORT}" \
  --debate-adapter "${DEBATE_ADAPTER}" \
  --solution-adapter "${SOLUTION_ADAPTER}" \
  --gold-summary "${RUN_ROOT}/scorer_gold_summary.json" 2>&1 | tee "${RUN_ROOT}/diagnostic.log"

if ! python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value.get("probes_pass") and value.get("phase0_pass") else 1)' "${RUN_ROOT}/diagnostic/FINAL_GATE.json"; then
  log "bounded_diagnostic_rejected optimizer_arms_skipped=true phase1_forbidden=true"
  finalize_preflight
  exit 0
fi

COMMON_TRAIN_ARGS=(
  --model-path "${MODEL_4B}"
  --tokenizer-path "${MODEL_4B}"
  --env constrained_writing
  --mode debate
  --adapter-layout split
  --steps 1
  --num-groups 16
  --group-size 4
  --rollout-batch-size 16
  --max-tokens 768
  --debate-r1-max-tokens 768
  --debate-r23-max-tokens 1536
  --temperature 1.0
  --top-p 1.0
  --min-p 0.0
  --seed 2026070901
  --request-seed-mode none
  --learning-rate 1e-5
  --weight-decay 0.01
  --max-grad-norm 1.0
  --advantage-mode zscore
  --ppo-clip-epsilon 0.1
  --debate-rounds 3
  --debate-r1-reward judge_rejection_task
  --debate-r23-reward constant
  --debate-r23-constant 1.0
  --debate-r23-mode symmetric
  --debate-r23-advantage-scope per_round
  --debate-round-adapter-names solution debate debate
  --debate-prompt-format qwen35_base_text_prefill
  --debate-stop-on-concluded
  --base-r2-prefill "${DEBATE_PREFILL}"
  --base-r3-prefill "${DEBATE_PREFILL}"
  --rollout-assistant-prefill "${R1_PREFILL}"
  --rollout-grad-accum-steps 1
  --train-minibatch-size 4
  --train-logprob-backend selective_lm_head
  --no-trace-model-io
  --train-adapter-names solution debate
  --gradient-checkpointing
  --on-policy-logprob-check
  --on-policy-logprob-abs-tol 0.001
  --sampler-backend sglang
  --sampler-sglang-base-url "http://127.0.0.1:${POLICY_PORT}"
  --sampler-sglang-timeout-s 900
  --sampler-sleep-before-training
  --sampler-max-model-len 16384
  --init-adapter-dir "solution=${SOLUTION_ADAPTER}"
  --init-adapter-dir "debate=${DEBATE_ADAPTER}"
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --lora-rank 32
  --resource-log-interval-s 2.0
)

run_arm() {
  local arm=$1
  shift
  log "phase0_arm_start arm=${arm} optimizer_steps=1"
  python3 "${ROOT}/scripts/run_train.py" \
    --output-dir "${RUN_ROOT}/arms/${arm}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "$@" 2>&1 | tee "${RUN_ROOT}/arm_${arm}.log"
  log "phase0_arm_done arm=${arm}"
}

start_judge_server real08b "${MODEL_08B}"
run_arm real08b \
  --debate-judge-server-url "http://127.0.0.1:${JUDGE_PORT}" \
  --debate-judge-server-adapter-path "${JUDGE_08B_ADAPTER}"
stop_judge

start_judge_server real4b "${MODEL_4B}"
run_arm real4b \
  --debate-judge-server-url "http://127.0.0.1:${JUDGE_PORT}" \
  --debate-judge-server-adapter-path "${JUDGE_4B_ADAPTER}"
stop_judge

run_arm mock2026071001 --debate-mock-judge-seed 2026071001
run_arm mock2026071002 --debate-mock-judge-seed 2026071002

finalize_preflight
log "phase0_bounded_run_complete raw_review_required=true phase1_forbidden=true"
