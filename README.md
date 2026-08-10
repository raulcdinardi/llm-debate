# llm-local-rl

Local-only RL rewrite for:
- `vllm` sampling
- LoRA training
- explicit token masking
- split adapter routing

Current priorities:
- keep the `ht_sequence` reward-hacking environment
- make masking first-class rather than implicit
- make adapter routing explicit per turn and per train batch
- make parity tests easy to run after engine changes

## Architecture

The key split is:
- environments define task instances, prompt content, parsing, and rewards
- episode builders define turn structure such as `single_turn` or `debate`
- samplers generate tokens for a named adapter
- trainers consume explicit masks for a named adapter

This avoids the previous shape where debate mechanics and task/environment logic leaked into each other.

## Test strata

- unit tests: mask construction, routing, environment behavior
- integration tests: `vllm` + LoRA loadability, sampling parity, replayed training parity

Heavy integration tests are opt-in and require a local model path.

## Tested vLLM lifecycle

Install the pinned GPU integration stack with `pip install -e '.[integration]'`.
It follows the vLLM 0.26 CUDA compatibility set (`torch==2.11.0`) and the exact
Transformers/PEFT versions used by the LFM2.5 training runtime.

vLLM now uses level-1 sleep between rollout and HF backward by default. The
base model remains CPU-offloaded, the KV cache is discarded, and only LoRAs
that will be trained are explicitly evicted. Frozen adapters such as `judge`
remain registered across the cycle. Use `--sampler-teardown-before-training`
only as a compatibility fallback; use `--sampler-sleep-level 2` only when CPU
memory cannot retain the base weights.

## PPO behavior-policy contract

PPO rollouts and trainer recomputation share one serialized
`BehaviorPolicySpec`. The stored old logprob and the trainer's current
logprob must both describe the same normalized distribution that generated
the action. Temperature is therefore applied in both paths. The parity gate
checks every sampled completion token and raises before ratio calculation,
backward, or an optimizer step if the values differ.

The current trainer reconstructs temperature scaling exactly. Trainable PPO
runs fail configuration validation when `top_p != 1`, `min_p != 0`, top-k is
enabled, or a repetition penalty is active, because those processors are not
yet identically normalized in both the sampler and trainer. These options
remain usable for non-trainable judge/evaluation sampling, but their returned
logprobs must not be consumed as PPO behavior logprobs.

Renderer-inserted R2/R3 continuation tokens carry an explicit
`behavior_logprob_mask=0`; sampled tokens carry `1`. The gate never infers
provenance from a numeric logprob such as zero. For SGLang, launchers pin
`SGLANG_RETURN_ORIGINAL_LOGPROB=false`; that declaration is not trusted as
proof by itself—the zero-update parity gate remains the runtime authority.

## Winner-modulated R1 reward

For split-adapter debate runs, `--debate-r1-reward judge_rejection_task`
implements literal winner rejection for R1: the judge-selected trajectory keeps
its objective task reward, the losing R1 trajectory is omitted, and selected
winner rewards are population-z-scored within each task-instance group. R2/R3 remain
independent: `--debate-r23-mode symmetric` still trains both speakers with
positive/negative constant rewards. The mode requires the round mapping
`solution, debate, ...` and is not supported by the shared-adapter layout.
