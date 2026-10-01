# Separate trainable judge backbone

Warm-up and coherence RL use the same `TrainingDriver`, rollout assembler and `MultiAdapterTrainer`. The objective controls whether judge updates use supervised A/B label CE with optional JS, unsupervised JS coherence, or GRPO. A separate HTTP judge already supported inference, but did not provide a second trainable backbone or optimizer.

Set `judge_model_path` (CLI: `--judge-model-path`) to opt into separate actor and judge backbones. Optional `judge_tokenizer_path` overrides the judge tokenizer. Actor rounds use the actor tokenizer; judge prompts, label constraints, EOS and judge training rows use the judge tokenizer. The `judge` and optional `judge_shadow` adapters belong to the judge model. Other adapters belong to the actor. Each component retains its own Adam state and capacity memory. This first version deliberately shares learning rate, rank, optimizer settings, PEFT target modules and kernel options. Both backbones must support the configured targets/kernels; it does not infer different scientific hyperparameters for the judge. For different architectures, select a verified common target list and use automatic GDN dispatch unless both models support the selected FLA implementation. One checkpoint transaction saves both components plus the existing global RNG state.

The first version supports vLLM engines managed by the driver only, with persistent engines and sleep level 1. Only one sampler or trainer backbone is active at a time. vLLM V1 must keep its default separate worker processes; `VLLM_ENABLE_V1_MULTIPROCESSING=0` is rejected. Trainable LoRAs are unloaded before sleep; a later request to unload those already-evicted adapters is a no-op. The router rejects removal of a frozen adapter from a sleeping engine. Explicitly set `trace_model_io=false` (`--no-trace-model-io`): the global token tracer does not yet support two vocabularies, and config validation rejects an ambiguous trace. External/mock judge settings and placing judge adapters in actor rounds are rejected.

Example additions to a valid existing Qwen warm-up config:

```json
{
  "model_path": "Qwen/Qwen3.5-4B",
  "judge_model_path": "Qwen/Qwen3.5-0.8B",
  "judge_sampler_gpu_memory_utilization": 0.25,
  "debate_judge_adapter": "judge",
  "train_judge": true,
  "trace_model_io": false
}
```

Keep the existing objective/harness/label-contract checks. Supervised warm-up requires labels; JS coherence does not obtain labels from the supervised objective. A frozen heterogeneous judge uses the same routing with `train_judge=false`. Initialize each adapter from its corresponding backbone; no actor LoRA can be loaded into the smaller judge by renaming it.

The unchanged single-backbone path remains the default. Neutral new fields preserve compatibility with earlier exact-resume fingerprints. Model identities, non-default options, Adam state and adapter inventories remain checked. Resume preserves the training RNG across sampler construction.

Validation includes two native tiny Llama backbones with different hidden sizes/vocabularies, actual actor PPO and judge CE/JS/GRPO updates, cross-model parameter isolation, both Adam/capacity states and identical next-update resume, and native judge prompt/label/EOS routing. This does **not** certify a production Qwen4B+0.8B GPU launch: real dual-vLLM engine loading, sleep/wake, memory fit and behavior-logprob parity still require the normal local Phase-0 gate before full training. Phase 0 should retain durable evidence outside the GPU disk and should not create a W&B run.
