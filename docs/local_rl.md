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

## Inference execution

vLLM inference defaults to `sampler_enforce_eager=false`, allowing its CUDA graph
and compilation optimizations. Use `--sampler-enforce-eager` to explicitly select
eager execution. JSON configurations that omit the field use the graph-capable
default; explicitly saved values are preserved when loading existing runs.

## Training activation memory

Optional [fused training kernels](fused_training.md) add strict FLA GDN selection
and a Triton LM-head scoring path; these are independent of inference caching
and backbone activation checkpointing.

Backbone activation checkpointing defaults to `gradient_checkpointing=false` in
the shared trainer, run config, JSON loader and CLI. This retains intermediate
activations to avoid replaying decoder layers during backward, trading more GPU
memory for less recomputation. Use `--gradient-checkpointing` or an explicit
JSON `true` to enable it. Explicit values in saved run configs are preserved.

The selective LM-head path still checkpoints projection/loss chunks independently,
so vocabulary-sized intermediates do not accumulate across chunks. This default
does not change model/optimizer checkpoint saving or inference KV caching.
Memory pressure lowers the physical microbatch through the existing OOM fallback;
it does not automatically enable backbone checkpointing.

## Multiple optimizer updates per rollout

`--train-optimizer-batch-size 32 --train-minibatch-size 8` takes four optimizer
updates from a rollout with 128 training examples **for that adapter**. Each
update accumulates four physical forward/backward minibatches of eight examples.
There is one pass through the rollout: each example is used once, with the
original rollout logprobs, rewards and group-computed advantages held fixed.
The default optimizer batch size `0` retains one update over the full batch.

The size counts `TrainExample` rows, not prompts or tokens. Adapter batches may
have different row counts and therefore different update counts. A short final
optimizer batch gets its own update, normalized by its actual row count. An
all-zero-advantage PPO optimizer batch performs no update (including no weight
decay). Existing row order is retained; optional length bucketing still applies.
Direct CE/JS judge training keeps forward/reverse pairs together and requires
both batch sizes to be zero or even. No extra epochs or shuffling are added.

**Cost:** multiple PPO updates add a forward-only pass over the rollout, in
physical minibatches, to verify all rollout logprobs before any weights change.
Later PPO ratios/clipping use the updated policy against those same original
logprobs. This parity pass is unnecessary for direct CE/JS judge objectives.

`--steps`, rollout accumulation and checkpoint cadence still count rollout
batches. Metrics include per-adapter `num_optimizer_steps` and
`num_preflight_minibatches`. Loss is the example-weighted mean across optimizer
batches, token diagnostics include the whole rollout, and gradient norm/max
metrics report the maximum across updates. Selected-layer optimizer diagnostics
report the last update. Existing forward-token/minibatch counters count training
passes; the additional parity pass is counted separately. Old exact-resume
checkpoints remain compatible with the default; changing optimizer batch size
changes the configuration fingerprint and requires a new run.

### Four updates of 32 trajectories, keeping groups intact

For a new round-wise debate run, use:

```bash
--train-optimizer-batch-size 32 --train-minibatch-size 32 --train-keep-groups-together
```

For the HS3/OBQA layout of 64 groups of two trajectories, this produces four
optimizer batches containing 16 complete debate pairs each. Eight groups of
16 trajectories also pack into four batches, with two groups each. Physical batch 32 gives one
forward/backward and one optimizer step per batch. Both rounds of one agent
remain one merged trajectory row. Both orderings of a supervised CE/JS judge
pair remain together; with 128 judge rows this also gives four judge updates.

The assembler attaches a stable ID for the full original question/instance
normalization group. The trainer packs complete groups in first-seen order,
then optionally length-sorts **within** each optimizer batch. Rewards and
advantages are computed once on the original groups and are not recomputed
after an update. All PPO parity checks run before the first optimizer step;
later ratios use the frozen rollout logprobs against the updated policy.

An OOM fallback to physical batch 16 accumulates two backwards within each
fixed 32-row optimizer batch; it never repartitions the optimizer groups.
Unequal groups can produce short batches, normalized by their actual row count.
An oversized group, missing group metadata, or partial overlength filtering
fails before that adapter's first optimizer step. Plans log exact group IDs
and row counts in `optimizer_group_plan`. All-zero-advantage PPO batches retain
the existing no-update behavior.

This changes optimization: four Adam updates, clipping and weight-decay
applications replace one. It is not four times less backward computation;
PPO additionally performs the existing no-gradient parity pass. Learning rate
is not automatically rescaled. Rollout/checkpoint step counts stay unchanged.
The option defaults off for compatibility; exact-resume fingerprints distinguish
the new behavior. Legacy shared projection is unsupported. The older global
judge-GRPO objective forms one global group, which cannot be split into 32-row
updates under this option; supervised label CE/JS uses the prompt groups.

### Remembered physical capacity

The shared trainer treats `train_minibatch_size` as a physical ceiling. On an
out-of-memory failure **before any optimizer step starts**, it discards partial
gradients, restores the RNG state and retries with half as many rows. It remembers
the reduced ceiling separately for each adapter, including in exact-resume
checkpoints. Later batches reuse it; larger inputs may lower it further. Limits
never increase automatically, trading some potential throughput on shorter
batches for avoiding repeated failed capacity searches. Older checkpoints without
this field start without a learned limit.

The effective optimizer batch, row coverage and forward/reverse judge pairing
are preserved. Reference logprob scoring also honors the physical limit. An OOM
inside or after an optimizer step propagates without replay, since weights or
optimizer state may already have changed. Retries stop at one row (two for paired
judge objectives); non-OOM errors are never treated as capacity failures.

Per-adapter metrics report `physical_microbatch_size`, `microbatch_initial_size`,
`microbatch_oom_attempts` and `microbatch_retry_seconds` (failed attempt time,
excluding cleanup); each failed attempt also emits a `train_microbatch_oom`
event. This is built into `MultiAdapterTrainer`:
new launches must not install the old experiment-specific `backward_batching`
wrapper on top. Historical frozen experiments retain their original source.

## Test strata

- unit tests: mask construction, routing, environment behavior
- integration tests: `vllm` + LoRA loadability, sampling parity, replayed training parity

Heavy integration tests are opt-in and require a local model path.

## Default observability and recovery

Training enables W&B by default (`--no-wandb` disables it) and reads
authentication only from the environment. Local files remain canonical: every
committed optimizer step is appended to `step_records.jsonl`, while W&B gets a
single scalar commit at the same global step. Complete rollout records are
published as immutable, compressed 10-step artifacts plus a small deterministic
Table preview. A transient W&B failure is recorded under `observability/` and
does not stop training.

Trainable LoRAs are retained under `checkpoints/lora/` every 10 steps and at the
configured final step. Intervening exports live under `.live_adapters/` only as
rollout-engine handoff files and are removed after replacement. Every 50 steps
and at the final step, `checkpoints/exact_resume/` receives an atomically
published bundle containing all active adapters, AdamW state, Python/NumPy/Torch
CPU/CUDA RNG state, configuration fingerprint, per-file hashes, and a `READY`
marker. Resume rolls back any uncheckpointed tail and refuses to silently call
an adapter-only state an exact resume.
PEFT `target_parameters` adapters cannot currently be reconstructed with the
same multi-adapter optimizer topology; such runs must explicitly use
`--optimizer-checkpoint-every 0` until that PEFT limitation is removed.

The trainer logs reward/component distributions, completion and group
statistics, entropy and rolling changes, PPO ratios and tails, clip-high/low,
advantages, sampled old-policy KL/logprob deltas, gradient clipping, LR,
nonfinite counters, and representative LoRA-layer gradient/Adam diagnostics.
Sampled-token KL to the frozen initialization is measured every 10 steps by
default (`--reference-kl-every`; `0` disables it), since measuring it every
step requires an additional policy forward.

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

## Variable debate depth

`--debate-rounds` is the maximum depth for a rollout. Round 2 uses the opening
debate prompt; round 3 and every later active round use the same rebuttal prompt
contract. Debates that have reached their assigned depth are omitted from later
generation batches.

Depth assignment is a named, serialized policy invoked once for every task
group. Built-ins are `fixed`, `shuffled_multiset`, and `categorical`. For
example, this independently samples each debate's depth within every group:

```bash
--debate-rounds 6 \
--debate-depth-policy categorical \
--debate-depth-policy-params-json '{"depths":[3,4,6],"weights":[1,2,1]}'
```

`--debate-rounds-per-group 3 3 4 4` is a convenience alias for a
`shuffled_multiset` policy. It preserves that exact composition and independently
permutes it within every group.

Custom policies register a callable under a stable name with
`register_debate_depth_policy`. They receive `DebateDepthContext`, including the
optimizer step, rollout and group indices, sampled task-instance metadata, and a
deterministically seeded group-local RNG. Register the policy before constructing
or restoring `TrainRunConfig`; persist only its JSON parameters in
`debate_depth_policy_params`. The policy name and parameters are stored in the run
manifest, while the implementation is pinned by the run's source revision.

## Reusing the R2 prefix in R3

For the vLLM sampler, opt in with `--sampler-prefix-caching`. This passes
`enable_prefix_caching=True` and `mamba_cache_mode="align"` to the engine and
checks that both settings took effect. LFM2.5 needs aligned hybrid-state caching;
its vLLM 0.26 implementation rejects the `all` mode.

R3 already extends the exact R2 prompt and generated token IDs, using the same
version of the debate LoRA. Prefix caching can reuse matching resident blocks.
It is an engine-wide setting: other same-adapter matching prefixes can also hit.
It does not guarantee reuse of the entire prefix, prevent eviction, or reuse
state across different LoRA identities. The existing adapter identity changes
when training saves a new version, preventing reuse of stale policy state.
Prompts, token counts, rewards and optimizer behavior stay unchanged. Floating
point differences between cached and uncached execution still require validation.

Omitting the flag preserves the previous engine defaults. Explicit
`--no-sampler-prefix-caching` disables it. Both flags are vLLM-only; specifying
one for another backend fails configuration validation. Existing exact-resume
fingerprints remain compatible when the flag is omitted.

The GPU regression requires vLLM 0.26, a compatible GPU and local model/LoRA
artifacts. Set `LLM_LOCAL_RL_BASE_MODEL`, `LLM_LOCAL_RL_ADAPTER_A` (debate), and
`LLM_LOCAL_RL_ADAPTER_B` (a different compatible LoRA), then run:

```bash
PYTHONPATH=src python -m pytest -s tests/integration/test_vllm_prefix_cache.py
```

It checks actual R3 cache hits, cached versus cold greedy token equality and
logprobs (absolute tolerance 0.01), plus zero cache hits across different and
refreshed LoRA identities. This is a correctness/cache-hit gate, not a throughput
benchmark. Run it on the intended GPU environment before enabling experimental
training. CPU configuration tests do not establish real cache hits or speedup.

Backend references: [LFM2 aligned-cache support](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/lfm2.py#L424),
[vLLM hybrid cache defaults](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/engine/arg_utils.py#L2391).

### Grouped-update OOM recovery

When one rollout produces multiple optimizer updates, the trainer keeps a CPU
snapshot of the active adapter, its Adam state and the pre-iteration RNG state.
An OOM outside `optimizer.step()` restores that snapshot before retrying with a
smaller physical batch, preserving the original optimizer-group order. The
capacity ceiling remains remembered. An exception inside `optimizer.step()` is
fatal because the update may have partially mutated state.

Intermediate optimizer calls are provisional until the complete adapter batch
returns; rollback events explicitly record discarded updates. Snapshot bytes,
copy time and discarded-update counts are logged. This adds CPU memory and
transfer work, without changing the intended four sequential updates.
