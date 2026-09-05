# Debate and judge GRPO modes

The split-adapter debate runner can train the solution, debate, and judge LoRAs
independently while serving all rollout and judge requests from the same vLLM
engine.

## Frozen-judge, debate-only training

Load a dedicated judge adapter for inference and restrict optimizer work to the
debate adapter:

```text
--adapter-layout split
--debate-judge-adapter judge
--debate-judge-harness solution_r1_rationale_v1
--init-adapter-dir solution=/path/to/solution
--init-adapter-dir debate=/path/to/debate
--init-adapter-dir judge=/path/to/judge
--train-adapter-names debate
```

The judge remains available in vLLM. Before backward, only adapters that both
produced training examples and appear in `train_adapter_names` are evicted. The
frozen judge therefore remains loaded while the updated debate adapter is
refreshed after the optimizer step.

## Bidirectional coherence sampling

`--debate-judge-bidirectional` renders each completed transcript in the original
and A/B-reversed order. The reverse verdict is mapped back to the original
referent. Agreement is recorded as `order_invariant`; disagreement selects a
deterministic seeded fallback winner for policy rewards and remains visible in
the audit metrics.

The exact forward and reverse judge behavior-policy turns are retained only in
memory. Step-record JSON omits those token arrays. Bidirectional mode does not
replace invalid outputs with a greedy retry because that retry would come from a
different behavior distribution.

## Judge coherence GRPO

Enable judge optimization with this complete prerequisite set:

```text
--adapter-layout split
--debate-judge-adapter judge
--debate-judge-harness solution_r1_rationale_v1
--debate-judge-bidirectional
--train-judge-coherence-grpo
--temperature 1.0
--debate-judge-temperature 1.0
--train-adapter-names solution debate judge
```

Each of the two judgment turns receives the pair-level reward `+1` when the
mapped judgments agree and `-1` otherwise. Rewards are z-scored once across the
complete judgment population for the optimizer step, not within each equal-
reward pair. If every debate has the same coherence label, the reward standard
deviation is zero and judge advantages are intentionally zero.

Judge GRPO requires the judge sampler and trainer reconstruction to use the same
behavior-policy settings. Configuration validation rejects mismatched sampling
distributions. If sampling controls are overridden, set their rollout and judge
counterparts to the same values (for example, `--temperature` together with
`--debate-judge-temperature`, and `--top-p` together with
`--debate-judge-top-p`). Both temperatures default to `1.0`, but they are shown
explicitly above so the documented contract stays self-contained.

Judge LoRAs are bound to their training harness by a `judge_harness.json`
sidecar. Bind an existing adapter once before using it:

```text
python scripts/bind_judge_harness.py \
  --adapter-dir /path/to/judge \
  --harness solution_r1_rationale_v1
```

Loading fails closed if the sidecar is missing, names another harness, or has a
stale prompt fingerprint. Newly saved judge checkpoints inherit the configured
harness automatically.

## Related reward controls

- `judge_delta_task` gives each R1 task reward an additive
  `+/- q * abs(task_reward_a - task_reward_b)` modulation based on the selected
  winner.
- `--debate-incoherent-r23-reward` controls the reward assigned to both debate
  paths when bidirectional judgments disagree.
- `--debate-r2-max-tokens` and `--debate-r3-max-tokens` override the shared R2/R3
  cap independently.

The GRPO and reward behaviors above are opt in. New policy and judge sampling
both default to temperature `1.0`; explicit values in existing run
configurations are preserved.
