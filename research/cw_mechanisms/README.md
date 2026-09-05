# Preserved CW mechanism tools

These recipes preserve the actual September experiments without replacing
running or archived copies. They cover causal count interventions, late
cross-play, early checkpoints, effective-weight amplification through ±32,
resumable scoring, and the archived-score orthogonal R1 pilot.

Each manifest records original and staged code hashes plus the precise frozen
runtime files it requires. The runtime is intentionally **not** replaced with
current main: historical prompt/reward semantics are part of the experiment.
Models, checkpoints, outputs, credentials and provider launch scripts are not
in this repository. Recover frozen sources and inputs from the corresponding
experiment archive. Manifests reject the wrong runtime before writing anything.

Stage in a fresh directory (this command does not execute code or call APIs):

```bash
python scripts/stage_research_recipe.py early_amplification \
  --source-directory /recovered/frozen_source \
  --destination /new/experiment_directory
```

Supply the specification's exact input files under `inputs/`, including model
and adapter assets; preserve the specified seeds, prompt panels and checkpoint
identity. The early evaluator accepts `CW_HOST=vm02` or `vm03` to select the
original shard/window assignment. Use the existing dependency environment
matching the archived run. Run from the staged directory using `control/`;
`evaluate.py --pilot` measures the early experiment. Full release remains
explicit via the original `execution/release.json` gate. No release is staged.
Use `score_v2.py` for the repaired scoring transport, with an environment API
key and the original cumulative budget state. The orthogonal pilot reads its
key from the environment; it selects from archived scores, without re-scoring
baseline ties. These are preserved experiment tools, not a new authorization
for model inference, API spending or a provider fallback.

Recipes retain historical model/provider names and schema assumptions. Provider
availability is not established by this migration. A replay still needs the
normal experiment preflight and its own frozen specification. Do not interpret
staging or CPU tests as a completed scientific replication.
