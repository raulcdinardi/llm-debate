# Drive Snapshot vs Local Worktree

Date: 2026-03-08

Compared directories:

- Local desktop repo: `/mnt/c/Users/raulc/Desktop/llm-debate`
- Drive snapshot copy: `/mnt/c/Users/raulc/Desktop/tinker_debate_drive`
- Original Drive location: `G:\My Drive\tinker_debate`

Scope of comparison:

- The local side is the current working tree, including uncommitted and untracked files.
- The Drive side is a plain folder snapshot, not a Git checkout.

## High-level conclusion

These are not just two nearby revisions of the same checkout.

- The local desktop repo is a full Git repository with source code, prompts, scripts, docs, local backend code, and local run artifacts.
- The Drive snapshot is a smaller project snapshot oriented around logs, archived run outputs, and a few top-level docs/files.
- There is some overlap in top-level files (`AGENTS.md`, `README.md`, `debate_types.py`, `decisions.md`, `requirements.txt`, `rl_basic.py`, `viz.py`), but the local repo contains most of the code directories and active work.

## Structure differences

Present only in the local desktop repo:

- `.git/`
- `.tinker-local/`
- `.venv/`
- `docs/`
- `logs/`
- `prompts/`
- `scripts/`
- `src/`
- `tinker-local/`

Present only in the Drive snapshot:

- `.env`
- `logs_remote/`
- `logs_summary_train_30steps_t12.tgz`
- `outcodex.txt`
- `tinker_logs_20260120_231826_summary_s8.tgz`
- `tinker_logs_20260120_233615_debate_s3_strict.tgz`

Implication:

- The Drive folder looks like a partial snapshot plus experiment artifacts.
- The local desktop folder is the active development repo.

## Overlapping files with different contents

Files present in both trees but with different contents:

- `AGENTS.md`
- `README.md`
- `decisions.md`

Files present in both trees and apparently matching:

- `debate_types.py`
- `requirements.txt`
- `rl_basic.py`
- `viz.py`
- `.gitignore`
- `.env.example`

## README.md differences

The Drive snapshot README is more explicit about named runs and result indexing.

Drive-only or Drive-newer points:

- Example commands append `--name ...` in more places.
- Local backend wording says `with_local_tinker` sets `TINKER_BACKEND=local` without shadowing the real `tinker` import.
- Logging section describes the newer run directory naming pattern:
  `logs/<timestamp>_n<num_rollouts>_g<num_groups>_<mode>_<env>_<dataset_or_custom>_<name>/`
- Logging section includes `run_metadata.json`, `baseline_*.json`, and `summary_*.json`.
- README documents a results search hub built via `scripts/build_results_search_hub.py`.
- README documents notebook visualizer behavior and lazy step indexing.

Local-only or local-newer points:

- The current local README is shorter and does not include the results search hub or notebook visualizer sections.
- The current local README still describes an older logs layout (`logs/test_runs`, `logs/gpqa_runs`, `logs/experiments`).

## AGENTS.md differences

The Drive snapshot `AGENTS.md` contains additional workflow rules not present in the local repo copy.

Drive-only additions include:

- Naming guidance for folders/files, emphasizing descriptive, sortable `snake_case` names.
- Git hygiene guidance:
  - one commit per logical change
  - descriptive commit messages
  - inspect `git status --short` and staged diff before committing
  - avoid rewriting shared history
- Remote-run guidance:
  - prefer named `tmux` sessions over `nohup`
  - enforce a single queue owner per host/workdir
  - prefer `rsync -av --relative` over brittle `scp`
  - use `scripts/vastai_autostop.py` in both local and cloud contexts
  - use GPU wattage as a primary health/activity signal
  - verify remote launches twice
  - include host/session/PID/command/log path in decision logs

## decisions.md differences

This is the largest content divergence.

Drive snapshot `decisions.md`:

- Contains a much longer history centered around February 2026.
- Includes graph-path/spec-gaming work.
- Includes constrained-writing experiments.
- Includes GPU / vast.ai / remote queueing / notebook / logging work.
- Includes result syncs into `logs_remote/` and `results/`.
- Does not include the latest March 4-5 local work now present in the desktop repo.

Local desktop `decisions.md`:

- Contains March 4-5, 2026 entries about:
  - repo familiarization
  - CPU coin training run
  - local backend dependency/setup issues
  - LoRA target module fixes for local backend
  - creation of the `ht_sequence` environment
  - prompt refinement for fair-coin H/T outputs
  - optimizer switch support for local backend (`adamw` / `sgd`)
  - fast coin and `ht_sequence` SGD runs
- The local file currently has uncommitted additions beyond the checked-in version.

Net effect:

- The Drive snapshot preserves a broader older experiment log.
- The local repo preserves newer task/runtime development and local test history.

## Local uncommitted changes

Current local modified tracked files:

- `decisions.md`
- `scripts/train.py`
- `src/tinker_debate/tasks/__init__.py`
- `src/tinker_debate/train/orthogonal_driver.py`
- `tinker-local/src/tinker/_transformers_backend.py`

Current local untracked files:

- `prompts/tasks/ht_sequence_user.md`
- `src/tinker_debate/tasks/ht_sequence_task.py`
- `.venv/` (environment folder, not source)

`git diff --stat` summary for tracked uncommitted changes:

- 5 files changed
- 103 insertions
- 3 deletions

### What the local uncommitted code changes do

#### 1. New `ht_sequence` environment

Local-only changes add a new single-turn task named `ht_sequence`.

Files involved:

- `scripts/train.py`
- `src/tinker_debate/tasks/__init__.py`
- `src/tinker_debate/train/orthogonal_driver.py`
- `src/tinker_debate/tasks/ht_sequence_task.py` (untracked)
- `prompts/tasks/ht_sequence_user.md` (untracked)

Behavior added locally:

- CLI `--env` now accepts `ht_sequence`.
- CLI adds `--ht-seq-len` with default `8`.
- Driver wiring creates `HTSequenceTask(sequence_len=...)` when `--env ht_sequence` is selected.
- New prompt asks the model to output exactly a sequence of H/T coin flips.
- Reward is computed as the count of `H` characters parsed from the first `N` H/T characters in the completion.
- Metrics logged include:
  - parse success
  - target length
  - parsed length
  - parsed sequence
  - number of H
  - number of T
  - full decoded text

Interpretation:

- This is a reward-hacking canary task intended to make it easy to observe optimization pressure toward emitting more `H`.

#### 2. Local optimizer switch support

Local-only changes in `scripts/train.py` and `tinker-local/src/tinker/_transformers_backend.py` add optimizer selection for the local backend.

Behavior added locally:

- New CLI flag `--opt` with choices `adamw` or `sgd`.
- New CLI flag `--sgd-momentum`.
- New CLI flag `--sgd-nesterov`.
- Validation prevents `--sgd-momentum` or `--sgd-nesterov` without `--opt=sgd`.
- Validation rejects `--opt=sgd` outside local backend mode.
- CLI populates local backend env vars:
  - `TINKER_LOCAL_OPTIMIZER`
  - `TINKER_LOCAL_SGD_MOMENTUM`
  - `TINKER_LOCAL_SGD_NESTEROV`
- Local backend optimizer builder now supports:
  - `torch.optim.AdamW`
  - `torch.optim.SGD`

Interpretation:

- The local repo is actively experimenting with optimizer behavior in local training runs.

#### 3. LoRA target module fallback changes

Local-only changes in `tinker-local/src/tinker/_transformers_backend.py` expand LoRA target-module handling.

Behavior added locally:

- Existing explicit handling for `lfm2` is preserved.
- New explicit handling is added for `qwen3_5`.
- If no architecture-specific mapping is found, local code falls back to `target_modules = "all-linear"`.

Interpretation:

- This was added to make PEFT/LoRA setup work more reliably across model families in the local backend.

#### 4. Local decisions log additions

The local `decisions.md` includes uncommitted entries documenting:

- implementation of `ht_sequence`
- prompt updates for fair-coin wording
- local optimizer switch work
- coin and `ht_sequence` SGD runs

These entries are not present in the Drive snapshot.

## Relationship between the two trees

The Drive snapshot does not look like a direct replacement for the local repo.

More accurate interpretation:

- The Drive snapshot preserves older documentation and experiment artifacts that are absent from the current local repo.
- The local repo contains newer source-code work and uncommitted task/runtime changes that are absent from the Drive snapshot.
- Any merge should be selective, especially for `README.md`, `AGENTS.md`, and `decisions.md`, because each side contains useful but different information.

## Practical merge guidance

If this comparison is used for integration later, likely candidates are:

- bring selected documentation improvements from the Drive `README.md` into the local `README.md`
- review whether the extra workflow guidance in the Drive `AGENTS.md` should be restored locally
- merge older historical entries from the Drive `decisions.md` only if preserving the full experiment timeline matters
- keep the local uncommitted code changes as the current source-of-truth for active development unless they are intentionally being discarded
