# LLM debate

Two supported entry points share this repository:

- **Current local RL:** `PYTHONPATH=src python scripts/run_train.py --help`.
  See [local RL usage](docs/local_rl.md) for vLLM/LoRA training, checkpoints,
  judge contracts and arbitrary debate depths.
- **Legacy Tinker:** existing `scripts/train.py`, `src/tinker_debate`,
  `tinker-local`, and `requirements.txt` remain at their original paths.
  See [the preserved Tinker guide](docs/legacy_tinker.md).

The two dependency environments should remain separate. Installing local RL
is not an instruction to upgrade an existing experiment environment.

Current runtime templates live in `prompts/local_rl`; legacy templates remain
in `prompts`. The template contents and legacy command paths are preserved.

See [migration and compatibility](docs/migration/README.md) for source identities,
validation, experiment tooling and rollback. Existing experiment branches and
worktrees remain usable; this migration does not change running jobs.
