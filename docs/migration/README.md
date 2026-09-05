# Main consolidation

The integration preserves both `5796701` (legacy main) and `10ede7c` (latest
full-gap CW local RL) as ancestors. Their histories were independently rooted.
Legacy files stay in place rather than being deleted by the runtime migration.

`legacy_inventory.json` binds unchanged legacy implementation, scripts and
prompts. `runtime_inventory.json` binds the imported runtime snapshot; paths
record relocations. The only production Python change during initial import
is the local RL prompt directory, from `prompts` to `prompts/local_rl`.
All template bytes remain identical to their respective source snapshots.

No existing branch is rewritten or deleted. Existing worktrees, dependency
environments, model files, outputs, timers and remote jobs are untouched.
Keep runs pinned to their recorded commit and environment. Roll back a future
main deployment by checking out its recorded previous SHA in a fresh directory;
do not reset a dirty experiment checkout.

This preserves legacy main, not arbitrary uncommitted changes in a developer
checkout. Those remain in their existing checkout and require separate review.
GPU/model-dependent tests remain distinct from CPU compatibility validation.

## Recovered scientific branch

`f7b7975` is also an ancestor: it supplies the direct unlabeled-JS objective,
MMLU prompt-group soft-score standardization and canonical token validation.
The merge retains CW raw later-round scores and generalized depth, and tests
all three soft reward modes on mixed depths. The September source's intended
8×16 default is restored; historical profiles specify geometry explicitly.
No reward mode is silently substituted in a stored configuration.

## Verification and tools

- `python scripts/verify_migration.py` checks legacy file hashes and both
  prompt inventories offline.
- `python -m pytest -q -ra tests/unit` discovers the whole unit suite; CI runs
  this on Python 3.11/3.12. The former parity test's laptop-only source paths
  are replaced with its hash-bound, test-only legacy snapshot.
- [Profiles](../../profiles/README.md) validate the real CLI-to-config wiring
  without constructing samplers or trainers in tests.
- [Research recipes](../../research/cw_mechanisms/README.md) preserve the
  mechanism runners and scoring tools with frozen-source staging.

Open PR ancestry incorporated: #3, #4, #6, #7, #8, #9, #10, #11, #12, as well
as their already-merged dependency commits #2/#5. No old branch or PR is
removed by this migration; remote state may be retired separately only after
unique work is accounted for. Local dirty work is intentionally not published.
