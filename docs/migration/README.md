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
