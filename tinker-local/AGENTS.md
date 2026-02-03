# Agent Instructions (tinker-local)

Follow these rules for all files under `tinker-local/`:

- Prefer explicit typing; avoid `Any` and `type: ignore`.
- Keep APIs small and compatible with the subset of Tinker used by `tinker_debate`.
- Async methods should be truly awaitable; preserve the SDK’s “double-await” pattern where callers do `fut = await ...; await fut`.
- Keep dependencies optional when feasible (import heavy libs lazily).
- Fail visibly: do not program defensively (`try/except`, `.get(..., default)`); avoid silent fallbacks.
- Keep scripts/file roles narrow (Unix philosophy); prefer small composable modules.
- Prefer minimal diffs and reproducibility (explicit seeds, log config, deterministic output names).
