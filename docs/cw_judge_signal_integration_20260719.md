# Constrained-writing judge-signal integration decisions (2026-07-19)

## Purpose and provenance

This integration makes the reviewed `judge_rejection_task` R1 projection
usable with the newer constrained-writing, SGLang, trainer, and sampling
runtime. Commit `cdc1d89` was the source-only boundary. The subsequent
Phase-0 preparation adds the bounded diagnostic, one-step four-arm launcher,
immutable-image recipe, artifact finalizer, and tests. It still stops before
GPU rental, Phase-0 execution, or Phase 1.

The integration branch starts at merge commit `36d6bc1` (PR #2), whose first
parent is `990f784`. The newer runtime was an uncommitted working-tree snapshot
on top of `990f784` in `llm-debate-sft-countdown`; therefore this PR is also the
first reviewable commit boundary for that imported runtime. This provenance is
important: the large runtime/trainer diff is not newly designed by the conflict
resolution, even though it becomes visible in this PR.

## Resolved definitions

### R1 judge-rejection projection

- Only debates with a parsed `A` or `B` verdict enter training. A judge result
  that remains invalid after the existing retry is retained as `INVALID` and
  the whole debate is omitted; this mode does not invent a random winner.
- Exactly the judge-selected R1 trajectory is emitted. The losing R1
  trajectory is absent rather than assigned a negative or zero reward.
- The emitted R1 trajectory receives its own objective constrained-writing
  task reward. Judge confidence or a fixed win reward is not substituted.
- Selected-winner rewards are population-z-scored within the actual task
  instance. `instance_id` is used when available, with the legacy question
  string only as a fallback. This avoids pooling constrained-writing tasks that
  share a rules-blind topic but have different rule sets.
- A selected-winner group with zero reward variance receives zero R1
  advantages. No artificial variance is introduced.
- R1 is always routed through `round_adapter_names[0]`. The public
  `judge_rejection_task` configuration remains deliberately stricter and
  requires `solution, debate, ...`; its R1 adapter must be distinct from every
  later-round adapter.

### R2/R3 symmetric mode

Winner rejection applies only to R1. For each valid debate, R2 and R3 still
train both speakers. With `debate_r23_mode=symmetric`, the winner gets the
configured positive constant and the loser gets its negative. When R2 and R3
share the `debate` adapter, their transitions remain merged into one training
example per speaker; the inherited default `per_round` scope divides each
round's reward independently over that round's generated tokens.

### Telemetry

Projection telemetry classifies the examples that were actually emitted by
their `agent` and `verdict` metadata. In particular,
`loser_r1_example_count` is measured from output examples; it is not a literal
zero asserting the intended behavior. Expected-vs-actual deltas, unclassified
examples, missing group metadata, and live/zero-variance group counts are also
recorded.

### Qwen3.5 Base prompt contract

- The only selected debate Base-text renderer is the existing internal
  `qwen35_base_text_prefill` path. No second "official chat template" renderer
  was retained.
- R1 is ordinary text encoded once with `add_special_tokens=False`:
  `User:\n{task}\nAssistant:\n{prefill}`. It does not call
  `apply_chat_template`, does not inject Qwen role special tokens, and does not
  use `enable_thinking=False`.
- The experiment-level non-empty R1 prefill is pinned separately as the exact
  bytes `Ok, I will produce a 3-sentence story adhering to the rules:\n`.
  Runtime code accepts it through `rollout_assistant_prefill`; it is jointly
  encoded with the prompt and remains prompt context, outside the trainable
  completion/loss mask.
- This PR does not silently make that experiment-specific string a global
  runtime default. The later bounded Phase-0 spec must explicitly define its
  non-empty cell and paired empty-prefill control.

### `CONCLUDED` stopping

- Real `CONCLUDED` stopping is an explicit switch,
  `debate_stop_on_concluded`, independent of the prompt renderer selection.
- Configuration fails closed unless the run is debate mode with at least two
  rounds, the internal Base-text renderer, and the SGLang backend.
- The string stop applies only to R2/R3. SGLang is asked to retain the sentinel
  in generated tokens/text while stopping immediately after it. The visible
  debate text strips the sentinel for judging, while raw completion text and
  generation metadata remain available for diagnostics.
- The old heuristic numbered-list truncation remains only when real stopping
  is disabled. vLLM and Transformers reject string-stop requests instead of
  approximating incompatible behavior.

### Sampling and scorer contracts

- `top_p` is explicit in config/CLI and is propagated through SGLang, vLLM,
  and Transformers along with the existing `min_p`.
- In the constrained-writing scorer, positional rule entries equal to zero
  mean "not applicable," not failure. Exact three-sentence parsing remains a
  prerequisite for all-rules-satisfied metrics.

## Conflict-resolution map

| Area | Resolution |
|---|---|
| `scripts/run_train.py` | Keep the newer runtime CLI, retain testable `parse_args(argv)`, add judge-rejection, `top_p`, and explicit stop flags. |
| `debate_parity.py` and projection tests | Preserve reviewed winner-only R1 semantics; keep newer same-adapter R2/R3 merging; group by task instance; measure actual emissions. |
| `driver.py` and config tests | Keep newer runtime/sampler/trainer wiring; pass the effective R1 prefill and explicit stop flag; retain reviewed fail-closed reward-mode validation. |
| `debate_runtime.py` | Use only the existing internal Base-text renderer; remove the competing official-template path; decouple real stopping from renderer identity. |

## Bounded Phase-0 follow-up

- The diagnostic tests the exact D036 R2/R3 token-prefix extension and the
  exact newline-terminated R1 prefill, plus scorer correctness, zero-B parity,
  real `CONCLUDED` stopping, and reduced-seed robustness.
- Only after every diagnostic gate passes does the launcher run one optimizer
  step for each of the four judge arms.
- The finalizer measures actual winner/loser emissions, non-zero R1
  advantages, adapter deltas, on-policy logprob agreement, resource headroom,
  and arm completion. It always emits `phase1_forbidden=true`.
- The image is based on the immutable SGLang runtime digest and contains all
  Python dependencies; runtime pip installation is prohibited.

## Explicitly excluded

- No cloud instance, GPU run, model download, or billable action has occurred
  as part of source preparation.
- No automatic Phase-1 continuation exists.
- No authorization of the four 100-step Phase-1 arms is implied by a
  quantitative Phase-0 pass; local archive validation, manual review, Fable
  critique, and a later explicit human release decision remain mandatory.

## Remaining empirical risks

- Unit tests can verify exact bytes, routing, stop request payloads, and reward
  projection, but they cannot establish model behavior. The internal R2/R3
  renderer's compatibility with the debate-SFT adapter and SGLang's real
  no-bytes-after-sentinel behavior remain Phase-0 questions.
- The imported runtime is intentionally broad because constrained writing,
  SGLang LoRA hot-swap, selective logprobs, initialization, and training are
  coupled in the current working snapshot. Review should treat this PR as a
  source consolidation boundary, not as a minimal one-function patch.
- Passing this PR's tests is a prerequisite for preparing a freeze; it is not
  quantitative or qualitative Phase-0 acceptance.

## Local verification

- Python compile check over `src`, `scripts/run_train.py`, and unit tests.
- Complete unit suite: `148 passed, 5 skipped` after the D036 and Phase-0
  lifecycle additions.
- Constrained-writing scorer gold suite: `24/24` passed.
