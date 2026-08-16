# Versioned judge harnesses

A judge harness is one immutable contract covering all behavior that must match
the judge's training data:

- prompt serialization and decision objective;
- structured A/B transcript rendering;
- assistant prefill and output contract;
- verdict parser and default output-token budget;
- a deterministic fingerprint stored with judge adapters.

Select the contract with one flag:

```text
--debate-judge-harness solution_r1_rationale_v1
```

The registered harnesses are:

| Harness | Objective | Output |
|---|---|---|
| `chat_solution_tagged_v1` | Best Round-1 solution; debate is evidence only | Verdict tag |
| `chat_pointwise_tagged_v1` | Best answer in a one-round comparison | Verdict tag |
| `solution_r1_rationale_v1` | Best Round-1 solution; debate is evidence only | Rationale, then verdict tag |
| `constitution_single_token_v1` | Best case for constitution adherence, matching its one-token SFT corpus | One token: A or B |

The names intentionally encode semantics rather than model family. In
particular, raw-Base formatting is not enough to identify whether a judge was
trained to evaluate solution quality or debate persuasiveness.

## Adapter binding

Every pretrained judge adapter must contain `judge_harness.json`. Bind a legacy
adapter once with the harness actually used to create its SFT data:

```text
python scripts/bind_judge_harness.py \
  --adapter-dir /path/to/judge \
  --harness solution_r1_rationale_v1
```

The runtime rejects a missing sidecar, a different harness ID, or a stale
template fingerprint before loading the adapter. Saved judge checkpoints write
the sidecar automatically. The harness ID and fingerprint are also recorded in
the run configuration, model-I/O trace metadata, and each step record.

Binding is a provenance assertion, not automatic discovery. The operator must
choose the harness that generated the adapter's training examples; the command
cannot infer that history from LoRA weights.

## Compatibility

Old serialized configurations containing `debate_judge_prompt_format` migrate
as follows when read:

- `chat` -> `chat_solution_tagged_v1`
- `base_model_sft` -> `solution_r1_rationale_v1`
- `single_token_sft` -> `constitution_single_token_v1`

New configurations serialize only `debate_judge_harness`. The old CLI flag is
accepted as a hidden migration alias, but new invocations should use the
versioned harness flag.
