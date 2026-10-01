# One actor adapter across debate rounds

Use the existing trained debate weights once, under one adapter identity:

```json
{
  "adapter_layout": "split",
  "debate_round_adapter_names": ["debate", "debate", "debate"],
  "debate_judge_adapter": "judge",
  "init_adapter_dirs": {
    "debate": "/restored/step100/debate",
    "judge": "/restored/step100/judge"
  },
  "train_adapter_names": ["debate"]
}
```

This is a configuration fragment, not a full experiment spec. Include
`judge_shadow` in initialization and training lists when enabling paired-judge
training; include `judge` in the training list when training the active judge.
The driver derives the required adapter inventory from round routing plus the
judge selection. It no longer requires an unused `solution` adapter.

`adapter_layout="shared"` also supports an independent judge and shadow judge;
it names the single actor `shared`, so use that key for initialization and
training filters. Historical shared-policy hard-judge configurations retain
their legacy reward projection for reproducibility. Shared actors with separate
judges or the newer reward modes use the round-wise projection. To explicitly
use the round-wise projection for any objective, use the mapping above.

Loading the same checkpoint under two names still creates two parameter sets.
Identity sharing must be explicit in the round mapping. A separate judge uses
its own adapter; sharing the actor does not merge actor and judge parameters.

## Training rows and backward

Each agent trajectory contributes one training row per adapter used in that
trajectory. Same-adapter rounds use the longest causal history and the union
of their sampled-token loss masks, including nonadjacent rounds. Opponent text,
user instructions, and another adapter's tokens are context, not loss targets.
Round-specific advantages and behavior logprobs remain unchanged. Exact token
prefixes are checked before merging; rewritten histories fail visibly.

The loss is the sum of the eligible round objectives, averaged over agent
trajectories for that adapter. Minibatch and optimizer batch sizes therefore
count those trajectories, not individual rounds. A single actor's R1/R2/R3
must not be averaged over three duplicated training rows: that would change
both objective scale and optimizer update boundaries. Each trajectory receives
one gradient-enabled forward/backward per training pass (batched with others),
not one per round. Distinct adapters still require their own parameter-specific
training passes.

The fail-closed, no-gradient behavior-logprob preflight is retained. Optional
reference-KL diagnostics and activation-checkpoint recomputation also remain;
"one backward row" does not mean exactly one model invocation of any kind.

Winner-only `judge_rejection_task` now works with a shared actor: a rejected
R1 has no training mask, while that agent's eligible later rounds can still
train. Supplied `fixed_r1` answers are never sampled policy targets. This does
not turn the completed fixed-R1 MMLU-Pro experiment into a generated-R1 run.

## Validation

CPU tests compare exact masks, logprobs and advantages for 1–4 rounds, check
fixed/rejected R1 and repeated prompts, preserve active/shadow judge isolation,
and compare actual parameter gradients against an independent sum of the
original round losses for both full-logit and selective-head trainer backends.
A model/GPU smoke remains necessary before launching a new scientific run;
these tests make no claim about GPU throughput or bf16 numerical parity.

Validated 2026-09-10: full CPU unit suite **439 passed, 10 skipped**.
