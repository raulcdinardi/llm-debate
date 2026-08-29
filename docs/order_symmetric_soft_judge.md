# Order-symmetric soft judge reward

This path is opt-in. Existing hard-verdict experiments are unchanged.

## Temporary LFM2.5 label-token compatibility contract

The first implementation deliberately silos the current LFM2.5 tokenizer mapping behind:

```text
--judge-label-token-contract lfm25_ab_whitespace_compat_v1
```

That contract resolves and asserts exactly:

```text
A  -> 41      " A" -> 334
B  -> 42      " B" -> 378
```

It is **temporary**. Do not reuse it with a different tokenizer, answer stem, or judge
harness. Such a change requires a new named contract and a Phase-0 candidate-logprob probe.
The old contract fails closed if any surface form maps to a different token ID.

## Scoring

For each ordering, the sampler returns all four candidate log-probabilities independently
of model-I/O tracing. Whitespace variants are aggregated semantically:

```text
log P(A) = logsumexp(logp[41], logp[334])
log P(B) = logsumexp(logp[42], logp[378])
z        = log P(A) - log P(B)
```

With forward and swapped-order logits `z_forward` and `z_reverse`:

```text
z_symmetric = (z_forward - z_reverse) / 2
s           = tanh(z_symmetric / 2)
```

No invalid-verdict or order-disagreement fallback participates in this score.

## Opt-in flags

```text
--debate-judge-score-mode order_sym_soft_logit
--judge-label-token-contract lfm25_ab_whitespace_compat_v1
--debate-judge-bidirectional
--debate-judge-constrain-single-token
--debate-judge-temperature 0
--debate-judge-max-tokens 1
```

Select either or both reward projections:

```text
--debate-r1-reward judge_soft_task_gap
--debate-r23-reward soft_judge
```

`judge_soft_task_gap` allocates the existing absolute task-reward gap while conserving its
pair sum before the existing group normalization. `soft_judge` assigns `+s` to A and `-s`
to B for an exactly zero-sum debate reward.

## Trainable OpenBookQA labeled-debate mode

The labeled OpenBookQA judge uses a separate strict contract:

```text
--debate-judge-harness constitution_single_token_v1
--debate-judge-constrain-single-token
--judge-label-token-contract lfm25_openbookqa_spaced_ab_v1
--debate-judge-score-mode order_sym_soft_logit
--train-judge-coherence-grpo
--judge-grpo-reward-mode label_js
--debate-r23-reward soft_judge
```

This contract allows exactly the canonical leading-space tokens `" A"` (334)
and `" B"` (378). Before sampling, the runtime retokenizes `prompt + label` and
requires it to equal `prompt_tokens + [label_token]` for both labels. The fixed
prefill therefore cannot agglutinate with either answer token.

For forward order, let `p = (P(A), P(B))`. For reversed display order, map the
labels back to original referents and let `q = (P_reverse(B), P_reverse(A))`.
The consistency penalty is the bounded quantity

```text
J = JS(p, q) / ln(2)        # 0 <= J <= 1
```

Each sampled judge action keeps the objective OpenBookQA label reward and
subtracts the same pair-level consistency penalty:

```text
r_judge = (+1 if sampled referent is gold else -1) - J
```

(Gold-label ties retain label reward 0.) The debate adapter independently uses
the granular zero-sum signal `r_A = s`, `r_B = -s`, where
`s = tanh((z_forward - z_reverse) / 4)`.

Judge PPO logprobs are reconstructed under the exact two-token conditional
normalization in both full-logits and selective-LM-head trainer backends. This
keeps the constrained sampler and trainer behavior policies identical instead
of scoring a two-token sample under the full vocabulary.

## Direct gold-label judge gradient

For labeled tasks, add:

```text
--judge-training-objective supervised_label_ce
```

This replaces sampled-action judge GRPO with direct two-class cross-entropy on
the known correct referent. Both transcript orders remain in the batch. The
target is swapped in the reversed transcript so both examples supervise the
same underlying trajectory:

```text
forward order: target = gold referent (A or B)
reverse order: target = swap(gold referent)
loss:          -log P(target | {" A", " B"})
```

Only the judge objective changes. Debate R2/R3 continues using the granular
zero-sum GRPO signal. Sampled judge accuracy, order coherence, and
referent-aligned JS remain rollout diagnostics; they do not enter the judge CE
loss.
