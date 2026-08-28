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
