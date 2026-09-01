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

Both projections attenuate the judge preference by its order-coherence reliability
`c = 1 - J`. `judge_soft_task_gap` first constructs a pair-sum-conserving task-gap
preference, standardizes it within the question/instance group, and then multiplies
the final R1 advantage by `c`. Applying `c` before the group z-score would cancel
the reliability gate for a single A/B pair. `soft_judge` assigns `+c*s` to A and
`-c*s` to B for an exactly zero-sum reward.

## Trainable OpenBookQA labeled-debate mode

The labeled OpenBookQA judge uses a separate strict contract:

```text
--debate-judge-harness constitution_single_token_v1
--debate-judge-constrain-single-token
--judge-label-token-contract lfm25_openbookqa_spaced_ab_v1
--debate-judge-score-mode order_sym_soft_logit
--train-judge
--judge-training-objective supervised_label_ce_js
--judge-coherence-js-weight 1.0
--debate-r23-reward soft_judge
```

This contract allows exactly the canonical leading-space tokens `" A"` (334)
and `" B"` (378). Before sampling, the runtime retokenizes `prompt + label` and
requires it to equal `prompt_tokens + [label_token]` for both labels. The fixed
prefill therefore cannot agglutinate with either answer token.

For forward order, let `p = (P(A), P(B))`. For reversed display order, map the
labels back to original referents and let `q = (P_reverse(B), P_reverse(A))`.
The coherence loss and debate reliability are

```text
J = JS(p, q) / ln(2)        # 0 <= J <= 1
c = 1 - J                   # 0 <= c <= 1
```

The judge is trained directly from the known gold referent and current model
probabilities:

```text
forward target = gold referent
reverse target = swap(gold referent)
L_judge = mean(label CE) + lambda_js * mean(J)
```

Forward/reverse pairs remain adjacent within every trainer minibatch, including
length-bucketed training. An overlength row or incomplete pair fails closed.
Minimizing `J` alone would admit the trivial constant 50/50 judge, so label CE
is always present in this objective.

The debate adapter consumes the same rollout audit through the reliability-gated
preference

```text
r_A = +(1-J)*s
r_B = -(1-J)*s
```

The R1 task-gap projection uses the same coefficient. JS is therefore neither
mixed into sampled-action rewards nor duplicated as an unrelated diagnostic:
it has exactly two explicit roles—judge coherence loss and debate-reward
reliability.
