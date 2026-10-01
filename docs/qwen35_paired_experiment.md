# Qwen3.5 instruction-model paired-judge experiment support

The opt-in `--debate-prompt-format qwen35_instruct_three_points` uses the native
Qwen3.5 chat template with `--thinking-mode no_think`, exactly three rounds, and
native EOS termination. Both later-round instructions are identical except the
opponent payload and the final R3 response instruction. The assistant prefill is
exactly `<think>\n\n</think>\n\n`; no numbered point is prefilled. Prior prompt and
completion token IDs remain unchanged when the next user turn is appended.
If a completion already contains the native turn-end token, the continuation
does not add a second copy. R2/R3 text is not rewritten or truncated after sampling.

`--debate-r23-format-failure-penalty -0.2` checks three numbered, nonempty lines,
1), 2), 3), with at most 30 whitespace-delimited words per point, followed by a
fourth standalone CONCLUDED line. Each failed round receives one additive penalty;
multiple violations in one round do not multiply it. No grammar or token allowlist
is applied to debaters. Meaning, grounding, and argumentative relevance are not
claimed to be validated by this syntactic check.

Use `--debate-judge-harness qwen35_chat_single_token_v1` together with
`--judge-label-token-contract qwen35_instruct_ab_v1`. The native instruction judge
compares original R1 response quality/correctness using the debate as evidence.
Its generated label is bare A or B. The tokenizer and appended-label boundary are
validated explicitly (Qwen3.5 A=32, B=33), including the empty thinking block.
The LFM harness and label-token contract remain separately supported.

Use the paired shadow options in [paired_shadow_judge.md](paired_shadow_judge.md),
`--debate-r23-reward soft_judge_raw`, and `--judge-coherence-js-weight 0` for the
requested CE-only comparison. Raw order-symmetric soft margin sets the debate
reward; the shadow never influences rollouts or rewards. The new Qwen loader uses
the native conditional-generation class and supports the existing selective
language-model head path. A tiny hybrid Qwen model tests native checkpoint loading,
zero effective initial judge delta, selective/full logits agreement, and isolated
active/shadow CE updates. Full-size H200 runtime and vLLM parity remain Phase-0 checks.

`paired_judge_eval.py` reports binary Brier, log loss, confidence ECE with ten fixed
bins, hard A/B-order consistency, and probability disagreement. Score ties are
retained with target 0.5 and separately reported; no post-hoc calibrator is fitted.
The paired Brier difference uses identical IDs and targets for both judges.
Negative shadow-minus-active differences favor the shadow on that evaluation panel.
This measures generalization on the active judge's training stream, not the outcome
of two independent co-evolving training runs.
