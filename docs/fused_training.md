# Optional fused training kernels

These options apply to the Torch training/reference path, independently of vLLM
inference kernels. Existing frozen configs are unchanged. The default remains
`train_gdn_backend=auto`, `train_lm_head_kernel=torch` until real GPU validation.

```bash
python scripts/run_train.py ... \
  --train-gdn-backend fla \
  --train-logprob-backend selective_lm_head \
  --train-lm-head-kernel triton
```

Both options round-trip through saved run configuration and exact resume. Do not
change them silently when resuming an existing frozen experiment. Backbone
activation checkpointing remains off by default; independent LM-head chunk
checkpointing remains on. No changes to effective optimizer batch, PPO clipping,
judge CE/JS objectives, or the mandatory behavior-policy parity check.

## GDN

Install the `fused-training` extra into the intended CUDA runtime before starting
Python. `causal-conv1d` may require building against that runtime's Torch/CUDA;
record the resolved package versions when freezing a launch.

Hugging Face Qwen3.5 already selects FLA's `chunk_gated_delta_rule` and
`causal_conv1d_fn` when the dependencies are available. `--train-gdn-backend fla`
requires these actual callable identities on **every** GDN layer, otherwise it
fails before training. It supports Transformers 5.14's per-layer dispatch and
5.16's wrapped optional-package dispatch; an unknown dispatch fails visibly.
It emits a `training_kernel_receipt` with layer count, callable names and package
versions. Package installation or fast vLLM inference alone is not evidence
that the training GDN uses FLA. No parameters/modules are replaced and LoRA
names/state remain intact.

## LM head

The `triton` option keeps the chunked cuBLAS matrix multiplication and fuses its
temperature scaling, log-sum-exp, target log probability, diagnostic entropy,
and arbitrary per-token backward weights. It is **not** an end-to-end fused
linear/cross-entropy GEMM kernel. Projection still materializes one bounded
chunk of logits, then backward recomputes that chunk under checkpointing.

The reduction splits large vocabularies into 4,096-column tiles. Only small
FP32 row/tile statistics survive inside a chunk; it avoids full-vocabulary
FP32 log-softmax, probability, and entropy-product buffers. Backward produces
the gradient in the projection dtype. There is no tiny-gradient filtering,
sampled softmax, vocabulary approximation, or scalar-CE substitution for PPO.
Floating-point reductions can differ from Torch; parity remains mandatory.
Entropy retains its existing **diagnostic-only** semantics.

Restricted judgments group identical allowed-token sets and project only those
LM-head rows, using small dense Torch CE. Mixed restricted/unrestricted rows
are restored to their original order. A/B target remapping, bias, temperature,
selected positions and the entropy mask are preserved. Restricted GEMM shape
changes can alter rounding; GPU parity tests must cover them too.

Supported: explicit FP32/BF16 CUDA tensors, frozen or trainable linear head,
first-order gradients. Autocast, higher-order gradients and combining this option
with `compile_train_logprob_helper` are unsupported. CUDA is required; there is
no silent CPU fallback. Triton imports are lazy on the normal Torch path.

## Validation and measurement

```bash
# Routing/config plus ordinary trainer regressions (no GPU needed).
PYTHONPATH=src:. pytest tests/unit/test_training_kernels.py \
  tests/unit/test_trainer.py tests/unit/test_lm_head_checkpointing.py

# Execute the actual Triton arithmetic in the CPU interpreter. This does NOT
# establish CUDA compilation, BF16 GPU correctness, throughput or peak VRAM.
TRITON_INTERPRET=1 PYTHONPATH=src:. pytest tests/unit/test_training_kernels.py

# On an existing idle CUDA host with FLA/causal-conv installed:
PYTHONPATH=src:. pytest tests/unit/test_training_kernels.py
```

Tests cover full Qwen vocabulary, non-power-of-two tails, temperature, signed
per-token gradients, hidden/head/bias gradients, selected/entropy masks,
mixed/full A/B restrictions, no-grad reference scoring, saved activation bounds,
and detection of inactive GDN kernels. Dedicated CUDA tests check BF16 and FLA
backward. CPU interpreter success is reported separately from CUDA execution.

Sources: [HF Qwen3.5 dispatch](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
[FLA](https://github.com/fla-org/flash-linear-attention),
[Triton reductions](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
