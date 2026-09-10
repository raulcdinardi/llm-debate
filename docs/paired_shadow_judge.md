# Paired judge initialization comparison

Train two judge LoRAs on the same gold-labeled debate stream. Only `judge`
supplies debate rewards. `judge_shadow` receives the same supervised updates
but cannot influence generation, reward computation, or example selection.
All adapters share one frozen backbone and are updated sequentially.

## Enable

Add these settings to a valid split-adapter, bidirectional, single-token judge
training configuration:

```text
--train-judge
--judge-training-objective supervised_label_ce_js
--judge-coherence-js-weight 0
--debate-r1-reward none
--debate-r23-reward soft_judge_raw
--train-adapter-names debate judge judge_shadow
--train-shadow-judge
--shadow-judge-init-seed <explicit integer>
--shadow-judge-init-std <explicit positive standard deviation>
```

The last two values are required scientific choices, with no implicit defaults.
This is an addition to an existing launch configuration, not a full launch command.
The flag does not change model identity, prompts, token contracts, generation caps,
format penalties, LoRA rank, or training hyperparameters. In particular, the
existing tokenizer-specific A/B contract must still match the selected model;
this feature does not port the OpenBookQA/LFM token IDs to Qwen.

`soft_judge_raw` uses the active judge's existing order-symmetrized log odds:
`z = (z_forward - z_reverse) / 2`, `s = tanh(z / 2)`. Candidate rewards are
`+s` and `-s`, with existing independent formatting penalties. There is no
JS/reliability multiplier in this mode. Existing `soft_judge` is also supported.
The paired feature requires CE-only; JS remains a measured coherence diagnostic.

## Initialization and updates

On a fresh run without imported adapters, the active judge uses standard LoRA
initialization: random A, zero B, hence exactly zero effective weight delta.
The shadow copies A from the active judge and draws B from `Normal(0, std)`
using its own CPU Torch generator. The backbone is never randomized. The usual
LoRA scaling still multiplies BA. A receipt records seed, configured standard
deviation, paired module count and realized B norm in
`shadow_judge_initialization.json`.

Both judges receive identical prompt/target tokens, masks, pair ordering and
batch sizes. Forward and reversed presentations stay paired under the existing
CE batching rules. Active sampled verdict metadata is removed from shadow rows
so it cannot be mistaken for a shadow prediction. CE recomputes the shadow's
own logits; it never uses active-judge importance ratios.

Rollouts and rewards are computed before any optimizer updates. The existing
policy and active-judge update order is retained, followed by shadow training.
Only the selected adapter has gradients. Adam moments are independent per
parameter within the existing optimizer; an inactive judge's moments, step
counters and weights remain unchanged. Python, NumPy and Torch RNG states are
restored around shadow updates, including on failure. Shadow adapter creation
and loading also preserve RNG state.

The shadow is excluded from inference-engine adapter registration and policy
entropy/KL aggregates. No additional debate sampling or judge-label generation
is required. It does require another judge forward/backward pass and storage
for its LoRA weights and Adam state. Physical GPU fit and runtime need a smoke
test with the intended model and batch geometry.

## Metrics, checkpoints and evaluation

Step records and W&B retain separate `train/judge/...` and
`train/judge_shadow/...` metrics, including loss, CE NLL, correct-label
probability, accuracy, binary Brier score, JS coherence and optimizer counts.
These are training-batch diagnostics collected during the update, not held-out
calibration measurements. Binary Brier uses `(1 - p_correct)^2` (not twice that
value, as in summed two-class Brier).

Both judges are saved at the configured LoRA cadence and included in exact-resume
bundles, along with all parameter-specific Adam states. Both exports receive the
same judge-harness manifest and can be selected independently for evaluation.
Resume loads the saved shadow weights; it never randomizes them again. An
explicit imported-adapter run must provide the complete adapter inventory,
including `judge_shadow`; it loads those exact weights rather than applying
fresh paired initialization. Existing runs with this feature disabled retain
their prior checkpoint fingerprint.

Use identical held-out/OOD panels for the two exports and compare matching
checkpoints, including initialization. Historical and regenerated CW debates
over the same R1 pair should reuse the same existing DeepSeek preference target.
The result compares initialization conditional on the active judge's debate
stream. It does not compare two independently controlled RL trajectories.

## Validation

`tests/unit/test_shadow_judge.py` uses a tiny local Llama model with real PEFT
LoRA and AdamW, without downloading weights. It checks paired initialization,
unchanged initial active outputs and RNG, separate CE updates, parameter/Adam
isolation, and exact-resume equivalence. The debate projection test checks
unchanged active reward examples and identical gold targets in both orders.
No full Qwen GPU training is claimed by these CPU tests.

Local validation on 2026-09-09: **417 unit tests passed, 9 skipped**.
Runtime: Torch 2.13.0+cu130, PEFT 0.20.0, Transformers 5.16.1, pytest 9.1.1.
Tests run on CPU; localhost mock-server tests require socket permission.
The existing PEFT weight-converter compatibility wrapper now forwards native
keyword arguments, needed for Transformers 5.16.1 `force_cpu`.
