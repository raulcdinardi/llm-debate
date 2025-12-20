# Logging Structure

This document describes the logging format for debate training runs. All data crossing the API boundary is logged for reproducibility and debugging.

## Directory Structure

```
logs/
├── experiments/           # Named experiments (--experiment flag)
│   └── <experiment_name>/
│       ├── debate_*.json
│       └── training_step_*.json
└── test_runs/             # Unnamed runs (auto-timestamped)
    └── <YYYYMMDD_HHMMSS>/
        ├── debate_*.json
        └── training_step_*.json
```

## File Types

### 1. `debate_*.json` — Inference Logs

Captures each debate rollout, including all sampling API calls.

```json
{
  "timestamp": "20251218_150726_575375",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "config": {
    "num_rounds": 3,
    "max_tokens_per_turn": 256,
    "temperature": 0.7,
    "kl_coef": 0.01,
    "learning_rate": 1e-05
  },
  "question": "What is 7 * 8?",
  "ground_truth": "56",
  "verdict": "B",
  "judge_reasoning": "...",
  "agent_a": { ... },
  "agent_b": { ... },
  "metrics": { ... }
}
```

#### Agent Structure

Each agent contains text (for readability) and raw tokens (for exactness):

```json
{
  "frozen_solution": "56",
  "r1": "<THINKING>...</THINKING>\n<SOLUTION>56</SOLUTION>",
  "r2": "...",
  "r3": "...",
  "transitions": [
    {
      "round": 1,
      "prompt_tokens": [151644, 8948, ...],      // API request
      "completion_tokens": [3125, 39, ...],      // API response
      "completion_logprobs": [-0.217, 0.0, ...]  // API response
    },
    // ... rounds 2 and 3
  ]
}
```

#### API Mapping (Inference)

| Field | API | Direction |
|-------|-----|-----------|
| `config.max_tokens_per_turn` | `SamplingParams.max_tokens` | Request |
| `config.temperature` | `SamplingParams.temperature` | Request |
| `prompt_tokens` | `ModelInput.from_ints()` | Request |
| `completion_tokens` | `result.sequences[0].tokens` | Response |
| `completion_logprobs` | `result.sequences[0].logprobs` | Response |

---

### 2. `training_step_*.json` — Training Logs

Captures each training step, including exact tensors sent to `forward_backward()`.

```json
{
  "step": 1,
  "timestamp": "20251218_150729_420837",
  "num_datums": 3,
  "datums": [ ... ],
  "results": {
    "loss": -1.0016,
    "num_tokens": 1895,
    "num_trained_tokens": 651
  },
  "learning_rate": 1e-05
}
```

#### Datum Structure

Each datum represents one training example (one round from the winning trajectory):

```json
{
  // Exact tensors sent to API
  "input_tokens": [151644, 8948, ..., 3125],   // full_seq[:-1]
  "target_tokens": [8948, 198, ..., 39],       // full_seq[1:]
  "advantages": [0.0, 0.0, ..., 0.00153, ...], // mask × advantage
  "sampling_logprobs": [0.0, 0.0, ..., -0.217, ...],

  // Provenance
  "source": {
    "question": "What is 7 * 8?",
    "agent": "B",
    "verdict": "B",
    "round": 1
  },

  // Raw components (for cross-checking)
  "prompt_tokens": [151644, 8948, ...],
  "completion_tokens": [3125, 39, ...],
  "advantage_value": 0.0015290519877675841
}
```

#### API Mapping (Training)

| Field | API | Direction |
|-------|-----|-----------|
| `input_tokens` | `Datum.model_input` | Request |
| `target_tokens` | `Datum.loss_fn_inputs["target_tokens"]` | Request |
| `advantages` | `Datum.loss_fn_inputs["advantages"]` | Request |
| `sampling_logprobs` | `Datum.loss_fn_inputs["logprobs"]` | Request |
| `results.loss` | Computed from `forward_backward()` output | Response |
| `learning_rate` | `AdamParams.learning_rate` | Request (`optim_step`) |

#### Understanding the Mask

The `advantages` tensor encodes which tokens are trained:

```
Position:     0    1    2   ...  63   64   65   ...  202
Content:    [BOS][sys][tem]...  [?] [<T][HIN][K] ...  [>]
                                     ↑
                              prompt ends here

advantages:  0.0  0.0  0.0  ... 0.0  0.0015 0.0015 ... 0.0015
             ←── prompt (masked) ──→ ←── completion (trained) ──→
```

- `0.0` = prompt token, not trained (mask=0)
- `0.0015...` = completion token, trained with advantage = 1/total_completion_tokens

---

## Cross-Referencing

To trace a training datum back to its source debate:

1. Look at `datum.source` for `agent`, `round`, `verdict`
2. Find the debate file with matching `verdict` and timestamp (same run)
3. Verify: `datum.prompt_tokens` == `debate.agent_X.transitions[round-1].prompt_tokens`

---

## File Sizes (Typical)

| File | Size | Contents |
|------|------|----------|
| `debate_*.json` | ~100KB | 2 agents × 3 rounds of tokens + logprobs |
| `training_step_*.json` | ~150KB | 3 datums with full tensors |

Estimated storage: **~250KB per debate** with training, **~24,000 debates per GB**.
