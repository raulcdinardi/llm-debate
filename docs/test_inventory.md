# Test Inventory

## Unit tests

### `tests.unit.test_ht_sequence_env:test_ht_sequence_reward_counts_h`

- Builds an `HTSequenceEnv` with `sequence_len=4` and `reward_mode="num_h"`.
- Feeds a completion that decodes to `HHTH`.
- Verifies strict parsing accepts the exact four-character `H/T` string.
- Verifies `parse_success == 1.0`.
- Verifies the reward is exactly the count of `H` symbols, `3.0`.

### `tests.unit.test_ht_sequence_env:test_ht_sequence_strict_format_rejects_noisy_text`

- Builds an `HTSequenceEnv` with strict formatting enabled.
- Feeds a completion that decodes to `H H T H`.
- Verifies the environment rejects the noisy spaced completion.
- Verifies:
  - `parse_success == 0.0`
  - `parsed_sequence == ""`
  - reward is `0.0`

### `tests.unit.test_ht_sequence_env:test_ht_sequence_reward_counts_transitions`

- Builds an `HTSequenceEnv` with `sequence_len=4` and `reward_mode="num_transitions"`.
- Feeds a completion that decodes to `HHTT`.
- Verifies the reward is the number of symbol changes between adjacent parsed tokens.
- For `HHTT`, that transition count is `1`.

### `tests.unit.test_masking:test_make_train_example_builds_explicit_mask`

- Builds an `EpisodeTurn` with prompt tokens `[10, 11, 12]` and completion tokens `[20, 21]`.
- Converts it into a `TrainExample`.
- Verifies:
  - `input_ids` is prompt+completion shifted right
  - `target_ids` is prompt+completion shifted left
  - prompt positions are masked out with `loss_mask == [0, 0, 1, 1]`
  - prompt positions get dummy `old_logprobs == 0.0`
  - completion positions get the supplied advantage

### `tests.unit.test_masking:test_make_train_example_rejects_empty_completion`

- Builds an `EpisodeTurn` with an empty completion.
- Verifies that `make_train_example(...)` raises `ValueError`.
- This protects against silently constructing degenerate train batches.

### `tests.unit.test_episode_routing:test_single_turn_builder_uses_shared_adapter`

- Builds a `SingleTurnEpisodeBuilder`.
- Uses a recording fake sampler.
- Verifies the only sampling request uses adapter name `"shared"`.

### `tests.unit.test_episode_routing:test_debate_builder_routes_solution_then_debate`

- Builds a `DebateEpisodeBuilder`.
- Uses a recording fake sampler.
- Verifies the first turn uses adapter `"solution"` and the second uses `"debate"`.
- Verifies both the outgoing requests and the stored episode turns follow that routing.

### `tests.unit.test_driver:test_config_and_manifest_roundtrip`

- Builds a `TrainRunConfig`, serializes it, and stores it inside a `CheckpointManifest`.
- Reads the manifest back from disk.
- Verifies:
  - `current_step` is preserved
  - adapter checkpoint paths survive roundtrip
  - `TrainRunConfig.from_dict(...)` correctly rehydrates nested `rollout` data back into a `RolloutConfig`

### `tests.unit.test_driver:test_registry_uses_rollout_fields`

- Builds a `TrainRunConfig` whose `rollout` requests `env_name="coin_flip"` and `mode="debate"`.
- Verifies environment construction reads `config.rollout.env_name`, not a nonexistent top-level field.
- Verifies episode-builder construction reads `config.rollout.mode`.
- Verifies split-adapter debate routing becomes `solution` then `debate`.

## Integration tests

### `tests.integration.test_vllm_lora_load:test_vllm_loads_two_real_loras_on_one_engine`

- Reads a real base model path plus two real LoRA adapter paths from environment variables.
- Starts one real `vllm.LLM` engine with LoRA enabled.
- Runs one generation with adapter A and one generation with adapter B on the same engine.
- Verifies:
  - both calls return at least one generated token
  - generated token count equals returned logprob count
  - switching adapters on one live engine does not error

This test currently checks loadability and adapter switching, not semantic quality.

### `tests.integration.test_sampling_parity:test_sampling_pipeline_matches_direct_vllm_at_temp_zero`

- Starts one real `vllm.LLM` engine and one wrapper `VllmSampler` around that engine.
- Uses the same real LoRA adapter in both paths.
- Builds several deterministic `temperature=0.0` requests.
- For each request:
  - samples once through the wrapper path
  - samples once through a direct raw-`vllm` path
- Verifies:
  - the mismatch rate in generated token sequences stays below a warning/fail threshold
  - for prompts whose generated token sequences match exactly, the per-token generated logprobs match up to epsilon

This test currently checks wrapper-vs-direct generation parity, not arbitrary teacher-forced scoring of an externally supplied completion.

### `tests.integration.test_training_replay_parity:test_training_replay_gradient_matches_reference_loop`

- Loads a real base model and a real LoRA adapter into Transformers/PEFT twice from the same starting weights.
- Builds one explicit-mask replay example from a short prompt/completion pair.
- Uses real model-computed completion logprobs as the replay baseline.
- Forces the last generated token to have zero advantage so the test exercises both:
  - prompt masking
  - zero-advantage masking inside the completion
- Computes replay loss in two independent ways:
  - vectorized masked implementation
  - scalar reference loop
- Backpropagates each loss on a separate copy of the same model.
- Verifies:
  - scalar losses match up to epsilon
  - every trainable parameter gradient matches up to epsilon

This test currently checks algebraic replay-parity for the explicit-mask loss, not optimizer-step parity across a full training loop.

### `tests.unit.test_trainer:test_adapter_snapshot_keys_disjoint_when_split`

- Builds a split-adapter trainer with adapters `solution` and `debate`.
- Collects the trainable parameter names attributed to each adapter.
- Verifies the two adapter parameter-name sets are disjoint.

### `tests.unit.test_trainer:test_save_and_reload_adapter_snapshot_preserves_weights`

- Builds a trainer, saves one adapter, reloads it, and snapshots parameters before and after.
- Verifies every saved adapter parameter matches exactly up to epsilon.

### `tests.unit.test_trainer:test_trainer_smoke_requires_model_env`

- Loads a real base model path from `LLM_LOCAL_RL_BASE_MODEL` when available.
- Builds a multi-adapter trainer on CPU.
- Verifies tokenizer initialization succeeds and exposes a usable `pad_token_id`.
