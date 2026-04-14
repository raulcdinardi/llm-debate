# llm-local-rl

Local-only RL rewrite for:
- `vllm` sampling
- LoRA training
- explicit token masking
- split adapter routing

Current priorities:
- keep the `ht_sequence` reward-hacking environment
- make masking first-class rather than implicit
- make adapter routing explicit per turn and per train batch
- make parity tests easy to run after engine changes

## Architecture

The key split is:
- environments define task instances, prompt content, parsing, and rewards
- episode builders define turn structure such as `single_turn` or `debate`
- samplers generate tokens for a named adapter
- trainers consume explicit masks for a named adapter

This avoids the previous shape where debate mechanics and task/environment logic leaked into each other.

## Test strata

- unit tests: mask construction, routing, environment behavior
- integration tests: `vllm` + LoRA loadability, sampling parity, replayed training parity

Heavy integration tests are opt-in and require a local model path.
