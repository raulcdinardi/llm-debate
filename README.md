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

## Winner-modulated R1 reward

For split-adapter debate runs, `--debate-r1-reward judge_rejection_task`
implements literal winner rejection for R1: the judge-selected trajectory keeps
its objective task reward, the losing R1 trajectory is omitted, and selected
winner rewards are population-z-scored within each question group. R2/R3 remain
independent: `--debate-r23-mode symmetric` still trains both speakers with
positive/negative constant rewards. The mode requires the round mapping
`solution, debate, ...` and is not supported by the shared-adapter layout.
