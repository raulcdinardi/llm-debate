# Architecture

## Why this split

The previous repo mixed:
- task/environment concerns
- turn-structure concerns
- adapter-routing concerns
- implicit masking concerns

This rewrite separates them:

- `envs.py`
  task prompts, parsing, reward logic
- `episodes.py`
  turn sequencing such as `single_turn` and `debate`
- `masking.py`
  explicit conversion from an `EpisodeTurn` into a trainable example
- `interfaces.py`
  thin protocols for tokenizer, sampler, trainer

## Better environment interface

The cleaner boundary is:

- environment:
  "what is the task and how is reward computed?"
- episode builder:
  "what sequence of turns do we run and which adapter owns each turn?"

That is better than making a task implement debate-specific methods directly.

## Testing priorities

- unit:
  routing, masks, parsing
- integration:
  vLLM adapter activation, sampling parity, replayed training parity
