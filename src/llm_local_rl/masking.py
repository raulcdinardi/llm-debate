from __future__ import annotations

from llm_local_rl.types import EpisodeTurn, TrainExample


def make_train_example(
    *,
    turn: EpisodeTurn,
    advantage_per_token: float,
    extra_metadata: dict | None = None,
) -> TrainExample:
    if len(turn.completion_token_ids) == 0:
        raise ValueError("Cannot build a train example from an empty completion.")
    if len(turn.completion_token_ids) != len(turn.completion_logprobs):
        raise ValueError("Completion tokens and logprobs must have equal length.")

    full_token_ids = turn.prompt_token_ids + turn.completion_token_ids
    input_ids = full_token_ids[:-1]
    target_ids = full_token_ids[1:]
    prompt_prefix_len = len(turn.prompt_token_ids) - 1
    if prompt_prefix_len < 0:
        raise ValueError("Prompt must contain at least one token.")

    loss_mask = ([0] * prompt_prefix_len) + ([1] * len(turn.completion_token_ids))
    behavior_logprob_mask = ([0] * prompt_prefix_len) + ([1] * len(turn.completion_token_ids))
    old_logprobs = ([0.0] * prompt_prefix_len) + list(turn.completion_logprobs)
    advantages = ([0.0] * prompt_prefix_len) + ([advantage_per_token] * len(turn.completion_token_ids))

    if not (
        len(input_ids)
        == len(target_ids)
        == len(loss_mask)
        == len(behavior_logprob_mask)
        == len(old_logprobs)
        == len(advantages)
    ):
        raise AssertionError("Train example fields must have equal length.")

    metadata = {"turn_name": turn.turn_name, **turn.metadata}
    if extra_metadata is not None:
        metadata.update(extra_metadata)

    return TrainExample(
        adapter_name=turn.adapter_name,
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        behavior_logprob_mask=behavior_logprob_mask,
        old_logprobs=old_logprobs,
        advantages=advantages,
        metadata=metadata,
    )
