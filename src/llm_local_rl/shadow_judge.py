"""Passive supervised judge training on the active judge's exact examples."""

from contextlib import contextmanager
from dataclasses import replace

from llm_local_rl.checkpointing import capture_rng_state, restore_rng_state
from llm_local_rl.types import TrainExample


SHADOW_JUDGE = "judge_shadow"


@contextmanager
def preserve_rng_state():
    # A passive observer must not change the next training/sampling RNG draw.
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def shadow_label_examples(examples: list[TrainExample]) -> list[TrainExample]:
    result = []
    for example in examples:
        metadata = dict(example.metadata)
        # These were sampled by the active judge; they are not shadow predictions.
        metadata.pop("judge_sampled_verdict", None)
        metadata.pop("judge_sampled_label_correct", None)
        metadata["judge_training_role"] = "shadow"
        result.append(replace(example, adapter_name=SHADOW_JUDGE, metadata=metadata))
    return result
