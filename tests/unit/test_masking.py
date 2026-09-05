from llm_local_rl.masking import make_train_example
from llm_local_rl.types import EpisodeTurn


def test_make_train_example_builds_explicit_mask() -> None:
    turn = EpisodeTurn(
        turn_name="solution",
        adapter_name="solution",
        prompt_token_ids=[10, 11, 12],
        completion_token_ids=[20, 21],
        completion_logprobs=[-0.1, -0.2],
        trainable=True,
    )
    ex = make_train_example(turn=turn, advantage_per_token=0.75)
    assert ex.input_ids == [10, 11, 12, 20]
    assert ex.target_ids == [11, 12, 20, 21]
    assert ex.loss_mask == [0, 0, 1, 1]
    assert ex.old_logprobs == [0.0, 0.0, -0.1, -0.2]
    assert ex.advantages == [0.0, 0.0, 0.75, 0.75]


def test_make_train_example_rejects_empty_completion() -> None:
    turn = EpisodeTurn(
        turn_name="solution",
        adapter_name="solution",
        prompt_token_ids=[10, 11],
        completion_token_ids=[],
        completion_logprobs=[],
        trainable=True,
    )
    try:
        make_train_example(turn=turn, advantage_per_token=1.0)
    except ValueError as exc:
        assert "empty completion" in str(exc)
    else:
        raise AssertionError("Expected ValueError.")
