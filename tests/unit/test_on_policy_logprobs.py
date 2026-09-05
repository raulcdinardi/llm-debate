from __future__ import annotations

import pytest

from llm_local_rl.on_policy_logprobs import check_on_policy_logprobs, truncation_start
from llm_local_rl.types import TrainExample


class FakeTokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return " ".join(f"tok{token_id}" for token_id in token_ids)


def _example() -> TrainExample:
    return TrainExample(
        adapter_name="shared",
        input_ids=[10, 11, 12, 13, 14],
        target_ids=[11, 12, 13, 14, 15],
        loss_mask=[0, 0, 1, 1, 1],
        behavior_logprob_mask=[0, 0, 1, 1, 1],
        old_logprobs=[0.0, 0.0, -0.3, -0.4, -0.5],
        advantages=[0.0, 0.0, 0.1, 0.1, 0.1],
        metadata={"instance_id": "case_0", "round_num": 2},
    )


def test_truncation_start_keeps_suffix() -> None:
    assert truncation_start(token_count=5, max_tokens=0) == 0
    assert truncation_start(token_count=5, max_tokens=8) == 0
    assert truncation_start(token_count=5, max_tokens=3) == 2


def test_on_policy_logprob_check_has_zero_violations_when_logprobs_match() -> None:
    result = check_on_policy_logprobs(
        adapter_name="shared",
        examples=[_example()],
        current_logprob_rows=[[0.0, 0.0, -0.3, -0.4, -0.5]],
        tokenizer=FakeTokenizer(),
        abs_tol=1e-6,
        max_tokens=0,
        max_records=8,
        minibatch_start=4,
    )

    assert result.num_checked_tokens == 3
    assert result.num_violations == 0
    assert result.max_abs_logprob_diff == 0.0
    assert result.first_offending_trained_token is None
    assert result.records == []
    assert result.metrics()["trained_tokens_checked"] == 3.0
    assert result.metrics()["zero_advantage_loss_mask_tokens_skipped"] == 0.0
    assert result.metrics()["trained_token_mean_abs_diff"] == 0.0
    assert result.metrics()["on_policy_logprob_checked_tokens"] == 3.0


def test_on_policy_logprob_check_covers_zero_advantage_completion_tokens() -> None:
    example = TrainExample(
        adapter_name="debate",
        input_ids=[10, 11, 12, 13, 14],
        target_ids=[11, 12, 13, 14, 15],
        loss_mask=[0, 1, 1, 1, 1],
        behavior_logprob_mask=[0, 1, 1, 1, 1],
        old_logprobs=[0.0, -99.0, -0.3, -99.0, -0.5],
        advantages=[0.0, 0.0, 0.1, 0.0, 0.1],
        metadata={"instance_id": "case_merged", "rounds_merged": 3},
    )

    result = check_on_policy_logprobs(
        adapter_name="debate",
        examples=[example],
        current_logprob_rows=[[0.0, -0.1, -0.3, -0.2, -0.9]],
        tokenizer=FakeTokenizer(),
        abs_tol=1e-3,
        max_tokens=0,
        max_records=8,
        minibatch_start=0,
    )

    assert result.num_checked_tokens == 4
    assert result.num_trained_tokens_checked == 2
    assert result.num_zero_advantage_loss_mask_tokens_checked == 2
    assert result.num_violations == 3
    assert result.num_trained_token_violations == 1
    assert result.max_abs_logprob_diff == pytest.approx(98.9)
    assert result.metrics()["trained_tokens_checked"] == 2.0
    assert result.metrics()["zero_advantage_loss_mask_tokens_checked"] == 2.0
    assert result.metrics()["zero_advantage_loss_mask_tokens_skipped"] == 0.0
    assert result.metrics()["trained_token_max_abs_diff"] == pytest.approx(0.4)
    assert result.first_offending_token == result.records[0]
    assert result.records[0]["trained_token"] is False
    trained_record = result.first_offending_trained_token
    assert trained_record is not None
    assert trained_record["original_token_position"] == 4
    assert trained_record["completion_token_position"] == 3
    assert trained_record["advantage"] == pytest.approx(0.1)
    assert trained_record["trained_token"] is True
    assert trained_record["metadata"]["rounds_merged"] == 3


def test_on_policy_logprob_check_excludes_injected_renderer_tokens() -> None:
    example = TrainExample(
        adapter_name="debate",
        input_ids=[10, 11, 12, 13, 14],
        target_ids=[11, 12, 13, 14, 15],
        loss_mask=[0, 1, 1, 1, 1],
        behavior_logprob_mask=[0, 1, 0, 0, 1],
        old_logprobs=[0.0, -0.3, 0.0, 0.0, -0.5],
        advantages=[0.0, 0.1, 0.0, 0.0, 0.1],
        metadata={"instance_id": "case_injected", "rounds_merged": 2},
    )

    result = check_on_policy_logprobs(
        adapter_name="debate",
        examples=[example],
        current_logprob_rows=[[0.0, -0.3, -99.0, 99.0, -0.5]],
        tokenizer=FakeTokenizer(),
        abs_tol=1e-6,
        max_tokens=0,
        max_records=8,
        minibatch_start=0,
    )

    assert result.num_checked_tokens == 2
    assert result.num_injected_loss_mask_tokens_skipped == 2
    assert result.num_violations == 0
    assert result.metrics()["injected_loss_mask_tokens_skipped"] == 2.0
    assert result.metrics()["on_policy_logprob_injected_loss_mask_tokens_skipped"] == 2.0


def test_on_policy_logprob_check_rejects_trained_token_without_behavior_logprob() -> None:
    example = TrainExample(
        adapter_name="debate",
        input_ids=[10, 11],
        target_ids=[11, 12],
        loss_mask=[0, 1],
        behavior_logprob_mask=[0, 0],
        old_logprobs=[0.0, 0.0],
        advantages=[0.0, 0.1],
    )

    with pytest.raises(ValueError, match="nonzero-advantage token"):
        check_on_policy_logprobs(
            adapter_name="debate",
            examples=[example],
            current_logprob_rows=[[0.0, 0.0]],
            tokenizer=FakeTokenizer(),
            abs_tol=1e-6,
            max_tokens=0,
            max_records=8,
            minibatch_start=0,
        )


def test_on_policy_logprob_check_keeps_first_offender_when_records_disabled() -> None:
    result = check_on_policy_logprobs(
        adapter_name="shared",
        examples=[_example()],
        current_logprob_rows=[[0.0, 0.0, -0.9, -0.4, -0.5]],
        tokenizer=FakeTokenizer(),
        abs_tol=1e-3,
        max_tokens=0,
        max_records=0,
        minibatch_start=2,
    )

    assert result.num_violations == 1
    assert result.records == []
    assert result.first_offending_trained_token is not None
    assert result.first_offending_trained_token["example_index"] == 2
    assert result.first_offending_trained_token["trained_token"] is True


def test_on_policy_logprob_check_records_truncated_drift_context() -> None:
    result = check_on_policy_logprobs(
        adapter_name="shared",
        examples=[_example()],
        current_logprob_rows=[[-0.3004, -0.41, -0.9]],
        tokenizer=FakeTokenizer(),
        abs_tol=1e-3,
        max_tokens=3,
        max_records=1,
        minibatch_start=8,
    )

    assert result.num_checked_tokens == 3
    assert result.num_violations == 2
    assert result.max_abs_logprob_diff == pytest.approx(0.4)
    assert len(result.records) == 1
    record = result.records[0]
    assert record["example_index"] == 8
    assert record["original_token_position"] == 3
    assert record["sliced_token_position"] == 1
    assert record["target_token_id"] == 14
    assert record["target_token_text"] == "tok14"
    assert record["sequence_text"] == "tok12 tok13 tok14 tok15"
    assert record["metadata"]["instance_id"] == "case_0"


def test_on_policy_logprob_check_fails_on_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="align after truncation"):
        check_on_policy_logprobs(
            adapter_name="shared",
            examples=[_example()],
            current_logprob_rows=[[-0.3]],
            tokenizer=FakeTokenizer(),
            abs_tol=1e-3,
            max_tokens=3,
            max_records=8,
            minibatch_start=0,
        )
