from __future__ import annotations

import math

import pytest

from llm_local_rl.soft_judge import (
    LFM25_AB_WHITESPACE_COMPAT_V1,
    order_symmetric_soft_judge_score,
    resolve_judge_label_token_contract,
)


class _PinnedLfmTokenizer:
    _ids = {"A": [41], " A": [334], "B": [42], " B": [378]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self._ids[text])


def test_temporary_lfm_contract_resolves_exact_four_ids() -> None:
    contract = resolve_judge_label_token_contract(
        tokenizer=_PinnedLfmTokenizer(),
        contract_name=LFM25_AB_WHITESPACE_COMPAT_V1,
    )
    assert contract.a_token_ids == (41, 334)
    assert contract.b_token_ids == (42, 378)
    assert contract.allowed_token_ids == (41, 334, 42, 378)
    assert contract.temporary is True


def test_temporary_lfm_contract_fails_closed_when_tokenizer_changes() -> None:
    tokenizer = _PinnedLfmTokenizer()
    tokenizer._ids = {**tokenizer._ids, " A": [999]}
    with pytest.raises(ValueError, match="tokenizer mapping changed"):
        resolve_judge_label_token_contract(
            tokenizer=tokenizer,
            contract_name=LFM25_AB_WHITESPACE_COMPAT_V1,
        )


def test_order_symmetric_score_aggregates_whitespace_variants() -> None:
    contract = resolve_judge_label_token_contract(
        tokenizer=_PinnedLfmTokenizer(),
        contract_name=LFM25_AB_WHITESPACE_COMPAT_V1,
    )
    score = order_symmetric_soft_judge_score(
        forward_candidate_logprobs={41: 0.0, 334: 0.0, 42: -2.0, 378: -2.0},
        reverse_candidate_logprobs={41: -2.0, 334: -2.0, 42: 0.0, 378: 0.0},
        contract=contract,
    )
    assert score.z_forward == pytest.approx(2.0)
    assert score.z_reverse == pytest.approx(-2.0)
    assert score.z_symmetric == pytest.approx(2.0)
    assert score.score == pytest.approx(math.tanh(1.0))
    assert score.order_bias_logit == pytest.approx(0.0)


def test_equal_order_logits_cancel_as_position_bias() -> None:
    contract = resolve_judge_label_token_contract(
        tokenizer=_PinnedLfmTokenizer(),
        contract_name=LFM25_AB_WHITESPACE_COMPAT_V1,
    )
    row = {41: 0.0, 334: 0.0, 42: -2.0, 378: -2.0}
    score = order_symmetric_soft_judge_score(
        forward_candidate_logprobs=row,
        reverse_candidate_logprobs=row,
        contract=contract,
    )
    assert score.z_symmetric == pytest.approx(0.0)
    assert score.score == pytest.approx(0.0)
    assert score.order_bias_logit == pytest.approx(2.0)


def test_soft_score_requires_every_contract_token() -> None:
    contract = resolve_judge_label_token_contract(
        tokenizer=_PinnedLfmTokenizer(),
        contract_name=LFM25_AB_WHITESPACE_COMPAT_V1,
    )
    with pytest.raises(ValueError, match="missing contract token ids"):
        order_symmetric_soft_judge_score(
            forward_candidate_logprobs={41: 0.0, 334: 0.0, 42: -1.0},
            reverse_candidate_logprobs={41: 0.0, 334: 0.0, 42: -1.0, 378: -1.0},
            contract=contract,
        )
