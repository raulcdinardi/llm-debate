from __future__ import annotations

import math

import pytest

from llm_local_rl.soft_judge import (
    LFM25_AB_WHITESPACE_COMPAT_V1,
    LFM25_OPENBOOKQA_SPACED_AB_V1,
    bernoulli_js_divergence,
    order_symmetric_soft_judge_score,
    resolve_judge_label_token_contract,
    validate_judge_prompt_label_boundary,
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


class _BoundaryTokenizer(_PinnedLfmTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text in self._ids:
            return list(self._ids[text])
        rows = {
            "Answer by Agent": [700, 701],
            "Answer by Agent A": [700, 701, 334],
            "Answer by Agent B": [700, 701, 378],
        }
        return list(rows[text])


def test_strict_openbookqa_contract_has_only_spaced_a_and_b_and_safe_boundary() -> None:
    tokenizer = _BoundaryTokenizer()
    contract = resolve_judge_label_token_contract(
        tokenizer=tokenizer,
        contract_name=LFM25_OPENBOOKQA_SPACED_AB_V1,
    )
    assert contract.a_token_ids == (334,)
    assert contract.b_token_ids == (378,)
    assert contract.allowed_token_ids == (334, 378)
    validate_judge_prompt_label_boundary(
        tokenizer=tokenizer,
        prompt_text="Answer by Agent",
        prompt_token_ids=[700, 701],
        contract=contract,
    )


def test_strict_openbookqa_boundary_fails_closed_on_agglutination() -> None:
    tokenizer = _BoundaryTokenizer()
    contract = resolve_judge_label_token_contract(
        tokenizer=tokenizer,
        contract_name=LFM25_OPENBOOKQA_SPACED_AB_V1,
    )
    tokenizer._ids = dict(tokenizer._ids)
    with pytest.raises(ValueError, match="agglutinated"):
        validate_judge_prompt_label_boundary(
            tokenizer=tokenizer,
            prompt_text="Answer by Agent",
            prompt_token_ids=[700, 999],
            contract=contract,
        )


def test_referent_js_aligns_reverse_labels_before_comparison() -> None:
    contract = resolve_judge_label_token_contract(
        tokenizer=_PinnedLfmTokenizer(),
        contract_name=LFM25_OPENBOOKQA_SPACED_AB_V1,
    )
    aligned = order_symmetric_soft_judge_score(
        forward_candidate_logprobs={334: 0.0, 378: -2.0},
        reverse_candidate_logprobs={334: -2.0, 378: 0.0},
        contract=contract,
    )
    biased = order_symmetric_soft_judge_score(
        forward_candidate_logprobs={334: 0.0, 378: -2.0},
        reverse_candidate_logprobs={334: 0.0, 378: -2.0},
        contract=contract,
    )
    assert aligned.referent_js_divergence_normalized == pytest.approx(0.0)
    assert biased.referent_js_divergence_normalized > 0.0
    assert bernoulli_js_divergence(0.0, 1.0) == pytest.approx(math.log(2.0))
