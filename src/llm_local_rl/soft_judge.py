from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol


JUDGE_LABEL_TOKEN_CONTRACT_NONE = "none"
LFM25_AB_WHITESPACE_COMPAT_V1 = "lfm25_ab_whitespace_compat_v1"
JUDGE_LABEL_TOKEN_CONTRACTS = (
    JUDGE_LABEL_TOKEN_CONTRACT_NONE,
    LFM25_AB_WHITESPACE_COMPAT_V1,
)


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class JudgeLabelTokenContract:
    name: str
    a_token_ids: tuple[int, ...]
    b_token_ids: tuple[int, ...]
    temporary: bool

    @property
    def allowed_token_ids(self) -> tuple[int, ...]:
        return (*self.a_token_ids, *self.b_token_ids)

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "temporary": self.temporary,
            "a_token_ids": self.a_token_ids,
            "b_token_ids": self.b_token_ids,
            "allowed_token_ids": self.allowed_token_ids,
        }


@dataclass(frozen=True)
class OrderSymmetricSoftJudgeScore:
    logp_forward_a: float
    logp_forward_b: float
    logp_reverse_a: float
    logp_reverse_b: float
    z_forward: float
    z_reverse: float
    z_symmetric: float
    order_bias_logit: float
    score: float

    def record(self) -> dict[str, float | str]:
        return {
            "mode": "order_sym_soft_logit",
            "logp_forward_a": self.logp_forward_a,
            "logp_forward_b": self.logp_forward_b,
            "logp_reverse_a": self.logp_reverse_a,
            "logp_reverse_b": self.logp_reverse_b,
            "z_forward": self.z_forward,
            "z_reverse": self.z_reverse,
            "z_symmetric": self.z_symmetric,
            "order_bias_logit": self.order_bias_logit,
            "score": self.score,
        }


def resolve_judge_label_token_contract(
    *, tokenizer: TokenizerLike, contract_name: str
) -> JudgeLabelTokenContract:
    if contract_name == JUDGE_LABEL_TOKEN_CONTRACT_NONE:
        return JudgeLabelTokenContract(
            name=contract_name,
            a_token_ids=(),
            b_token_ids=(),
            temporary=False,
        )
    if contract_name != LFM25_AB_WHITESPACE_COMPAT_V1:
        raise ValueError(f"Unknown judge label token contract: {contract_name!r}")

    # TEMPORARY COMPATIBILITY CONTRACT.  Do not generalize these IDs.  Replace
    # this mode with a harness-bound, tokenizer-derived contract when the judge
    # tokenizer or answer stem changes.
    expected = {"A": 41, " A": 334, "B": 42, " B": 378}
    observed: dict[str, int] = {}
    for surface, expected_id in expected.items():
        encoded = tuple(int(token_id) for token_id in tokenizer.encode(surface, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(
                f"{contract_name} requires {surface!r} to encode to one token; observed={encoded}"
            )
        observed[surface] = encoded[0]
        if encoded[0] != expected_id:
            raise ValueError(
                f"{contract_name} tokenizer mapping changed for {surface!r}: "
                f"expected={expected_id}, observed={encoded[0]}. Define and Phase-0 validate a new contract."
            )
    return JudgeLabelTokenContract(
        name=contract_name,
        a_token_ids=(observed["A"], observed[" A"]),
        b_token_ids=(observed["B"], observed[" B"]),
        temporary=True,
    )


def _logsumexp(values: tuple[float, ...]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Semantic label logprobs must be finite and non-empty: {values!r}")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def semantic_label_logprobs(
    candidate_logprobs: Mapping[int, float],
    *,
    contract: JudgeLabelTokenContract,
) -> tuple[float, float]:
    missing = [
        token_id
        for token_id in contract.allowed_token_ids
        if token_id not in candidate_logprobs
    ]
    if missing:
        raise ValueError(f"Candidate logprob row is missing contract token ids: {missing}")
    logp_a = _logsumexp(tuple(float(candidate_logprobs[token_id]) for token_id in contract.a_token_ids))
    logp_b = _logsumexp(tuple(float(candidate_logprobs[token_id]) for token_id in contract.b_token_ids))
    return logp_a, logp_b


def order_symmetric_soft_judge_score(
    *,
    forward_candidate_logprobs: Mapping[int, float],
    reverse_candidate_logprobs: Mapping[int, float],
    contract: JudgeLabelTokenContract,
) -> OrderSymmetricSoftJudgeScore:
    logp_forward_a, logp_forward_b = semantic_label_logprobs(
        forward_candidate_logprobs, contract=contract
    )
    logp_reverse_a, logp_reverse_b = semantic_label_logprobs(
        reverse_candidate_logprobs, contract=contract
    )
    z_forward = logp_forward_a - logp_forward_b
    z_reverse = logp_reverse_a - logp_reverse_b
    z_symmetric = 0.5 * (z_forward - z_reverse)
    score = math.tanh(0.5 * z_symmetric)
    if not all(
        math.isfinite(value)
        for value in (z_forward, z_reverse, z_symmetric, score)
    ):
        raise ValueError("Non-finite order-symmetric soft judge score")
    return OrderSymmetricSoftJudgeScore(
        logp_forward_a=logp_forward_a,
        logp_forward_b=logp_forward_b,
        logp_reverse_a=logp_reverse_a,
        logp_reverse_b=logp_reverse_b,
        z_forward=z_forward,
        z_reverse=z_reverse,
        z_symmetric=z_symmetric,
        order_bias_logit=0.5 * (z_forward + z_reverse),
        score=score,
    )
