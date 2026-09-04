from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol


JUDGE_LABEL_TOKEN_CONTRACT_NONE = "none"
LFM25_AB_WHITESPACE_COMPAT_V1 = "lfm25_ab_whitespace_compat_v1"
LFM25_OPENBOOKQA_SPACED_AB_V1 = "lfm25_openbookqa_spaced_ab_v1"
JUDGE_LABEL_TOKEN_CONTRACTS = (
    JUDGE_LABEL_TOKEN_CONTRACT_NONE,
    LFM25_AB_WHITESPACE_COMPAT_V1,
    LFM25_OPENBOOKQA_SPACED_AB_V1,
)


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class JudgeLabelTokenContract:
    name: str
    a_token_ids: tuple[int, ...]
    b_token_ids: tuple[int, ...]
    temporary: bool
    canonical_a: str = ""
    canonical_b: str = ""
    required_prompt_suffix: str = ""

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
            "canonical_a": self.canonical_a,
            "canonical_b": self.canonical_b,
            "required_prompt_suffix": self.required_prompt_suffix,
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
    forward_referent_a_probability: float
    reverse_referent_a_probability: float
    referent_js_divergence: float
    referent_js_divergence_normalized: float
    coherence_reliability: float

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
            "forward_referent_a_probability": self.forward_referent_a_probability,
            "reverse_referent_a_probability": self.reverse_referent_a_probability,
            "referent_js_divergence": self.referent_js_divergence,
            "referent_js_divergence_normalized": self.referent_js_divergence_normalized,
            "coherence_reliability": self.coherence_reliability,
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
    if contract_name not in (LFM25_AB_WHITESPACE_COMPAT_V1, LFM25_OPENBOOKQA_SPACED_AB_V1):
        raise ValueError(f"Unknown judge label token contract: {contract_name!r}")

    if contract_name == LFM25_OPENBOOKQA_SPACED_AB_V1:
        expected = {" A": 334, " B": 378}
        observed: dict[str, int] = {}
        for surface, expected_id in expected.items():
            encoded = tuple(
                int(token_id)
                for token_id in tokenizer.encode(surface, add_special_tokens=False)
            )
            if encoded != (expected_id,):
                raise ValueError(
                    f"{contract_name} tokenizer mapping changed for {surface!r}: "
                    f"expected={(expected_id,)}, observed={encoded}. Define and Phase-0 "
                    "validate a new contract."
                )
            observed[surface] = encoded[0]
        return JudgeLabelTokenContract(
            name=contract_name,
            a_token_ids=(observed[" A"],),
            b_token_ids=(observed[" B"],),
            temporary=False,
            canonical_a=" A",
            canonical_b=" B",
            required_prompt_suffix="Agent",
        )

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
        canonical_a=" A",
        canonical_b=" B",
    )


def validate_judge_prompt_label_boundary(
    *,
    tokenizer: TokenizerLike,
    prompt_text: str,
    prompt_token_ids: list[int] | tuple[int, ...],
    contract: JudgeLabelTokenContract,
) -> None:
    """Prove that each canonical label is exactly one appended token.

    Checking the label in isolation is insufficient: a BPE/SentencePiece boundary
    can retokenize the end of the prefill together with the label.  This equality
    is the mechanical non-agglutination contract used by the strict OpenBookQA
    judge harness.
    """
    if not contract.canonical_a or not contract.canonical_b:
        raise ValueError(f"Contract {contract.name!r} has no canonical label surfaces")
    if contract.required_prompt_suffix and not prompt_text.endswith(contract.required_prompt_suffix):
        raise ValueError(
            f"Judge prompt for {contract.name!r} must end with "
            f"{contract.required_prompt_suffix!r}; observed suffix={prompt_text[-32:]!r}"
        )
    prompt_ids = tuple(int(token_id) for token_id in prompt_token_ids)
    for surface, allowed_ids in (
        (contract.canonical_a, contract.a_token_ids),
        (contract.canonical_b, contract.b_token_ids),
    ):
        isolated = tuple(tokenizer.encode(surface, add_special_tokens=False))
        if len(isolated) != 1 or isolated[0] not in allowed_ids:
            raise ValueError(f"Canonical label {surface!r} is not one allowed token")
        expected_id = isolated[0]
        combined = tuple(
            int(token_id)
            for token_id in tokenizer.encode(prompt_text + surface, add_special_tokens=False)
        )
        expected = (*prompt_ids, int(expected_id))
        if combined != expected:
            raise ValueError(
                f"Judge prompt/label boundary agglutinated for {surface!r}: "
                f"expected final tokens={expected[-4:]!r}, observed={combined[-4:]!r}"
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
    forward_referent_a_probability = _sigmoid(z_forward)
    # In reverse display order, output label B refers to original Agent A.
    reverse_referent_a_probability = _sigmoid(-z_reverse)
    referent_js_divergence = bernoulli_js_divergence(
        forward_referent_a_probability,
        reverse_referent_a_probability,
    )
    referent_js_divergence_normalized = referent_js_divergence / math.log(2.0)
    coherence_reliability = 1.0 - referent_js_divergence_normalized
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
        forward_referent_a_probability=forward_referent_a_probability,
        reverse_referent_a_probability=reverse_referent_a_probability,
        referent_js_divergence=referent_js_divergence,
        referent_js_divergence_normalized=referent_js_divergence_normalized,
        coherence_reliability=coherence_reliability,
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def bernoulli_js_divergence(p: float, q: float) -> float:
    """Natural-log Jensen-Shannon divergence for Bernoulli distributions."""
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (p, q)):
        raise ValueError(f"Bernoulli probabilities must be finite in [0, 1]: p={p}, q={q}")

    def _kl_term(probability: float, midpoint: float) -> float:
        return 0.0 if probability == 0.0 else probability * math.log(probability / midpoint)

    midpoint_a = 0.5 * (p + q)
    midpoint_b = 1.0 - midpoint_a
    divergence = 0.5 * (
        _kl_term(p, midpoint_a)
        + _kl_term(1.0 - p, midpoint_b)
        + _kl_term(q, midpoint_a)
        + _kl_term(1.0 - q, midpoint_b)
    )
    if not math.isfinite(divergence) or divergence < -1e-15 or divergence > math.log(2.0) + 1e-12:
        raise ValueError(f"Invalid Bernoulli Jensen-Shannon divergence: {divergence}")
    return max(0.0, divergence)
