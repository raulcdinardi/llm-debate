from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    text: str | None = None
    messages: list[dict[str, str]] | None = None


@dataclass(frozen=True)
class SamplingParamSpec:
    param_id: str
    max_tokens: int
    temperature: float
    min_p: float = 0.0
    seed: int | None = None


@dataclass(frozen=True)
class AdapterStateSpec:
    state_id: str
    adapter_paths: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorSample:
    model_case_id: str
    backend: str
    adapter_state_id: str
    prompt_id: str
    param_id: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    text: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def completion_len(self) -> int:
        return len(self.completion_token_ids)

    @property
    def completion_logprob_sum(self) -> float:
        return float(sum(self.completion_logprobs))

    def key_without_backend(self) -> tuple[str, str, str, str]:
        return (self.model_case_id, self.adapter_state_id, self.prompt_id, self.param_id)


def sample_to_json(sample: BehaviorSample) -> dict[str, Any]:
    return asdict(sample)


def token_ngrams(token_ids: list[int], *, n: int) -> list[tuple[int, ...]]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(token_ids) < n:
        return []
    return [tuple(token_ids[idx : idx + n]) for idx in range(0, len(token_ids) - n + 1)]


def ngram_counts(samples: list[BehaviorSample], *, n: int) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for sample in samples:
        counts.update(token_ngrams(sample.completion_token_ids, n=n))
    return counts


def _normalized_counter(counter: Counter[tuple[int, ...]]) -> dict[tuple[int, ...], float]:
    total = float(sum(counter.values()))
    if total == 0.0:
        return {}
    return {key: float(value) / total for key, value in counter.items()}


def _kl_divergence(p: dict[Any, float], q: dict[Any, float], *, epsilon: float = 1e-12) -> float:
    support = set(p) | set(q)
    if not support:
        return 0.0
    p_smooth = {key: float(p[key]) if key in p else 0.0 for key in support}
    q_smooth = {key: float(q[key]) if key in q else 0.0 for key in support}
    p_total = sum(p_smooth.values()) + epsilon * len(support)
    q_total = sum(q_smooth.values()) + epsilon * len(support)
    out = 0.0
    for key in support:
        pk = (p_smooth[key] + epsilon) / p_total
        qk = (q_smooth[key] + epsilon) / q_total
        out += pk * math.log(pk / qk)
    return float(out)


def jensen_shannon_divergence(
    left: dict[Any, float],
    right: dict[Any, float],
    *,
    epsilon: float = 1e-12,
) -> float:
    support = set(left) | set(right)
    if not support:
        return 0.0
    left_total = sum(float(left[key]) for key in left) + epsilon * len(support)
    right_total = sum(float(right[key]) for key in right) + epsilon * len(support)
    p = {key: (float(left[key]) if key in left else 0.0) + epsilon for key in support}
    q = {key: (float(right[key]) if key in right else 0.0) + epsilon for key in support}
    p = {key: value / left_total for key, value in p.items()}
    q = {key: value / right_total for key, value in q.items()}
    midpoint = {key: 0.5 * (p[key] + q[key]) for key in support}
    return 0.5 * _kl_divergence(p, midpoint, epsilon=epsilon) + 0.5 * _kl_divergence(q, midpoint, epsilon=epsilon)


def length_ks_statistic(left_lengths: list[int], right_lengths: list[int]) -> float:
    if len(left_lengths) == 0 or len(right_lengths) == 0:
        return 0.0
    values = sorted(set(left_lengths) | set(right_lengths))
    left_sorted = sorted(left_lengths)
    right_sorted = sorted(right_lengths)
    left_idx = 0
    right_idx = 0
    best = 0.0
    for value in values:
        while left_idx < len(left_sorted) and left_sorted[left_idx] <= value:
            left_idx += 1
        while right_idx < len(right_sorted) and right_sorted[right_idx] <= value:
            right_idx += 1
        best = max(best, abs(left_idx / len(left_sorted) - right_idx / len(right_sorted)))
    return float(best)


def length_summary(samples: list[BehaviorSample]) -> dict[str, float]:
    lengths = [sample.completion_len for sample in samples]
    if len(lengths) == 0:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(lengths)),
        "mean": float(sum(lengths) / len(lengths)),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def exact_match_fraction(left: list[BehaviorSample], right: list[BehaviorSample]) -> float:
    right_by_key = {sample.key_without_backend(): sample for sample in right}
    compared = 0
    matched = 0
    for left_sample in left:
        key = left_sample.key_without_backend()
        if key not in right_by_key:
            continue
        compared += 1
        if left_sample.completion_token_ids == right_by_key[key].completion_token_ids:
            matched += 1
    if compared == 0:
        return 0.0
    return float(matched / compared)


def _top_logprob_dist(row: list[dict[str, Any]]) -> dict[int, float]:
    raw = {int(item["token_id"]): math.exp(float(item["logprob"])) for item in row}
    total = sum(raw.values())
    if total == 0.0:
        return {}
    return {token_id: prob / total for token_id, prob in raw.items()}


def truncated_topk_kl_for_samples(left: BehaviorSample, right: BehaviorSample) -> list[float]:
    left_rows = left.raw["completion_top_logprobs"] if "completion_top_logprobs" in left.raw else []
    right_rows = right.raw["completion_top_logprobs"] if "completion_top_logprobs" in right.raw else []
    limit = min(len(left_rows), len(right_rows))
    out = []
    for idx in range(limit):
        left_dist = _top_logprob_dist(left_rows[idx])
        right_dist = _top_logprob_dist(right_rows[idx])
        out.append(_kl_divergence(left_dist, right_dist))
    return out


def compare_sample_sets(left: list[BehaviorSample], right: list[BehaviorSample]) -> dict[str, float]:
    left_lengths = [sample.completion_len for sample in left]
    right_lengths = [sample.completion_len for sample in right]
    left_unigrams = _normalized_counter(ngram_counts(left, n=1))
    right_unigrams = _normalized_counter(ngram_counts(right, n=1))
    left_bigrams = _normalized_counter(ngram_counts(left, n=2))
    right_bigrams = _normalized_counter(ngram_counts(right, n=2))

    right_by_key = {sample.key_without_backend(): sample for sample in right}
    topk_kl_values: list[float] = []
    logprob_sum_abs_diffs: list[float] = []
    for left_sample in left:
        key = left_sample.key_without_backend()
        if key not in right_by_key:
            continue
        right_sample = right_by_key[key]
        topk_kl_values.extend(truncated_topk_kl_for_samples(left_sample, right_sample))
        logprob_sum_abs_diffs.append(abs(left_sample.completion_logprob_sum - right_sample.completion_logprob_sum))

    return {
        "left_count": float(len(left)),
        "right_count": float(len(right)),
        "exact_token_match_fraction": exact_match_fraction(left, right),
        "length_ks": length_ks_statistic(left_lengths, right_lengths),
        "left_mean_len": length_summary(left)["mean"],
        "right_mean_len": length_summary(right)["mean"],
        "unigram_jsd": jensen_shannon_divergence(left_unigrams, right_unigrams),
        "bigram_jsd": jensen_shannon_divergence(left_bigrams, right_bigrams),
        "mean_abs_logprob_sum_diff": (
            float(sum(logprob_sum_abs_diffs) / len(logprob_sum_abs_diffs))
            if len(logprob_sum_abs_diffs) > 0
            else 0.0
        ),
        "mean_truncated_topk_kl": (
            float(sum(topk_kl_values) / len(topk_kl_values))
            if len(topk_kl_values) > 0
            else 0.0
        ),
    }


def generate_sampling_param_specs(*, seed: int, count: int) -> list[SamplingParamSpec]:
    rng = random.Random(seed)
    specs = [
        SamplingParamSpec(param_id="greedy_16", max_tokens=16, temperature=0.0, min_p=0.0, seed=seed),
        SamplingParamSpec(param_id="sample_t07_16", max_tokens=16, temperature=0.7, min_p=0.0, seed=seed + 1),
    ]
    temperatures = [0.2, 0.7, 1.0]
    min_ps = [0.0, 0.02, 0.08]
    max_tokens_values = [8, 16, 32]
    while len(specs) < count:
        idx = len(specs)
        specs.append(
            SamplingParamSpec(
                param_id=f"fuzz_{idx:03d}",
                max_tokens=rng.choice(max_tokens_values),
                temperature=rng.choice(temperatures),
                min_p=rng.choice(min_ps),
                seed=seed + 1000 + idx,
            )
        )
    return specs[:count]
