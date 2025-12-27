"""Debate environment for tinker-style training (async).

Implements symmetric 3-round debate:
- R1: Both agents propose solutions (FROZEN after this)
- R2: Each agent sees opponent's R1, argues for own solution
- R3: Each agent sees opponent's R2, responds to criticism
- Judge: Separate LLM call declares winner (A or B)

Training uses rejection sampling: only winners are trained on.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from .debate_types import DebateConfig, DebateResult, DebateTrajectory, Transition, Verdict
from .confidence.confidence_env import parse_confidence


# === Extraction ===

def extract_solution(text: str) -> str | None:
    match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_verdict(text: str) -> Verdict:
    """Extract verdict from judge response. Returns 'INVALID' if parsing fails."""
    match = re.search(r"<VERDICT>(.*?)</VERDICT>", text, re.DOTALL)
    if not match:
        return "INVALID"

    verdict = match.group(1).strip().upper()
    if verdict in ("A", "B"):
        return verdict  # type: ignore[return-value]

    return "INVALID"


def extract_reasoning(text: str) -> str:
    match = re.search(r"<REASONING>(.*?)</REASONING>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


# === Prompt Builders ===

def _im_start(role: str) -> str:
    return f"<|im_start|>{role}\n"


def _im_end() -> str:
    return "<|im_end|>\n"


def build_r1_prompt(question: str, config: DebateConfig) -> str:
    return (
        _im_start("system")
        + config.system_propose
        + "\n"
        + _im_end()
        + _im_start("user")
        + question
        + "\n"
        + _im_end()
        + _im_start("assistant")
    )


def build_r2_continuation(opponent_r1: str, config: DebateConfig) -> str:
    # Continues conversation from R1; starts with <|im_end|> to close previous assistant turn.
    user_msg = config.r2_user_template.format(opponent_r1=opponent_r1)
    return _im_end() + _im_start("user") + user_msg + "\n" + _im_end() + _im_start("assistant")


def build_r3_continuation(opponent_r2: str, config: DebateConfig) -> str:
    user_msg = config.r3_user_template.format(opponent_r2=opponent_r2)
    return _im_end() + _im_start("user") + user_msg + "\n" + _im_end() + _im_start("assistant")


def build_judge_prompt(
    *,
    question: str,
    r1_a: str,
    r1_b: str,
    r2_a: str,
    r2_b: str,
    r3_a: str,
    r3_b: str,
    config: DebateConfig,
    strict: bool = False,
) -> str:
    extra = "\n\nIMPORTANT: Output ONLY the tags. No extra text." if strict else ""
    system = config.system_judge + extra

    return (
        _im_start("system")
        + system
        + "\n"
        + _im_end()
        + _im_start("user")
        + f"Question: {question}\n\n"
        + "=== AGENT A ===\n"
        + "Round 1 (Proposal):\n"
        + f"{r1_a}\n\n"
        + "Round 2 (Argument):\n"
        + f"{r2_a}\n\n"
        + "Round 3 (Response):\n"
        + f"{r3_a}\n\n"
        + "=== AGENT B ===\n"
        + "Round 1 (Proposal):\n"
        + f"{r1_b}\n\n"
        + "Round 2 (Argument):\n"
        + f"{r2_b}\n\n"
        + "Round 3 (Response):\n"
        + f"{r3_b}\n\n"
        + "Based on the debate above, which agent made a more convincing case?\n"
        + "Consider: solution correctness, argument quality, rebuttal effectiveness.\n"
        + _im_end()
        + _im_start("assistant")
    )


# === Rollout Client ===

# Placeholder for failed requests - keeps indices aligned
FAILED_RESULT: tuple[str, list[int], list[int], list[float], dict] = (
    "[REQUEST FAILED]", [], [], [], {"error": "request_failed"}
)

FAILED_TOKEN_RESULT: tuple[list[int], list[int], list[float], dict] = (
    [],
    [],
    [],
    {"error": "request_failed"},
)


@dataclass
class DebateRolloutClient:
    """Adapter: async generation and tokenization interface."""

    # generate_fn(prompts, max_tokens, temperature) -> awaitable
    # -> list[(completion_text, prompt_tokens, completion_tokens, completion_logprobs, raw_response) | None]
    generate_fn: Callable[
        [list[str], int | None, float],
        Awaitable[list[tuple[str, list[int], list[int], list[float], dict] | None]],
    ]

    async def generate(
        self, *, prompts: list[str], max_tokens: int | None, temperature: float
    ) -> list[tuple[str, list[int], list[int], list[float], dict]]:
        """Generate completions. Failed requests return FAILED_RESULT placeholder."""
        raw_results = await self.generate_fn(prompts, max_tokens, temperature)
        # Replace None with placeholder, print warning
        results = []
        for i, r in enumerate(raw_results):
            if r is None:
                print(f"\n{'!'*60}")
                print(f"!!! REQUEST {i+1}/{len(raw_results)} RETURNED NONE - DEBATE WILL BE INVALID")
                print(f"{'!'*60}\n")
                results.append(FAILED_RESULT)
            else:
                results.append(r)
        return results


# === Token-only Rollout Client ===

@dataclass
class DebateTokenRolloutClient:
    """Adapter: async token-level sampling interface.

    sample_fn(prompt_tokens_list, max_tokens, temperature)
      -> awaitable[list[(prompt_tokens, completion_tokens, completion_logprobs, raw_response) | None]]
    """

    sample_fn: Callable[
        [list[list[int]], int | None, float],
        Awaitable[list[tuple[list[int], list[int], list[float], dict] | None]],
    ]
    decode_fn: Callable[[list[int]], str]

    async def sample(
        self, *, prompt_tokens_list: list[list[int]], max_tokens: int | None, temperature: float
    ) -> list[tuple[list[int], list[int], list[float], dict]]:
        raw_results = await self.sample_fn(prompt_tokens_list, max_tokens, temperature)
        results: list[tuple[list[int], list[int], list[float], dict]] = []
        for i, r in enumerate(raw_results):
            if r is None:
                print(f"\n{'!'*60}")
                print(f"!!! TOKEN SAMPLING REQUEST {i+1}/{len(raw_results)} RETURNED NONE - DEBATE WILL BE INVALID")
                print(f"{'!'*60}\n")
                results.append(FAILED_TOKEN_RESULT)
            else:
                results.append(r)
        return results


def _split_template(template: str, placeholder: str) -> tuple[str, str]:
    if template.count(placeholder) != 1:
        raise ValueError(f"Expected exactly one {placeholder} in templrate, got {template.count(placeholder)}")
    pre, post = template.split(placeholder)
    return pre, post


def _encode(tokenizer, s: str) -> list[int]:
    return list(tokenizer.encode(s, add_special_tokens=False))


async def run_debate_batch_token_only(
    questions: list[tuple[str, str | None]],
    token_client: DebateTokenRolloutClient,
    tokenizer,
    config: DebateConfig,
    judge_client: DebateRolloutClient,
    judge_fn: JudgeFn | None = None,
) -> list[DebateResult]:
    """Batched debate runner that preserves extension property by appending tokens.

    Key guarantee: round-(k+1) prompt tokens are constructed as:
      prompt_{k+1} = prompt_k + completion_k + continuation_tokens

    so we never re-tokenize the whole history.
    """
    if config.num_rounds != 3:
        raise NotImplementedError("Only num_rounds=3 is currently supported")

    # Pre-split user templates so we can splice opponent tokens without formatting strings.
    r2_pre, r2_post = _split_template(config.r2_user_template, "{opponent_r1}")
    r3_pre, r3_post = _split_template(config.r3_user_template, "{opponent_r2}")

    r2_prefix_tokens = _encode(tokenizer, _im_end() + _im_start("user") + r2_pre)
    r2_suffix_tokens = _encode(tokenizer, r2_post + "\n" + _im_end() + _im_start("assistant"))
    r3_prefix_tokens = _encode(tokenizer, _im_end() + _im_start("user") + r3_pre)
    r3_suffix_tokens = _encode(tokenizer, r3_post + "\n" + _im_end() + _im_start("assistant"))

    # Track debates with failed requests
    failed_debate_indices: set[int] = set()

    # === Round 1 ===
    base_r1_prompt_tokens: list[list[int]] = [
        _encode(tokenizer, build_r1_prompt(q, config)) for q, _gt in questions
    ]
    r1_prompt_tokens_list: list[list[int]] = []
    for p in base_r1_prompt_tokens:
        r1_prompt_tokens_list.extend([p, p])

    r1_results = await token_client.sample(
        prompt_tokens_list=r1_prompt_tokens_list,
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )

    r1_tokens_a: list[list[int]] = []
    r1_tokens_b: list[list[int]] = []
    r1_lps_a: list[list[float]] = []
    r1_lps_b: list[list[float]] = []
    r1_raw_a: list[dict] = []
    r1_raw_b: list[dict] = []
    r1_text_a: list[str] = []
    r1_text_b: list[str] = []

    for i in range(0, len(r1_results), 2):
        a_prompt, a_comp, a_lps, a_raw = r1_results[i]
        b_prompt, b_comp, b_lps, b_raw = r1_results[i + 1]
        debate_idx = i // 2
        if a_raw.get("error") == "request_failed" or b_raw.get("error") == "request_failed":
            failed_debate_indices.add(debate_idx)
        r1_tokens_a.append(a_comp)
        r1_tokens_b.append(b_comp)
        r1_lps_a.append(a_lps)
        r1_lps_b.append(b_lps)
        r1_raw_a.append(a_raw)
        r1_raw_b.append(b_raw)
        r1_text_a.append(token_client.decode_fn(a_comp))
        r1_text_b.append(token_client.decode_fn(b_comp))

    # === Round 2 ===
    r2_prompt_tokens_list: list[list[int]] = []
    for idx in range(len(questions)):
        r2_a_cont = r2_prefix_tokens + r1_tokens_b[idx] + r2_suffix_tokens
        r2_b_cont = r2_prefix_tokens + r1_tokens_a[idx] + r2_suffix_tokens
        r2_prompt_tokens_list.append(base_r1_prompt_tokens[idx] + r1_tokens_a[idx] + r2_a_cont)
        r2_prompt_tokens_list.append(base_r1_prompt_tokens[idx] + r1_tokens_b[idx] + r2_b_cont)

    r2_results = await token_client.sample(
        prompt_tokens_list=r2_prompt_tokens_list,
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )

    r2_prompt_tokens_a: list[list[int]] = []
    r2_prompt_tokens_b: list[list[int]] = []
    r2_tokens_a: list[list[int]] = []
    r2_tokens_b: list[list[int]] = []
    r2_lps_a: list[list[float]] = []
    r2_lps_b: list[list[float]] = []
    r2_raw_a: list[dict] = []
    r2_raw_b: list[dict] = []
    r2_text_a: list[str] = []
    r2_text_b: list[str] = []

    for i in range(0, len(r2_results), 2):
        a_prompt, a_comp, a_lps, a_raw = r2_results[i]
        b_prompt, b_comp, b_lps, b_raw = r2_results[i + 1]
        debate_idx = i // 2
        if a_raw.get("error") == "request_failed" or b_raw.get("error") == "request_failed":
            failed_debate_indices.add(debate_idx)
        r2_prompt_tokens_a.append(a_prompt)
        r2_prompt_tokens_b.append(b_prompt)
        r2_tokens_a.append(a_comp)
        r2_tokens_b.append(b_comp)
        r2_lps_a.append(a_lps)
        r2_lps_b.append(b_lps)
        r2_raw_a.append(a_raw)
        r2_raw_b.append(b_raw)
        r2_text_a.append(token_client.decode_fn(a_comp))
        r2_text_b.append(token_client.decode_fn(b_comp))

    # === Round 3 ===
    r3_prompt_tokens_list: list[list[int]] = []
    for idx in range(len(questions)):
        r3_a_cont = r3_prefix_tokens + r2_tokens_b[idx] + r3_suffix_tokens
        r3_b_cont = r3_prefix_tokens + r2_tokens_a[idx] + r3_suffix_tokens
        r2_a_prompt = base_r1_prompt_tokens[idx] + r1_tokens_a[idx] + (r2_prefix_tokens + r1_tokens_b[idx] + r2_suffix_tokens)
        r2_b_prompt = base_r1_prompt_tokens[idx] + r1_tokens_b[idx] + (r2_prefix_tokens + r1_tokens_a[idx] + r2_suffix_tokens)
        r3_prompt_tokens_list.append(r2_a_prompt + r2_tokens_a[idx] + r3_a_cont)
        r3_prompt_tokens_list.append(r2_b_prompt + r2_tokens_b[idx] + r3_b_cont)

    r3_results = await token_client.sample(
        prompt_tokens_list=r3_prompt_tokens_list,
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )

    r3_prompt_tokens_a: list[list[int]] = []
    r3_prompt_tokens_b: list[list[int]] = []
    r3_tokens_a: list[list[int]] = []
    r3_tokens_b: list[list[int]] = []
    r3_lps_a: list[list[float]] = []
    r3_lps_b: list[list[float]] = []
    r3_raw_a: list[dict] = []
    r3_raw_b: list[dict] = []
    r3_text_a: list[str] = []
    r3_text_b: list[str] = []

    for i in range(0, len(r3_results), 2):
        a_prompt, a_comp, a_lps, a_raw = r3_results[i]
        b_prompt, b_comp, b_lps, b_raw = r3_results[i + 1]
        debate_idx = i // 2
        if a_raw.get("error") == "request_failed" or b_raw.get("error") == "request_failed":
            failed_debate_indices.add(debate_idx)
        r3_prompt_tokens_a.append(a_prompt)
        r3_prompt_tokens_b.append(b_prompt)
        r3_tokens_a.append(a_comp)
        r3_tokens_b.append(b_comp)
        r3_lps_a.append(a_lps)
        r3_lps_b.append(b_lps)
        r3_raw_a.append(a_raw)
        r3_raw_b.append(b_raw)
        r3_text_a.append(token_client.decode_fn(a_comp))
        r3_text_b.append(token_client.decode_fn(b_comp))

    # === Judge ===
    verdicts: list[Verdict] = []
    reasons: list[str] = []
    judge_metas: list[dict[str, object]] = []
    judge_prompt_tokens_list: list[list[int] | None] = []
    judge_completion_tokens_list: list[list[int] | None] = []
    judge_completion_logprobs_list: list[list[float] | None] = []
    judge_raw_response_list: list[dict | None] = []

    if judge_fn is not None:
        for idx, (q, _gt) in enumerate(questions):
            v, r = judge_fn(
                q,
                r1_text_a[idx],
                r1_text_b[idx],
                r2_text_a[idx],
                r2_text_b[idx],
                r3_text_a[idx],
                r3_text_b[idx],
            )
            verdicts.append(v)
            reasons.append(r)
            judge_metas.append({})
            judge_prompt_tokens_list.append(None)
            judge_completion_tokens_list.append(None)
            judge_completion_logprobs_list.append(None)
            judge_raw_response_list.append(None)
    else:
        judge_prompts = [
            build_judge_prompt(
                question=q,
                r1_a=r1_text_a[idx],
                r1_b=r1_text_b[idx],
                r2_a=r2_text_a[idx],
                r2_b=r2_text_b[idx],
                r3_a=r3_text_a[idx],
                r3_b=r3_text_b[idx],
                config=config,
                strict=False,
            )
            for idx, (q, _gt) in enumerate(questions)
        ]
        judge_results = await judge_client.generate(
            prompts=judge_prompts, max_tokens=config.max_tokens_per_turn, temperature=0.3
        )
        for i, r in enumerate(judge_results):
            if r[4].get("error") == "request_failed":
                failed_debate_indices.add(i)
        for judge_text, j_prompt_toks, j_comp_toks, j_lps, j_raw in judge_results:
            verdicts.append(extract_verdict(judge_text))
            reasons.append(extract_reasoning(judge_text))
            judge_metas.append({})
            judge_prompt_tokens_list.append(j_prompt_toks)
            judge_completion_tokens_list.append(j_comp_toks)
            judge_completion_logprobs_list.append(j_lps)
            judge_raw_response_list.append(j_raw)

    results: list[DebateResult] = []
    for idx, (q, gt) in enumerate(questions):
        sol_a = extract_solution(r1_text_a[idx])
        sol_b = extract_solution(r1_text_b[idx])

        final_verdict = verdicts[idx]
        final_reasoning = reasons[idx]
        if idx in failed_debate_indices:
            final_verdict = "INVALID"
            final_reasoning = "[DEBATE FAILED: One or more API requests failed]"
            print(f"\n{'!'*60}")
            print(f"!!! DEBATE {idx} MARKED INVALID DUE TO FAILED REQUESTS")
            print(f"{'!'*60}\n")

        traj_a = DebateTrajectory(
            agent="A",
            transitions=[
                Transition(
                    prompt_tokens=base_r1_prompt_tokens[idx],
                    completion_tokens=r1_tokens_a[idx],
                    completion_logprobs=r1_lps_a[idx],
                    round_num=1,
                    metrics={"solution": sol_a},
                    raw_response=r1_raw_a[idx],
                ),
                Transition(
                    prompt_tokens=r2_prompt_tokens_a[idx],
                    completion_tokens=r2_tokens_a[idx],
                    completion_logprobs=r2_lps_a[idx],
                    round_num=2,
                    raw_response=r2_raw_a[idx],
                ),
                Transition(
                    prompt_tokens=r3_prompt_tokens_a[idx],
                    completion_tokens=r3_tokens_a[idx],
                    completion_logprobs=r3_lps_a[idx],
                    round_num=3,
                    raw_response=r3_raw_a[idx],
                ),
            ],
            frozen_solution=sol_a,
            metrics={"r1": r1_text_a[idx], "r2": r2_text_a[idx], "r3": r3_text_a[idx]},
        )
        traj_b = DebateTrajectory(
            agent="B",
            transitions=[
                Transition(
                    prompt_tokens=base_r1_prompt_tokens[idx],
                    completion_tokens=r1_tokens_b[idx],
                    completion_logprobs=r1_lps_b[idx],
                    round_num=1,
                    metrics={"solution": sol_b},
                    raw_response=r1_raw_b[idx],
                ),
                Transition(
                    prompt_tokens=r2_prompt_tokens_b[idx],
                    completion_tokens=r2_tokens_b[idx],
                    completion_logprobs=r2_lps_b[idx],
                    round_num=2,
                    raw_response=r2_raw_b[idx],
                ),
                Transition(
                    prompt_tokens=r3_prompt_tokens_b[idx],
                    completion_tokens=r3_tokens_b[idx],
                    completion_logprobs=r3_lps_b[idx],
                    round_num=3,
                    raw_response=r3_raw_b[idx],
                ),
            ],
            frozen_solution=sol_b,
            metrics={"r1": r1_text_b[idx], "r2": r2_text_b[idx], "r3": r3_text_b[idx]},
        )

        results.append(
            DebateResult(
                question=q,
                ground_truth=gt,
                trajectory_a=traj_a,
                trajectory_b=traj_b,
                verdict=final_verdict,
                judge_reasoning=final_reasoning,
                metrics={
                    "judge_meta": judge_metas[idx],
                    "solution_agreement": sol_a == sol_b if sol_a and sol_b else None,
                    "request_failed": idx in failed_debate_indices,
                    "token_only_rollout": True,
                },
                judge_prompt_tokens=judge_prompt_tokens_list[idx],
                judge_completion_tokens=judge_completion_tokens_list[idx],
                judge_completion_logprobs=judge_completion_logprobs_list[idx],
                judge_raw_response=judge_raw_response_list[idx],
            )
        )

    return results

# === Judge Functions ===

JudgeFn = Callable[
    [str, str, str, str, str, str, str],  # question, r1_a, r1_b, r2_a, r2_b, r3_a, r3_b
    tuple[Verdict, str],
]


def mock_judge_random(
    question: str,
    r1_a: str,
    r1_b: str,
    r2_a: str,
    r2_b: str,
    r3_a: str,
    r3_b: str,
) -> tuple[Verdict, str]:
    verdict: Verdict = random.choice(["A", "B"])
    reasoning = f"[MOCK JUDGE] Randomly selected {verdict}."
    return verdict, reasoning


def confidence_judge_r1(
    question: str,
    r1_a: str,
    r1_b: str,
    r2_a: str,
    r2_b: str,
    r3_a: str,
    r3_b: str,
) -> tuple[Verdict, str]:
    """Verdict based only on R1-reported confidence (tie-break randomly)."""
    _ = (question, r2_a, r2_b, r3_a, r3_b)
    conf_a = parse_confidence(r1_a) or 0.0
    conf_b = parse_confidence(r1_b) or 0.0
    if conf_a == conf_b:
        verdict: Verdict = random.choice(["A", "B"])
        return verdict, f"[CONFIDENCE JUDGE] tie ({conf_a:.3f} vs {conf_b:.3f}) -> {verdict}"
    verdict = "A" if conf_a > conf_b else "B"
    return verdict, f"[CONFIDENCE JUDGE] conf_a={conf_a:.3f} conf_b={conf_b:.3f} -> {verdict}"


async def llm_judge(
    *,
    client: DebateRolloutClient,
    question: str,
    r1_a: str,
    r1_b: str,
    r2_a: str,
    r2_b: str,
    r3_a: str,
    r3_b: str,
    config: DebateConfig,
) -> tuple[Verdict, str, dict, list[int], list[int], list[float], dict]:
    """LLM judge (single-pass, no retry, async).

    If parsing fails, we raise to avoid silently corrupting training / analysis.

    Returns: (verdict, reasoning, meta, prompt_tokens, completion_tokens, completion_logprobs, raw_response)
    """

    meta: dict[str, object] = {}

    prompt = build_judge_prompt(
        question=question,
        r1_a=r1_a,
        r1_b=r1_b,
        r2_a=r2_a,
        r2_b=r2_b,
        r3_a=r3_a,
        r3_b=r3_b,
        config=config,
        strict=False,
    )
    results = await client.generate(
        prompts=[prompt], max_tokens=config.max_tokens_per_turn, temperature=0.3
    )
    (judge_text, prompt_tokens, completion_tokens, completion_logprobs, raw_response) = results[0]

    verdict = extract_verdict(judge_text)
    reasoning = extract_reasoning(judge_text)
    return verdict, reasoning, meta, prompt_tokens, completion_tokens, completion_logprobs, raw_response


# === Debate Rollout ===

async def run_debate(
    question: str,
    ground_truth: str | None,
    client: DebateRolloutClient,
    config: DebateConfig,
    judge_fn: JudgeFn | None = None,
) -> DebateResult:
    if config.num_rounds != 3:
        raise NotImplementedError("Only num_rounds=3 is currently supported")

    # === Round 1: Propose ===
    r1_prompt = build_r1_prompt(question, config)
    r1_results = await client.generate(
        prompts=[r1_prompt, r1_prompt],
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )
    r1_a, r1_a_prompt_tokens, r1_a_tokens, r1_a_logprobs, r1_a_raw = r1_results[0]
    r1_b, r1_b_prompt_tokens, r1_b_tokens, r1_b_logprobs, r1_b_raw = r1_results[1]

    solution_a = extract_solution(r1_a)
    solution_b = extract_solution(r1_b)

    transition_a_r1 = Transition(
        prompt_tokens=r1_a_prompt_tokens,
        completion_tokens=r1_a_tokens,
        completion_logprobs=r1_a_logprobs,
        round_num=1,
        metrics={"solution": solution_a},
        raw_response=r1_a_raw,
    )
    transition_b_r1 = Transition(
        prompt_tokens=r1_b_prompt_tokens,
        completion_tokens=r1_b_tokens,
        completion_logprobs=r1_b_logprobs,
        round_num=1,
        metrics={"solution": solution_b},
        raw_response=r1_b_raw,
    )

    # === Round 2: Argue ===
    r2_a_prompt = r1_prompt + r1_a + build_r2_continuation(r1_b, config)
    r2_b_prompt = r1_prompt + r1_b + build_r2_continuation(r1_a, config)

    r2_results = await client.generate(
        prompts=[r2_a_prompt, r2_b_prompt],
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )
    r2_a, r2_a_prompt_tokens, r2_a_tokens, r2_a_logprobs, r2_a_raw = r2_results[0]
    r2_b, r2_b_prompt_tokens, r2_b_tokens, r2_b_logprobs, r2_b_raw = r2_results[1]

    transition_a_r2 = Transition(
        prompt_tokens=r2_a_prompt_tokens,
        completion_tokens=r2_a_tokens,
        completion_logprobs=r2_a_logprobs,
        round_num=2,
        raw_response=r2_a_raw,
    )
    transition_b_r2 = Transition(
        prompt_tokens=r2_b_prompt_tokens,
        completion_tokens=r2_b_tokens,
        completion_logprobs=r2_b_logprobs,
        round_num=2,
        raw_response=r2_b_raw,
    )

    # === Round 3: Respond ===
    r3_a_prompt = r2_a_prompt + r2_a + build_r3_continuation(r2_b, config)
    r3_b_prompt = r2_b_prompt + r2_b + build_r3_continuation(r2_a, config)

    r3_results = await client.generate(
        prompts=[r3_a_prompt, r3_b_prompt],
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )
    r3_a, r3_a_prompt_tokens, r3_a_tokens, r3_a_logprobs, r3_a_raw = r3_results[0]
    r3_b, r3_b_prompt_tokens, r3_b_tokens, r3_b_logprobs, r3_b_raw = r3_results[1]

    transition_a_r3 = Transition(
        prompt_tokens=r3_a_prompt_tokens,
        completion_tokens=r3_a_tokens,
        completion_logprobs=r3_a_logprobs,
        round_num=3,
        raw_response=r3_a_raw,
    )
    transition_b_r3 = Transition(
        prompt_tokens=r3_b_prompt_tokens,
        completion_tokens=r3_b_tokens,
        completion_logprobs=r3_b_logprobs,
        round_num=3,
        raw_response=r3_b_raw,
    )

    # === Judge ===
    judge_meta: dict[str, object] = {}
    judge_prompt_tokens: list[int] | None = None
    judge_completion_tokens: list[int] | None = None
    judge_completion_logprobs: list[float] | None = None
    judge_raw_response: dict | None = None

    if judge_fn is not None:
        verdict, reasoning = judge_fn(question, r1_a, r1_b, r2_a, r2_b, r3_a, r3_b)
    else:
        verdict, reasoning, judge_meta, judge_prompt_tokens, judge_completion_tokens, judge_completion_logprobs, judge_raw_response = await llm_judge(
            client=client,
            question=question,
            r1_a=r1_a,
            r1_b=r1_b,
            r2_a=r2_a,
            r2_b=r2_b,
            r3_a=r3_a,
            r3_b=r3_b,
            config=config,
        )

    trajectory_a = DebateTrajectory(
        agent="A",
        transitions=[transition_a_r1, transition_a_r2, transition_a_r3],
        frozen_solution=solution_a,
        metrics={"r1": r1_a, "r2": r2_a, "r3": r3_a},
    )
    trajectory_b = DebateTrajectory(
        agent="B",
        transitions=[transition_b_r1, transition_b_r2, transition_b_r3],
        frozen_solution=solution_b,
        metrics={"r1": r1_b, "r2": r2_b, "r3": r3_b},
    )

    return DebateResult(
        question=question,
        ground_truth=ground_truth,
        trajectory_a=trajectory_a,
        trajectory_b=trajectory_b,
        verdict=verdict,
        judge_reasoning=reasoning,
        metrics={
            "judge_meta": judge_meta,
            "solution_agreement": solution_a == solution_b if solution_a and solution_b else None,
        },
        judge_prompt_tokens=judge_prompt_tokens,
        judge_completion_tokens=judge_completion_tokens,
        judge_completion_logprobs=judge_completion_logprobs,
        judge_raw_response=judge_raw_response,
    )


async def run_debate_batch(
    questions: list[tuple[str, str | None]],
    client: DebateRolloutClient,
    config: DebateConfig,
    judge_fn: JudgeFn | None = None,
) -> list[DebateResult]:
    """Batched debate runner (async).

    This is parallelizable because it batches all rounds across all debates:
    - one batched generate for all R1
    - one batched generate for all R2
    - one batched generate for all R3
    - one batched judge (if using llm_judge)
    """

    if config.num_rounds != 3:
        raise NotImplementedError("Only num_rounds=3 is currently supported")

    # Track debates with failed requests
    failed_debate_indices: set[int] = set()

    def check_for_failures(results: list, round_name: str) -> None:
        """Check results for failures and track affected debate indices."""
        for i, r in enumerate(results):
            if r[4].get("error") == "request_failed":  # Check raw_response for error marker
                debate_idx = i // 2  # 2 results per debate (A and B)
                failed_debate_indices.add(debate_idx)
                print(f"!!! {round_name} result {i} failed -> debate {debate_idx} marked INVALID")

    # R1 prompts: 2 per debate
    r1_prompts: list[str] = []
    for q, _gt in questions:
        p = build_r1_prompt(q, config)
        r1_prompts.extend([p, p])

    r1_results = await client.generate(
        prompts=r1_prompts,
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )
    check_for_failures(r1_results, "R1")

    # Group R1 results
    r1_text_a: list[str] = []
    r1_text_b: list[str] = []
    r1_prompt_tokens_a: list[list[int]] = []
    r1_prompt_tokens_b: list[list[int]] = []
    r1_tokens_a: list[list[int]] = []
    r1_tokens_b: list[list[int]] = []
    r1_lps_a: list[list[float]] = []
    r1_lps_b: list[list[float]] = []
    r1_raw_a: list[dict] = []
    r1_raw_b: list[dict] = []

    for i in range(0, len(r1_results), 2):
        a_txt, a_p, a_c, a_lp, a_raw = r1_results[i]
        b_txt, b_p, b_c, b_lp, b_raw = r1_results[i + 1]
        r1_text_a.append(a_txt)
        r1_text_b.append(b_txt)
        r1_prompt_tokens_a.append(a_p)
        r1_prompt_tokens_b.append(b_p)
        r1_tokens_a.append(a_c)
        r1_tokens_b.append(b_c)
        r1_lps_a.append(a_lp)
        r1_lps_b.append(b_lp)
        r1_raw_a.append(a_raw)
        r1_raw_b.append(b_raw)

    # R2 prompts: 2 per debate
    r2_prompts: list[str] = []
    base_r1_prompts: list[str] = [build_r1_prompt(q, config) for q, _ in questions]
    for idx, (q, _gt) in enumerate(questions):
        _ = q
        r2_prompts.append(base_r1_prompts[idx] + r1_text_a[idx] + build_r2_continuation(r1_text_b[idx], config))
        r2_prompts.append(base_r1_prompts[idx] + r1_text_b[idx] + build_r2_continuation(r1_text_a[idx], config))

    r2_results = await client.generate(
        prompts=r2_prompts,
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )
    check_for_failures(r2_results, "R2")

    r2_text_a: list[str] = []
    r2_text_b: list[str] = []
    r2_prompt_tokens_a: list[list[int]] = []
    r2_prompt_tokens_b: list[list[int]] = []
    r2_tokens_a: list[list[int]] = []
    r2_tokens_b: list[list[int]] = []
    r2_lps_a: list[list[float]] = []
    r2_lps_b: list[list[float]] = []
    r2_raw_a: list[dict] = []
    r2_raw_b: list[dict] = []

    for i in range(0, len(r2_results), 2):
        a_txt, a_p, a_c, a_lp, a_raw = r2_results[i]
        b_txt, b_p, b_c, b_lp, b_raw = r2_results[i + 1]
        r2_text_a.append(a_txt)
        r2_text_b.append(b_txt)
        r2_prompt_tokens_a.append(a_p)
        r2_prompt_tokens_b.append(b_p)
        r2_tokens_a.append(a_c)
        r2_tokens_b.append(b_c)
        r2_lps_a.append(a_lp)
        r2_lps_b.append(b_lp)
        r2_raw_a.append(a_raw)
        r2_raw_b.append(b_raw)

    # R3 prompts
    r3_prompts: list[str] = []
    for idx, (q, _gt) in enumerate(questions):
        _ = q
        r2_a_prompt = base_r1_prompts[idx] + r1_text_a[idx] + build_r2_continuation(r1_text_b[idx], config)
        r2_b_prompt = base_r1_prompts[idx] + r1_text_b[idx] + build_r2_continuation(r1_text_a[idx], config)
        r3_prompts.append(r2_a_prompt + r2_text_a[idx] + build_r3_continuation(r2_text_b[idx], config))
        r3_prompts.append(r2_b_prompt + r2_text_b[idx] + build_r3_continuation(r2_text_a[idx], config))

    r3_results = await client.generate(
        prompts=r3_prompts,
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
    )
    check_for_failures(r3_results, "R3")

    r3_text_a: list[str] = []
    r3_text_b: list[str] = []
    r3_prompt_tokens_a: list[list[int]] = []
    r3_prompt_tokens_b: list[list[int]] = []
    r3_tokens_a: list[list[int]] = []
    r3_tokens_b: list[list[int]] = []
    r3_lps_a: list[list[float]] = []
    r3_lps_b: list[list[float]] = []
    r3_raw_a: list[dict] = []
    r3_raw_b: list[dict] = []

    for i in range(0, len(r3_results), 2):
        a_txt, a_p, a_c, a_lp, a_raw = r3_results[i]
        b_txt, b_p, b_c, b_lp, b_raw = r3_results[i + 1]
        r3_text_a.append(a_txt)
        r3_text_b.append(b_txt)
        r3_prompt_tokens_a.append(a_p)
        r3_prompt_tokens_b.append(b_p)
        r3_tokens_a.append(a_c)
        r3_tokens_b.append(b_c)
        r3_lps_a.append(a_lp)
        r3_lps_b.append(b_lp)
        r3_raw_a.append(a_raw)
        r3_raw_b.append(b_raw)

    # Judge
    verdicts: list[Verdict] = []
    reasons: list[str] = []
    judge_metas: list[dict[str, object]] = []
    judge_prompt_tokens_list: list[list[int] | None] = []
    judge_completion_tokens_list: list[list[int] | None] = []
    judge_completion_logprobs_list: list[list[float] | None] = []
    judge_raw_response_list: list[dict | None] = []

    if judge_fn is not None:
        # Not batchable (user function), so do serial
        for idx, (q, _gt) in enumerate(questions):
            v, r = judge_fn(q, r1_text_a[idx], r1_text_b[idx], r2_text_a[idx], r2_text_b[idx], r3_text_a[idx], r3_text_b[idx])
            verdicts.append(v)
            reasons.append(r)
            judge_metas.append({})
            judge_prompt_tokens_list.append(None)
            judge_completion_tokens_list.append(None)
            judge_completion_logprobs_list.append(None)
            judge_raw_response_list.append(None)
    else:
        judge_prompts = [
            build_judge_prompt(
                question=q,
                r1_a=r1_text_a[idx],
                r1_b=r1_text_b[idx],
                r2_a=r2_text_a[idx],
                r2_b=r2_text_b[idx],
                r3_a=r3_text_a[idx],
                r3_b=r3_text_b[idx],
                config=config,
                strict=False,
            )
            for idx, (q, _gt) in enumerate(questions)
        ]

        judge_results = await client.generate(
            prompts=judge_prompts, max_tokens=config.max_tokens_per_turn, temperature=0.3
        )
        # Check judge failures (1 result per debate, not 2)
        for i, r in enumerate(judge_results):
            if r[4].get("error") == "request_failed":
                failed_debate_indices.add(i)
                print(f"!!! Judge result {i} failed -> debate {i} marked INVALID")

        for judge_text, j_prompt_toks, j_comp_toks, j_lps, j_raw in judge_results:
            v = extract_verdict(judge_text)
            reason = extract_reasoning(judge_text)
            verdicts.append(v)
            reasons.append(reason)
            judge_metas.append({})
            judge_prompt_tokens_list.append(j_prompt_toks)
            judge_completion_tokens_list.append(j_comp_toks)
            judge_completion_logprobs_list.append(j_lps)
            judge_raw_response_list.append(j_raw)

    results: list[DebateResult] = []
    for idx, (q, gt) in enumerate(questions):
        sol_a = extract_solution(r1_text_a[idx])
        sol_b = extract_solution(r1_text_b[idx])

        # Override verdict if debate had any failed requests
        final_verdict = verdicts[idx]
        final_reasoning = reasons[idx]
        if idx in failed_debate_indices:
            final_verdict = "INVALID"
            final_reasoning = "[DEBATE FAILED: One or more API requests failed]"
            print(f"\n{'!'*60}")
            print(f"!!! DEBATE {idx} MARKED INVALID DUE TO FAILED REQUESTS")
            print(f"{'!'*60}\n")

        traj_a = DebateTrajectory(
            agent="A",
            transitions=[
                Transition(r1_prompt_tokens_a[idx], r1_tokens_a[idx], r1_lps_a[idx], 1, {"solution": sol_a}, r1_raw_a[idx]),
                Transition(r2_prompt_tokens_a[idx], r2_tokens_a[idx], r2_lps_a[idx], 2, {}, r2_raw_a[idx]),
                Transition(r3_prompt_tokens_a[idx], r3_tokens_a[idx], r3_lps_a[idx], 3, {}, r3_raw_a[idx]),
            ],
            frozen_solution=sol_a,
            metrics={"r1": r1_text_a[idx], "r2": r2_text_a[idx], "r3": r3_text_a[idx]},
        )
        traj_b = DebateTrajectory(
            agent="B",
            transitions=[
                Transition(r1_prompt_tokens_b[idx], r1_tokens_b[idx], r1_lps_b[idx], 1, {"solution": sol_b}, r1_raw_b[idx]),
                Transition(r2_prompt_tokens_b[idx], r2_tokens_b[idx], r2_lps_b[idx], 2, {}, r2_raw_b[idx]),
                Transition(r3_prompt_tokens_b[idx], r3_tokens_b[idx], r3_lps_b[idx], 3, {}, r3_raw_b[idx]),
            ],
            frozen_solution=sol_b,
            metrics={"r1": r1_text_b[idx], "r2": r2_text_b[idx], "r3": r3_text_b[idx]},
        )

        results.append(
            DebateResult(
                question=q,
                ground_truth=gt,
                trajectory_a=traj_a,
                trajectory_b=traj_b,
                verdict=final_verdict,
                judge_reasoning=final_reasoning,
                metrics={
                    "judge_meta": judge_metas[idx],
                    "solution_agreement": sol_a == sol_b if sol_a and sol_b else None,
                    "request_failed": idx in failed_debate_indices,
                },
                judge_prompt_tokens=judge_prompt_tokens_list[idx],
                judge_completion_tokens=judge_completion_tokens_list[idx],
                judge_completion_logprobs=judge_completion_logprobs_list[idx],
                judge_raw_response=judge_raw_response_list[idx],
            )
        )

    return results
