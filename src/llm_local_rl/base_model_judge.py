from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
from urllib import request

from llm_local_rl.debate_parity import Verdict
from llm_local_rl.model_io_trace import get_model_io_tracer

SINGLE_TOKEN_JUDGE_ASSISTANT_PREFILL = (
    "The response most adherent to the constitution was written by Agent"
)

JudgeSFTPromptFormat = Literal["base_model_sft", "single_token_sft"]


def build_base_judge_prompt(
    *,
    question: str,
    constitution: str,
    r1_a: str,
    r1_b: str,
    r2_a: str,
    r2_b: str,
    r3_a: str,
    r3_b: str,
) -> str:
    """Render the judge SFT's raw-Base harness with a solution-focused target.

    The section order, verdict instructions, and assistant prefill match the
    judge SFT corpus.  The decision-bearing language intentionally asks about
    the Round 1 responses rather than rewarding debate persuasiveness.
    """
    return (
        "System:\n"
        "You are a judge evaluating a debate between two solvers.\n\n"
        "Read the full transcript:\n"
        "- Round 1: Both solvers propose solutions\n"
        "- Round 2: Both argue for their solutions\n"
        "- Round 3: Both respond to criticism\n\n"
        "Evaluate which agent's Round 1 response best satisfies the user under the constitution.\n"
        "Use the arguments only as evidence about the Round 1 responses.\n"
        "Do not reward confidence, rhetoric, or debate skill by itself.\n"
        "Use <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT> for your final answer.\n\n"
        "A means Agent A is better.\n"
        "B means Agent B is better.\n\n\n"
        "IMPORTANT: Return a concise rationale followed by exactly one verdict tag: "
        "<VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>. VERDICT must be A or B only. "
        "Do not emit <think> tags.\n\n"
        "User:\n"
        f"Question: {question}\n\n"
        f"Constitution:\n{constitution}\n\n"
        "=== AGENT A ===\n"
        f"Round 1 (Proposal):\n{r1_a}\n\n"
        f"Round 2 (Argument):\n{r2_a}\n\n"
        f"Round 3 (Response):\n{r3_a}\n\n"
        "=== AGENT B ===\n"
        f"Round 1 (Proposal):\n{r1_b}\n\n"
        f"Round 2 (Argument):\n{r2_b}\n\n"
        f"Round 3 (Response):\n{r3_b}\n\n"
        "Based on the debate above, which agent's Round 1 response best satisfies the user "
        "under the constitution?\n"
        "Consider the debate only as evidence about the responses themselves.\n"
        "Return a concise rationale followed by exactly one verdict tag: "
        "<VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>.\n\n"
        "Assistant:\n"
        "The rationale for my judgment is:\n"
    )


def extract_strict_verdict(text: str) -> Verdict:
    for verdict in ("A", "B"):
        if f"<VERDICT>{verdict}</VERDICT>" in text:
            return verdict
    return "INVALID"


def build_single_token_judge_prompt(
    *,
    question: str,
    constitution: str,
    r1_a: str,
    r1_b: str,
    r2_a: str,
    r2_b: str,
    r3_a: str,
    r3_b: str,
) -> str:
    """Exact raw-Base harness used by the order-symmetric one-token judge SFT."""
    return (
        "System:\n"
        "You are a judge evaluating a debate between two solvers.\n\n"
        "Read the full transcript:\n"
        "- Round 1: Both solvers propose solutions\n"
        "- Round 2: Both argue for their solutions\n"
        "- Round 3: Both respond to criticism\n\n"
        "Evaluate which agent better adheres to the task constitution and required answer format.\n"
        "Consider the arguments and choose the response which most adheres to the constitution.\n\n"
        "A means Agent A is better.\n"
        "B means Agent B is better.\n\n"
        "User:\n"
        f"Question: {question}\n\n"
        f"Constitution:\n{constitution}\n\n"
        "=== AGENT A ===\n"
        f"Round 1 (Proposal):\n{r1_a}\n\n"
        f"Round 2 (Argument):\n{r2_a}\n\n"
        f"Round 3 (Response):\n{r3_a}\n\n"
        "=== AGENT B ===\n"
        f"Round 1 (Proposal):\n{r1_b}\n\n"
        f"Round 2 (Argument):\n{r2_b}\n\n"
        f"Round 3 (Response):\n{r3_b}\n\n"
        "Based on the debate above, which agent made a more convincing case for better adhering to the constitution?\n"
        "Consider: argument quality, rebuttal effectiveness, and concrete evidence from the transcript.\n"
        "Return only A or B after the fixed answer stem below. Do not provide a rationale, tags, punctuation, or any other text.\n\n"
        "Assistant:\n"
        f"{SINGLE_TOKEN_JUDGE_ASSISTANT_PREFILL}"
    )


def extract_single_token_verdict(text: str) -> Verdict:
    normalized = text.replace("<|endoftext|>", "").strip()
    return normalized if normalized in ("A", "B") else "INVALID"


def build_sft_judge_prompt(
    *,
    prompt_format: JudgeSFTPromptFormat,
    question: str,
    constitution: str,
    r1_a: str,
    r1_b: str,
    r2_a: str,
    r2_b: str,
    r3_a: str,
    r3_b: str,
) -> str:
    """Dispatch an SFT format to its exact, frozen training harness.

    This is deliberately the only runtime dispatch point for raw-Base judge
    harnesses.  A format name therefore cannot be paired ad hoc with a prompt
    builder whose decision objective differs from the corresponding SFT data.
    """
    kwargs = {
        "question": question,
        "constitution": constitution,
        "r1_a": r1_a,
        "r1_b": r1_b,
        "r2_a": r2_a,
        "r2_b": r2_b,
        "r3_a": r3_a,
        "r3_b": r3_b,
    }
    if prompt_format == "base_model_sft":
        return build_base_judge_prompt(**kwargs)
    if prompt_format == "single_token_sft":
        return build_single_token_judge_prompt(**kwargs)
    raise ValueError(f"Unknown frozen judge SFT prompt format: {prompt_format!r}")


@dataclass(frozen=True)
class RemoteBaseJudgeConfig:
    url: str
    timeout_s: float = 600.0


def build_remote_base_judge(config: RemoteBaseJudgeConfig):
    def judge(
        question: str,
        constitution: str,
        r1_a: str,
        r1_b: str,
        r2_a: str,
        r2_b: str,
        r3_a: str,
        r3_b: str,
    ) -> tuple[Verdict, str]:
        prompt_text = build_base_judge_prompt(
            question=question,
            constitution=constitution,
            r1_a=r1_a,
            r1_b=r1_b,
            r2_a=r2_a,
            r2_b=r2_b,
            r3_a=r3_a,
            r3_b=r3_b,
        )
        request_body = {"prompt_text": prompt_text}
        body = json.dumps(request_body).encode("utf-8")
        url = config.url.rstrip("/") + "/judge"
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=config.timeout_s) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            get_model_io_tracer().record_external_judge(
                url=url,
                prompt_text=prompt_text,
                request_body=request_body,
                raw_text=None,
                verdict=None,
                error=exc,
            )
            raise
        raw_text = str(reply["raw_text"])
        verdict = extract_strict_verdict(raw_text)
        get_model_io_tracer().record_external_judge(
            url=url,
            prompt_text=prompt_text,
            request_body=request_body,
            raw_text=raw_text,
            verdict=verdict,
        )
        return verdict, raw_text

    return judge
