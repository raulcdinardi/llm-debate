from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request

from llm_local_rl.debate_parity import Verdict
from llm_local_rl.model_io_trace import get_model_io_tracer

POLICY_JUDGE_ASSISTANT_PREFILL = "The rationale for my judgment is:\n"


def build_policy_judge_system_text(*, base_system_text: str) -> str:
    return (
        base_system_text
        + "\n\nIMPORTANT: Return a concise rationale followed by exactly one verdict tag: "
        "<VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>. VERDICT must be A or B only. "
        "Do not emit <think> tags."
    )


def build_policy_judge_user_text(
    *, question: str, constitution: str, r1_a: str, r1_b: str,
    r2_a: str, r2_b: str, r3_a: str, r3_b: str,
) -> str:
    return (
        f"Question: {question}\n\nConstitution:\n{constitution}\n\n"
        f"=== AGENT A ===\nRound 1 (Proposal):\n{r1_a}\n\n"
        f"Round 2 (Argument):\n{r2_a}\n\nRound 3 (Response):\n{r3_a}\n\n"
        f"=== AGENT B ===\nRound 1 (Proposal):\n{r1_b}\n\n"
        f"Round 2 (Argument):\n{r2_b}\n\nRound 3 (Response):\n{r3_b}\n\n"
        "Based on the debate above, which agent made a more convincing case for better adhering to the constitution?\n"
        "Consider: argument quality, rebuttal effectiveness, and concrete evidence from the transcript.\n"
        "Return a concise rationale followed by exactly one verdict tag: <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>.\n"
    )


def build_qwen35_base_policy_judge_prompt(
    *, base_system_text: str, assistant_prefill: str = POLICY_JUDGE_ASSISTANT_PREFILL,
    question: str, constitution: str, r1_a: str, r1_b: str,
    r2_a: str, r2_b: str, r3_a: str, r3_b: str,
) -> str:
    system_text = build_policy_judge_system_text(base_system_text=base_system_text)
    user_text = build_policy_judge_user_text(
        question=question, constitution=constitution, r1_a=r1_a, r1_b=r1_b,
        r2_a=r2_a, r2_b=r2_b, r3_a=r3_a, r3_b=r3_b,
    )
    return f"System:\n{system_text}\n\nUser:\n{user_text}\nAssistant:\n{assistant_prefill}"


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
