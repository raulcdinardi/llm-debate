from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence
from urllib import request

from llm_local_rl.debate_parity import Verdict
from llm_local_rl.model_io_trace import get_model_io_tracer


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
    return (
        "System:\n"
        "You are a judge evaluating a debate between two solvers.\n\n"
        "Read the full transcript:\n"
        "- Round 1: Both solvers propose solutions\n"
        "- Round 2: Both argue for their solutions\n"
        "- Round 3: Both respond to criticism\n\n"
        "Evaluate which agent better adheres to the task constitution and required answer format.\n"
        "Consider the arguments and choose the response which most adheres to the constitution.\n"
        "Use <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT> for your final answer.\n\n"
        "A means Agent A is better.\n"
        "B means Agent B is better.\n\n"
        "User:\n"
        f"Original task prompt:\n{question}\n\n"
        f"Constitution:\n{constitution}\n\n"
        "=== AGENT A ===\n"
        f"Round 1 (Proposal):\n{r1_a}\n\n"
        f"Round 2 (Argument):\n{r2_a}\n\n"
        f"Round 3 (Response):\n{r3_a}\n\n"
        "=== AGENT B ===\n"
        f"Round 1 (Proposal):\n{r1_b}\n\n"
        f"Round 2 (Argument):\n{r2_b}\n\n"
        f"Round 3 (Response):\n{r3_b}\n\n"
        "Based on the debate above, which agent made a more convincing case?\n"
        "Consider the arguments and choose the response which most adheres to the constitution.\n"
        "Use <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT> for your final answer.\n\n"
        "Assistant:\n"
        "Ok, I will list 3 short considerations then immediately emit my verdict inside <VERDICT> tags:\n"
        "1)"
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
    class RemoteBaseJudge:
        def _post(self, *, request_body: dict) -> dict:
            body = json.dumps(request_body).encode("utf-8")
            url = config.url.rstrip("/") + "/judge"
            req = request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=config.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def __call__(
            self,
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
            url = config.url.rstrip("/") + "/judge"
            try:
                reply = self._post(request_body=request_body)
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

        def judge_many(
            self,
            debates: Sequence[tuple[str, str, str, str, str, str, str, str]],
        ) -> list[tuple[Verdict, str]]:
            prompt_texts = [
                build_base_judge_prompt(
                    question=question,
                    constitution=constitution,
                    r1_a=r1_a,
                    r1_b=r1_b,
                    r2_a=r2_a,
                    r2_b=r2_b,
                    r3_a=r3_a,
                    r3_b=r3_b,
                )
                for question, constitution, r1_a, r1_b, r2_a, r2_b, r3_a, r3_b in debates
            ]
            request_body = {"prompt_texts": prompt_texts}
            url = config.url.rstrip("/") + "/judge"
            try:
                reply = self._post(request_body=request_body)
                results = reply["results"]
                if not isinstance(results, list) or len(results) != len(prompt_texts):
                    raise ValueError(
                        f"External judge returned {len(results) if isinstance(results, list) else 'non-list'} "
                        f"results for {len(prompt_texts)} prompts"
                    )
            except Exception as exc:
                for prompt_text in prompt_texts:
                    get_model_io_tracer().record_external_judge(
                        url=url,
                        prompt_text=prompt_text,
                        request_body=request_body,
                        raw_text=None,
                        verdict=None,
                        error=exc,
                    )
                raise

            judged: list[tuple[Verdict, str]] = []
            for prompt_text, result in zip(prompt_texts, results, strict=True):
                raw_text = str(result["raw_text"])
                verdict = extract_strict_verdict(raw_text)
                get_model_io_tracer().record_external_judge(
                    url=url,
                    prompt_text=prompt_text,
                    request_body=request_body,
                    raw_text=raw_text,
                    verdict=verdict,
                )
                judged.append((verdict, raw_text))
            return judged

    return RemoteBaseJudge()
