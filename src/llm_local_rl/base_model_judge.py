from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request

from llm_local_rl.debate_parity import Verdict
from llm_local_rl.judge_harness import (
    AgentDebateText,
    JudgeTranscript,
    SOLUTION_R1_RATIONALE_V1,
    extract_tagged_verdict,
    get_judge_harness,
)
from llm_local_rl.model_io_trace import get_model_io_tracer


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
        harness = get_judge_harness(SOLUTION_R1_RATIONALE_V1)
        transcript = JudgeTranscript(
            question=question,
            constitution=constitution,
            agent_a=AgentDebateText(r1=r1_a, r2=r2_a, r3=r3_a),
            agent_b=AgentDebateText(r1=r1_b, r2=r2_b, r3=r3_b),
        )
        rendered = harness.render_checked(transcript=transcript, base_system_text="")
        assert rendered.raw_text is not None
        prompt_text = rendered.raw_text
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
        verdict = harness.parse_verdict(raw_text)
        get_model_io_tracer().record_external_judge(
            url=url,
            prompt_text=prompt_text,
            request_body=request_body,
            raw_text=raw_text,
            verdict=verdict,
        )
        return verdict, raw_text

    return judge
