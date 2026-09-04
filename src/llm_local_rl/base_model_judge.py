from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request

from llm_local_rl.debate_parity import Verdict
from llm_local_rl.judge_harness import (
    AgentDebateText,
    JudgeTranscript,
    SOLUTION_R1_RATIONALE_V1,
    get_judge_harness,
    harness_fingerprint,
)
from llm_local_rl.model_io_trace import get_model_io_tracer


@dataclass(frozen=True)
class RemoteBaseJudgeConfig:
    url: str
    harness_id: str
    timeout_s: float = 600.0
    max_rounds: int = 3

    def __post_init__(self) -> None:
        harness = get_judge_harness(self.harness_id)
        if harness.harness_id != SOLUTION_R1_RATIONALE_V1:
            raise ValueError(
                "External HTTP judge supports only "
                f"{SOLUTION_R1_RATIONALE_V1!r}; got {harness.harness_id!r}"
            )
        if self.max_rounds < harness.required_rounds:
            raise ValueError(
                f"External judge harness requires at least {harness.required_rounds} rounds"
            )


def build_remote_base_judge(config: RemoteBaseJudgeConfig):
    harness = get_judge_harness(config.harness_id)
    fingerprint = harness_fingerprint(
        harness.harness_id,
        max_rounds=config.max_rounds,
    )

    def judge(
        question: str,
        constitution: str,
        *interleaved_round_texts: str,
    ) -> tuple[Verdict, str]:
        if len(interleaved_round_texts) % 2 != 0 or not interleaved_round_texts:
            raise ValueError("Judge requires one A/B text pair per debate round")
        actual_rounds = len(interleaved_round_texts) // 2
        if actual_rounds > config.max_rounds:
            raise ValueError(
                f"Judge received {actual_rounds} rounds above configured maximum "
                f"{config.max_rounds}"
            )
        transcript = JudgeTranscript(
            question=question,
            constitution=constitution,
            agent_a=AgentDebateText(rounds=interleaved_round_texts[0::2]),
            agent_b=AgentDebateText(rounds=interleaved_round_texts[1::2]),
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
                harness_id=harness.harness_id,
                harness_fingerprint=fingerprint,
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
            harness_id=harness.harness_id,
            harness_fingerprint=fingerprint,
        )
        return verdict, raw_text

    return judge
