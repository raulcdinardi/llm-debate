from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Literal

from llm_local_rl.debate_parity import Verdict


CHAT_SOLUTION_TAGGED_V1 = "chat_solution_tagged_v1"
CHAT_POINTWISE_TAGGED_V1 = "chat_pointwise_tagged_v1"
SOLUTION_R1_RATIONALE_V1 = "solution_r1_rationale_v1"
CONSTITUTION_SINGLE_TOKEN_V1 = "constitution_single_token_v1"
CONSULTANCY_SINGLE_TOKEN_V1 = "consultancy_single_token_v1"
PAIRWISE_SINGLE_TOKEN_V1 = "pairwise_single_token_v1"
JUDGE_HARNESS_MANIFEST = "judge_harness.json"
JUDGE_HARNESS_MANIFEST_SCHEMA = "llm_local_rl_judge_harness_v1"

JudgeHarnessId = Literal[
    "chat_solution_tagged_v1",
    "chat_pointwise_tagged_v1",
    "solution_r1_rationale_v1",
    "constitution_single_token_v1",
    "consultancy_single_token_v1",
    "pairwise_single_token_v1",
]
PromptSerialization = Literal["chat", "raw_base"]

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAIL_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, init=False)
class AgentDebateText:
    rounds: tuple[str, ...]

    def __init__(
        self,
        r1: str = "",
        r2: str = "",
        r3: str = "",
        r4: str = "",
        *,
        rounds: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        values = (
            tuple(str(value) for value in rounds)
            if rounds is not None
            else (r1, r2, r3, *((r4,) if r4 else ()))
        )
        if not values:
            raise ValueError("Agent debate transcript must contain at least one round")
        object.__setattr__(self, "rounds", values)

    @property
    def r1(self) -> str:
        return self.rounds[0] if self.rounds else ""

    @property
    def r2(self) -> str:
        return self.rounds[1] if len(self.rounds) > 1 else ""

    @property
    def r3(self) -> str:
        return self.rounds[2] if len(self.rounds) > 2 else ""

    @property
    def r4(self) -> str:
        return self.rounds[3] if len(self.rounds) > 3 else ""


@dataclass(frozen=True)
class JudgeTranscript:
    question: str
    constitution: str
    agent_a: AgentDebateText
    agent_b: AgentDebateText

    def swapped(self) -> "JudgeTranscript":
        return JudgeTranscript(
            question=self.question,
            constitution=self.constitution,
            agent_a=self.agent_b,
            agent_b=self.agent_a,
        )


@dataclass(frozen=True)
class RenderedJudgePrompt:
    raw_text: str | None = None
    messages: tuple[dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (self.raw_text is None) == (not self.messages):
            raise ValueError("Rendered judge prompt must contain exactly one representation")


@dataclass(frozen=True)
class JudgeHarnessSpec:
    harness_id: JudgeHarnessId
    serialization: PromptSerialization
    objective: str
    output_contract: str
    assistant_prefill: str
    default_max_tokens: int | None
    required_rounds: int
    render: Callable[[JudgeTranscript, str], RenderedJudgePrompt]
    parse_verdict: Callable[[str], Verdict]
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()

    def render_checked(
        self, *, transcript: JudgeTranscript, base_system_text: str
    ) -> RenderedJudgePrompt:
        # Validate only harness-authored template text. Real questions,
        # constitutions, and model outputs are untrusted payload: searching the
        # completed prompt would let data both trigger forbidden phrases and
        # satisfy required phrases accidentally.
        contract_rendered = self.render(
            _contract_probe_transcript(required_rounds=self.required_rounds),
            base_system_text,
        )
        contract_text = contract_rendered.raw_text or "\n".join(
            message["content"] for message in contract_rendered.messages
        )
        missing = [phrase for phrase in self.required_phrases if phrase not in contract_text]
        forbidden = [phrase for phrase in self.forbidden_phrases if phrase in contract_text]
        if missing or forbidden:
            raise ValueError(
                f"Judge harness {self.harness_id!r} violated its semantic contract: "
                f"missing={missing!r}, forbidden={forbidden!r}"
            )
        return self.render(transcript, base_system_text)


def _contract_probe_transcript(*, required_rounds: int = 3) -> JudgeTranscript:
    """Return inert payload used only to inspect trusted harness instructions."""
    return JudgeTranscript(
        question="__JUDGE_HARNESS_QUESTION_PAYLOAD__",
        constitution="__JUDGE_HARNESS_CONSTITUTION_PAYLOAD__",
        agent_a=AgentDebateText(rounds=tuple(
            f"__JUDGE_HARNESS_A_R{round_num}_PAYLOAD__"
            for round_num in range(1, required_rounds + 1)
        )),
        agent_b=AgentDebateText(rounds=tuple(
            f"__JUDGE_HARNESS_B_R{round_num}_PAYLOAD__"
            for round_num in range(1, required_rounds + 1)
        )),
    )


def _round_summary(transcript: JudgeTranscript) -> str:
    count = max(len(transcript.agent_a.rounds), len(transcript.agent_b.rounds))
    lines = []
    for round_num in range(1, count + 1):
        if round_num == 1:
            lines.append("- Round 1: Both solvers propose solutions")
        elif round_num == 2:
            lines.append("- Round 2: Both argue for their solutions")
        else:
            lines.append(f"- Round {round_num}: Both respond to criticism")
    return "\n".join(lines) + "\n"


def extract_tagged_verdict(text: str) -> Verdict:
    cleaned = _THINK_TAIL_RE.sub("", _THINK_BLOCK_RE.sub("", text))
    for verdict in ("A", "B"):
        if f"<VERDICT>{verdict}</VERDICT>" in cleaned:
            return verdict
    return "INVALID"


def extract_single_token_verdict(text: str) -> Verdict:
    normalized = text.replace("<|endoftext|>", "").strip()
    return normalized if normalized in ("A", "B") else "INVALID"


def _transcript_body(transcript: JudgeTranscript) -> str:
    def agent_body(name: str, agent: AgentDebateText) -> str:
        labels = {1: "Proposal", 2: "Argument"}
        return f"=== AGENT {name} ===\n" + "".join(
            f"Round {round_num} ({labels.get(round_num, 'Response')}):\n{text}\n\n"
            for round_num, text in enumerate(agent.rounds, start=1)
        )

    return (
        f"Question: {transcript.question}\n\n"
        f"Constitution:\n{transcript.constitution}\n\n"
        + agent_body("A", transcript.agent_a)
        + agent_body("B", transcript.agent_b)
    )


def _render_solution_r1_rationale(
    transcript: JudgeTranscript, _base_system_text: str
) -> RenderedJudgePrompt:
    return RenderedJudgePrompt(
        raw_text=(
            "System:\n"
            "You are a judge evaluating a debate between two solvers.\n\n"
            "Read the full transcript:\n"
            + _round_summary(transcript)
            + "\n"
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
            + _transcript_body(transcript)
            + "Based on the debate above, which agent's Round 1 response best satisfies the user "
            "under the constitution?\n"
            "Consider the debate only as evidence about the responses themselves.\n"
            "Return a concise rationale followed by exactly one verdict tag: "
            "<VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>.\n\n"
            "Assistant:\n"
            "The rationale for my judgment is:\n"
        )
    )


def _render_constitution_single_token(
    transcript: JudgeTranscript, _base_system_text: str
) -> RenderedJudgePrompt:
    return RenderedJudgePrompt(
        raw_text=(
            "System:\n"
            "You are a judge evaluating a debate between two solvers.\n\n"
            "Read the full transcript:\n"
            + _round_summary(transcript)
            + "\n"
            "Evaluate which agent better adheres to the task constitution and required answer format.\n"
            "Consider the arguments and choose the response which most adheres to the constitution.\n\n"
            "A means Agent A is better.\n"
            "B means Agent B is better.\n\n"
            "User:\n"
            + _transcript_body(transcript)
            + "Based on the debate above, which agent made a more convincing case for better adhering to the constitution?\n"
            "Consider: argument quality, rebuttal effectiveness, and concrete evidence from the transcript.\n"
            "Return only A or B after the fixed answer stem below. Do not provide a rationale, tags, punctuation, or any other text.\n\n"
            "Assistant:\n"
            "The response most adherent to the constitution was written by Agent"
        )
    )


def _render_consultancy_single_token(
    transcript: JudgeTranscript, _base_system_text: str
) -> RenderedJudgePrompt:
    def agent_block(name: str, agent: AgentDebateText) -> str:
        return (
            f"=== AGENT {name} ===\n"
            f"Round 1 (Proposal):\n{agent.r1}\n\n"
            f"Round 2 (Argument):\n{agent.r2}"
        )

    return RenderedJudgePrompt(
        raw_text=(
            "System:\n"
            "You are a judge comparing two candidate responses and their supporting arguments.\n\n"
            "Read the available transcript:\n"
            "- Round 1: Both agents propose responses\n"
            "- Round 2: Both agents present supporting arguments\n\n"
            "Evaluate which agent better adheres to the task constitution and required answer format.\n"
            "Consider each proposal and its supporting argument, then choose the response which most adheres to the constitution.\n\n"
            "A means Agent A is better.\n"
            "B means Agent B is better.\n\n"
            "User:\nQuestion: " + transcript.question + "\n\n"
            "Constitution:\n" + transcript.constitution + "\n\n"
            + agent_block("A", transcript.agent_a) + "\n\n"
            + agent_block("B", transcript.agent_b) + "\n\n"
            "Considering each proposal and its supporting argument, which agent better adheres to the constitution?\n"
            "Consider: task compliance, proposal quality, supporting-argument quality, and concrete evidence from the transcript.\n"
            "Return only A or B after the fixed answer stem below. Do not provide a rationale, tags, punctuation, or any other text.\n\n"
            "Assistant:\n"
            "The response most adherent to the constitution was written by Agent"
        )
    )


def _render_pairwise_single_token(
    transcript: JudgeTranscript, _base_system_text: str
) -> RenderedJudgePrompt:
    return RenderedJudgePrompt(
        raw_text=(
            "System:\n"
            "You are a judge comparing two candidate responses.\n\n"
            "Read both responses and evaluate which agent better adheres to the task constitution and required answer format.\n"
            "Choose the response which most adheres to the constitution.\n\n"
            "A means Agent A is better.\n"
            "B means Agent B is better.\n\n"
            "User:\nQuestion: " + transcript.question + "\n\n"
            "Constitution:\n" + transcript.constitution + "\n\n"
            "=== AGENT A ===\nResponse:\n" + transcript.agent_a.r1 + "\n\n"
            "=== AGENT B ===\nResponse:\n" + transcript.agent_b.r1 + "\n\n"
            "Based only on the two responses above, which agent better adheres to the constitution?\n"
            "Consider: task compliance, answer quality, and concrete details in each response.\n"
            "Return only A or B after the fixed answer stem below. Do not provide a rationale, tags, punctuation, or any other text.\n\n"
            "Assistant:\n"
            "The response most adherent to the constitution was written by Agent"
        )
    )


def _render_chat_solution(
    transcript: JudgeTranscript, base_system_text: str
) -> RenderedJudgePrompt:
    if len(transcript.agent_a.rounds) > 3 or len(transcript.agent_b.rounds) > 3:
        r3_line = "- Round 3: Both respond to criticism\n"
        later_lines = "".join(
            f"- Round {round_num}: Both respond to criticism\n"
            for round_num in range(
                4,
                max(len(transcript.agent_a.rounds), len(transcript.agent_b.rounds)) + 1,
            )
        )
        base_system_text = base_system_text.replace(r3_line, r3_line + later_lines, 1)
    system = (
        base_system_text
        + "\n\nIMPORTANT: Output exactly one tag: <VERDICT>...</VERDICT>. "
        "VERDICT must be A or B only. Do not emit <think> tags or any reasoning."
    )
    user = (
        _transcript_body(transcript)
        + "Based on the debate above, which agent's Round 1 response best satisfies the user "
        "under the constitution?\n"
        "Consider the debate only as evidence about the responses themselves.\n"
        "Return only <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>.\n"
    )
    return RenderedJudgePrompt(
        messages=(
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        )
    )


def _render_chat_pointwise(
    transcript: JudgeTranscript, _base_system_text: str
) -> RenderedJudgePrompt:
    system = (
        "You are a judge comparing two candidate answers to the same task.\n\n"
        "IMPORTANT: Output exactly one tag: <VERDICT>...</VERDICT>. "
        "Judge task compliance first. VERDICT must be A or B only."
    )
    user = (
        f"Task context:\n{transcript.question}\n\n"
        f"Constitution:\n{transcript.constitution}\n\n"
        "=== AGENT A ===\n"
        f"Answer:\n{transcript.agent_a.r1}\n\n"
        "=== AGENT B ===\n"
        f"Answer:\n{transcript.agent_b.r1}\n\n"
        "Compare the two answers only.\n"
        "Which agent gave the better answer?\n"
        "Return only <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>.\n"
    )
    return RenderedJudgePrompt(
        messages=(
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        )
    )


_HARNESSES: dict[JudgeHarnessId, JudgeHarnessSpec] = {
    CHAT_SOLUTION_TAGGED_V1: JudgeHarnessSpec(
        harness_id=CHAT_SOLUTION_TAGGED_V1,
        serialization="chat",
        objective="select_best_round1_solution",
        output_contract="verdict_tag_only",
        assistant_prefill="",
        default_max_tokens=None,
        required_rounds=3,
        render=_render_chat_solution,
        parse_verdict=extract_tagged_verdict,
        required_phrases=("Round 1 response best satisfies the user",),
        forbidden_phrases=("more convincing case", "rebuttal effectiveness"),
    ),
    CHAT_POINTWISE_TAGGED_V1: JudgeHarnessSpec(
        harness_id=CHAT_POINTWISE_TAGGED_V1,
        serialization="chat",
        objective="select_best_pointwise_answer",
        output_contract="verdict_tag_only",
        assistant_prefill="",
        default_max_tokens=None,
        required_rounds=1,
        render=_render_chat_pointwise,
        parse_verdict=extract_tagged_verdict,
        required_phrases=("Compare the two answers only.",),
        forbidden_phrases=("Round 2", "more convincing case"),
    ),
    SOLUTION_R1_RATIONALE_V1: JudgeHarnessSpec(
        harness_id=SOLUTION_R1_RATIONALE_V1,
        serialization="raw_base",
        objective="select_best_round1_solution",
        output_contract="rationale_then_verdict_tag",
        assistant_prefill="The rationale for my judgment is:\n",
        default_max_tokens=512,
        required_rounds=3,
        render=_render_solution_r1_rationale,
        parse_verdict=extract_tagged_verdict,
        required_phrases=(
            "Round 1 response best satisfies the user",
            "Do not reward confidence, rhetoric, or debate skill by itself.",
        ),
        forbidden_phrases=("more convincing case", "rebuttal effectiveness"),
    ),
    CONSTITUTION_SINGLE_TOKEN_V1: JudgeHarnessSpec(
        harness_id=CONSTITUTION_SINGLE_TOKEN_V1,
        serialization="raw_base",
        objective="select_best_constitution_adherence_case",
        output_contract="single_token_a_or_b",
        assistant_prefill="The response most adherent to the constitution was written by Agent",
        default_max_tokens=1,
        required_rounds=3,
        render=_render_constitution_single_token,
        parse_verdict=extract_single_token_verdict,
        required_phrases=("more convincing case", "rebuttal effectiveness"),
    ),
    CONSULTANCY_SINGLE_TOKEN_V1: JudgeHarnessSpec(
        harness_id=CONSULTANCY_SINGLE_TOKEN_V1,
        serialization="raw_base",
        objective="select_best_r1_r2_consultancy_view",
        output_contract="single_token_a_or_b",
        assistant_prefill="The response most adherent to the constitution was written by Agent",
        default_max_tokens=1,
        required_rounds=2,
        render=_render_consultancy_single_token,
        parse_verdict=extract_single_token_verdict,
        required_phrases=("supporting argument",),
        forbidden_phrases=("Round 3", "rebuttal effectiveness"),
    ),
    PAIRWISE_SINGLE_TOKEN_V1: JudgeHarnessSpec(
        harness_id=PAIRWISE_SINGLE_TOKEN_V1,
        serialization="raw_base",
        objective="select_best_r1_pairwise_view",
        output_contract="single_token_a_or_b",
        assistant_prefill="The response most adherent to the constitution was written by Agent",
        default_max_tokens=1,
        required_rounds=1,
        render=_render_pairwise_single_token,
        parse_verdict=extract_single_token_verdict,
        required_phrases=("Based only on the two responses above",),
        forbidden_phrases=("Round 2", "Round 3", "rebuttal effectiveness"),
    ),
}

LEGACY_PROMPT_FORMAT_TO_HARNESS: dict[str, JudgeHarnessId] = {
    "chat": CHAT_SOLUTION_TAGGED_V1,
    "base_model_sft": SOLUTION_R1_RATIONALE_V1,
    "single_token_sft": CONSTITUTION_SINGLE_TOKEN_V1,
}


def judge_harness_ids() -> tuple[str, ...]:
    return tuple(_HARNESSES)


def get_judge_harness(harness_id: str) -> JudgeHarnessSpec:
    try:
        return _HARNESSES[harness_id]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(
            f"Unknown judge harness {harness_id!r}; expected one of {judge_harness_ids()!r}"
        ) from exc


def resolve_judge_harness_id(
    *,
    harness_id: str | None,
    legacy_prompt_format: str | None = None,
    num_rounds: int = 3,
) -> str:
    if harness_id is not None and legacy_prompt_format is not None:
        raise ValueError(
            "--debate-judge-harness and legacy --debate-judge-prompt-format "
            "are mutually exclusive"
        )
    if harness_id is not None:
        get_judge_harness(harness_id)
        return harness_id
    legacy = legacy_prompt_format or "chat"
    if legacy == "chat" and num_rounds == 1:
        return CHAT_POINTWISE_TAGGED_V1
    try:
        return LEGACY_PROMPT_FORMAT_TO_HARNESS[legacy]
    except KeyError as exc:
        raise ValueError(f"Unknown legacy judge prompt format: {legacy!r}") from exc


def harness_fingerprint(harness_id: str) -> str:
    spec = get_judge_harness(harness_id)
    sentinel = JudgeTranscript(
        question="__QUESTION__",
        constitution="__CONSTITUTION__",
        agent_a=AgentDebateText("__A_R1__", "__A_R2__", "__A_R3__"),
        agent_b=AgentDebateText("__B_R1__", "__B_R2__", "__B_R3__"),
    )
    rendered = spec.render_checked(transcript=sentinel, base_system_text="__SYSTEM__")
    payload = {
        "harness_id": spec.harness_id,
        "serialization": spec.serialization,
        "objective": spec.objective,
        "output_contract": spec.output_contract,
        "assistant_prefill": spec.assistant_prefill,
        "default_max_tokens": spec.default_max_tokens,
        "required_rounds": spec.required_rounds,
        "raw_text": rendered.raw_text,
        "messages": rendered.messages,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_judge_harness_manifest(*, adapter_dir: str | Path, harness_id: str) -> Path:
    spec = get_judge_harness(harness_id)
    path = Path(adapter_dir) / JUDGE_HARNESS_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": JUDGE_HARNESS_MANIFEST_SCHEMA,
        "harness_id": spec.harness_id,
        "harness_fingerprint": harness_fingerprint(spec.harness_id),
        "objective": spec.objective,
        "output_contract": spec.output_contract,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_judge_harness_manifest(*, adapter_dir: str | Path, harness_id: str) -> dict:
    expected = get_judge_harness(harness_id)
    path = Path(adapter_dir) / JUDGE_HARNESS_MANIFEST
    if not path.is_file():
        raise ValueError(
            f"Judge adapter {str(adapter_dir)!r} has no {JUDGE_HARNESS_MANIFEST}; "
            "bind the adapter to its training harness before use"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fingerprint = harness_fingerprint(expected.harness_id)
    if payload.get("schema") != JUDGE_HARNESS_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported judge harness manifest schema in {path}")
    if payload.get("harness_id") != expected.harness_id:
        raise ValueError(
            f"Judge adapter harness mismatch: adapter={payload.get('harness_id')!r}, "
            f"requested={expected.harness_id!r}"
        )
    if payload.get("harness_fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Judge adapter harness fingerprint mismatch for {expected.harness_id!r}"
        )
    return payload
