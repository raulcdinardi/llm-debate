from __future__ import annotations

from dataclasses import replace
import json

import pytest

from llm_local_rl.judge_harness import (
    AgentDebateText,
    CHAT_POINTWISE_TAGGED_V1,
    CHAT_SOLUTION_TAGGED_V1,
    CONSTITUTION_SINGLE_TOKEN_V1,
    JUDGE_HARNESS_MANIFEST,
    JudgeTranscript,
    SOLUTION_R1_RATIONALE_V1,
    extract_single_token_verdict,
    extract_tagged_verdict,
    get_judge_harness,
    harness_fingerprint,
    resolve_judge_harness_id,
    validate_judge_harness_manifest,
    write_judge_harness_manifest,
)
from llm_local_rl.lora_identity import adapter_identity


def _transcript() -> JudgeTranscript:
    return JudgeTranscript(
        question="Question text",
        constitution="Follow the rules",
        agent_a=AgentDebateText(r1="A1", r2="A2", r3="A3"),
        agent_b=AgentDebateText(r1="B1", r2="B2", r3="B3"),
    )


def test_tagged_verdict_parser_requires_exact_tag() -> None:
    assert extract_tagged_verdict("<VERDICT>A</VERDICT>") == "A"
    assert extract_tagged_verdict("prefix <VERDICT>B</VERDICT> suffix") == "B"
    assert extract_tagged_verdict("A") == "INVALID"


def test_solution_harness_is_canonical_and_solution_focused() -> None:
    harness = get_judge_harness(SOLUTION_R1_RATIONALE_V1)
    rendered = harness.render_checked(transcript=_transcript(), base_system_text="ignored")
    assert rendered.raw_text is not None
    prompt = rendered.raw_text
    assert "Question text" in prompt
    assert "Follow the rules" in prompt
    assert "Round 1 (Proposal):\nA1" in prompt
    assert "Round 2 (Argument):\nB2" in prompt
    assert "Round 3 (Response):\nA3" in prompt
    assert "which agent's Round 1 response best satisfies the user" in prompt
    assert "Do not reward confidence, rhetoric, or debate skill by itself." in prompt
    assert prompt.endswith(harness.assistant_prefill)
    assert "more convincing case" not in prompt
    assert "argument quality" not in prompt
    assert "rebuttal effectiveness" not in prompt


@pytest.mark.parametrize(
    "harness_id",
    [CHAT_SOLUTION_TAGGED_V1, CHAT_POINTWISE_TAGGED_V1, SOLUTION_R1_RATIONALE_V1],
)
def test_semantic_validation_never_inspects_transcript_payload(harness_id: str) -> None:
    adversarial = JudgeTranscript(
        question="Round 2 asks who made a more convincing case.",
        constitution="Reward rebuttal effectiveness.",
        agent_a=AgentDebateText(
            r1="The other answer made a more convincing case.",
            r2="Discuss rebuttal effectiveness.",
            r3="Round 2",
        ),
        agent_b=AgentDebateText(r1="B1", r2="B2", r3="B3"),
    )

    rendered = get_judge_harness(harness_id).render_checked(
        transcript=adversarial,
        base_system_text="Judge the solutions.",
    )

    searchable = rendered.raw_text or "\n".join(
        message["content"] for message in rendered.messages
    )
    assert "more convincing case" in searchable


def test_transcript_payload_cannot_satisfy_a_missing_required_instruction() -> None:
    harness = replace(
        get_judge_harness(CHAT_POINTWISE_TAGGED_V1),
        required_phrases=("PAYLOAD_ONLY_REQUIRED_PHRASE",),
    )
    transcript = replace(_transcript(), question="PAYLOAD_ONLY_REQUIRED_PHRASE")

    with pytest.raises(ValueError, match="missing=.*PAYLOAD_ONLY_REQUIRED_PHRASE"):
        harness.render_checked(transcript=transcript, base_system_text="Judge the answers.")


def test_structured_transcript_swap_is_complete_and_involutive() -> None:
    transcript = _transcript()
    swapped = transcript.swapped()
    assert swapped.agent_a == transcript.agent_b
    assert swapped.agent_b == transcript.agent_a
    assert swapped.swapped() == transcript


def test_new_and_legacy_harness_flags_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_judge_harness_id(
            harness_id=SOLUTION_R1_RATIONALE_V1,
            legacy_prompt_format="base_model_sft",
        )


def test_versioned_harness_fingerprints_are_golden() -> None:
    assert harness_fingerprint(CHAT_SOLUTION_TAGGED_V1) == (
        "ddf47bbc791349aae04e6adf081cc9764a39e6cf225f24ebcf12c4fd4b26156f"
    )
    assert harness_fingerprint(CHAT_POINTWISE_TAGGED_V1) == (
        "f95725a66460a20617f3feea8fb8f333d83d03dc64c8a7da68d561aebfdb66d8"
    )
    assert harness_fingerprint(SOLUTION_R1_RATIONALE_V1) == (
        "28afa8006098fddeec5630680651bc3c1db30608e45463da3f91fc18d2dab67e"
    )
    assert harness_fingerprint(CONSTITUTION_SINGLE_TOKEN_V1) == (
        "2b8f61287b142bf6c82251d146b470b017cab6ea321fba8421e20490bdada930"
    )
    assert harness_fingerprint(CHAT_SOLUTION_TAGGED_V1, max_rounds=4) == (
        "a29bee8de7864beef424d998d712eca7099c41db66e49acad5eabc36f446fbe9"
    )
    assert harness_fingerprint(SOLUTION_R1_RATIONALE_V1, max_rounds=4) == (
        "ce0b8715b6276eed47c5d369f99d2f2aefc49d4437eb1be194bac1cb5c56c6c7"
    )
    assert harness_fingerprint(CONSTITUTION_SINGLE_TOKEN_V1, max_rounds=4) == (
        "834a868857ef85924906ec593fa939a659ef9934053f9488aeeb023f7aded03b"
    )


def test_extended_harness_fingerprint_binds_exact_maximum_depth() -> None:
    legacy = harness_fingerprint(CHAT_SOLUTION_TAGGED_V1)
    round_four = harness_fingerprint(CHAT_SOLUTION_TAGGED_V1, max_rounds=4)
    round_six = harness_fingerprint(CHAT_SOLUTION_TAGGED_V1, max_rounds=6)

    assert len({legacy, round_four, round_six}) == 3
    with pytest.raises(ValueError, match="requires at least 3 rounds"):
        harness_fingerprint(CHAT_SOLUTION_TAGGED_V1, max_rounds=2)


def test_single_token_harness_owns_its_parser_and_output_budget() -> None:
    harness = get_judge_harness(CONSTITUTION_SINGLE_TOKEN_V1)
    rendered = harness.render_checked(transcript=_transcript(), base_system_text="ignored")
    assert rendered.raw_text is not None
    assert rendered.raw_text.endswith(harness.assistant_prefill)
    assert harness.default_max_tokens == 1
    assert harness.parse_verdict(" A") == "A"
    assert extract_single_token_verdict("B<|endoftext|>") == "B"
    assert harness.parse_verdict("A because") == "INVALID"


def test_unknown_harness_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown judge harness"):
        get_judge_harness("persuasion_harness")


def test_adapter_manifest_binds_exact_harness_and_fingerprint(tmp_path) -> None:
    adapter = tmp_path / "judge"
    adapter.mkdir()
    write_judge_harness_manifest(
        adapter_dir=adapter,
        harness_id=SOLUTION_R1_RATIONALE_V1,
    )

    payload = validate_judge_harness_manifest(
        adapter_dir=adapter,
        harness_id=SOLUTION_R1_RATIONALE_V1,
    )
    assert payload["objective"] == "select_best_round1_solution"
    with pytest.raises(ValueError, match="harness mismatch"):
        validate_judge_harness_manifest(
            adapter_dir=adapter,
            harness_id=CONSTITUTION_SINGLE_TOKEN_V1,
        )

    manifest_path = adapter / JUDGE_HARNESS_MANIFEST
    tampered = json.loads(manifest_path.read_text())
    tampered["harness_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_judge_harness_manifest(
            adapter_dir=adapter,
            harness_id=SOLUTION_R1_RATIONALE_V1,
        )


def test_extended_adapter_manifest_binds_debate_depth_contract(tmp_path) -> None:
    adapter = tmp_path / "judge"
    adapter.mkdir()
    write_judge_harness_manifest(
        adapter_dir=adapter,
        harness_id=SOLUTION_R1_RATIONALE_V1,
        max_rounds=4,
    )

    payload = validate_judge_harness_manifest(
        adapter_dir=adapter,
        harness_id=SOLUTION_R1_RATIONALE_V1,
        max_rounds=4,
    )
    assert payload["max_rounds"] == 4
    assert payload["harness_fingerprint"] == harness_fingerprint(
        SOLUTION_R1_RATIONALE_V1,
        max_rounds=4,
    )
    with pytest.raises(ValueError, match="debate-depth contract mismatch"):
        validate_judge_harness_manifest(
            adapter_dir=adapter,
            harness_id=SOLUTION_R1_RATIONALE_V1,
            max_rounds=5,
        )


def test_pre_depth_manifest_remains_valid_for_legacy_three_round_contract(tmp_path) -> None:
    adapter = tmp_path / "judge"
    adapter.mkdir()
    manifest_path = write_judge_harness_manifest(
        adapter_dir=adapter,
        harness_id=SOLUTION_R1_RATIONALE_V1,
    )
    payload = json.loads(manifest_path.read_text())
    payload.pop("max_rounds")
    manifest_path.write_text(json.dumps(payload))

    validate_judge_harness_manifest(
        adapter_dir=adapter,
        harness_id=SOLUTION_R1_RATIONALE_V1,
        max_rounds=3,
    )


def test_adapter_manifest_is_required(tmp_path) -> None:
    with pytest.raises(ValueError, match="has no judge_harness.json"):
        validate_judge_harness_manifest(
            adapter_dir=tmp_path,
            harness_id=SOLUTION_R1_RATIONALE_V1,
        )


def test_adapter_identity_tracks_harness_manifest(tmp_path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    manifest = write_judge_harness_manifest(
        adapter_dir=tmp_path,
        harness_id=SOLUTION_R1_RATIONALE_V1,
    )
    before = adapter_identity(str(tmp_path))

    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    after = adapter_identity(str(tmp_path))

    assert before != after
