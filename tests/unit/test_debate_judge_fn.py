from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from llm_local_rl.behavior_policy import BEHAVIOR_POLICY_LOGPROBS, BehaviorPolicySpec
from llm_local_rl.debate_parity import DebateConfig
from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
from llm_local_rl.debate_tasks import HTSequenceDebateTask
from llm_local_rl.judge_harness import (
    CONSTITUTION_SINGLE_TOKEN_V1,
    PAIRWISE_SINGLE_TOKEN_V1,
    SOLUTION_R1_RATIONALE_V1,
)
from llm_local_rl.types import SamplingRequest, SamplingResult


class TinyChatTokenizer:
    all_special_tokens = ["<|im_end|>"]
    additional_special_tokens: list[str] = []

    _SPECIAL_TO_ID = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        out: list[int] = []
        idx = 0
        while idx < len(text):
            matched = False
            for token, token_id in self._SPECIAL_TO_ID.items():
                if text.startswith(token, idx):
                    out.append(token_id)
                    idx += len(token)
                    matched = True
                    break
            if matched:
                continue
            out.append(ord(text[idx]))
            idx += 1
        return out

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        inverse = {value: key for key, value in self._SPECIAL_TO_ID.items()}
        out: list[str] = []
        for token_id in token_ids:
            if token_id in inverse:
                if not skip_special_tokens:
                    out.append(inverse[token_id])
            else:
                out.append(chr(token_id))
        return "".join(out)

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ) -> str:
        _ = (tokenize, enable_thinking)
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


class PinnedLfmLabelTokenizer(TinyChatTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        labels = {"A": [41], " A": [334], "B": [42], " B": [378]}
        if text in labels:
            return list(labels[text])
        if text.endswith("Agent A"):
            return super().encode(text[:-2], add_special_tokens=add_special_tokens) + [334]
        if text.endswith("Agent B"):
            return super().encode(text[:-2], add_special_tokens=add_special_tokens) + [378]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        labels = {41: "A", 334: " A", 42: "B", 378: " B"}
        if len(token_ids) == 1 and token_ids[0] in labels:
            return labels[token_ids[0]]
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens)


@dataclass
class RecordingSampler:
    tokenizer: object
    requests: list[SamplingRequest]
    batches: list[list[SamplingRequest]] = field(default_factory=list)
    judge_texts: list[str] = field(default_factory=list)
    candidate_rows: list[dict[int, float]] = field(default_factory=list)

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        self.batches.append(list(requests))
        self.requests.extend(requests)
        outputs = []
        for request in requests:
            if request.adapter_name in {"shared", "solution"}:
                text = "<SOLUTION>H, T, H, T</SOLUTION>"
            elif request.adapter_name == "judge":
                text = self.judge_texts.pop(0) if self.judge_texts else "<VERDICT>A</VERDICT>"
            else:
                text = "Their answer violates the format."
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            outputs.append(
                SamplingResult(
                    adapter_name=request.adapter_name,
                    prompt_token_ids=request.prompt_token_ids,
                    completion_token_ids=token_ids,
                    completion_logprobs=[-0.1] * len(token_ids),
                    text=text,
                    behavior_policy=BehaviorPolicySpec.from_sampling_request(request),
                    completion_logprob_semantics=BEHAVIOR_POLICY_LOGPROBS,
                    candidate_logprobs=(
                        [self.candidate_rows.pop(0)]
                        if request.candidate_logprob_token_ids
                        else []
                    ),
                    raw={},
                )
            )
        return outputs


def test_external_judge_fn_short_circuits_r1_only_policy_judge_sampling() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(tokenizer=tokenizer, requests=[])
    judge_calls: list[tuple[str, str]] = []

    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(num_rounds=1, num_groups=1, group_size=2, judge_adapter="policy"),
        adapter_layout="split",
        judge_fn=lambda question, constitution, r1_a, r1_b, r2_a, r2_b, r3_a, r3_b: (
            judge_calls.append((question, constitution)) or True
        ) and ("B", "external judge"),
    )

    output = runtime.rollout(step_seed=0)

    assert len(output.debates) == 1
    assert output.debates[0].verdict == "B"
    assert output.debates[0].judge_reasoning == "external judge"
    assert len(judge_calls) == 1
    assert len(sampler.requests) == 2
    assert all(request.adapter_name == "solution" for request in sampler.requests)


def test_external_judge_fn_short_circuits_three_round_policy_judge_sampling() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(tokenizer=tokenizer, requests=[])
    seen_round_payloads: list[tuple[str, str, str]] = []

    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(num_rounds=3, num_groups=1, group_size=2, judge_adapter="policy"),
        adapter_layout="split",
        judge_fn=lambda question, constitution, r1_a, r1_b, r2_a, r2_b, r3_a, r3_b: (
            seen_round_payloads.append((r1_a, r2_a, r3_a)) or True
        ) and ("A", "external judge"),
    )

    output = runtime.rollout(step_seed=0)

    assert len(output.debates) == 1
    assert output.debates[0].verdict == "A"
    assert output.debates[0].judge_reasoning == "external judge"
    assert len(seen_round_payloads) == 1
    assert len(sampler.requests) == 6
    assert [request.adapter_name for request in sampler.requests] == [
        "solution",
        "solution",
        "debate",
        "debate",
        "debate",
        "debate",
    ]


def test_r1_only_bidirectional_judge_samples_both_orders_and_records_audit() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(
        tokenizer=tokenizer,
        requests=[],
        judge_texts=["A", "B"],
    )
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=1,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=PAIRWISE_SINGLE_TOKEN_V1,
            judge_bidirectional=True,
            judge_constrain_single_token=True,
        ),
        adapter_layout="split",
    )

    debate = runtime.rollout(step_seed=7).debates[0]

    assert len(sampler.batches) == 2
    assert len(sampler.batches[-1]) == 2
    assert debate.verdict == "A"
    assert debate.judge_raw_response["bidirectional_judge"] is True
    assert debate.judge_raw_response["forward_verdict"] == "A"
    assert debate.judge_raw_response["reverse_verdict"] == "B"
    assert debate.judge_raw_response["reverse_mapped_verdict"] == "A"
    assert debate.judge_raw_response["order_invariant"] is True


def test_order_symmetric_soft_judge_requests_four_logits_and_never_falls_back() -> None:
    tokenizer = PinnedLfmLabelTokenizer()
    sampler = RecordingSampler(
        tokenizer=tokenizer,
        requests=[],
        judge_texts=["A", "B"],
        candidate_rows=[
            {41: 0.0, 334: 0.0, 42: -2.0, 378: -2.0},
            {41: -2.0, 334: -2.0, 42: 0.0, 378: 0.0},
        ],
    )
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=1,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=PAIRWISE_SINGLE_TOKEN_V1,
            judge_temperature=0.0,
            judge_bidirectional=True,
            judge_constrain_single_token=True,
            judge_score_mode="order_sym_soft_logit",
            judge_label_token_contract="lfm25_ab_whitespace_compat_v1",
        ),
        adapter_layout="split",
    )

    debate = runtime.rollout(step_seed=7).debates[0]

    judge_requests = sampler.batches[-1]
    assert all(request.allowed_token_ids == (41, 334, 42, 378) for request in judge_requests)
    assert all(request.candidate_logprob_token_ids == (41, 334, 42, 378) for request in judge_requests)
    assert all(request.max_tokens == 1 and request.temperature == 0.0 for request in judge_requests)
    audit = debate.judge_raw_response
    assert audit["aggregation"] == "order_sym_soft_logit_no_fallback"
    assert audit["judge_label_token_contract"]["temporary"] is True
    assert audit["soft_score"]["z_symmetric"] == pytest.approx(2.0)
    assert audit["debate_reward_a"] == pytest.approx(0.7615941559557649)
    assert audit["debate_reward_a"] + audit["debate_reward_b"] == pytest.approx(0.0)


def test_trainable_openbookqa_soft_judge_uses_exactly_two_spaced_tokens() -> None:
    tokenizer = PinnedLfmLabelTokenizer()
    sampler = RecordingSampler(
        tokenizer=tokenizer,
        requests=[],
        judge_texts=[" A", " B"],
        candidate_rows=[
            {334: -0.1, 378: -2.1},
            {334: -2.1, 378: -0.1},
        ],
    )
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=1.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=CONSTITUTION_SINGLE_TOKEN_V1,
            judge_temperature=1.0,
            judge_bidirectional=True,
            judge_constrain_single_token=True,
            judge_score_mode="order_sym_soft_logit",
            judge_label_token_contract="lfm25_openbookqa_spaced_ab_v1",
        ),
        adapter_layout="split",
    )

    debate = runtime.rollout(step_seed=7).debates[0]
    judge_requests = sampler.batches[-1]
    assert all(request.allowed_token_ids == (334, 378) for request in judge_requests)
    assert all(request.candidate_logprob_token_ids == (334, 378) for request in judge_requests)
    turns = debate.judge_raw_response["_training_judge_turns"]
    assert [turn["behavior_policy_allowed_token_ids"] for turn in turns] == [
        [334, 378],
        [334, 378],
    ]
    assert debate.judge_raw_response["soft_score"]["referent_js_divergence_normalized"] == pytest.approx(0.0)


def test_frozen_base_sft_judge_uses_same_sampler_in_one_ordered_batch() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(tokenizer=tokenizer, requests=[])
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=2,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=SOLUTION_R1_RATIONALE_V1,
            judge_max_tokens=512,
            judge_temperature=1.0,
            judge_top_p=0.95,
            judge_top_k=20,
            judge_presence_penalty=1.5,
            judge_seed=0,
        ),
        adapter_layout="split",
    )

    runtime.rollout(step_seed=7)

    assert len(sampler.batches) == 4
    judge_batch = sampler.batches[-1]
    assert len(judge_batch) == 2
    assert all(request.adapter_name == "judge" for request in judge_batch)
    assert all(request.max_tokens == 512 for request in judge_batch)
    assert all(request.temperature == 1.0 for request in judge_batch)
    assert all(request.top_p == 0.95 for request in judge_batch)
    assert all(request.top_k == 20 for request in judge_batch)
    assert all(request.presence_penalty == 1.5 for request in judge_batch)
    assert all(request.seed == 0 for request in judge_batch)
    prompts = [tokenizer.decode(request.prompt_token_ids) for request in judge_batch]
    assert all(prompt.startswith("System:\n") for prompt in prompts)
    assert all(prompt.endswith("The rationale for my judgment is:\n") for prompt in prompts)
    assert all("Evaluate which agent's Round 1 response best satisfies the user" in prompt for prompt in prompts)
    assert all("Do not reward confidence, rhetoric, or debate skill by itself." in prompt for prompt in prompts)
    assert all("which agent made a more convincing case" not in prompt for prompt in prompts)
    assert all("argument quality" not in prompt for prompt in prompts)
    assert all("rebuttal effectiveness" not in prompt for prompt in prompts)
    assert all("Round 2 (Argument):\nTheir answer violates the format." in prompt for prompt in prompts)
    assert all("Round 3 (Response):\nTheir answer violates the format." in prompt for prompt in prompts)


def test_single_token_harness_owns_parser_prefill_and_one_token_budget() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(tokenizer=tokenizer, requests=[], judge_texts=["A"])
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=CONSTITUTION_SINGLE_TOKEN_V1,
            judge_constrain_single_token=True,
        ),
        adapter_layout="split",
    )

    debate = runtime.rollout(step_seed=7).debates[0]

    judge_request = sampler.batches[-1][0]
    prompt = tokenizer.decode(judge_request.prompt_token_ids)
    assert judge_request.max_tokens == 1
    assert judge_request.allowed_token_ids == (ord("A"), ord("B"))
    assert prompt.endswith(
        "The response most adherent to the constitution was written by Agent"
    )
    assert debate.verdict == "A"


def test_base_sft_reverse_order_swaps_every_a_b_round_argument() -> None:
    tokenizer = TinyChatTokenizer()
    task = HTSequenceDebateTask(sequence_len=4)
    runtime = DebateRuntime(
        task=task,
        tokenizer=tokenizer,
        sampler=RecordingSampler(tokenizer=tokenizer, requests=[]),
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=SOLUTION_R1_RATIONALE_V1,
        ),
        adapter_layout="split",
    )
    instances = task.sample_instances(n=2, seed=3)
    inst_pairs = [(instances[0], instances[1])]

    [forward_tokens] = runtime._judge_prompts(
        inst_pairs=inst_pairs,
        r1_visible_text=["A1", "B1"],
        r2_visible_text=["A2", "B2"],
        r3_visible_text=["A3", "B3"],
    )
    [reverse_tokens] = runtime._judge_prompts(
        inst_pairs=inst_pairs,
        r1_visible_text=["A1", "B1"],
        r2_visible_text=["A2", "B2"],
        r3_visible_text=["A3", "B3"],
        reverse_order=True,
    )
    forward = tokenizer.decode(forward_tokens)
    reverse = tokenizer.decode(reverse_tokens)

    assert "=== AGENT A ===\nRound 1 (Proposal):\nA1" in forward
    assert "=== AGENT B ===\nRound 1 (Proposal):\nB1" in forward
    assert "=== AGENT A ===\nRound 1 (Proposal):\nB1" in reverse
    assert "Round 2 (Argument):\nB2" in reverse
    assert "Round 3 (Response):\nB3" in reverse
    assert "=== AGENT B ===\nRound 1 (Proposal):\nA1" in reverse


def test_bidirectional_judge_maps_reverse_labels_and_records_coherence() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(
        tokenizer=tokenizer,
        requests=[],
        judge_texts=[
            "forward one <VERDICT>A</VERDICT>",
            "forward two <VERDICT>B</VERDICT>",
            "reverse one <VERDICT>B</VERDICT>",
            "reverse two <VERDICT>A</VERDICT>",
        ],
    )
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=2,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=SOLUTION_R1_RATIONALE_V1,
            judge_bidirectional=True,
        ),
        adapter_layout="split",
    )

    output = runtime.rollout(step_seed=7)

    assert len(sampler.batches[-1]) == 4
    assert [debate.verdict for debate in output.debates] == ["A", "B"]
    assert all(debate.judge_raw_response["order_invariant"] is True for debate in output.debates)
    assert all(
        debate.judge_raw_response["aggregation"] == "order_invariant_agreement"
        for debate in output.debates
    )
    assert all(len(debate.judge_raw_response["_training_judge_turns"]) == 2 for debate in output.debates)
    assert [
        turn["order"] for turn in output.debates[0].judge_raw_response["_training_judge_turns"]
    ] == ["forward", "reverse"]


def test_bidirectional_judge_uses_seeded_random_winner_on_disagreement() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(
        tokenizer=tokenizer,
        requests=[],
        judge_texts=["<VERDICT>A</VERDICT>", "<VERDICT>A</VERDICT>"],
    )
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=SOLUTION_R1_RATIONALE_V1,
            judge_bidirectional=True,
        ),
        adapter_layout="split",
    )

    debate = runtime.rollout(step_seed=11).debates[0]

    assert debate.verdict in ("A", "B")
    assert debate.judge_raw_response["forward_verdict"] == "A"
    assert debate.judge_raw_response["reverse_verdict"] == "A"
    assert debate.judge_raw_response["reverse_mapped_verdict"] == "B"
    assert debate.judge_raw_response["order_invariant"] is False
    assert debate.judge_raw_response["aggregation"] == "seeded_random_on_order_disagreement"


def test_bidirectional_judge_does_not_replace_invalid_grpo_turn_with_greedy_retry() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingSampler(
        tokenizer=tokenizer,
        requests=[],
        judge_texts=["not a verdict", "<VERDICT>B</VERDICT>"],
    )
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.8),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=SOLUTION_R1_RATIONALE_V1,
            judge_bidirectional=True,
            judge_temperature=0.8,
        ),
        adapter_layout="split",
    )

    debate = runtime.rollout(step_seed=13).debates[0]

    # R1/R2/R3 plus exactly one two-order judge batch: no temperature-0 retry.
    assert len(sampler.batches) == 4
    assert debate.judge_raw_response["forward_verdict"] == "INVALID"
    assert debate.judge_raw_response["order_invariant"] is False
    turns = debate.judge_raw_response["_training_judge_turns"]
    assert tokenizer.decode(turns[0]["completion_tokens"]) == "not a verdict"


def test_solution_harness_rejects_incomplete_debate_instead_of_blank_rounds() -> None:
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=TinyChatTokenizer(),
        sampler=RecordingSampler(tokenizer=TinyChatTokenizer(), requests=[]),
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=1,
            num_groups=1,
            group_size=2,
            judge_adapter="judge",
            judge_harness_id=SOLUTION_R1_RATIONALE_V1,
        ),
        adapter_layout="split",
    )

    with pytest.raises(ValueError, match="requires at least 3 rounds"):
        runtime.rollout(step_seed=0)
