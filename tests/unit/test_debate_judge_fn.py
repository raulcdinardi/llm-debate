from __future__ import annotations

from dataclasses import dataclass, field

from llm_local_rl.behavior_policy import BEHAVIOR_POLICY_LOGPROBS, BehaviorPolicySpec
from llm_local_rl.debate_parity import DebateConfig
from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
from llm_local_rl.debate_tasks import HTSequenceDebateTask
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


@dataclass
class RecordingSampler:
    tokenizer: object
    requests: list[SamplingRequest]
    batches: list[list[SamplingRequest]] = field(default_factory=list)

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        self.batches.append(list(requests))
        self.requests.extend(requests)
        outputs = []
        for request in requests:
            if request.adapter_name in {"shared", "solution"}:
                text = "<SOLUTION>H, T, H, T</SOLUTION>"
            elif request.adapter_name == "judge":
                text = "<VERDICT>A</VERDICT>"
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
            judge_prompt_format="base_model_sft",
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
    assert all(prompt.endswith("1)") for prompt in prompts)
