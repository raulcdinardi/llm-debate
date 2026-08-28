from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_local_rl import debate_runtime as debate_runtime_module
from llm_local_rl.behavior_policy import BEHAVIOR_POLICY_LOGPROBS, BehaviorPolicySpec
from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.debate_parity import DebateConfig
from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
from llm_local_rl.judge_harness import CHAT_POINTWISE_TAGGED_V1, CHAT_SOLUTION_TAGGED_V1
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
class RecordingPolicySampler:
    tokenizer: TinyChatTokenizer
    requests: list[SamplingRequest]

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        self.requests.extend(requests)
        outputs = []
        for request in requests:
            if request.adapter_name in {"shared", "solution"}:
                text = "<SOLUTION>H, T, H, T</SOLUTION>"
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
                    raw={"fake_policy": True},
                )
            )
        return outputs


class FakeSglangSampler:
    instances: list["FakeSglangSampler"] = []

    def __init__(self, *, runtime, adapter_paths: dict[str, str] | None = None) -> None:
        self.runtime = runtime
        self.adapter_paths = {} if adapter_paths is None else dict(adapter_paths)
        self.requests: list[SamplingRequest] = []
        self.instances.append(self)

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        self.requests.extend(requests)
        outputs = []
        for request in requests:
            text = "PASS" if request.max_tokens <= 8 else "<VERDICT>A</VERDICT>"
            outputs.append(
                SamplingResult(
                    adapter_name=request.adapter_name,
                    prompt_token_ids=request.prompt_token_ids,
                    completion_token_ids=[42],
                    completion_logprobs=[-0.1],
                    text=text,
                    raw={"fake_sglang": True},
                )
            )
        return outputs


def _install_fake_sglang(monkeypatch: pytest.MonkeyPatch) -> type[FakeSglangSampler]:
    FakeSglangSampler.instances = []
    monkeypatch.setattr(debate_runtime_module, "SglangSampler", FakeSglangSampler)
    return FakeSglangSampler


def _runtime(
    *,
    tokenizer: TinyChatTokenizer,
    sampler: RecordingPolicySampler,
    num_rounds: int,
    judge_server_url: str | None = "http://judge.test:30000",
    judge_server_adapter_path: str | None = "/tmp/judge-adapter",
) -> DebateRuntime:
    return DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=num_rounds,
            num_groups=1,
            group_size=2,
            judge_adapter="policy",
            judge_harness_id=(
                CHAT_POINTWISE_TAGGED_V1 if num_rounds == 1 else CHAT_SOLUTION_TAGGED_V1
            ),
            debate_judge_server_url=judge_server_url,
            debate_judge_server_adapter_path=judge_server_adapter_path,
        ),
        adapter_layout="split",
    )


def test_cross_base_judge_config_round_trips() -> None:
    config = TrainRunConfig(
        model_path="/tmp/model",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="debate"),
        debate_judge_server_url="http://judge.test:30000",
        debate_judge_server_adapter_path="/tmp/judge-adapter",
    )

    restored = TrainRunConfig.from_dict(config.to_dict())

    assert restored.debate_judge_server_url == "http://judge.test:30000"
    assert restored.debate_judge_server_adapter_path == "/tmp/judge-adapter"


def test_cross_base_judge_config_rejects_external_judge_url() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainRunConfig(
            model_path="/tmp/model",
            output_dir="/tmp/out",
            rollout=RolloutConfig(mode="debate"),
            debate_external_judge_url="http://external-judge.test:8123",
            debate_judge_server_url="http://judge.test:30000",
        )


def test_default_judge_sampler_is_policy_sampler() -> None:
    tokenizer = TinyChatTokenizer()
    sampler = RecordingPolicySampler(tokenizer=tokenizer, requests=[])
    runtime = _runtime(
        tokenizer=tokenizer,
        sampler=sampler,
        num_rounds=1,
        judge_server_url=None,
        judge_server_adapter_path=None,
    )

    assert runtime.judge_sampler is sampler


def test_dedicated_judge_sampler_uses_base_model_when_adapter_path_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sglang = _install_fake_sglang(monkeypatch)
    tokenizer = TinyChatTokenizer()
    sampler = RecordingPolicySampler(tokenizer=tokenizer, requests=[])

    runtime = _runtime(
        tokenizer=tokenizer,
        sampler=sampler,
        num_rounds=1,
        judge_server_adapter_path=None,
    )

    assert runtime.judge_sampler is fake_sglang.instances[0]
    assert fake_sglang.instances[0].runtime.base_url == "http://judge.test:30000"
    assert fake_sglang.instances[0].adapter_paths == {}
    assert runtime._judge_adapter_name() == "base"


def test_dedicated_judge_sampler_routes_r1_three_round_and_pointwise_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sglang = _install_fake_sglang(monkeypatch)

    r1_tokenizer = TinyChatTokenizer()
    r1_policy_sampler = RecordingPolicySampler(tokenizer=r1_tokenizer, requests=[])
    r1_runtime = _runtime(tokenizer=r1_tokenizer, sampler=r1_policy_sampler, num_rounds=1)

    r1_output = r1_runtime.rollout(step_seed=0)
    r1_judge_sampler = fake_sglang.instances[-1]

    assert len(r1_output.debates) == 1
    assert r1_output.debates[0].verdict == "A"
    assert [request.adapter_name for request in r1_policy_sampler.requests] == ["solution", "solution"]
    assert [request.adapter_name for request in r1_judge_sampler.requests] == ["judge"]
    assert r1_judge_sampler.runtime.base_url == "http://judge.test:30000"
    assert r1_judge_sampler.adapter_paths == {"judge": "/tmp/judge-adapter"}

    three_tokenizer = TinyChatTokenizer()
    three_policy_sampler = RecordingPolicySampler(tokenizer=three_tokenizer, requests=[])
    three_runtime = _runtime(tokenizer=three_tokenizer, sampler=three_policy_sampler, num_rounds=3)

    three_output = three_runtime.rollout(step_seed=0)
    three_judge_sampler = fake_sglang.instances[-1]

    assert len(three_output.debates) == 1
    assert three_output.debates[0].verdict == "A"
    assert [request.adapter_name for request in three_policy_sampler.requests] == [
        "solution",
        "solution",
        "debate",
        "debate",
        "debate",
        "debate",
    ]
    assert [request.adapter_name for request in three_judge_sampler.requests] == ["judge"]

    reward_map = three_runtime.sample_pointwise_judge_rewards(debates=three_output.debates, step_seed=0)

    assert reward_map
    assert len(three_policy_sampler.requests) == 6
    assert [request.adapter_name for request in three_judge_sampler.requests] == ["judge", "judge", "judge"]
    assert [request.max_tokens for request in three_judge_sampler.requests] == [16, 8, 8]


def test_driver_debate_runtime_forwards_cross_base_judge_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_local_rl.driver import TrainingDriver

    fake_sglang = _install_fake_sglang(monkeypatch)
    tokenizer = TinyChatTokenizer()
    policy_sampler = RecordingPolicySampler(tokenizer=tokenizer, requests=[])
    driver = object.__new__(TrainingDriver)
    driver.config = TrainRunConfig(
        model_path="/tmp/model",
        output_dir="/tmp/out",
        adapter_layout="split",
        rollout=RolloutConfig(mode="debate", num_groups=1, group_size=2),
        debate_judge_server_url="http://judge.test:30000",
        debate_judge_server_adapter_path="/tmp/judge-adapter",
    )
    driver.debate_task = HTSequenceDebateTask(sequence_len=4)
    driver.tokenizer = tokenizer
    driver.sampler = policy_sampler

    runtime = driver._debate_runtime()

    assert runtime.runtime_config.debate_judge_server_url == "http://judge.test:30000"
    assert runtime.runtime_config.debate_judge_server_adapter_path == "/tmp/judge-adapter"
    assert runtime.judge_sampler is fake_sglang.instances[0]
    assert fake_sglang.instances[0].runtime.base_url == "http://judge.test:30000"
    assert fake_sglang.instances[0].adapter_paths == {"judge": "/tmp/judge-adapter"}
