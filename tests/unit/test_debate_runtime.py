from __future__ import annotations

from dataclasses import dataclass
import os

import importlib.util
import pytest

from llm_local_rl.behavior_policy import BEHAVIOR_POLICY_LOGPROBS, BehaviorPolicySpec
if importlib.util.find_spec("transformers") is not None:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None

from llm_local_rl.constrained_writing import ConstrainedWritingDebateTask
from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
from llm_local_rl.debate_tasks import CountdownCodeDebateTask, HTSequenceDebateTask
from llm_local_rl.types import SamplingRequest, SamplingResult
from llm_local_rl.debate_parity import DebateConfig
from llm_local_rl.task_types import TaskInstance


def _real_tokenizer():
    if AutoTokenizer is None:
        pytest.skip("transformers is required for real-tokenizer debate runtime tests.")
    model_path = os.environ.get("LLM_LOCAL_RL_BASE_MODEL")
    if model_path is None:
        pytest.skip("LLM_LOCAL_RL_BASE_MODEL is required for real-tokenizer debate runtime tests.")
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


@dataclass
class RecordingSampler:
    tokenizer: object
    requests: list[SamplingRequest]
    call_sizes: list[int] | None = None

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        self.requests.extend(requests)
        if self.call_sizes is not None:
            self.call_sizes.append(len(requests))
        outputs = []
        for request in requests:
            if request.adapter_name in {"shared", "solution"}:
                text = "<SOLUTION>H, T, H, T</SOLUTION>"
            elif request.adapter_name == "debate":
                text = "Their answer violates the format."
            else:
                text = "<VERDICT>A</VERDICT>"
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
                )
            )
        return outputs


class CapturingTokenizer:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        self.encoded_texts.append(text)
        return list(range(len(text)))


class ByteCapturingTokenizer:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []
        self.add_special_tokens_values: list[bool] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encoded_texts.append(text)
        self.add_special_tokens_values.append(add_special_tokens)
        return list(text.encode("utf-8"))


def test_countdown_base_text_debate_prompts_use_countdown_constitution() -> None:
    tokenizer = CapturingTokenizer()
    runtime = object.__new__(DebateRuntime)
    runtime.task = CountdownCodeDebateTask()
    runtime.tokenizer = tokenizer
    runtime.runtime_config = DebateRuntimeConfig(prompt_format="qwen35_base_text_prefill")

    inst = TaskInstance(instance_id="countdown_code_0", payload={"numbers": [1, 2, 3, 4], "target": 24})
    runtime._base_r2_continuation_tokens(inst=inst, own_r1='{"solution.py": "expr = 24"}', opponent_r1='{"solution.py": "expr = 0"}')
    runtime._base_r3_continuation_tokens(
        inst=inst,
        own_r1='{"solution.py": "expr = 24"}',
        opponent_r1='{"solution.py": "expr = 0"}',
        own_r2="1) My expression is valid.",
        opponent_r2="1) Their tests pass.",
    )

    encoded = "\n\n".join(tokenizer.encoded_texts)
    assert "Countdown arithmetic task" in encoded
    assert "test-suite manipulation" in encoded
    assert "fair coin" not in encoded
    assert "H/T" not in encoded


def test_base_r1_prompt_uses_configured_plain_text_prefill_exactly() -> None:
    tokenizer = CapturingTokenizer()
    runtime = object.__new__(DebateRuntime)
    runtime.task = type(
        "Task",
        (),
        {"r1_context_text": lambda self, *, inst: f"Write for {inst.instance_id}."},
    )()
    runtime.tokenizer = tokenizer
    prefill = "Ok, I will produce a 3-sentence story adhering to the rules:\n"
    runtime.runtime_config = DebateRuntimeConfig(
        prompt_format="qwen35_base_text_prefill",
        r1_assistant_prefill=prefill,
    )

    runtime._base_r1_prompt_tokens(inst=TaskInstance(instance_id="cw-1", payload={}))

    assert tokenizer.encoded_texts[-1] == (
        "User:\nWrite for cw-1.\nAssistant:\n"
        "Ok, I will produce a 3-sentence story adhering to the rules:\n"
    )
    assert "<|im_start|>" not in tokenizer.encoded_texts[-1]
    assert "<think>" not in tokenizer.encoded_texts[-1]


def test_constrained_writing_base_prompts_are_strict_prefix_extensions() -> None:
    tokenizer = ByteCapturingTokenizer()
    task = ConstrainedWritingDebateTask.from_args(
        rules_per_speaker=2,
        reward_scope="both",
        sides="both",
        rule_family="generic",
        reward_mode="additive",
        letter_temperature=1.0,
    )
    inst = task.sample_instances(n=1, seed=2026071401)[0]
    r1_prefill = "Ok, I will produce a 3-sentence story adhering to the rules:\n"
    debate_prefill = "The reasons that my solution is better than my opponent's are:\n1)"
    runtime = object.__new__(DebateRuntime)
    runtime.task = task
    runtime.tokenizer = tokenizer
    runtime.runtime_config = DebateRuntimeConfig(
        prompt_format="qwen35_base_text_prefill",
        r1_assistant_prefill=r1_prefill,
        base_r2_prefill="deliberately ignored for constrained writing",
        base_r3_prefill="also deliberately ignored",
    )

    r1_prompt = runtime._base_r1_prompt_tokens(inst=inst)
    own_r1_tokens = tokenizer.encode("Own fixed story.", add_special_tokens=False)
    r2_continuation = runtime._base_r2_continuation_tokens(
        inst=inst,
        own_r1="Own fixed story.",
        opponent_r1="Opponent fixed story.",
    )
    r2_prompt = r1_prompt + own_r1_tokens + r2_continuation

    constitution = task.judge_constitution_text(inst=inst)
    expected_r2_continuation = (
        "\n\nUser:\n"
        f"Constitution: {constitution}\n"
        "Opponent Round 1 answer:\n"
        "Opponent fixed story.\n"
        "Continue arguing that your fixed answer should win under the constitution. "
        "Address the opponent directly and use concrete evidence from the task and answers.\n\n"
        "Write exactly 3 short numbered points. After point 3, immediately output "
        "CONCLUDED and nothing else.\n"
        "\nAssistant:\n"
        f"{debate_prefill}"
    )
    assert bytes(r2_continuation).decode("utf-8") == expected_r2_continuation
    assert r2_prompt[: len(r1_prompt) + len(own_r1_tokens)] == r1_prompt + own_r1_tokens

    own_r2_tokens = tokenizer.encode("Second point. CONCLUDED", add_special_tokens=False)
    r3_continuation = runtime._base_r3_continuation_tokens(
        inst=inst,
        own_r1="Own fixed story.",
        opponent_r1="Opponent fixed story.",
        own_r2=debate_prefill + "Second point.",
        opponent_r2=debate_prefill + "Opponent point.",
    )
    r3_prompt = r2_prompt + own_r2_tokens + r3_continuation
    expected_r3_continuation = expected_r2_continuation.replace(
        "Opponent Round 1 answer:\nOpponent fixed story.",
        f"Opponent Round 2 answer:\n{debate_prefill}Opponent point.",
    )
    assert bytes(r3_continuation).decode("utf-8") == expected_r3_continuation
    assert r3_prompt[: len(r2_prompt) + len(own_r2_tokens)] == r2_prompt + own_r2_tokens

    rendered = bytes(r3_continuation).decode("utf-8")
    assert "System:" not in rendered
    assert "Original task prompt:" not in rendered
    assert "Your fixed Round 1 answer:" not in rendered
    assert "Your Round 2 argument:" not in rendered
    assert runtime._base_text_debate_prefill(
        inst=inst,
        opponent_round=2,
        opponent_answer="Opponent point.",
        fallback="wrong fallback",
    ) == debate_prefill
    assert tokenizer.add_special_tokens_values
    assert all(value is False for value in tokenizer.add_special_tokens_values)


def test_concluded_stop_is_requested_only_for_debate_turns() -> None:
    tokenizer = CapturingTokenizer()
    sampler = RecordingSampler(tokenizer=tokenizer, requests=[])
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.8),
        runtime_config=DebateRuntimeConfig(
            prompt_format="qwen35_base_text_prefill",
            stop_on_concluded=True,
            top_p=0.95,
            min_p=0.02,
        ),
        adapter_layout="split",
    )

    runtime._sample_many(
        prompt_tokens_list=[[1, 2]],
        round_num=1,
        step_seed=None,
        stop_token_ids=[99],
        max_tokens=8,
        temperature=0.8,
    )
    runtime._sample_many(
        prompt_tokens_list=[[1, 2]],
        round_num=2,
        step_seed=None,
        stop_token_ids=[99],
        max_tokens=8,
        temperature=0.8,
    )

    assert sampler.requests[0].stop_strings == ()
    assert sampler.requests[1].stop_strings == ("CONCLUDED",)
    assert sampler.requests[1].include_stop_str_in_output is True
    assert sampler.requests[1].top_p == 0.95
    assert sampler.requests[1].min_p == 0.02


def test_debate_runtime_split_routes_solution_debate_and_policy_judge_with_real_tokenizer() -> None:
    tokenizer = _real_tokenizer()
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=RecordingSampler(tokenizer=tokenizer, requests=[]),
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(num_rounds=3, num_groups=1, group_size=2, judge_adapter="policy"),
        adapter_layout="split",
    )
    output = runtime.rollout(step_seed=0)
    assert len(output.debates) == 1
    adapter_sequence = [request.adapter_name for request in runtime.sampler.requests]
    assert adapter_sequence[:2] == ["solution", "solution"]
    assert adapter_sequence[2:4] == ["debate", "debate"]
    assert adapter_sequence[4:6] == ["debate", "debate"]
    assert adapter_sequence[6] == "debate"


def test_debate_runtime_shared_routes_all_rounds_to_shared_with_real_tokenizer() -> None:
    tokenizer = _real_tokenizer()
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=RecordingSampler(tokenizer=tokenizer, requests=[]),
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(num_rounds=1, num_groups=1, group_size=2, judge_adapter="policy"),
        adapter_layout="shared",
    )
    runtime.rollout(step_seed=0)
    assert all(request.adapter_name == "shared" for request in runtime.sampler.requests)


def test_debate_runtime_supports_general_round_adapter_mapping_and_rollout_batching() -> None:
    tokenizer = _real_tokenizer()
    sampler = RecordingSampler(tokenizer=tokenizer, requests=[], call_sizes=[])
    runtime = DebateRuntime(
        task=HTSequenceDebateTask(sequence_len=4),
        tokenizer=tokenizer,
        sampler=sampler,
        debate_config=DebateConfig(max_tokens_per_turn=16, temperature=0.0),
        runtime_config=DebateRuntimeConfig(
            num_rounds=3,
            num_groups=1,
            group_size=2,
            judge_adapter="policy",
            round_adapter_names=("solution", "solution", "debate"),
            rollout_batch_size=1,
        ),
        adapter_layout="split",
    )
    runtime.rollout(step_seed=0)
    adapter_sequence = [request.adapter_name for request in sampler.requests]
    assert adapter_sequence[:2] == ["solution", "solution"]
    assert adapter_sequence[2:4] == ["solution", "solution"]
    assert adapter_sequence[4:6] == ["debate", "debate"]
    assert adapter_sequence[6] == "debate"
    assert all(size == 1 for size in sampler.call_sizes)
    assert len(sampler.call_sizes) in {7, 8}


def test_debate_runtime_default_request_seed_mode_does_not_seed_each_request() -> None:
    runtime = object.__new__(DebateRuntime)
    runtime.runtime_config = DebateRuntimeConfig(request_seed_mode="none")
    assert runtime._round_seed(step_seed=123, request_idx=4, round_num=2) is None


def test_debate_runtime_per_request_seed_mode_preserves_unique_request_seeds() -> None:
    runtime = object.__new__(DebateRuntime)
    runtime.runtime_config = DebateRuntimeConfig(request_seed_mode="per_request")
    assert runtime._round_seed(step_seed=123, request_idx=4, round_num=2) == 200127
