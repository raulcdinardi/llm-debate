from dataclasses import dataclass

from llm_local_rl.behavior_policy import (
    BEHAVIOR_POLICY_LOGPROBS,
    UNSPECIFIED_LOGPROBS,
    BehaviorPolicySpec,
)
from llm_local_rl.envs import HTSequenceEnv
from llm_local_rl.episodes import DebateEpisodeBuilder, SingleTurnEpisodeBuilder
from llm_local_rl.types import SamplingRequest, SamplingResult


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "".join(chr(tok) for tok in token_ids)


@dataclass
class RecordingSampler:
    requests: list[SamplingRequest]
    logprob_semantics: str = BEHAVIOR_POLICY_LOGPROBS

    def sample(self, request: SamplingRequest) -> SamplingResult:
        self.requests.append(request)
        text = "HHHT" if request.adapter_name in {"shared", "solution"} else "critique"
        token_ids = [ord(ch) for ch in text]
        return SamplingResult(
            adapter_name=request.adapter_name,
            prompt_token_ids=request.prompt_token_ids,
            completion_token_ids=token_ids,
            completion_logprobs=[-0.1] * len(token_ids),
            text=text,
            behavior_policy=BehaviorPolicySpec.from_sampling_request(request),
            completion_logprob_semantics=self.logprob_semantics,  # type: ignore[arg-type]
        )


def test_single_turn_builder_uses_shared_adapter() -> None:
    env = HTSequenceEnv(sequence_len=4)
    sampler = RecordingSampler(requests=[])
    builder = SingleTurnEpisodeBuilder()
    sample = builder.build_and_score(
        env=env,
        tokenizer=TinyTokenizer(),
        sampler=sampler,
        instance=env.sample_instances(n=1, seed=0)[0],
        max_tokens=8,
        temperature=0.0,
        seed=0,
        min_p=0.2,
    )
    assert len(sample.turns) == 1
    assert sampler.requests[0].adapter_name == "shared"
    assert sampler.requests[0].min_p == 0.2


def test_single_turn_eval_builder_does_not_apply_ppo_logprob_validation() -> None:
    env = HTSequenceEnv(sequence_len=4)
    sampler = RecordingSampler(requests=[], logprob_semantics=UNSPECIFIED_LOGPROBS)
    builder = SingleTurnEpisodeBuilder(validate_behavior_policy_contract=False)

    sample = builder.build_and_score(
        env=env,
        tokenizer=TinyTokenizer(),
        sampler=sampler,
        instance=env.sample_instances(n=1, seed=0)[0],
        max_tokens=8,
        temperature=0.7,
        seed=0,
        min_p=0.2,
    )

    assert len(sample.turns) == 1
    assert sample.turns[0].completion_token_ids


def test_debate_builder_routes_solution_then_debate() -> None:
    env = HTSequenceEnv(sequence_len=4)
    sampler = RecordingSampler(requests=[])
    builder = DebateEpisodeBuilder()
    sample = builder.build_and_score(
        env=env,
        tokenizer=TinyTokenizer(),
        sampler=sampler,
        instance=env.sample_instances(n=1, seed=0)[0],
        max_tokens=8,
        temperature=0.0,
        seed=0,
    )
    assert [turn.adapter_name for turn in sample.turns] == ["solution", "debate"]
    assert [req.adapter_name for req in sampler.requests] == ["solution", "debate"]
