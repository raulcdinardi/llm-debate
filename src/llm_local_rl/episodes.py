from __future__ import annotations

from dataclasses import dataclass

from llm_local_rl.behavior_policy import validate_sampling_result_contract
from llm_local_rl.interfaces import Environment, Sampler, Tokenizer
from llm_local_rl.types import EpisodeSample, EpisodeTurn, SamplingRequest


@dataclass(frozen=True)
class SingleTurnEpisodeBuilder:
    adapter_name: str = "shared"

    def build_and_score(
        self,
        *,
        env: Environment,
        tokenizer: Tokenizer,
        sampler: Sampler,
        instance,
        max_tokens: int,
        temperature: float,
        seed: int | None,
        min_p: float = 0.0,
    ) -> EpisodeSample:
        prompt_builder = getattr(env, "build_initial_prompt_token_ids", None)
        if callable(prompt_builder):
            prompt_token_ids = prompt_builder(instance=instance, tokenizer=tokenizer, enable_thinking=None)
        else:
            prompt_text = env.build_initial_prompt(instance=instance)
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        request = SamplingRequest(
            adapter_name=self.adapter_name,
            prompt_token_ids=prompt_token_ids,
            stop_token_ids=env.stop_token_ids(tokenizer=tokenizer),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            min_p=min_p,
        )
        result = sampler.sample(request)
        validate_sampling_result_contract(request=request, result=result)
        reward, reward_metrics = env.score_completion(
            instance=instance,
            tokenizer=tokenizer,
            completion_token_ids=result.completion_token_ids,
        )
        return EpisodeSample(
            instance_id=instance.instance_id,
            turns=[
                EpisodeTurn(
                    turn_name="response",
                    adapter_name=result.adapter_name,
                    prompt_token_ids=result.prompt_token_ids,
                    completion_token_ids=result.completion_token_ids,
                    completion_logprobs=result.completion_logprobs,
                    trainable=True,
                    metadata={"text": result.text},
                )
            ],
            reward=reward,
            reward_metrics=reward_metrics,
        )


@dataclass(frozen=True)
class DebateEpisodeBuilder:
    solution_adapter: str = "solution"
    debate_adapter: str = "debate"

    def build_and_score(
        self,
        *,
        env: Environment,
        tokenizer: Tokenizer,
        sampler: Sampler,
        instance,
        max_tokens: int,
        temperature: float,
        seed: int | None,
        min_p: float = 0.0,
    ) -> EpisodeSample:
        # The environment remains task-only. Debate mechanics live here.
        prompt_builder = getattr(env, "build_initial_prompt_token_ids", None)
        if callable(prompt_builder):
            prompt_token_ids = prompt_builder(instance=instance, tokenizer=tokenizer, enable_thinking=None)
        else:
            prompt_text = env.build_initial_prompt(instance=instance)
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        stop_token_ids = env.stop_token_ids(tokenizer=tokenizer)

        solution_request = SamplingRequest(
            adapter_name=self.solution_adapter,
            prompt_token_ids=prompt_token_ids,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            min_p=min_p,
        )
        solution_result = sampler.sample(solution_request)
        validate_sampling_result_contract(
            request=solution_request,
            result=solution_result,
        )
        critique_prefix = tokenizer.encode(" Critique the previous answer.", add_special_tokens=False)
        critique_prompt = prompt_token_ids + solution_result.completion_token_ids + critique_prefix
        debate_request = SamplingRequest(
            adapter_name=self.debate_adapter,
            prompt_token_ids=critique_prompt,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=None if seed is None else seed + 1,
            min_p=min_p,
        )
        debate_result = sampler.sample(debate_request)
        validate_sampling_result_contract(
            request=debate_request,
            result=debate_result,
        )
        reward, reward_metrics = env.score_completion(
            instance=instance,
            tokenizer=tokenizer,
            completion_token_ids=solution_result.completion_token_ids,
        )
        return EpisodeSample(
            instance_id=instance.instance_id,
            turns=[
                EpisodeTurn(
                    turn_name="solution",
                    adapter_name=solution_result.adapter_name,
                    prompt_token_ids=solution_result.prompt_token_ids,
                    completion_token_ids=solution_result.completion_token_ids,
                    completion_logprobs=solution_result.completion_logprobs,
                    trainable=True,
                    metadata={"text": solution_result.text},
                ),
                EpisodeTurn(
                    turn_name="critique",
                    adapter_name=debate_result.adapter_name,
                    prompt_token_ids=debate_result.prompt_token_ids,
                    completion_token_ids=debate_result.completion_token_ids,
                    completion_logprobs=debate_result.completion_logprobs,
                    trainable=True,
                    metadata={"text": debate_result.text},
                ),
            ],
            reward=reward,
            reward_metrics=reward_metrics,
        )
