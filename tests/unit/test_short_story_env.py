from __future__ import annotations

from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.debate_runtime import DebateRuntime
from llm_local_rl.registry import build_debate_task, build_environment
from llm_local_rl.short_story_format import contains_word, extract_solution


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "".join(chr(tok) for tok in token_ids)


def test_short_story_extracts_tagged_solution_and_secret_word() -> None:
    text = "<SOLUTION>The opal glowed under the bridge.</SOLUTION>"
    assert extract_solution(text) == "The opal glowed under the bridge."
    assert contains_word(text, "opal")
    assert not contains_word(text, "pal")


def test_short_story_env_rewards_secret_word_in_solution() -> None:
    config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(env_name="short_story", mode="single_turn"),
    )
    env = build_environment(config)
    tokenizer = TinyTokenizer()
    instance = env.sample_instances(n=1, seed=0)[0]
    completion = f"<SOLUTION>A quiet traveler found a {instance.secret_word} beside the road.</SOLUTION>"

    reward, metrics = env.score_completion(
        instance=instance,
        tokenizer=tokenizer,
        completion_token_ids=tokenizer.encode(completion),
    )

    assert reward == 1.0
    assert metrics["parse_success"] == 1.0
    assert metrics["used_secret"] == 1.0


def test_short_story_base_r1_prompt_includes_secret_word_but_judge_context_hides_it() -> None:
    config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(env_name="short_story", mode="debate"),
        debate_prompt_format="qwen35_base_text_prefill",
    )
    task = build_debate_task(config)
    inst = task.sample_instances(n=1, seed=1)[0]
    tokenizer = TinyTokenizer()
    runtime = object.__new__(DebateRuntime)
    runtime.task = task
    runtime.tokenizer = tokenizer
    runtime.runtime_config = type("RuntimeConfig", (), {"prompt_format": "qwen35_base_text_prefill"})()

    r1_prompt = tokenizer.decode(DebateRuntime._base_r1_prompt_tokens(runtime, inst=inst))
    judge_context = task.judge_context_text(inst=inst)

    assert str(inst.payload["secret_word"]) in r1_prompt
    assert str(inst.payload["secret_word"]) not in judge_context
