from __future__ import annotations

import pytest

from llm_local_rl.constrained_writing import ConstrainedWritingDebateTask, ConstrainedWritingEnv
from llm_local_rl.task_types import TaskInstance


class FakeTokenizer:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def decode(self, tokens, skip_special_tokens=True):  # noqa: ARG002
        return self._text


def make_task(*, reward_mode: str = "additive", anchors: str = "on") -> ConstrainedWritingDebateTask:
    return ConstrainedWritingDebateTask.from_args(
        rules_per_speaker=2,
        reward_scope="both",
        sides="both",
        rule_family="generic",
        reward_mode=reward_mode,
        letter_temperature=1.0,
        anchors=anchors,
    )


def make_instance(alice_rules: list[str], bob_rules: list[str]) -> TaskInstance:
    return TaskInstance(
        instance_id="cw_test",
        payload={
            "topic": "a lost map",
            "end_word": "home",
            "color_set": ["red", "blue", "green"],
            "alice_rules": alice_rules,
            "bob_rules": bob_rules,
            "order": "alice_first",
            "sides": "both",
            "reward_scope": "both",
            "rule_family": "generic",
            "reward_mode": "additive",
            "letter_temperature": 1.0,
            "alice_banned_letters": [],
            "bob_banned_letters": [],
        },
    )


def score(task: ConstrainedWritingDebateTask, inst: TaskInstance, text: str):
    return task.compute_reward(inst=inst, completion_tokens=[0], tokenizer=FakeTokenizer(text))


HONEST = (
    "I wandered through the quiet town and searched for the lost map. "
    "I climbed the old stone tower and studied the faded roads below. "
    "I followed the final marked trail until the winding path ended at home."
)

DEGENERATE_SHORT = "I rested. I rested. I rested."


def test_i_only_pronoun_ban_enforced():
    task = make_task(anchors="off")
    inst = make_instance(["i_only"], ["no_commas"])
    bad = score(task, inst, "I walked and he followed. I slept well. I woke up early.")
    good = score(task, inst, HONEST)
    assert bad.metrics["rule_scores"]["i_only"][0] == -1
    assert good.metrics["rule_scores"]["i_only"] == [1, 1, 1]


def test_anchors_off_matches_pure_rule_reward():
    task = make_task(anchors="off")
    inst = make_instance(["i_only", "no_commas"], ["past_tense_ed", "end_word"])
    r = score(task, inst, DEGENERATE_SHORT)
    assert r.metrics["anchor_penalty"] == 0.0
    assert r.reward == 6.0


def test_degenerate_short_cannot_beat_honest():
    task = make_task()
    inst = make_instance(["i_only", "no_commas"], ["past_tense_ed", "end_word"])
    degenerate = score(task, inst, DEGENERATE_SHORT)
    honest = score(task, inst, HONEST)
    assert degenerate.metrics["anchor_word_window_violations"] == 3
    assert degenerate.metrics["anchor_repetition_violated"]
    assert degenerate.metrics["anchor_duplicate_sentences"]
    assert honest.metrics["anchor_penalty"] == 0.0
    assert degenerate.reward < honest.reward
    assert degenerate.reward <= 0.0


def test_wrong_sentence_count_penalized():
    task = make_task()
    inst = make_instance(["i_only", "no_commas"], ["past_tense_ed", "end_word"])
    one_sentence = score(task, inst, "I walked home.")
    rambling = score(task, inst, HONEST + " I paused. I turned. I waited. I smiled. I left.")
    assert one_sentence.metrics["anchor_sentence_count_penalty"] > 0
    assert rambling.metrics["anchor_sentence_count_penalty"] > 0
    assert one_sentence.metrics["parse_success"] == 0.0


def test_binary_mode_anchor_scale():
    task = make_task(reward_mode="binary")
    inst = make_instance(["i_only", "no_commas"], ["past_tense_ed", "end_word"])
    degenerate = score(task, inst, DEGENERATE_SHORT)
    assert degenerate.reward <= -1.5


def test_sampling_deterministic_by_seed():
    task = make_task()
    a = task.sample_instances(n=8, seed=13)
    b = task.sample_instances(n=8, seed=13)
    c = task.sample_instances(n=8, seed=14)
    assert [x.payload for x in a] == [x.payload for x in b]
    assert [x.payload for x in a] != [x.payload for x in c]


def test_debate_protocol_surface():
    task = make_task()
    inst = task.sample_instances(n=1, seed=3)[0]
    r1 = task.r1_context_text(inst=inst)
    assert "exactly three sentences" in r1
    jc = task.judge_context_text(inst=inst)
    # judge must stay rules-blind: context is the topic only
    assert "rule" not in jc.lower()
    assert jc == str(inst.payload["topic"])
    constitution = task.judge_constitution_text(inst=inst)
    assert "A or B" in constitution
    assert task.debate_r2_user_template() is None
    assert task.debate_r3_user_template() is None


def test_base_text_debate_extension_contract_is_exact_and_round_specific():
    task = make_task()
    inst = make_instance(["i_only", "no_commas"], ["past_tense_ed", "end_word"])
    constitution = task.judge_constitution_text(inst=inst)
    expected_prefill = "The reasons that my solution is better than my opponent's are:\n1)"

    r2 = task.build_base_text_debate_extension(
        inst=inst,
        opponent_round=1,
        opponent_answer="Opponent story.",
    )
    assert r2.user_text == (
        f"Constitution: {constitution}\n"
        "Opponent Round 1 answer:\n"
        "Opponent story.\n"
        "Continue arguing that your fixed answer should win under the constitution. "
        "Address the opponent directly and use concrete evidence from the task and answers.\n\n"
        "Write exactly 3 short numbered points. After point 3, immediately output "
        "CONCLUDED and nothing else.\n"
    )
    assert r2.assistant_prefill == expected_prefill

    r3 = task.build_base_text_debate_extension(
        inst=inst,
        opponent_round=2,
        opponent_answer="Opponent argument.",
    )
    assert "Opponent Round 2 answer:\nOpponent argument.\n" in r3.user_text
    assert r3.assistant_prefill == expected_prefill

    with pytest.raises(ValueError, match="opponent_round must be 1 or 2"):
        task.build_base_text_debate_extension(
            inst=inst,
            opponent_round=3,
            opponent_answer="invalid",
        )


def test_env_wrapper_shadow_scoring():
    task = make_task()
    env = ConstrainedWritingEnv(task=task)
    inst = make_instance(["i_only", "no_commas"], ["past_tense_ed", "end_word"])
    reward, metrics = env.score_completion(
        instance=inst, tokenizer=FakeTokenizer(HONEST), completion_token_ids=[0]
    )
    assert reward > 0
    assert "rule_scores" in metrics and "anchor_penalty" in metrics


def test_registry_builds_cw():
    from llm_local_rl.config import RolloutConfig, TrainRunConfig
    from llm_local_rl.registry import build_debate_task, build_environment

    config = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="debate", env_name="constrained_writing"),
    )
    task = build_debate_task(config)
    assert isinstance(task, ConstrainedWritingDebateTask)
    config2 = TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(mode="single_turn", env_name="constrained_writing"),
    )
    env = build_environment(config2)
    assert isinstance(env, ConstrainedWritingEnv)

    # config knobs flow through from_dict round-trip
    payload = config.to_dict()
    payload["constrained_writing_rules_per_speaker"] = 1
    payload["constrained_writing_rule_family"] = "ban_letters"
    restored = TrainRunConfig.from_dict(payload)
    task2 = build_debate_task(restored)
    assert task2.rules_per_speaker == 1
    assert task2.rule_family == "ban_letters"


POSITIONAL_STORY = (
    "A patient ranger walked beside the old trail. "
    "B quiet lanterns glowed above the stone trail. "
    "C careful hikers rested near the winding trail."
)


def test_positional_rule_zeroes_are_not_applicable_not_failures():
    task = make_task(anchors="off")
    inst = make_instance(["start_a"], ["no_commas"])

    result = score(task, inst, POSITIONAL_STORY)

    assert result.metrics["rule_scores"]["start_a"] == [1, 0, 0]
    assert result.metrics["rule_satisfaction"]["start_a"] == 1.0
    assert result.metrics["reward_all_rules_satisfied"] is True


def test_positional_rule_failure_and_wrong_sentence_count_fail_global_satisfaction():
    task = make_task(anchors="off")
    inst = make_instance(["start_a"], ["no_commas"])

    bad_start = score(task, inst, POSITIONAL_STORY.replace("A patient", "The patient", 1))
    two_sentences = score(
        task,
        inst,
        "A patient ranger walked beside the old trail. B quiet lanterns glowed above the trail.",
    )

    assert bad_start.metrics["reward_all_rules_satisfied"] is False
    assert two_sentences.metrics["parse_success"] == 0.0
    assert two_sentences.metrics["reward_all_rules_satisfied"] is False


VALID_STORY = (
    "I wandered through the quiet town and searched for the lost map. "
    "I climbed the old stone tower and studied the faded roads below. "
    "I followed the final marked trail until the winding path ended at home."
)


class _CwTinyTokenizer:
    all_special_tokens = ["<|im_end|>"]
    additional_special_tokens: list[str] = []
    _SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2}

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        out = []
        i = 0
        while i < len(text):
            for tok, tid in self._SPECIAL.items():
                if text.startswith(tok, i):
                    out.append(tid)
                    i += len(tok)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self._SPECIAL.items()}
        return "".join(
            (inv[t] if not skip_special_tokens else "") if t in inv else chr(t) for t in ids
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking=None):  # noqa: ARG002
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


def test_debate_runtime_integration_fake_sampler():
    from llm_local_rl.debate_parity import DebateConfig
    from llm_local_rl.debate_runtime import DebateRuntime, DebateRuntimeConfig
    from llm_local_rl.types import SamplingRequest, SamplingResult

    tokenizer = _CwTinyTokenizer()
    issued: list[SamplingRequest] = []

    class ScriptedSampler:
        def sample_many(self, requests):
            issued.extend(requests)
            outs = []
            for req in requests:
                prompt_text = tokenizer.decode(req.prompt_token_ids)
                text = "<VERDICT>A</VERDICT>" if "VERDICT" in prompt_text else VALID_STORY
                outs.append(SamplingResult(
                    adapter_name=req.adapter_name,
                    prompt_token_ids=req.prompt_token_ids,
                    completion_token_ids=tokenizer.encode(text),
                    completion_logprobs=[-0.1] * len(tokenizer.encode(text)),
                    text=text,
                    raw={"scripted": True},
                ))
            return outs

        def sample(self, request):
            return self.sample_many([request])[0]

    runtime = DebateRuntime(
        task=make_task(),
        tokenizer=tokenizer,
        sampler=ScriptedSampler(),
        debate_config=DebateConfig(max_tokens_per_turn=64, temperature=0.0),
        runtime_config=DebateRuntimeConfig(num_rounds=3, num_groups=1, group_size=2, judge_adapter="policy"),
        adapter_layout="shared",
    )
    out = runtime.rollout(step_seed=7)

    assert len(out.debates) == 1
    debate = out.debates[0]
    assert debate.verdict in ("A", "B")
    # shadow programmatic scoring must be present regardless of reward mode
    for traj_metrics in (debate.trajectory_a.metrics, debate.trajectory_b.metrics):
        rm = traj_metrics["task_reward_metrics"]
        assert "rule_scores" in rm and "anchor_penalty" in rm
        assert isinstance(traj_metrics["task_reward"], float)
    # judge prompt was issued and is rules-blind
    judge_prompts = [tokenizer.decode(r.prompt_token_ids) for r in issued
                     if "VERDICT" in tokenizer.decode(r.prompt_token_ids)]
    assert judge_prompts, "no judge request issued"
    assert all("Each sentence" not in p for p in judge_prompts)
