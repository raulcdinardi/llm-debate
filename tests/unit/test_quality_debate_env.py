from __future__ import annotations

import json
from pathlib import Path

from llm_local_rl.config import RolloutConfig, TrainRunConfig
from llm_local_rl.debate_runtime import DebateRuntime
from llm_local_rl.quote_verifier import verify_quotes
from llm_local_rl.registry import build_debate_task, build_environment


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "".join(chr(tok) for tok in token_ids)


def _write_quality_fixture(root: Path) -> None:
    row = {
        "article_id": "fixture_article",
        "set_unique_id": "fixture_set",
        "source": "Gutenberg",
        "title": "Fixture Story",
        "topic": "Science fiction; Short stories",
        "article": "Captain Mira opened the silver door. The engine failed before dawn.",
        "questions": [
            {
                "question": "What did Captain Mira open?",
                "question_unique_id": "fixture_q1",
                "options": ["a red box", "the silver door", "a locked gate", "the engine"],
                "gold_label": 2,
                "validation": [
                    {"untimed_best_distractor": 4},
                    {"untimed_best_distractor": 4},
                    {"untimed_best_distractor": 1},
                ],
                "difficult": 1,
            }
        ],
    }
    for split in ("train", "dev", "test"):
        (root / f"QuALITY.v1.0.1.htmlstripped.{split}").write_text(json.dumps(row) + "\n")


def _quality_config(root: Path) -> TrainRunConfig:
    return TrainRunConfig(
        model_path="/tmp/nonexistent_model_for_shape_only",
        output_dir="/tmp/out",
        rollout=RolloutConfig(env_name="quality_debate", mode="debate"),
        quality_data_dir=str(root),
        quality_split="train",
        quality_hard_only=True,
        quality_source="Gutenberg",
        quality_topic_contains="Science fiction",
    )


def test_quote_verifier_rewrites_exact_and_failed_quotes() -> None:
    result = verify_quotes(
        'Evidence: <quote>Captain Mira opened the silver door.</quote> Bad: <quote>not in source</quote>',
        source_text="Captain Mira opened the silver door. The engine failed before dawn.",
    )
    assert "<v_quote>Captain Mira opened the silver door.</v_quote>" in result.text
    assert "<u_quote>not in source</u_quote>" in result.text
    assert result.metrics["verified_quote_count"] == 1
    assert result.metrics["unverified_quote_count"] == 1


def test_quality_debate_task_hides_article_from_judge_and_expands_answer_sides(tmp_path: Path) -> None:
    _write_quality_fixture(tmp_path)
    task = build_debate_task(_quality_config(tmp_path))
    inst = task.sample_instances(n=1, seed=0)[0]
    expanded = task.expand_group_instances(inst=inst, group_size=4, seed=0)

    assert [item.payload["assigned_label"] for item in expanded] == ["A", "B", "A", "B"]
    assert "Captain Mira opened" in task.r1_context_text(inst=expanded[0])
    assert "Captain Mira opened" not in task.judge_context_text(inst=expanded[0])
    assert expanded[0].payload["answer_a"] != expanded[0].payload["answer_b"]
    assert expanded[0].payload["correct_label"] in {"A", "B"}


def test_quality_base_prompt_uses_private_article_but_judge_context_does_not(tmp_path: Path) -> None:
    _write_quality_fixture(tmp_path)
    task = build_debate_task(_quality_config(tmp_path))
    inst = task.expand_group_instances(inst=task.sample_instances(n=1, seed=1)[0], group_size=2, seed=1)[0]
    tokenizer = TinyTokenizer()
    runtime = object.__new__(DebateRuntime)
    runtime.task = task
    runtime.tokenizer = tokenizer
    runtime.runtime_config = type("RuntimeConfig", (), {"prompt_format": "qwen35_base_text_prefill"})()

    r1_prompt = tokenizer.decode(DebateRuntime._base_r1_prompt_tokens(runtime, inst=inst))

    assert "Captain Mira opened" in r1_prompt
    assert "Captain Mira opened" not in task.judge_context_text(inst=inst)


def test_quality_single_turn_env_scores_choice(tmp_path: Path) -> None:
    _write_quality_fixture(tmp_path)
    env = build_environment(
        TrainRunConfig(
            model_path="/tmp/nonexistent_model_for_shape_only",
            output_dir="/tmp/out",
            rollout=RolloutConfig(env_name="quality_debate", mode="single_turn"),
            quality_data_dir=str(tmp_path),
            quality_split="train",
        )
    )
    tokenizer = TinyTokenizer()
    inst = env.sample_instances(n=1, seed=0)[0]
    reward, metrics = env.score_completion(
        instance=inst,
        tokenizer=tokenizer,
        completion_token_ids=tokenizer.encode(str(inst.payload["correct_label"])),
    )

    assert reward == 1.0
    assert metrics["parse_success"] == 1.0
