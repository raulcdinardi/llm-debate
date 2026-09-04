from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import types

from llm_local_rl import debate_parity as new


def _source_compatible_training_data(data) -> list[dict]:
    rows = []
    for datum in data:
        row = asdict(datum)
        row.pop("completion_logprob_mask", None)
        rows.append(row)
    return rows


def _import_source_modules():
    candidates = [
        Path("/mnt/c/Users/raulc/Desktop/llm-debate/src"),
        Path("/home/vm02/Desktop/raul/mnt/c/Users/raulc/Desktop/llm-debate/src"),
        Path("/home/vm02/Desktop/raul/llm-debate-4070/src"),
    ]
    source_repo = next((path for path in candidates if path.exists()), None)
    if source_repo is None:
        raise RuntimeError(f"Source repo path missing; tried {candidates}")
    package = types.ModuleType("tinker_debate")
    package.__path__ = [str(source_repo / "tinker_debate")]
    sys.modules["tinker_debate"] = package

    prompts_spec = importlib.util.spec_from_file_location(
        "tinker_debate.prompts",
        source_repo / "tinker_debate" / "prompts.py",
    )
    prompts_module = importlib.util.module_from_spec(prompts_spec)
    assert prompts_spec.loader is not None
    sys.modules["tinker_debate.prompts"] = prompts_module
    prompts_spec.loader.exec_module(prompts_module)

    debate_types_spec = importlib.util.spec_from_file_location(
        "tinker_debate.debate_types",
        source_repo / "tinker_debate" / "debate_types.py",
    )
    debate_types_module = importlib.util.module_from_spec(debate_types_spec)
    assert debate_types_spec.loader is not None
    sys.modules["tinker_debate.debate_types"] = debate_types_module
    debate_types_spec.loader.exec_module(debate_types_module)

    return debate_types_module


def _make_transition(old_types, new_module, *, prompt_tokens: list[int], completion_tokens: list[int], logprobs: list[float], round_num: int):
    old_transition = old_types.Transition(
        prompt_tokens=list(prompt_tokens),
        completion_tokens=list(completion_tokens),
        completion_logprobs=list(logprobs),
        round_num=round_num,
    )
    new_transition = new_module.Transition(
        prompt_tokens=list(prompt_tokens),
        completion_tokens=list(completion_tokens),
        completion_logprobs=list(logprobs),
        round_num=round_num,
    )
    return old_transition, new_transition


def _make_debate_pair():
    old_types = _import_source_modules()
    config_old = old_types.DebateConfig.cheap(chat_preamble="PREAMBLE\n")
    config_new = new.DebateConfig.cheap(chat_preamble="PREAMBLE\n")

    old_t1a, new_t1a = _make_transition(old_types, new, prompt_tokens=[10, 11], completion_tokens=[12, 13], logprobs=[-0.1, -0.2], round_num=1)
    old_t2a, new_t2a = _make_transition(old_types, new, prompt_tokens=[10, 11, 12, 13, 20, 21], completion_tokens=[22, 23], logprobs=[-0.3, -0.4], round_num=2)
    old_t3a, new_t3a = _make_transition(old_types, new, prompt_tokens=[10, 11, 12, 13, 20, 21, 22, 23, 30], completion_tokens=[31, 32], logprobs=[-0.5, -0.6], round_num=3)

    old_t1b, new_t1b = _make_transition(old_types, new, prompt_tokens=[10, 11], completion_tokens=[14, 15], logprobs=[-0.7, -0.8], round_num=1)
    old_t2b, new_t2b = _make_transition(old_types, new, prompt_tokens=[10, 11, 14, 15, 24, 25], completion_tokens=[26, 27], logprobs=[-0.9, -1.0], round_num=2)
    old_t3b, new_t3b = _make_transition(old_types, new, prompt_tokens=[10, 11, 14, 15, 24, 25, 26, 27, 34], completion_tokens=[35, 36], logprobs=[-1.1, -1.2], round_num=3)

    old_a = old_types.DebateTrajectory(agent="A", transitions=[old_t1a, old_t2a, old_t3a], frozen_solution="A_SOL")
    old_b = old_types.DebateTrajectory(agent="B", transitions=[old_t1b, old_t2b, old_t3b], frozen_solution="B_SOL")
    new_a = new.DebateTrajectory(agent="A", transitions=[new_t1a, new_t2a, new_t3a], frozen_solution="A_SOL")
    new_b = new.DebateTrajectory(agent="B", transitions=[new_t1b, new_t2b, new_t3b], frozen_solution="B_SOL")

    old_debate = old_types.DebateResult(
        question="Q1",
        ground_truth="A_SOL",
        trajectory_a=old_a,
        trajectory_b=old_b,
        verdict="A",
        judge_reasoning="judge",
    )
    new_debate = new.DebateResult(
        question="Q1",
        ground_truth="A_SOL",
        trajectory_a=new_a,
        trajectory_b=new_b,
        verdict="A",
        judge_reasoning="judge",
    )
    return old_types, config_old, config_new, old_debate, new_debate


def test_prompt_builders_match_source_exactly() -> None:
    config_new = new.DebateConfig.cheap(chat_preamble="PREAMBLE\n")

    expected_r1 = (
        "PREAMBLE\n"
        "<|im_start|>system\n"
        + config_new.system_propose
        + "\n<|im_end|>\n<|im_start|>user\nQuestion?\n<|im_end|>\n<|im_start|>assistant\n"
    )
    expected_r2 = (
        "<|im_end|>\n<|im_start|>user\n"
        + config_new.r2_user_template.format(opponent_r1="<SOLUTION>A</SOLUTION>")
        + "\n<|im_end|>\n<|im_start|>assistant\n"
    )
    expected_r3 = (
        "<|im_end|>\n<|im_start|>user\n"
        + config_new.r3_user_template.format(
            round_num=3,
            opponent_round=2,
            opponent_response="Counterargument",
        )
        + "\n<|im_end|>\n<|im_start|>assistant\n"
    )

    assert new.build_r1_prompt("Question?", config_new) == expected_r1
    assert new.build_r2_continuation("<SOLUTION>A</SOLUTION>", config_new) == expected_r2
    assert new.build_r3_continuation("Counterargument", config_new) == expected_r3


def test_assemble_training_data_r1_r23_matches_source_exactly() -> None:
    old_types, _config_old, _config_new, old_debate, new_debate = _make_debate_pair()

    old_data = old_types.assemble_training_data_r1_r23(
        [old_debate],
        r1_reward_fn=lambda traj, _debate: 1.0 if traj.agent == "A" else 0.0,
        r23_reward=0.5,
        r23_symmetric=True,
    )
    new_data = new.assemble_training_data_r1_r23(
        [new_debate],
        r1_reward_fn=lambda traj, _debate: 1.0 if traj.agent == "A" else 0.0,
        r23_reward=0.5,
        r23_symmetric=True,
    )

    assert _source_compatible_training_data(new_data) == [asdict(x) for x in old_data]
    assert all(
        datum.completion_logprob_mask == [1, 1, 0, 0, 1, 1, 0, 1, 1]
        for datum in new_data
    )


def test_assemble_training_data_r1_r2_matches_source_exactly() -> None:
    old_types, _config_old, _config_new, old_debate, new_debate = _make_debate_pair()

    old_two = old_types.DebateResult(
        question=old_debate.question,
        ground_truth=old_debate.ground_truth,
        trajectory_a=old_types.DebateTrajectory(agent="A", transitions=old_debate.trajectory_a.transitions[:2], frozen_solution=old_debate.trajectory_a.frozen_solution),
        trajectory_b=old_types.DebateTrajectory(agent="B", transitions=old_debate.trajectory_b.transitions[:2], frozen_solution=old_debate.trajectory_b.frozen_solution),
        verdict=old_debate.verdict,
        judge_reasoning=old_debate.judge_reasoning,
    )
    new_two = new.DebateResult(
        question=new_debate.question,
        ground_truth=new_debate.ground_truth,
        trajectory_a=new.DebateTrajectory(agent="A", transitions=new_debate.trajectory_a.transitions[:2], frozen_solution=new_debate.trajectory_a.frozen_solution),
        trajectory_b=new.DebateTrajectory(agent="B", transitions=new_debate.trajectory_b.transitions[:2], frozen_solution=new_debate.trajectory_b.frozen_solution),
        verdict=new_debate.verdict,
        judge_reasoning=new_debate.judge_reasoning,
    )

    old_data = old_types.assemble_training_data_r1_r2(
        [old_two],
        r1_reward_fn=lambda traj, _debate: 1.0 if traj.agent == "A" else 0.0,
        r2_reward=0.5,
        r2_symmetric=False,
    )
    new_data = new.assemble_training_data_r1_r2(
        [new_two],
        r1_reward_fn=lambda traj, _debate: 1.0 if traj.agent == "A" else 0.0,
        r2_reward=0.5,
        r2_symmetric=False,
    )

    assert _source_compatible_training_data(new_data) == [asdict(x) for x in old_data]
    assert all(
        datum.completion_logprob_mask == [1, 1, 0, 0, 1, 1]
        for datum in new_data
    )


def test_assemble_training_data_r1_only_compare_matches_source_exactly() -> None:
    old_types, _config_old, _config_new, old_debate, new_debate = _make_debate_pair()

    old_result = old_types.assemble_training_data_r1_only_compare(
        [old_debate],
        r1_reward=0.25,
        r1_symmetric=True,
    )
    if isinstance(old_result, tuple):
        old_training_data, old_skipped = old_result
    else:
        old_training_data = old_result
        old_skipped = 0

    new_training_data, new_skipped = new.assemble_training_data_r1_only_compare(
        [new_debate],
        r1_reward=0.25,
        r1_symmetric=True,
    )

    assert (
        _source_compatible_training_data(new_training_data),
        new_skipped,
    ) == ([asdict(x) for x in old_training_data], old_skipped)
    assert all(
        datum.completion_logprob_mask == [1, 1]
        for datum in new_training_data
    )


def test_assemble_training_data_grpo_matches_source_exactly() -> None:
    old_types, _config_old, _config_new, old_debate, new_debate = _make_debate_pair()

    old_data = old_types.assemble_training_data_grpo([old_debate], reward_fn=lambda _traj, _debate: 1.5)
    new_data = new.assemble_training_data_grpo([new_debate], reward_fn=lambda _traj, _debate: 1.5)

    assert _source_compatible_training_data(new_data) == [asdict(x) for x in old_data]
    assert all(
        datum.completion_logprob_mask == [1, 1, 0, 0, 1, 1, 0, 1, 1]
        for datum in new_data
    )


def test_training_datum_to_train_example_preserves_exact_completion_fields() -> None:
    datum = new.TrainingDatum(
        prompt_tokens=[10, 11],
        completion_tokens=[12, 20, 21, 22],
        completion_logprobs=[-0.1, 0.0, -0.2, -0.3],
        completion_logprob_mask=[1, 0, 1, 1],
        completion_advantages=[0.25, 0.0, 0.25, 0.25],
        metadata={"k": "v"},
    )

    example = new.training_datum_to_train_example(datum=datum, adapter_name="debate")

    assert example.adapter_name == "debate"
    assert example.input_ids == [10, 11, 12, 20, 21]
    assert example.target_ids == [11, 12, 20, 21, 22]
    assert example.loss_mask == [0, 1, 1, 1, 1]
    assert example.behavior_logprob_mask == [0, 1, 0, 1, 1]
    assert example.old_logprobs == [0.0, -0.1, 0.0, -0.2, -0.3]
    assert example.advantages == [0.0, 0.25, 0.0, 0.25, 0.25]
    assert example.metadata == {"k": "v"}
