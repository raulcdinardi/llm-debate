from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import importlib
import math
from types import SimpleNamespace

import pytest

torch = importlib.import_module("torch") if importlib.util.find_spec("torch") is not None else None
peft_spec = importlib.util.find_spec("peft")
transformers_spec = importlib.util.find_spec("transformers")

if torch is None or peft_spec is None or transformers_spec is None:
    pytest.skip("trainer unit tests require torch, peft, and transformers", allow_module_level=True)

from llm_local_rl.trainer import (
    BehaviorPolicyLogprobMismatchError,
    MultiAdapterTrainer,
    TrainerConfig,
    _is_configured_adapter_parameter,
    _normalized_bernoulli_js_from_paired_correct_logprobs,
    _order_batch_for_minibatching,
    _order_paired_js_batch_for_minibatching,
    _pad_batch,
    _selected_lm_head_token_logprobs,
    _target_token_logprobs,
    _truncated_row_length,
)
from llm_local_rl.behavior_policy import BehaviorPolicySpec
from llm_local_rl.debate_parity import (
    DebateResult,
    DebateTrajectory,
    Transition,
    _merge_transition_pair_with_adv_values,
    training_datum_to_train_example,
)
from llm_local_rl.types import TrainExample
from llm_local_rl.soft_judge import bernoulli_js_divergence


def _base_model_path() -> str | None:
    import os

    return os.environ.get("LLM_LOCAL_RL_BASE_MODEL")


def _test_scratch_root() -> Path:
    root = Path.cwd() / ".tmp_test_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, *, input_ids, attention_mask):
        _ = attention_mask
        positive = torch.zeros((*input_ids.shape, 1), dtype=torch.float32, device=input_ids.device) + self.bias
        negative = torch.zeros((*input_ids.shape, 1), dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(logits=torch.cat([positive, negative], dim=-1))


class _TinyBackbone(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 64, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        _ = attention_mask, use_cache, return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyCausalModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 64, hidden_size: int = 8) -> None:
        super().__init__()
        self.model = _TinyBackbone(vocab_size=vocab_size, hidden_size=hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, *, input_ids, attention_mask):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden_states))

    def get_output_embeddings(self):
        return self.lm_head


def _fake_trainer(*, train_max_tokens: int = 0) -> MultiAdapterTrainer:
    trainer = object.__new__(MultiAdapterTrainer)
    trainer.config = TrainerConfig(
        base_model_path="/tmp/nonexistent_model_for_shape_only",
        device="cpu",
        torch_dtype="float32",
        learning_rate=0.0,
        train_max_tokens=train_max_tokens,
    )
    trainer.compute_device = "cpu"
    trainer.current_device = "cpu"
    trainer.tokenizer = _FakeTokenizer()
    trainer.model = _FakeModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    trainer._mem_rec = None
    trainer.wake_up = lambda: None
    trainer.set_adapter = lambda adapter_name: None
    return trainer


def _tiny_causal_trainer(*, backend: str, learning_rate: float = 0.0) -> MultiAdapterTrainer:
    trainer = object.__new__(MultiAdapterTrainer)
    trainer.config = TrainerConfig(
        base_model_path="/tmp/nonexistent_model_for_shape_only",
        device="cpu",
        torch_dtype="float32",
        learning_rate=learning_rate,
        train_logprob_backend=backend,
        train_minibatch_size=1,
        behavior_policy=BehaviorPolicySpec(temperature=0.8),
        on_policy_logprob_abs_tol=1e-6,
    )
    trainer.compute_device = "cpu"
    trainer.current_device = "cpu"
    trainer.tokenizer = _FakeTokenizer()
    trainer.model = _TinyCausalModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=learning_rate)
    trainer._mem_rec = None
    trainer.wake_up = lambda: None
    trainer.set_adapter = lambda adapter_name: None
    return trainer


def _real_merged_debate_example() -> TrainExample:
    first = Transition(
        prompt_tokens=[1, 2],
        completion_tokens=[3, 4],
        completion_logprobs=[-0.1, -0.2],
        round_num=2,
    )
    second = Transition(
        prompt_tokens=[1, 2, 3, 4, 5],
        completion_tokens=[6, 7],
        completion_logprobs=[-0.3, -0.4],
        round_num=3,
    )
    trajectory = DebateTrajectory(
        agent="A",
        transitions=[first, second],
        frozen_solution="fixed",
    )
    debate = DebateResult(
        question="Q",
        ground_truth=None,
        trajectory_a=trajectory,
        trajectory_b=DebateTrajectory(
            agent="B",
            transitions=[first, second],
            frozen_solution="other",
        ),
        verdict="A",
        judge_reasoning="judge",
    )
    datum = _merge_transition_pair_with_adv_values(
        debate=debate,
        traj=trajectory,
        first=first,
        second=second,
        first_adv_value=0.25,
        second_adv_value=0.25,
        metadata={"rounds_merged": 2},
    )
    return training_datum_to_train_example(datum=datum, adapter_name="debate")


def test_pad_batch_rejects_overlength_instead_of_truncating() -> None:
    example = TrainExample(
        adapter_name="shared",
        input_ids=[10, 11, 12, 13, 14],
        target_ids=[11, 12, 13, 14, 15],
        loss_mask=[0, 0, 0, 1, 1],
        behavior_logprob_mask=[0, 0, 0, 1, 1],
        old_logprobs=[0.0, 0.0, 0.0, -0.3, -0.2],
        advantages=[0.0, 0.0, 0.0, 0.5, 0.5],
    )

    with pytest.raises(ValueError, match="Over-length TrainExample"):
        _pad_batch(batch=[example], pad_token_id=0, device="cpu", max_tokens=3)


def test_truncated_row_length_matches_pad_batch_effective_length() -> None:
    example = TrainExample(
        adapter_name="shared",
        input_ids=[10, 11, 12, 13, 14],
        target_ids=[11, 12, 13, 14, 15],
        loss_mask=[0, 0, 0, 1, 1],
        behavior_logprob_mask=[0, 0, 0, 1, 1],
        old_logprobs=[0.0, 0.0, 0.0, -0.3, -0.2],
        advantages=[0.0, 0.0, 0.0, 0.5, 0.5],
    )

    assert _truncated_row_length(example=example, max_tokens=0) == 5
    with pytest.raises(ValueError, match="Over-length examples"):
        _truncated_row_length(example=example, max_tokens=3)


def test_train_batch_loss_is_normalized_by_kept_sample_count() -> None:
    old_logprob = -math.log(2.0)
    example = TrainExample(
        adapter_name="shared",
        input_ids=[0],
        target_ids=[1],
        loss_mask=[1],
        behavior_logprob_mask=[1],
        old_logprobs=[old_logprob],
        advantages=[1.0],
    )

    one_metrics = _fake_trainer().train_batch(adapter_name="shared", batch=[example])
    two_metrics = _fake_trainer().train_batch(adapter_name="shared", batch=[example, example])

    assert one_metrics["loss"] == pytest.approx(-1.0, abs=1e-6)
    assert two_metrics["loss"] == pytest.approx(-1.0, abs=1e-6)
    assert two_metrics["num_examples"] == 2.0
    assert two_metrics["num_dropped_overlength"] == 0.0


def test_supervised_label_ce_js_uses_direct_labels_and_zero_js_for_equal_distributions() -> None:
    examples = [
        TrainExample(
            adapter_name="judge",
            input_ids=[0],
            target_ids=[0],
            loss_mask=[1],
            behavior_logprob_mask=[0],
            old_logprobs=[0.0],
            advantages=[0.0],
            metadata={
                "training_objective": "supervised_label_ce_js",
                "behavior_policy_allowed_token_ids": [0, 1],
                "judge_coherence_pair_id": "pair-0",
                "judge_coherence_pair_member": member,
            },
        )
        for member in ("forward", "reverse")
    ]

    metrics = _fake_trainer().train_batch(
        adapter_name="judge",
        batch=examples,
        objective="supervised_label_ce_js",
    )

    assert metrics["training_objective"] == "supervised_label_ce_js"
    assert metrics["loss"] == pytest.approx(math.log(2.0), abs=1e-6)
    assert metrics["supervised_label_nll"] == pytest.approx(math.log(2.0), abs=1e-6)
    assert metrics["supervised_correct_label_probability_mean"] == pytest.approx(0.5)
    assert metrics["judge_coherence_js"] == pytest.approx(0.0)
    assert metrics["judge_coherence_reliability"] == pytest.approx(1.0)
    assert metrics["judge_coherence_pair_count"] == 1.0
    assert metrics["completion_tokens_checked"] == 0.0


def test_unsupervised_js_has_no_label_ce_term() -> None:
    examples = [
        TrainExample(
            adapter_name="judge",
            input_ids=[0],
            target_ids=[target],
            loss_mask=[1],
            behavior_logprob_mask=[0],
            old_logprobs=[0.0],
            advantages=[0.0],
            metadata={
                "training_objective": "unsupervised_js",
                "behavior_policy_allowed_token_ids": [0, 1],
                "judge_coherence_pair_id": "pair-0",
                "judge_coherence_pair_member": member,
            },
        )
        for member, target in (("forward", 0), ("reverse", 1))
    ]

    trainer = _fake_trainer()
    with torch.no_grad():
        trainer.model.bias.fill_(1.0)
    metrics = trainer.train_batch(
        adapter_name="judge",
        batch=examples,
        objective="unsupervised_js",
    )

    expected_js = bernoulli_js_divergence(
        1.0 / (1.0 + math.exp(-1.0)),
        1.0 / (1.0 + math.exp(1.0)),
    ) / math.log(2.0)
    assert metrics["training_objective"] == "unsupervised_js"
    assert metrics["loss"] == pytest.approx(expected_js, abs=1e-6)
    assert metrics["supervised_label_nll"] == 0.0
    assert metrics["supervised_label_accuracy"] == 0.0
    assert metrics["judge_coherence_js"] == pytest.approx(expected_js, abs=1e-6)
    assert metrics["judge_coherence_reliability"] == pytest.approx(1.0 - expected_js)
    assert metrics["judge_coherence_pair_count"] == 1.0


def test_direct_js_is_bounded_symmetric_and_differentiable() -> None:
    logprobs = torch.tensor([math.log(0.9), math.log(0.1)], requires_grad=True)
    js = _normalized_bernoulli_js_from_paired_correct_logprobs(logprobs)
    swapped = _normalized_bernoulli_js_from_paired_correct_logprobs(logprobs.flip(0))

    assert js.shape == (1,)
    assert js.item() == pytest.approx(
        bernoulli_js_divergence(0.9, 0.1) / math.log(2.0), abs=1e-6
    )
    assert 0.0 < js.item() < 1.0
    assert js.item() == pytest.approx(swapped.item())
    js.sum().backward()
    assert logprobs.grad is not None
    assert torch.isfinite(logprobs.grad).all()
    assert torch.count_nonzero(logprobs.grad).item() == 2


def test_direct_js_pair_ordering_keeps_members_adjacent_when_length_bucketed() -> None:
    def row(pair_id: str, member: str, length: int) -> TrainExample:
        return TrainExample(
            adapter_name="judge",
            input_ids=[0] * length,
            target_ids=[0] * length,
            loss_mask=([0] * (length - 1)) + [1],
            behavior_logprob_mask=[0] * length,
            old_logprobs=[0.0] * length,
            advantages=[0.0] * length,
            metadata={
                "training_objective": "supervised_label_ce_js",
                "behavior_policy_allowed_token_ids": [0, 1],
                "judge_coherence_pair_id": pair_id,
                "judge_coherence_pair_member": member,
            },
        )

    ordered = _order_paired_js_batch_for_minibatching(
        batch=[
            row("long", "reverse", 6),
            row("short", "forward", 2),
            row("long", "forward", 5),
            row("short", "reverse", 3),
        ],
        length_bucket_batches=True,
    )
    assert [example.metadata["judge_coherence_pair_id"] for example in ordered] == [
        "short",
        "short",
        "long",
        "long",
    ]
    assert [example.metadata["judge_coherence_pair_member"] for example in ordered] == [
        "forward",
        "reverse",
        "forward",
        "reverse",
    ]

    with pytest.raises(ValueError, match="one forward and one reverse"):
        _order_paired_js_batch_for_minibatching(
            batch=[row("broken", "forward", 2)]
        )


def test_train_batch_drops_overlength_samples_and_reports_counter() -> None:
    old_logprob = -math.log(2.0)
    kept = TrainExample(
        adapter_name="shared",
        input_ids=[0, 0],
        target_ids=[1, 1],
        loss_mask=[0, 1],
        behavior_logprob_mask=[0, 1],
        old_logprobs=[0.0, old_logprob],
        advantages=[0.0, 1.0],
    )
    overlength = TrainExample(
        adapter_name="shared",
        input_ids=[0, 0, 0],
        target_ids=[1, 1, 1],
        loss_mask=[0, 1, 1],
        behavior_logprob_mask=[0, 1, 1],
        old_logprobs=[0.0, old_logprob, old_logprob],
        advantages=[0.0, 1.0, 1.0],
    )

    metrics = _fake_trainer(train_max_tokens=2).train_batch(
        adapter_name="shared",
        batch=[kept, overlength],
    )

    assert metrics["num_input_examples"] == 2.0
    assert metrics["num_examples"] == 1.0
    assert metrics["num_dropped_overlength"] == 1.0
    assert metrics["num_forward_input_tokens"] == 2.0


def test_train_batch_raises_when_all_samples_are_overlength() -> None:
    example = TrainExample(
        adapter_name="shared",
        input_ids=[0, 0, 0],
        target_ids=[1, 1, 1],
        loss_mask=[0, 1, 1],
        behavior_logprob_mask=[0, 1, 1],
        old_logprobs=[0.0, -math.log(2.0), -math.log(2.0)],
        advantages=[0.0, 1.0, 1.0],
    )

    with pytest.raises(ValueError, match="All train samples exceed train_max_tokens"):
        _fake_trainer(train_max_tokens=2).train_batch(adapter_name="shared", batch=[example])


def test_target_token_logprobs_match_full_log_softmax_gather() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260625)
    logits = torch.randn((3, 5, 17), generator=generator, dtype=torch.float32)
    target_ids = torch.randint(0, 17, (3, 5), generator=generator)

    expected = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    actual = _target_token_logprobs(logits=logits, target_ids=target_ids)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_target_token_logprobs_match_full_log_softmax_gather_with_small_chunks() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260628)
    logits = torch.randn((3, 7, 19), generator=generator, dtype=torch.float32)
    target_ids = torch.randint(0, 19, (3, 7), generator=generator)

    expected = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    actual = _target_token_logprobs(logits=logits, target_ids=target_ids, max_positions_per_chunk=4)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_target_token_logprobs_reconstruct_temperature_scaled_behavior_distribution() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260723)
    logits = torch.randn((2, 4, 13), generator=generator, dtype=torch.float32)
    target_ids = torch.randint(0, 13, (2, 4), generator=generator)

    expected = torch.log_softmax(logits / 0.8, dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    actual = _target_token_logprobs(
        logits=logits,
        target_ids=target_ids,
        behavior_temperature=0.8,
        max_positions_per_chunk=3,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_target_token_logprobs_reconstruct_allowed_token_normalization() -> None:
    logits = torch.tensor([[[9.0, 2.0, -1.0, 7.0]]], dtype=torch.float32)
    target_ids = torch.tensor([[2]])
    expected = torch.log_softmax(logits[0, 0, [1, 2]], dim=-1)[1]
    actual = _target_token_logprobs(
        logits=logits,
        target_ids=target_ids,
        allowed_token_ids_by_position=[(1, 2)],
    )
    assert actual[0, 0] == pytest.approx(float(expected), abs=1e-6)


def test_selective_lm_head_reconstructs_allowed_token_normalization() -> None:
    hidden_states = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    lm_head = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.tensor([[9.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [7.0, 0.0]])
        )
    target_ids = torch.tensor([[2]])
    actual, entropy = _selected_lm_head_token_logprobs(
        hidden_states=hidden_states,
        lm_head=lm_head,
        target_ids=target_ids,
        selected_positions=torch.tensor([[True]]),
        allowed_token_ids_by_selected_position=[(1, 2)],
    )
    restricted = torch.log_softmax(torch.tensor([2.0, -1.0]), dim=-1)
    expected_entropy = float((-(restricted.exp() * restricted).sum()).item())
    assert float(actual[0].detach()) == pytest.approx(float(restricted[1]), abs=1e-6)
    assert entropy == pytest.approx(expected_entropy, abs=1e-6)


def test_temperature_mismatch_negative_control_reproduces_inverse_temperature_slope() -> None:
    logits = torch.tensor([6.0, 2.0, -3.0, -8.0], dtype=torch.float32)
    raw = torch.log_softmax(logits, dim=-1)
    behavior = torch.log_softmax(logits / 0.8, dim=-1)

    rare_token_pair_slope = float(
        ((behavior[3] - behavior[2]) / (raw[3] - raw[2])).item()
    )

    assert rare_token_pair_slope == pytest.approx(1.25, abs=1e-6)
    assert not torch.allclose(raw, behavior, atol=1e-3)


def test_selected_lm_head_token_logprobs_match_full_lm_head_logits() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260629)
    hidden_states = torch.randn((2, 4, 5), generator=generator, dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(5, 11, bias=True)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn((11, 5), generator=generator, dtype=torch.float32))
        lm_head.bias.copy_(torch.randn((11,), generator=generator, dtype=torch.float32))
    target_ids = torch.randint(0, 11, (2, 4), generator=generator)
    trained_positions = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ],
        dtype=torch.bool,
    )

    full_logits = lm_head(hidden_states)
    full_logprobs = _target_token_logprobs(logits=full_logits, target_ids=target_ids)
    expected_logprobs = full_logprobs[trained_positions]
    expected_log_probs = torch.log_softmax(full_logits[trained_positions].float(), dim=-1)
    expected_entropy = float(
        (-(expected_log_probs.exp() * expected_log_probs).sum(dim=-1)).sum().detach().cpu().item()
    )

    actual_logprobs, actual_entropy = _selected_lm_head_token_logprobs(
        hidden_states=hidden_states,
        lm_head=lm_head,
        target_ids=target_ids,
        selected_positions=trained_positions,
        max_positions_per_chunk=2,
    )

    assert torch.allclose(actual_logprobs, expected_logprobs, atol=1e-6)
    assert actual_entropy == pytest.approx(expected_entropy, abs=1e-6)


def test_selected_lm_head_token_logprobs_handles_empty_trained_positions() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260630)
    hidden_states = torch.randn((2, 4, 5), generator=generator, dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(5, 11, bias=False)
    target_ids = torch.randint(0, 11, (2, 4), generator=generator)
    trained_positions = torch.zeros((2, 4), dtype=torch.bool)

    actual_logprobs, actual_entropy = _selected_lm_head_token_logprobs(
        hidden_states=hidden_states,
        lm_head=lm_head,
        target_ids=target_ids,
        selected_positions=trained_positions,
    )

    assert actual_logprobs.shape == (0,)
    assert actual_logprobs.dtype == torch.float32
    assert actual_entropy == 0.0


def test_selective_lm_head_matches_full_logits_at_behavior_temperature() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260724)
    hidden_states = torch.randn((2, 3, 5), generator=generator, dtype=torch.float32)
    lm_head = torch.nn.Linear(5, 11, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn((11, 5), generator=generator))
    target_ids = torch.randint(0, 11, (2, 3), generator=generator)
    selected_positions = torch.tensor(
        [[True, False, True], [False, True, True]],
        dtype=torch.bool,
    )

    full_logits = lm_head(hidden_states)
    expected = _target_token_logprobs(
        logits=full_logits,
        target_ids=target_ids,
        behavior_temperature=0.8,
    )[selected_positions]
    actual, _entropy = _selected_lm_head_token_logprobs(
        hidden_states=hidden_states,
        lm_head=lm_head,
        target_ids=target_ids,
        selected_positions=selected_positions,
        behavior_temperature=0.8,
        max_positions_per_chunk=2,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_train_batch_fails_before_backward_or_optimizer_on_contract_violation() -> None:
    trainer = _fake_trainer()
    trainer.config = TrainerConfig(
        base_model_path="/tmp/nonexistent_model_for_shape_only",
        device="cpu",
        torch_dtype="float32",
        learning_rate=0.1,
        behavior_policy=BehaviorPolicySpec(temperature=0.8),
        on_policy_logprob_abs_tol=1e-6,
    )
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    step_calls = 0
    original_step = trainer.optimizer.step

    def tracked_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = tracked_step
    before = trainer.model.bias.detach().clone()
    example = TrainExample(
        adapter_name="shared",
        input_ids=[0],
        target_ids=[1],
        loss_mask=[1],
        behavior_logprob_mask=[1],
        old_logprobs=[-0.1],
        advantages=[1.0],
    )

    with pytest.raises(BehaviorPolicyLogprobMismatchError, match="before PPO ratio/backward"):
        trainer.train_batch(adapter_name="shared", batch=[example])

    assert step_calls == 0
    assert trainer.model.bias.grad is None
    assert torch.equal(trainer.model.bias.detach(), before)


def test_train_batch_warn_only_records_violation_and_updates() -> None:
    trainer = _fake_trainer()
    trainer.config = TrainerConfig(
        base_model_path="/tmp/nonexistent_model_for_shape_only",
        device="cpu",
        torch_dtype="float32",
        learning_rate=0.1,
        behavior_policy=BehaviorPolicySpec(temperature=0.8),
        on_policy_logprob_abs_tol=1e-6,
        on_policy_logprob_warn_only=True,
    )
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    before = trainer.model.bias.detach().clone()
    example = TrainExample(
        adapter_name="shared",
        input_ids=[0],
        target_ids=[1],
        loss_mask=[1],
        behavior_logprob_mask=[1],
        old_logprobs=[-0.1],
        advantages=[1.0],
    )

    metrics = trainer.train_batch(adapter_name="shared", batch=[example])

    assert metrics["on_policy_logprob_violations"] == 1.0
    assert not torch.equal(trainer.model.bias.detach(), before)


@pytest.mark.parametrize("backend", ["full_logits", "selective_lm_head"])
def test_real_merged_debate_example_checks_only_sampled_tokens(backend: str) -> None:
    trainer = _tiny_causal_trainer(backend=backend)
    example = _real_merged_debate_example()
    assert example.loss_mask[-5:] == [1, 1, 1, 1, 1]
    assert example.behavior_logprob_mask[-5:] == [1, 1, 0, 1, 1]

    current_rows = trainer.compute_logprobs(adapter_name="debate", batch=[example])
    example = replace(
        example,
        old_logprobs=[
            current if has_behavior_logprob else old
            for current, old, has_behavior_logprob in zip(
                current_rows[0],
                example.old_logprobs,
                example.behavior_logprob_mask,
                strict=True,
            )
        ],
    )

    metrics = trainer.train_batch(adapter_name="debate", batch=[example])

    assert metrics["completion_tokens_checked"] == 4.0
    assert metrics["injected_loss_mask_tokens_skipped"] == 1.0
    assert metrics["on_policy_logprob_violations"] == 0.0


@pytest.mark.parametrize("backend", ["full_logits", "selective_lm_head"])
def test_real_merged_debate_mismatch_fails_before_update_for_both_backends(
    backend: str,
) -> None:
    trainer = _tiny_causal_trainer(backend=backend, learning_rate=0.1)
    example = _real_merged_debate_example()
    current_rows = trainer.compute_logprobs(adapter_name="debate", batch=[example])
    old_logprobs = [
        current if has_behavior_logprob else old
        for current, old, has_behavior_logprob in zip(
            current_rows[0],
            example.old_logprobs,
            example.behavior_logprob_mask,
            strict=True,
        )
    ]
    first_sampled_position = example.behavior_logprob_mask.index(1)
    old_logprobs[first_sampled_position] += 0.1
    example = replace(example, old_logprobs=old_logprobs)
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    step_calls = 0
    original_step = trainer.optimizer.step

    def tracked_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = tracked_step

    with pytest.raises(BehaviorPolicyLogprobMismatchError, match="before PPO ratio/backward"):
        trainer.train_batch(adapter_name="debate", batch=[example])

    assert step_calls == 0
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert all(
        torch.equal(parameter.detach(), before[name])
        for name, parameter in trainer.model.named_parameters()
    )


@pytest.mark.parametrize("backend", ["full_logits", "selective_lm_head"])
def test_second_minibatch_mismatch_discards_first_minibatch_gradients(
    backend: str,
) -> None:
    trainer = _tiny_causal_trainer(backend=backend, learning_rate=0.1)
    first = _real_merged_debate_example()
    second = replace(_real_merged_debate_example(), metadata={"case": "second"})
    current_rows = trainer.compute_logprobs(
        adapter_name="debate",
        batch=[first, second],
    )

    def with_current_old(example: TrainExample, row: list[float]) -> TrainExample:
        return replace(
            example,
            old_logprobs=[
                current if has_behavior_logprob else old
                for current, old, has_behavior_logprob in zip(
                    row,
                    example.old_logprobs,
                    example.behavior_logprob_mask,
                    strict=True,
                )
            ],
        )

    first = with_current_old(first, current_rows[0])
    second = with_current_old(second, current_rows[1])
    second_old_logprobs = list(second.old_logprobs)
    second_old_logprobs[second.behavior_logprob_mask.index(1)] += 0.1
    second = replace(second, old_logprobs=second_old_logprobs)
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    step_calls = 0
    original_step = trainer.optimizer.step

    def tracked_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = tracked_step

    with pytest.raises(BehaviorPolicyLogprobMismatchError, match="minibatch_start=1"):
        trainer.train_batch(adapter_name="debate", batch=[first, second])

    assert step_calls == 0
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert all(
        torch.equal(parameter.detach(), before[name])
        for name, parameter in trainer.model.named_parameters()
    )


def test_trainer_config_records_moe_target_parameters() -> None:
    config = TrainerConfig(
        base_model_path="/tmp/nonexistent_model_for_shape_only",
        target_parameters=("experts.gate_up_proj", "experts.down_proj"),
    )

    assert config.target_parameters == ("experts.gate_up_proj", "experts.down_proj")


def test_configured_adapter_parameter_detection_catches_inactive_split_adapter() -> None:
    assert _is_configured_adapter_parameter(
        parameter_name="base_model.model.layers.0.self_attn.q_proj.lora_A.debate.weight",
        adapter_names=("solution", "debate", "judge"),
    )
    assert not _is_configured_adapter_parameter(
        parameter_name="base_model.model.layers.0.self_attn.q_proj.base_layer.weight",
        adapter_names=("solution", "debate", "judge"),
    )


def test_length_bucketed_minibatches_sort_by_effective_length() -> None:
    examples = [
        TrainExample("shared", [1] * 8, [1] * 8, [1] * 8, [1] * 8, [0.0] * 8, [1.0] * 8),
        TrainExample("shared", [1] * 3, [1] * 3, [1] * 3, [1] * 3, [0.0] * 3, [1.0] * 3),
        TrainExample("shared", [1] * 6, [1] * 6, [1] * 6, [1] * 6, [0.0] * 6, [1.0] * 6),
        TrainExample("shared", [1] * 2, [1] * 2, [1] * 2, [1] * 2, [0.0] * 2, [1.0] * 2),
    ]

    ordered = _order_batch_for_minibatching(
        batch=examples,
        max_tokens=0,
        length_bucket_batches=True,
    )

    assert [len(example.input_ids) for example in ordered] == [2, 3, 6, 8]


def test_unbucketed_minibatches_preserve_order_without_aliasing() -> None:
    examples = [
        TrainExample("shared", [1] * 8, [1] * 8, [1] * 8, [1] * 8, [0.0] * 8, [1.0] * 8),
        TrainExample("shared", [1] * 3, [1] * 3, [1] * 3, [1] * 3, [0.0] * 3, [1.0] * 3),
        TrainExample("shared", [1] * 6, [1] * 6, [1] * 6, [1] * 6, [0.0] * 6, [1.0] * 6),
    ]

    ordered = _order_batch_for_minibatching(
        batch=examples,
        max_tokens=0,
        length_bucket_batches=False,
    )

    assert ordered == examples
    assert ordered is not examples


def test_length_bucketed_minibatches_use_truncated_effective_length() -> None:
    examples = [
        TrainExample("shared", [1] * 100, [1] * 100, [1] * 100, [1] * 100, [0.0] * 100, [1.0] * 100),
        TrainExample("shared", [1] * 8, [1] * 8, [1] * 8, [1] * 8, [0.0] * 8, [1.0] * 8),
        TrainExample("shared", [1] * 60, [1] * 60, [1] * 60, [1] * 60, [0.0] * 60, [1.0] * 60),
    ]

    ordered = _order_batch_for_minibatching(
        batch=examples,
        max_tokens=32,
        length_bucket_batches=True,
    )

    assert [len(example.input_ids) for example in ordered] == [8, 100, 60]


def test_trainer_smoke_requires_model_env() -> None:
    if torch is None:
        pytest.skip("torch is unavailable")
    path = _base_model_path()
    if path is None:
        pytest.skip("LLM_LOCAL_RL_BASE_MODEL is not configured")
    trainer = MultiAdapterTrainer(
        config=TrainerConfig(
            base_model_path=path,
            adapter_names=("shared", "solution"),
            device="cpu",
            torch_dtype="float32",
        )
    )
    assert trainer.tokenizer.pad_token_id is not None


def test_adapter_snapshot_keys_disjoint_when_split() -> None:
    if torch is None:
        pytest.skip("torch is unavailable")
    path = _base_model_path()
    if path is None:
        pytest.skip("LLM_LOCAL_RL_BASE_MODEL is not configured")
    trainer = MultiAdapterTrainer(
        config=TrainerConfig(
            base_model_path=path,
            adapter_names=("solution", "debate"),
            device="cpu",
            torch_dtype="float32",
        )
    )
    solution_keys = set(trainer.adapter_parameter_snapshot(adapter_name="solution"))
    debate_keys = set(trainer.adapter_parameter_snapshot(adapter_name="debate"))
    assert solution_keys
    assert debate_keys
    assert solution_keys.isdisjoint(debate_keys)


def test_save_and_reload_adapter_snapshot_preserves_weights() -> None:
    if torch is None:
        pytest.skip("torch is unavailable")
    path = _base_model_path()
    if path is None:
        pytest.skip("LLM_LOCAL_RL_BASE_MODEL is not configured")
    trainer = MultiAdapterTrainer(
        config=TrainerConfig(
            base_model_path=path,
            adapter_names=("shared",),
            device="cpu",
            torch_dtype="float32",
        )
    )
    before = trainer.adapter_parameter_snapshot(adapter_name="shared")
    with tempfile.TemporaryDirectory(dir=_test_scratch_root()) as tmpdir:
        saved_dir = trainer.save_adapter(adapter_name="shared", output_dir=tmpdir)
        reloaded = MultiAdapterTrainer.from_saved_adapters(
            config=TrainerConfig(
                base_model_path=path,
                adapter_names=("shared",),
                device="cpu",
                torch_dtype="float32",
            ),
            adapter_dirs={"shared": saved_dir},
        )
    after = reloaded.adapter_parameter_snapshot(adapter_name="shared")
    assert before.keys() == after.keys()
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-7)


def test_train_minibatch_matches_full_batch_update() -> None:
    if torch is None:
        pytest.skip("torch is unavailable")
    path = _base_model_path()
    if path is None:
        pytest.skip("LLM_LOCAL_RL_BASE_MODEL is not configured")
    common_kwargs = dict(
        base_model_path=path,
        adapter_names=("shared",),
        device="cpu",
        torch_dtype="float32",
    )
    trainer_full = MultiAdapterTrainer(
        config=TrainerConfig(
            **common_kwargs,
            train_minibatch_size=0,
        )
    )
    with tempfile.TemporaryDirectory(dir=_test_scratch_root()) as tmpdir:
        saved_dir = trainer_full.save_adapter(adapter_name="shared", output_dir=tmpdir)
        trainer_mini = MultiAdapterTrainer.from_saved_adapters(
            config=TrainerConfig(
                **common_kwargs,
                train_minibatch_size=1,
                train_length_bucket_batches=True,
            ),
            adapter_dirs={"shared": saved_dir},
        )
    batch = [
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 3],
            target_ids=[2, 3, 4],
            loss_mask=[0, 1, 1],
            behavior_logprob_mask=[0, 1, 1],
            old_logprobs=[0.0, -1.0, -1.0],
            advantages=[0.0, 0.5, 0.5],
        ),
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 5],
            target_ids=[2, 5, 6],
            loss_mask=[0, 1, 1],
            behavior_logprob_mask=[0, 1, 1],
            old_logprobs=[0.0, -1.0, -1.0],
            advantages=[0.0, -0.25, -0.25],
        ),
    ]
    old_rows = trainer_full.compute_logprobs(adapter_name="shared", batch=batch)
    batch = [
        replace(
            example,
            old_logprobs=[
                current if has_behavior_logprob else 0.0
                for current, has_behavior_logprob in zip(
                    row,
                    example.behavior_logprob_mask,
                    strict=True,
                )
            ],
        )
        for example, row in zip(batch, old_rows, strict=True)
    ]
    trainer_full.train_batch(adapter_name="shared", batch=batch)
    trainer_mini.train_batch(adapter_name="shared", batch=batch)
    full_after = trainer_full.adapter_parameter_snapshot(adapter_name="shared")
    mini_after = trainer_mini.adapter_parameter_snapshot(adapter_name="shared")
    assert full_after.keys() == mini_after.keys()
    for key in full_after:
        assert torch.allclose(full_after[key], mini_after[key], atol=1e-4)


def test_selective_lm_head_train_batch_matches_full_logits_update() -> None:
    if torch is None:
        pytest.skip("torch is unavailable")
    path = _base_model_path()
    if path is None:
        pytest.skip("LLM_LOCAL_RL_BASE_MODEL is not configured")
    common_kwargs = dict(
        base_model_path=path,
        adapter_names=("shared",),
        device="cpu",
        torch_dtype="float32",
        learning_rate=1e-5,
        on_policy_logprob_check=True,
        on_policy_logprob_abs_tol=1e9,
    )
    trainer_full = MultiAdapterTrainer(
        config=TrainerConfig(
            **common_kwargs,
            train_logprob_backend="full_logits",
        )
    )
    with tempfile.TemporaryDirectory(dir=_test_scratch_root()) as tmpdir:
        saved_dir = trainer_full.save_adapter(adapter_name="shared", output_dir=tmpdir)
        trainer_selective = MultiAdapterTrainer.from_saved_adapters(
            config=TrainerConfig(
                **common_kwargs,
                train_logprob_backend="selective_lm_head",
            ),
            adapter_dirs={"shared": saved_dir},
        )
    batch = [
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 3, 4],
            target_ids=[2, 3, 4, 5],
            loss_mask=[0, 1, 1, 1],
            behavior_logprob_mask=[0, 1, 1, 1],
            old_logprobs=[0.0, -123.0, -1.0, -1.0],
            advantages=[0.0, 0.0, 0.5, -0.5],
        ),
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 5],
            target_ids=[2, 5, 6],
            loss_mask=[0, 1, 1],
            behavior_logprob_mask=[0, 1, 1],
            old_logprobs=[0.0, -1.0, -456.0],
            advantages=[0.0, -0.25, 0.0],
        ),
    ]

    full_metrics = trainer_full.train_batch(adapter_name="shared", batch=batch)
    selective_metrics = trainer_selective.train_batch(adapter_name="shared", batch=batch)

    for key in (
        "loss",
        "loss_per_trained_token",
        "approx_kl",
        "ratio_mean",
        "ratio_p95",
        "ratio_p99",
        "clipfrac",
        "entropy",
        "num_trained_tokens",
        "completion_tokens_checked",
        "trained_tokens_checked",
        "zero_advantage_loss_mask_tokens_checked",
    ):
        assert selective_metrics[key] == pytest.approx(full_metrics[key], abs=1e-5)
    assert selective_metrics["trained_tokens_checked"] == 3.0
    assert selective_metrics["zero_advantage_loss_mask_tokens_checked"] == 2.0
    assert selective_metrics["zero_advantage_loss_mask_tokens_skipped"] == 0.0
    assert full_metrics["lm_head_positions"] == full_metrics["num_padded_input_tokens"]
    assert selective_metrics["lm_head_positions"] == selective_metrics["completion_tokens_checked"]
    assert selective_metrics["lm_head_positions_avoided"] == (
        selective_metrics["num_padded_input_tokens"] - selective_metrics["completion_tokens_checked"]
    )

    full_after = trainer_full.adapter_parameter_snapshot(adapter_name="shared")
    selective_after = trainer_selective.adapter_parameter_snapshot(adapter_name="shared")
    assert full_after.keys() == selective_after.keys()
    for key in full_after:
        assert torch.allclose(full_after[key], selective_after[key], atol=1e-5)


def test_selective_lm_head_compute_logprobs_matches_full_logits_on_all_completion_positions() -> None:
    if torch is None:
        pytest.skip("torch is unavailable")
    path = _base_model_path()
    if path is None:
        pytest.skip("LLM_LOCAL_RL_BASE_MODEL is not configured")
    common_kwargs = dict(
        base_model_path=path,
        adapter_names=("shared",),
        device="cpu",
        torch_dtype="float32",
    )
    trainer_full = MultiAdapterTrainer(
        config=TrainerConfig(
            **common_kwargs,
            train_logprob_backend="full_logits",
        )
    )
    with tempfile.TemporaryDirectory(dir=_test_scratch_root()) as tmpdir:
        saved_dir = trainer_full.save_adapter(adapter_name="shared", output_dir=tmpdir)
        trainer_selective = MultiAdapterTrainer.from_saved_adapters(
            config=TrainerConfig(
                **common_kwargs,
                train_logprob_backend="selective_lm_head",
            ),
            adapter_dirs={"shared": saved_dir},
        )
    batch = [
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 3, 4],
            target_ids=[2, 3, 4, 5],
            loss_mask=[0, 1, 1, 1],
            behavior_logprob_mask=[0, 1, 1, 1],
            old_logprobs=[0.0, -123.0, -1.0, -1.0],
            advantages=[0.0, 0.0, 0.5, -0.5],
        ),
        TrainExample(
            adapter_name="shared",
            input_ids=[1, 2, 5],
            target_ids=[2, 5, 6],
            loss_mask=[0, 1, 1],
            behavior_logprob_mask=[0, 1, 1],
            old_logprobs=[0.0, -1.0, -456.0],
            advantages=[0.0, -0.25, 0.0],
        ),
    ]

    full_rows = trainer_full.compute_logprobs(adapter_name="shared", batch=batch)
    selective_rows = trainer_selective.compute_logprobs(adapter_name="shared", batch=batch)

    assert len(selective_rows) == len(full_rows)
    for example, full_row, selective_row in zip(batch, full_rows, selective_rows, strict=True):
        assert len(selective_row) == len(full_row)
        for idx, (full_logprob, selective_logprob) in enumerate(zip(full_row, selective_row, strict=True)):
            if example.loss_mask[idx]:
                assert selective_logprob == pytest.approx(full_logprob, abs=1e-5)
            else:
                assert selective_logprob == 0.0


def _ce_only_pair_rows():
    rows = []
    for pair, targets, lengths in [('p0', (0, 1), (1, 5)), ('p1', (0, 0), (4, 2)), ('p2', (1, 1), (3, 1))]:
        for member, target, length in zip(('forward', 'reverse'), targets, lengths, strict=True):
            rows.append(TrainExample(
                adapter_name='judge', input_ids=[0] * length,
                target_ids=[target] * length, loss_mask=[0] * (length - 1) + [1],
                behavior_logprob_mask=[0] * length, old_logprobs=[0.] * length,
                advantages=[0.] * length,
                metadata={'training_objective': 'supervised_label_ce_js',
                          'behavior_policy_allowed_token_ids': [0, 1],
                          'judge_coherence_pair_id': pair,
                          'judge_coherence_pair_member': member},
            ))
    return rows


@pytest.mark.parametrize('minibatch_size', [0, 2, 4])
@pytest.mark.parametrize('bucket', [False, True])
@pytest.mark.parametrize('seed', [0, 17])
def test_ce_only_keeps_pairs_in_physical_batches_and_accumulates_exact_ce_gradient(
    monkeypatch, minibatch_size, bucket, seed,
):
    import random
    import llm_local_rl.trainer as trainer_module
    rows = _ce_only_pair_rows()
    random.Random(seed).shuffle(rows)
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_minibatch_size=minibatch_size,
                             train_length_bucket_batches=bucket)
    with torch.no_grad():
        trainer.model.bias.fill_(1.)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.1)
    physical_batches = []
    pad_batch = trainer_module._pad_batch
    def capture_batch(**kwargs):
        physical_batches.append([(row.metadata['judge_coherence_pair_id'],
                                  row.metadata['judge_coherence_pair_member']) for row in kwargs['batch']])
        return pad_batch(**kwargs)
    monkeypatch.setattr(trainer_module, '_pad_batch', capture_batch)
    gradients = []
    step = trainer.optimizer.step
    def capture_step(*args, **kwargs):
        gradients.append(trainer.model.bias.grad.item())
        return step(*args, **kwargs)
    monkeypatch.setattr(trainer.optimizer, 'step', capture_step)
    metrics = trainer.train_batch(adapter_name='judge', batch=rows,
                                  objective='supervised_label_ce_js', judge_coherence_js_weight=0.)
    expected_batches = 1 if minibatch_size == 0 else math.ceil(6 / minibatch_size)
    assert len(physical_batches) == expected_batches
    for batch in physical_batches:
        assert len(batch) % 2 == 0
        for forward, reverse in zip(batch[::2], batch[1::2], strict=True):
            assert forward[0] == reverse[0]
            assert (forward[1], reverse[1]) == ('forward', 'reverse')
    assert sorted(pair for batch in physical_batches for pair, member in batch if member == 'forward') == ['p0', 'p1', 'p2']
    expected_gradient = torch.sigmoid(torch.tensor(1.)).item() - .5
    assert gradients == pytest.approx([expected_gradient], abs=1e-6)
    assert trainer.model.bias.item() == pytest.approx(1. - .1 * expected_gradient, abs=1e-6)
    assert metrics['loss'] == pytest.approx(math.log1p(math.exp(1.)) - .5, abs=1e-6)
    assert metrics['judge_coherence_js'] > 0.0  # Nonzero diagnostic contributes no gradient.
    assert metrics['judge_coherence_pair_count'] == 3


@pytest.mark.parametrize('invalid', ['missing_member', 'duplicate_member', 'overlength', 'odd_minibatch'])
def test_ce_only_rejects_pair_breaking_batches_before_optimizer_step(monkeypatch, invalid):
    rows = _ce_only_pair_rows()
    trainer = _fake_trainer()
    if invalid == 'missing_member':
        rows.pop()
    elif invalid == 'duplicate_member':
        rows.append(rows[0])
    elif invalid == 'overlength':
        trainer.config = replace(trainer.config, train_max_tokens=4)
    else:
        trainer.config = replace(trainer.config, train_minibatch_size=3)
    steps = []
    monkeypatch.setattr(trainer.optimizer, 'step', lambda: steps.append(True))
    with pytest.raises(ValueError):
        trainer.train_batch(adapter_name='judge', batch=rows,
                            objective='supervised_label_ce_js', judge_coherence_js_weight=0.)
    assert steps == []


@pytest.mark.parametrize('backend', ['full_logits', 'selective_lm_head'])
def test_ce_only_accumulated_update_matches_independent_ce_on_both_backends(backend):
    with torch.random.fork_rng():
        torch.manual_seed(7)
        trainer = _tiny_causal_trainer(backend=backend, learning_rate=.1)
    trainer.config = replace(trainer.config, train_minibatch_size=4,
                             train_length_bucket_batches=True, max_grad_norm=0.,
                             behavior_policy=BehaviorPolicySpec(temperature=1.0))
    rows = _ce_only_pair_rows()[::-1]
    initial = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
    losses = []
    for row in rows:
        inputs = torch.tensor([row.input_ids])
        logits = trainer.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
        losses.append(torch.nn.functional.cross_entropy(logits[:, -1, :2], torch.tensor([row.target_ids[-1]])))
    independent_ce = torch.stack(losses).mean()
    independent_ce.backward()
    expected = {name: initial[name] - .1 * value.grad for name, value in trainer.model.named_parameters()}
    metrics = trainer.train_batch(adapter_name='judge', batch=rows,
                                  objective='supervised_label_ce_js', judge_coherence_js_weight=0.)
    assert metrics['loss'] == pytest.approx(independent_ce.item(), abs=1e-6)
    assert metrics['num_train_minibatches'] == 2
    for name, value in trainer.model.named_parameters():
        torch.testing.assert_close(value, expected[name], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("physical_size", [0, 1, 2, 4])
def test_optimizer_batches_match_sequential_ppo_updates_with_frozen_rollout(physical_size):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_optimizer_batch_size=3,
                             train_minibatch_size=physical_size, max_grad_norm=0.0)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.7)
    # Deliberately large updates activate clipping against the ORIGINAL rollout policy.
    advantages = [1.0, 0.5, -0.2, 1.0, 0.3, -0.4, -1.0]
    batch = [TrainExample("shared", [0], [0], [1], [1], [-math.log(2)], [a])
             for a in advantages]
    expected = torch.tensor(0.0, requires_grad=True)
    reference_optimizer = torch.optim.SGD([expected], lr=0.7)
    observed_losses = []
    for start in range(0, len(batch), 3):
        adv = torch.tensor(advantages[start:start + 3])
        ratio = 2 * torch.sigmoid(expected)
        clipped = ratio.clamp(1 - trainer.config.ppo_clip_epsilon,
                              1 + trainer.config.ppo_clip_epsilon)
        loss = -torch.minimum(ratio * adv, clipped * adv).mean()
        observed_losses.append(loss.item() * len(adv) / len(batch))
        loss.backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()
    metrics = trainer.train_batch(adapter_name="shared", batch=batch)
    assert trainer.model.bias.item() == pytest.approx(expected.item(), abs=1e-7)
    assert metrics["loss"] == pytest.approx(sum(observed_losses), abs=1e-7)
    assert metrics["num_optimizer_steps"] == 3
    assert metrics["completion_tokens_checked"] == len(batch)
    assert metrics["on_policy_logprob_violations"] == 0
    assert metrics["ratio_max"] > 1.01
    assert all(row.old_logprobs == [-math.log(2)] for row in batch)
    assert [row.advantages[0] for row in batch] == advantages


@pytest.mark.parametrize("bad_advantage", [0.0, 1.0])
def test_optimizer_batches_check_last_rollout_row_before_first_update(bad_advantage):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_optimizer_batch_size=1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    batch = [TrainExample("shared", [0], [0], [1], [1], [-math.log(2)], [1.0]),
             TrainExample("shared", [0], [0], [1], [1], [-10.0], [bad_advantage])]
    with pytest.raises(BehaviorPolicyLogprobMismatchError):
        trainer.train_batch(adapter_name="shared", batch=batch)
    assert trainer.model.bias.item() == 0.0
    assert trainer.model.bias.grad is None


@pytest.mark.parametrize("physical_size", [0, 2, 4])
@pytest.mark.parametrize("bucket", [False, True])
def test_optimizer_batches_ce_preserve_pairs_and_match_individual_updates(physical_size, bucket):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_optimizer_batch_size=4,
                             train_minibatch_size=physical_size,
                             train_length_bucket_batches=bucket)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    baseline = _fake_trainer()
    baseline.optimizer = torch.optim.SGD(baseline.model.parameters(), lr=0.1)
    batch = _ce_only_pair_rows()
    ordered = _order_paired_js_batch_for_minibatching(
        batch=batch, max_tokens=0,
        length_bucket_batches=bucket and 0 < physical_size < len(batch))
    for start in range(0, len(ordered), 4):
        baseline.train_batch(adapter_name="judge", batch=ordered[start:start + 4],
                             objective="supervised_label_ce_js", judge_coherence_js_weight=0)
    metrics = trainer.train_batch(adapter_name="judge", batch=batch,
                                  objective="supervised_label_ce_js", judge_coherence_js_weight=0)
    assert metrics["num_optimizer_steps"] == 2
    assert metrics["judge_coherence_pair_count"] == 3
    assert trainer.model.bias.item() == pytest.approx(baseline.model.bias.item(), abs=1e-7)


def test_optimizer_batches_skip_zero_advantage_updates():
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_optimizer_batch_size=1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1, weight_decay=0.1)
    batch = [TrainExample("shared", [0], [0], [1], [1], [-math.log(2)], [a])
             for a in [0.0, 1.0, 0.0]]
    metrics = trainer.train_batch(adapter_name="shared", batch=batch)
    assert metrics["num_optimizer_steps"] == 1
    assert trainer.optimizer.state[trainer.model.bias]["step"] == 1
    assert metrics["completion_tokens_checked"] == 3


@pytest.mark.parametrize("size", [0, 7, 100])
def test_single_optimizer_batch_preserves_update_without_extra_forward(size):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_optimizer_batch_size=size, train_minibatch_size=2)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.compute_logprobs = lambda **kwargs: pytest.fail("single update must not preflight")
    batch = [TrainExample("shared", [0], [0], [1], [1], [-math.log(2)], [1.0]) for _ in range(7)]
    metrics = trainer.train_batch(adapter_name="shared", batch=batch)
    assert trainer.model.bias.item() == pytest.approx(0.05)
    assert metrics["num_optimizer_steps"] == 1
    assert metrics["num_train_minibatches"] == 4


@pytest.mark.parametrize("backend", ["full_logits", "selective_lm_head"])
def test_optimizer_batches_both_backends_match_independent_updates(backend):
    trainer = _tiny_causal_trainer(backend=backend, learning_rate=0.1)
    trainer.config = replace(trainer.config, train_optimizer_batch_size=2, max_grad_norm=0.1)
    batch = [TrainExample("shared", [i], [i + 1], [1], [1], [0.0], [a])
             for i, a in enumerate([1.0, -0.3, 0.5, -0.7, 0.4])]
    old = trainer.compute_logprobs(adapter_name="shared", batch=batch)
    batch = [replace(row, old_logprobs=lp) for row, lp in zip(batch, old, strict=True)]
    import copy
    expected = copy.deepcopy(trainer.model)
    optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    for start in range(0, len(batch), 2):
        rows = batch[start:start + 2]
        ids = torch.tensor([row.input_ids for row in rows])
        logits = expected(input_ids=ids, attention_mask=torch.ones_like(ids)).logits / 0.8
        logp = logits.log_softmax(-1).gather(-1, torch.tensor([[row.target_ids] for row in rows])).squeeze(-1)
        ratios = (logp - torch.tensor([row.old_logprobs for row in rows])).exp()
        advantages = torch.tensor([row.advantages for row in rows])
        epsilon = trainer.config.ppo_clip_epsilon
        loss = -torch.minimum(ratios * advantages, ratios.clamp(1-epsilon, 1+epsilon) * advantages).sum() / len(rows)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
    metrics = trainer.train_batch(adapter_name="shared", batch=batch)
    assert metrics["num_optimizer_steps"] == 3
    assert metrics["num_preflight_minibatches"] == 5
    assert metrics["on_policy_logprob_violations"] == 0
    for actual, wanted in zip(trainer.model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("backend", ["full_logits", "selective_lm_head"])
def test_all_rounds_shared_actor_one_backward_matches_union_objective(backend):
    from llm_local_rl.debate_parity import assemble_split_train_examples

    torch.manual_seed(1701)
    trainer = _tiny_causal_trainer(backend=backend, learning_rate=0.)
    # Every round extends the exact previous token sequence. Injected tokens 5
    # and 8 must contribute context but never a policy-gradient loss.
    transitions = [
        Transition(prompt_tokens=[1, 2], completion_tokens=[3, 4], completion_logprobs=[0., 0.], round_num=1),
        Transition(prompt_tokens=[1, 2, 3, 4, 5], completion_tokens=[6, 7], completion_logprobs=[0., 0.], round_num=2),
        Transition(prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8], completion_tokens=[9, 10], completion_logprobs=[0., 0.], round_num=3),
    ]
    debate = DebateResult(
        question="Q", ground_truth=None,
        trajectory_a=DebateTrajectory(agent="A", transitions=transitions, frozen_solution="A", metrics={"task_reward": 1.}),
        trajectory_b=DebateTrajectory(agent="B", transitions=transitions, frozen_solution="B", metrics={"task_reward": 0.}),
        verdict="A", judge_reasoning="",
    )
    rows = assemble_split_train_examples(
        debates=[debate], num_rounds=3, round_adapter_names=("debate",) * 3,
        r1_reward_mode="task", r23_reward_mode="constant", r23_constant=.7,
        r23_symmetric=True, task_reward_fn=lambda t, _: t.metrics["task_reward"],
    )["debate"]
    assert len(rows) == 2  # one per agent; test A to avoid cancelling +/- gradients
    example = rows[0]
    current = trainer.compute_logprobs(adapter_name="debate", batch=[example])[0]
    example = replace(example, old_logprobs=current)

    # An independent sum over the three original round losses is normalized by
    # one trajectory, rather than by three rows. Compare actual parameter grads.
    reference_gradients = None
    for transition, advantage in zip(transitions, (.5, .35, .35), strict=True):
        tokens = transition.prompt_tokens + transition.completion_tokens
        inputs = torch.tensor([tokens[:-1]])
        outputs = trainer.model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
        logps = torch.log_softmax(outputs.logits.float() / .8, dim=-1)
        targets = torch.tensor(tokens[1:])
        selected = logps[0, torch.arange(len(targets)), targets]
        loss = -(advantage * selected[-2:]).sum()
        loss.backward()
    reference_gradients = [p.grad.detach().clone() for p in trainer.model.parameters()]
    # Disable clipping only for this gradient-equivalence test.
    trainer.config = replace(trainer.config, max_grad_norm=1e9)
    grad_enabled_forwards = []
    hook = trainer.model.model.register_forward_hook(
        lambda _m, _i, _o: grad_enabled_forwards.append(torch.is_grad_enabled())
    )
    metrics = trainer.train_batch(adapter_name="debate", batch=[example])
    hook.remove()
    assert sum(grad_enabled_forwards) == 1
    assert metrics["num_examples"] == 1
    assert metrics["num_optimizer_steps"] == 1
    assert metrics["num_trained_tokens"] == 6
    assert metrics["completion_tokens_checked"] == 6
    assert metrics["on_policy_logprob_violations"] == 0
    for parameter, expected in zip(trainer.model.parameters(), reference_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-6)


# Capacity retries preserve optimizer groups; completed groups may only replay
# after a full rollback. A failure inside optimizer.step is always fatal.
def _capacity_rows(count):
    return [TrainExample('shared', [0], [0], [1], [1], [-math.log(2)], [1.0])
            for _ in range(count)]


def test_capacity_is_reused_per_adapter_and_only_decreases(monkeypatch):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_minibatch_size=128, train_optimizer_batch_size=128)
    configured = trainer.config
    calls = []
    limit = 32
    forward = trainer.model.forward

    def bounded_forward(**kwargs):
        size = kwargs['input_ids'].shape[0]
        calls.append(size)
        if size > limit:
            raise torch.OutOfMemoryError('injected capacity limit')
        return forward(**kwargs)

    monkeypatch.setattr(trainer.model, 'forward', bounded_forward)
    result = trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128))
    assert calls == [128, 64, 32, 32, 32, 32]
    assert result['microbatch_oom_attempts'] == 2
    assert result['num_optimizer_steps'] == 1
    assert trainer.config is configured
    calls.clear()
    result = trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128))
    assert calls == [32] * 4
    assert result['microbatch_oom_attempts'] == 0
    assert result['microbatch_initial_size'] == 32
    # A short tail is not evidence that the device's capacity fell to two rows.
    trainer.train_batch(adapter_name='shared', batch=_capacity_rows(2))
    calls.clear()
    trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128))
    assert calls == [32] * 4
    # Judge capacity is independent of the debate/shared adapter.
    calls.clear()
    trainer.train_batch(adapter_name='judge', batch=_capacity_rows(128))
    assert calls[:3] == [128, 64, 32]
    # Longer inputs or less available memory may force a further reduction.
    limit = 16
    calls.clear()
    trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128))
    assert calls == [32] + [16] * 8
    limit = 32
    calls.clear()
    trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128))
    assert calls == [16] * 8
    assert trainer._microbatch_limits == {'shared': 16, 'judge': 32}
    trainer.config = replace(configured, train_minibatch_size=8)
    calls.clear()
    trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128))
    assert calls == [8] * 16
    assert trainer._microbatch_limits['shared'] == 16


def test_capacity_retry_discards_partial_gradients_and_restores_all_rng(monkeypatch):
    import random
    import numpy as np

    actual = _fake_trainer()
    expected = _fake_trainer()
    for trainer, size in [(actual, 4), (expected, 2)]:
        trainer.config = replace(trainer.config, train_minibatch_size=size)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1)
    attempts = []

    def install(trainer, fail):
        forward = trainer.model.forward
        count = 0
        def stochastic_forward(**kwargs):
            nonlocal count
            count += 1
            draws = (random.random(), float(np.random.random()), torch.rand(()).item())
            if fail:
                attempts.append(draws)
            if fail and count == 2:
                assert trainer.model.bias.grad is not None  # First chunk already accumulated.
                raise torch.OutOfMemoryError('after partial backward')
            return forward(**kwargs)
        monkeypatch.setattr(trainer.model, 'forward', stochastic_forward)

    install(actual, True)
    install(expected, False)
    tails = []
    for trainer in (expected, actual):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        result = trainer.train_batch(adapter_name='shared', batch=_capacity_rows(8))
        tails.append((random.random(), float(np.random.random()), torch.rand(()).item()))
        assert result['num_optimizer_steps'] == 1
    assert attempts[0] == attempts[2]
    assert tails[0] == tails[1]
    torch.testing.assert_close(actual.model.bias, expected.model.bias, rtol=0, atol=0)
    a = actual.optimizer.state[actual.model.bias]
    b = expected.optimizer.state[expected.model.bias]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert a['step'] == 1
    assert actual.config.train_minibatch_size == 4


@pytest.mark.parametrize('optimizer_size', [0, 2])
def test_capacity_never_replays_a_failed_optimizer(monkeypatch, optimizer_size):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_minibatch_size=2,
                             train_optimizer_batch_size=optimizer_size)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    step = trainer.optimizer.step
    updates = []
    def mutating_step(*args, **kwargs):
        updates.append(True)
        step(*args, **kwargs)
        raise torch.OutOfMemoryError('optimizer partially mutated')
    monkeypatch.setattr(trainer.optimizer, 'step', mutating_step)
    with pytest.raises(torch.OutOfMemoryError):
        trainer.train_batch(adapter_name='shared', batch=_capacity_rows(4))
    assert len(updates) == 1
    assert trainer.model.bias.item() != 0
    assert trainer._microbatch_limits == {}
    assert trainer.config.train_minibatch_size == 2


@pytest.mark.parametrize('existing_state', [False, True])
def test_later_group_oom_rolls_back_and_matches_four_independent_adam_updates(monkeypatch, existing_state):
    import copy
    import random
    import numpy as np
    from llm_local_rl.optimizer_batching import pack_optimizer_groups

    actual = _fake_trainer()
    actual.single_target_parameter_adapter_mode = False
    actual.config = replace(actual.config, train_minibatch_size=4, train_optimizer_batch_size=4,
                            train_keep_groups_together=True, train_length_bucket_batches=True)
    actual.optimizer = torch.optim.AdamW(actual.model.parameters(), lr=.1)
    if existing_state:
        (actual.model.bias + actual.model.bias.square()).backward()
        actual.optimizer.step()
        actual.optimizer.zero_grad(set_to_none=True)
    expected = _fake_trainer()
    expected.config = actual.config
    expected.model.load_state_dict(actual.model.state_dict())
    expected.optimizer = torch.optim.AdamW(expected.model.parameters(), lr=.1)
    expected.optimizer.load_state_dict(copy.deepcopy(actual.optimizer.state_dict()))
    initial_logprob = float(torch.log_softmax(torch.stack([actual.model.bias.detach(), torch.tensor(0.)]), dim=0)[0])
    batch = []
    for index in range(16):
        length = 1 + (index // 4) % 3
        batch.append(TrainExample('shared', [0] * length, [0] * length, [1] * length,
                                  [1] * length, [initial_logprob] * length,
                                  [.2 + (index % 3) * .1] * length,
                                  metadata={'optimizer_group_id': str(index // 2)}))
    groups = [_order_batch_for_minibatching(batch=group, max_tokens=0, length_bucket_batches=True)
              for group in pack_optimizer_groups(batch, max_rows=4)]
    events = []
    def progress(event, **fields):
        events.append((event, fields))
        if event == 'attempt_begin' and fields['attempt_index'] == 1:
            state = actual.optimizer.state.get(actual.model.bias)
            assert (state is None) if not existing_state else state['step'].item() == 1
    actual.progress_callback = progress
    def draw_rng():
        return random.random(), float(np.random.random()), torch.rand(()).item()
    forward = actual.model.forward
    failed = False
    def bounded_forward(**kwargs):
        nonlocal failed
        draw_rng()
        if torch.is_grad_enabled() and kwargs['input_ids'].shape[0] == 4:
            state = actual.optimizer.state.get(actual.model.bias)
            if state is not None and state['step'].item() == int(existing_state) + 1:
                failed = True
                raise torch.OutOfMemoryError('longer second group after first Adam update')
        return forward(**kwargs)
    monkeypatch.setattr(actual.model, 'forward', bounded_forward)

    # Independent optimizer loop at the final physical size, with the same
    # frozen rollout policy and all parity forwards preceding any update.
    def seed():
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
    seed()
    for group in groups:
        for _ in range(0, len(group), 2):
            draw_rng()
    for group in groups:
        for start in range(0, len(group), 2):
            draw_rng()
            rows = group[start:start + 2]
            inputs = torch.tensor([row.input_ids for row in rows])
            logits = expected.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
            logprobs = torch.log_softmax(logits, dim=-1)[..., 0]
            ratios = torch.exp(logprobs - initial_logprob)
            advantages = torch.tensor([row.advantages for row in rows])
            epsilon = expected.config.ppo_clip_epsilon
            loss = -torch.minimum(ratios * advantages, ratios.clamp(1-epsilon, 1+epsilon) * advantages).sum() / len(group)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.model.parameters(), expected.config.max_grad_norm)
        expected.optimizer.step()
        expected.optimizer.zero_grad(set_to_none=True)
    expected_tail = draw_rng()
    seed()
    result = actual.train_batch(adapter_name='shared', batch=batch)
    assert draw_rng() == expected_tail
    assert failed
    assert result['num_optimizer_steps'] == 4
    assert result['microbatch_oom_attempts'] == 1
    assert result['microbatch_rolled_back_optimizer_steps'] == 1
    assert result['microbatch_rollback_snapshot_bytes'] == (16 if existing_state else 4)
    assert result['physical_microbatch_size'] == 2
    assert actual._microbatch_limits == {'shared': 2}
    assert actual.training_state_dict()['microbatch_limits'] == {'shared': 2}
    torch.testing.assert_close(actual.model.bias, expected.model.bias, atol=0, rtol=0)
    a = actual.optimizer.state[actual.model.bias]
    b = expected.optimizer.state[expected.model.bias]
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    assert a['step'].item() == int(existing_state) + 4
    rollback = [fields for event, fields in events if event == 'attempt_rollback']
    assert len(rollback) == 1 and rollback[0]['discarded_optimizer_updates'] == 1
    accepted = [(event, fields) for event, fields in events if fields['attempt_index'] == 1]
    committed = [fields['end_row'] for event, fields in accepted if event == 'optimizer_step_complete']
    assert committed == [4, 8, 12, 16]
    assert all(fields['provisional'] for event, fields in accepted if event == 'optimizer_step_complete')
    last_preflight = max(i for i, (event, _) in enumerate(accepted) if event == 'preflight_complete')
    first_update = min(i for i, (event, _) in enumerate(accepted) if event == 'optimizer_step_complete')
    assert last_preflight < first_update


def test_later_group_oom_exhaustion_restores_initial_state_and_is_bounded(monkeypatch):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_minibatch_size=4, train_optimizer_batch_size=4,
                             on_policy_logprob_warn_only=True)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.1)
    forward = trainer.model.forward
    sizes = []
    def always_fail_after_update(**kwargs):
        if torch.is_grad_enabled() and trainer.optimizer.state:
            sizes.append(kwargs['input_ids'].shape[0])
            raise torch.OutOfMemoryError('every later group fails')
        return forward(**kwargs)
    monkeypatch.setattr(trainer.model, 'forward', always_fail_after_update)
    with pytest.raises(torch.OutOfMemoryError, match='no retained updates'):
        trainer.train_batch(adapter_name='shared', batch=_capacity_rows(16))
    assert sizes == [4, 2, 1]
    assert trainer.model.bias.item() == 0
    assert not trainer.optimizer.state
    assert trainer.model.bias.grad is None
    assert trainer._microbatch_limits == {'shared': 1}
    assert trainer.config.train_minibatch_size == 4


def test_rollback_snapshot_excludes_inactive_adapter_and_preserves_state_device():
    from llm_local_rl.trainer import _OptimizerRollbackSnapshot
    active = torch.nn.Parameter(torch.tensor([1.]))
    inactive = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)
    optimizer = torch.optim.AdamW([active, inactive], lr=.1)
    active.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state[inactive] = {'step': torch.tensor(7.), 'exp_avg': torch.tensor([3.]),
                                 'exp_avg_sq': torch.tensor([4.])}
    inactive_state = optimizer.state[inactive]
    saved_value = active.detach().clone()
    saved_state = {key: value.clone() for key, value in optimizer.state[active].items()}
    snapshot = _OptimizerRollbackSnapshot(optimizer)
    assert snapshot.nbytes == 16
    (2 * active.sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    snapshot.restore(optimizer)
    torch.testing.assert_close(active, saved_value, atol=0, rtol=0)
    assert inactive.item() == 2
    assert optimizer.state[inactive] is inactive_state
    for key, value in optimizer.state[active].items():
        torch.testing.assert_close(value, saved_state[key], atol=0, rtol=0)
        assert value.device == saved_state[key].device


@pytest.mark.parametrize('paired', [False, True])
def test_capacity_exhaustion_is_bounded_and_non_oom_is_not_retried(monkeypatch, paired):
    trainer = _fake_trainer()
    attempted = []
    def always_oom(**kwargs):
        attempted.append(trainer.config.train_minibatch_size)
        raise torch.OutOfMemoryError('too large')
    monkeypatch.setattr(trainer, '_train_batch', always_oom)
    with pytest.raises(torch.OutOfMemoryError, match='minimum microbatch'):
        trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128),
                            objective='supervised_label_ce_js' if paired else 'ppo')
    assert attempted == ([128, 64, 32, 16, 8, 4, 2] if paired else [128, 64, 32, 16, 8, 4, 2, 1])
    assert trainer.config.train_minibatch_size == 0
    attempted.clear()
    def invalid(**kwargs):
        attempted.append(True)
        raise FloatingPointError('invalid gradients')
    monkeypatch.setattr(trainer, '_train_batch', invalid)
    with pytest.raises(FloatingPointError):
        trainer.train_batch(adapter_name='shared', batch=_capacity_rows(128))
    assert len(attempted) == 1


def test_capacity_reference_scoring_also_shrinks_and_stays_bounded(monkeypatch):
    trainer = _fake_trainer()
    trainer.reference_adapter_names = {'shared': 'reference__shared'}
    active = None
    calls = []
    def set_adapter(name):
        nonlocal active
        active = name
    trainer.set_adapter = set_adapter
    forward = trainer.model.forward
    def bounded_forward(**kwargs):
        size = kwargs['input_ids'].shape[0]
        calls.append((active, size))
        if active == 'reference__shared' and size > 2:
            raise torch.OutOfMemoryError('reference does not fit')
        return forward(**kwargs)
    monkeypatch.setattr(trainer.model, 'forward', bounded_forward)
    metrics = trainer.train_batch(adapter_name='shared', batch=_capacity_rows(8), measure_reference_kl=True)
    assert calls == [('reference__shared', 8), ('reference__shared', 4)] + [('reference__shared', 2)] * 4 + [('shared', 2)] * 4
    assert metrics['microbatch_oom_attempts'] == 2
    assert metrics['num_optimizer_steps'] == 1
    calls.clear()
    trainer.train_batch(adapter_name='shared', batch=_capacity_rows(8), measure_reference_kl=True)
    assert all(size == 2 for _, size in calls)
    calls.clear()
    trainer.compute_logprobs(adapter_name='reference__shared', batch=_capacity_rows(8))
    assert calls == [('reference__shared', 2)] * 4


@pytest.mark.parametrize('backend', ['full_logits', 'selective_lm_head'])
def test_logprob_microbatches_preserve_row_order_and_values(backend):
    trainer = _tiny_causal_trainer(backend=backend)
    rows = [TrainExample('shared', [i] * n, [(i + 1) % 16] * n, [1] * n, [1] * n,
                         [0.] * n, [1.] * n) for i, n in enumerate([5, 1, 3, 2, 4])]
    trainer.config = replace(trainer.config, train_minibatch_size=0)
    expected = trainer.compute_logprobs(adapter_name='shared', batch=rows)
    trainer.config = replace(trainer.config, train_minibatch_size=2)
    actual = trainer.compute_logprobs(adapter_name='shared', batch=rows)
    for a, b in zip(actual, expected, strict=True):
        assert a == pytest.approx(b, abs=1e-6)


def test_capacity_paired_judge_keeps_pairs_and_optimizer_grouping(monkeypatch):
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_optimizer_batch_size=4,
                             train_length_bucket_batches=True)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    baseline = _fake_trainer()
    baseline.config = trainer.config
    baseline.optimizer = torch.optim.SGD(baseline.model.parameters(), lr=0.1)
    rows = _ce_only_pair_rows()
    forward = trainer.model.forward
    sizes = []
    def bounded_forward(**kwargs):
        sizes.append(kwargs['input_ids'].shape[0])
        if sizes[-1] > 2:
            raise torch.OutOfMemoryError('pair memory')
        return forward(**kwargs)
    monkeypatch.setattr(trainer.model, 'forward', bounded_forward)
    baseline.train_batch(adapter_name='judge', batch=rows, objective='supervised_label_ce_js')
    metrics = trainer.train_batch(adapter_name='judge', batch=rows, objective='supervised_label_ce_js')
    assert sizes == [4, 2, 2, 2]
    assert metrics['num_optimizer_steps'] == 2
    assert metrics['judge_coherence_pair_count'] == 3
    torch.testing.assert_close(trainer.model.bias, baseline.model.bias, atol=1e-7, rtol=1e-6)


def test_capacity_limits_survive_serialized_resume_and_old_checkpoints_load(tmp_path, monkeypatch):
    import copy
    trainer = _fake_trainer()
    trainer.single_target_parameter_adapter_mode = False
    calls = []
    def bounded(**kwargs):
        size = trainer.config.train_minibatch_size
        calls.append(size)
        if size > 4:
            raise torch.OutOfMemoryError('injected')
        return {}
    monkeypatch.setattr(trainer, '_train_batch', bounded)
    trainer.train_batch(adapter_name='shared', batch=_capacity_rows(16))
    assert calls == [16, 8, 4]
    checkpoint = tmp_path / 'trainer_state.pt'
    torch.save(trainer.training_state_dict(), checkpoint)
    restored = _fake_trainer()
    state = torch.load(checkpoint, weights_only=False)
    restored.load_training_state_dict(state)
    result = restored.train_batch(adapter_name='shared', batch=_capacity_rows(16))
    assert result['microbatch_initial_size'] == 4
    assert result['microbatch_oom_attempts'] == 0
    assert result['num_train_minibatches'] == 4
    old = copy.deepcopy(state)
    del old['microbatch_limits']
    restored.load_training_state_dict(old)
    assert restored._microbatch_limits == {}
    state['microbatch_limits'] = {'shared': 0}
    with pytest.raises(ValueError, match='microbatch limits'):
        restored.load_training_state_dict(state)


@pytest.mark.parametrize("physical_size", [16, 32])
@pytest.mark.parametrize("bucket", [False, True])
@pytest.mark.parametrize("group_count", [8, 64])
def test_four_grouped_updates_match_independent_ppo(physical_size, bucket, group_count, capsys):
    from llm_local_rl.optimizer_batching import pack_optimizer_groups
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_keep_groups_together=True,
                             train_optimizer_batch_size=32, train_minibatch_size=physical_size,
                             train_length_bucket_batches=bucket, max_grad_norm=0.0)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.7)
    batch = []
    for index in range(128 // group_count):
        for group in range(group_count):
            n = 1 + (group + index) % 5
            advantage = [1.0, .4, -.8, .6, -.3, 1.2, .2, -1.0][group % 8]
            batch.append(TrainExample("shared", [0] * n, [0] * n,
                [0] * (n - 1) + [1], [0] * (n - 1) + [1],
                [0.] * (n - 1) + [-math.log(2)], [0.] * (n - 1) + [advantage],
                metadata={"optimizer_group_id": str(group)}))
    original = [(list(r.old_logprobs), list(r.advantages)) for r in batch]
    expected = torch.tensor(0., requires_grad=True)
    optimizer = torch.optim.SGD([expected], lr=0.7)
    for group in pack_optimizer_groups(batch, max_rows=32):
        advantages = torch.tensor([r.advantages[-1] for r in group])
        ratio = 2 * torch.sigmoid(expected)
        loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    result = trainer.train_batch(adapter_name="shared", batch=batch)
    assert trainer.model.bias.item() == pytest.approx(expected.item(), abs=1e-7)
    assert result["num_optimizer_steps"] == 4
    assert result["num_train_minibatches"] == 128 // physical_size
    assert result["completion_tokens_checked"] == 128
    assert original == [(r.old_logprobs, r.advantages) for r in batch]
    assert '"batch_rows": [32, 32, 32, 32]' in capsys.readouterr().out


def test_grouped_capacity_retry_preserves_membership(monkeypatch, capsys):
    import json
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_keep_groups_together=True,
                             train_optimizer_batch_size=32, train_minibatch_size=32,
                             train_length_bucket_batches=True)
    batch = [replace(r, metadata={"optimizer_group_id": str(i % 8)})
             for i, r in enumerate(_capacity_rows(128))]
    forward = trainer.model.forward
    def limited_forward(*args, **kwargs):
        if torch.is_grad_enabled() and kwargs["input_ids"].shape[0] > 16:
            raise torch.OutOfMemoryError("test physical capacity")
        return forward(*args, **kwargs)
    monkeypatch.setattr(trainer.model, "forward", limited_forward)
    result = trainer.train_batch(adapter_name="shared", batch=batch)
    plans = [json.loads(line) for line in capsys.readouterr().out.splitlines()
             if '"event": "optimizer_group_plan"' in line]
    assert len(plans) == 2 and plans[0] == plans[1]
    assert result["num_optimizer_steps"] == 4
    assert result["physical_microbatch_size"] == 16
    assert result["microbatch_oom_attempts"] == 1


def test_grouped_judge_updates_keep_all_pairs_and_match_reference():
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_keep_groups_together=True,
                             train_optimizer_batch_size=32, train_minibatch_size=32,
                             train_length_bucket_batches=True)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    reference = _fake_trainer()
    reference.optimizer = torch.optim.SGD(reference.model.parameters(), lr=0.1)
    templates = _ce_only_pair_rows()
    batch = []
    for pair in range(64):
        for template in templates[(pair % 3) * 2:(pair % 3) * 2 + 2]:
            batch.append(replace(template, metadata={**template.metadata,
                "optimizer_group_id": str(pair // 8), "judge_coherence_pair_id": str(pair)}))
    for start in range(0, 128, 32):
        reference.train_batch(adapter_name="judge", batch=batch[start:start + 32],
                              objective="supervised_label_ce_js", judge_coherence_js_weight=0.5)
    result = trainer.train_batch(adapter_name="judge", batch=batch,
                                  objective="supervised_label_ce_js", judge_coherence_js_weight=0.5)
    assert result["num_optimizer_steps"] == 4
    assert result["num_train_minibatches"] == 4
    assert result["judge_coherence_pair_count"] == 64
    assert trainer.model.bias.item() == pytest.approx(reference.model.bias.item(), abs=1e-7)


def test_grouped_batches_refuse_partial_groups_before_update():
    trainer = _fake_trainer()
    trainer.config = replace(trainer.config, train_keep_groups_together=True,
                             train_optimizer_batch_size=32)
    batch = [replace(r, metadata={"optimizer_group_id": "one"}) for r in _capacity_rows(33)]
    with pytest.raises(ValueError, match="cannot split"):
        trainer.train_batch(adapter_name="shared", batch=batch)
    assert trainer.model.bias.item() == 0.0
    trainer.config = replace(trainer.config, train_max_tokens=1)
    batch = [batch[0], replace(batch[1], input_ids=[0, 0])]
    with pytest.raises(ValueError, match="complete optimizer group"):
        trainer.train_batch(adapter_name="shared", batch=batch)
    assert trainer.model.bias.item() == 0.0
