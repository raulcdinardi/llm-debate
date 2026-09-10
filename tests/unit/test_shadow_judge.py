from dataclasses import replace
import copy
import random
from types import SimpleNamespace

import pytest

from llm_local_rl.checkpointing import config_fingerprint
from llm_local_rl.config import TrainRunConfig
from llm_local_rl.driver import TrainingDriver
from llm_local_rl.shadow_judge import SHADOW_JUDGE, preserve_rng_state, shadow_label_examples
from llm_local_rl.types import TrainExample
from scripts.run_train import parse_args


def paired_config(**changes):
    values = dict(
        model_path="/unused", output_dir="/unused", adapter_layout="split",
        debate_judge_adapter="judge", debate_judge_harness="constitution_single_token_v1",
        debate_judge_bidirectional=True, debate_judge_constrain_single_token=True,
        debate_judge_score_mode="order_sym_soft_logit",
        judge_label_token_contract="lfm25_openbookqa_spaced_ab_v1",
        debate_r1_reward="none", debate_r23_reward="soft_judge_raw",
        train_judge=True, judge_training_objective="supervised_label_ce_js",
        judge_coherence_js_weight=0.0, train_shadow_judge=True,
        shadow_judge_init_seed=17, shadow_judge_init_std=0.03,
        train_adapter_names=("debate", "judge", SHADOW_JUDGE),
    )
    values.update(changes)
    return TrainRunConfig(**values)


def label_batch():
    return [TrainExample(
        adapter_name="judge", input_ids=[2, 3 + index], target_ids=[3 + index, target],
        loss_mask=[0, 1], behavior_logprob_mask=[0, 0], old_logprobs=[0., 0.],
        advantages=[0., 0.], metadata={
            "training_objective": "supervised_label_ce_js",
            "behavior_policy_allowed_token_ids": [5, 6],
            "judge_coherence_pair_id": "pair-0", "judge_coherence_pair_member": member,
            "judge_sampled_verdict": "A", "judge_sampled_label_correct": False,
        },
    ) for index, (target, member) in enumerate(((5, "forward"), (6, "reverse")))]


def test_cli_config_and_legacy_checkpoint_compatibility():
    args = parse_args(["--model-path", "/unused", "--output-dir", "/unused",
                       "--train-shadow-judge", "--shadow-judge-init-seed", "17",
                       "--shadow-judge-init-std", "0.03"])
    assert (args.train_shadow_judge, args.shadow_judge_init_seed, args.shadow_judge_init_std) == (True, 17, .03)
    config = paired_config()
    assert TrainRunConfig.from_dict(config.to_dict()) == config
    old = TrainRunConfig(model_path="/unused", output_dir="/unused").to_dict()
    new = dict(old)
    for key in ("train_shadow_judge", "shadow_judge_init_seed", "shadow_judge_init_std"):
        del old[key]
    assert config_fingerprint(old) == config_fingerprint(new)
    assert config_fingerprint(config.to_dict()) != config_fingerprint(replace(config, shadow_judge_init_seed=18).to_dict())


@pytest.mark.parametrize("changes", [
    {"shadow_judge_init_seed": None}, {"shadow_judge_init_std": 0.},
    {"shadow_judge_init_std": float("nan")}, {"train_judge": False},
    {"judge_training_objective": "grpo"}, {"judge_coherence_js_weight": 1.},
    {"debate_round_adapter_names": ("solution", SHADOW_JUDGE, SHADOW_JUDGE)},
    {"train_adapter_names": ("debate", "judge")}, {"target_parameters": ("experts",)},
])
def test_invalid_shadow_config_rejected(changes):
    with pytest.raises(ValueError):
        paired_config(**changes)


def test_shadow_never_enters_sampler_and_uses_identical_labels():
    driver = object.__new__(TrainingDriver)
    driver.config = paired_config()
    driver.current_adapter_dirs = {name: f"/{name}" for name in driver._adapter_names()}
    assert driver._adapter_names() == ("solution", "debate", "judge", SHADOW_JUDGE)
    assert driver._train_adapter_names() == {"debate", "judge", SHADOW_JUDGE}
    assert SHADOW_JUDGE not in driver._sampler_adapter_dirs()
    active = label_batch()
    before = copy.deepcopy(active)
    shadow = shadow_label_examples(active)
    assert active == before
    for a, b in zip(active, shadow, strict=True):
        assert replace(b, adapter_name="judge", metadata=a.metadata) == a
        assert "judge_sampled_verdict" not in b.metadata


@pytest.fixture
def tiny_trainers(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM
    from llm_local_rl.trainer import MultiAdapterTrainer, TrainerConfig

    torch.set_num_threads(1)
    def backbone(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            return LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8,
                intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2, attention_dropout=0.0))
    monkeypatch.setattr(MultiAdapterTrainer, "_build_base_model", backbone)
    monkeypatch.setattr(MultiAdapterTrainer, "_load_tokenizer", staticmethod(
        lambda **kwargs: SimpleNamespace(pad_token_id=0)))

    def make(shadow=True):
        return MultiAdapterTrainer(config=TrainerConfig(
            base_model_path="/unused", device="cpu", torch_dtype="float32", lora_rank=2,
            adapter_names=("solution", "debate", "judge") + ((SHADOW_JUDGE,) if shadow else ()),
            learning_rate=.01, train_minibatch_size=2, gradient_checkpointing=False,
        ))
    return make


def test_initialization_preserves_active_model_rng_and_pairs_a(tiny_trainers):
    import torch
    torch.manual_seed(41)
    single = tiny_trainers(False)
    expected_rng = torch.get_rng_state().clone()
    expected_parameters = {n: p.detach().clone() for n, p in single.model.named_parameters()}
    torch.manual_seed(41)
    paired = tiny_trainers()
    receipt = paired.initialize_shadow_judge(seed=17, std=.03)
    assert torch.equal(expected_rng, torch.get_rng_state())
    params = dict(paired.model.named_parameters())
    for name, expected in expected_parameters.items():
        assert torch.equal(params[name], expected)
    for name, parameter in params.items():
        if ".lora_A.judge." in name:
            assert torch.equal(parameter, params[name.replace(".judge.", ".judge_shadow.")])
        if ".lora_B.judge." in name:
            assert torch.count_nonzero(parameter) == 0
            assert torch.count_nonzero(params[name.replace(".judge.", ".judge_shadow.")]) > 0
    paired.set_adapter("judge")
    paired.model.eval()
    inputs = dict(input_ids=torch.tensor([[2, 3]]), attention_mask=torch.ones(1, 2, dtype=torch.long))
    with torch.no_grad():
        initial = paired.model(**inputs).logits
        with paired.model.disable_adapter():
            original = paired.model(**inputs).logits
        paired.set_adapter(SHADOW_JUDGE)
        shadow = paired.model(**inputs).logits
    assert torch.equal(initial, original)
    assert not torch.equal(initial, shadow)
    assert receipt["shadow_b_l2"] > 0


def test_paired_ce_updates_isolate_parameters_and_adam_state(tiny_trainers):
    import torch
    trainer = tiny_trainers()
    trainer.initialize_shadow_judge(seed=17, std=.03)
    driver = object.__new__(TrainingDriver)
    driver.config = paired_config()
    driver.trainer = trainer
    active_batch = label_batch()
    active_metrics = driver._train_adapter_batch(adapter_name="judge", batch=active_batch, step_num=1)
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    active_states = {n: copy.deepcopy(trainer.optimizer.state[p]) for n, p in trainer.model.named_parameters() if ".judge." in n}
    rng = torch.get_rng_state().clone()
    metrics = driver._train_adapter_batch(adapter_name=SHADOW_JUDGE, batch=shadow_label_examples(active_batch), step_num=1)
    assert torch.equal(rng, torch.get_rng_state())
    assert metrics["num_optimizer_steps"] == active_metrics["num_optimizer_steps"] == 1
    assert metrics["num_examples"] == active_metrics["num_examples"] == 2
    assert metrics["completion_tokens_checked"] == 0
    assert 0 <= metrics["supervised_label_brier"] <= 1
    changed = []
    for name, parameter in trainer.model.named_parameters():
        if not torch.equal(before[name], parameter):
            changed.append(name)
            assert ".judge_shadow." in name
        if name in active_states:
            for key, value in active_states[name].items():
                assert torch.equal(trainer.optimizer.state[parameter][key], value)
    assert changed


def test_shadow_rng_restored_even_on_error():
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    random.seed(17)
    numpy.random.seed(17)
    torch.manual_seed(17)
    driver = object.__new__(TrainingDriver)
    driver.config = paired_config()
    def failing_update(**kwargs):
        assert kwargs["objective"] == "supervised_label_ce_js"
        assert kwargs["measure_reference_kl"] is False
        random.random(); numpy.random.rand(); torch.rand(1)
        raise RuntimeError("failed shadow step")
    driver.trainer = SimpleNamespace(train_batch=failing_update)
    with pytest.raises(RuntimeError):
        driver._train_adapter_batch(adapter_name=SHADOW_JUDGE, batch=label_batch(), step_num=1)
    actual = (random.random(), numpy.random.rand(), torch.rand(1))
    random.seed(17); numpy.random.seed(17); torch.manual_seed(17)
    assert actual[:2] == (random.random(), numpy.random.rand())
    assert torch.equal(actual[2], torch.rand(1))


def test_paired_exact_resume_matches_uninterrupted_updates(tiny_trainers, tmp_path):
    import torch
    from llm_local_rl.checkpointing import checkpoint_adapter_dirs, load_exact_resume_checkpoint, save_exact_resume_checkpoint
    from llm_local_rl.trainer import MultiAdapterTrainer
    trainer = tiny_trainers()
    trainer.initialize_shadow_judge(seed=17, std=.03)
    config = paired_config().to_dict()
    batch = label_batch()
    def train_pair(instance):
        for name, rows in (("judge", batch), (SHADOW_JUDGE, shadow_label_examples(batch))):
            instance.train_batch(adapter_name=name, batch=rows, objective="supervised_label_ce_js", judge_coherence_js_weight=0.)
    train_pair(trainer)
    dirs = {name: trainer.save_adapter(adapter_name=name, output_dir=str(tmp_path/name)) for name in trainer.config.adapter_names}
    checkpoint = save_exact_resume_checkpoint(root=tmp_path/"checkpoints", step=1, trainer=trainer,
        run_config=config, adapter_dirs=dirs)
    train_pair(trainer)
    restored = MultiAdapterTrainer.from_saved_adapters(config=trainer.config, adapter_dirs=checkpoint_adapter_dirs(checkpoint))
    load_exact_resume_checkpoint(path=checkpoint, trainer=restored, run_config=config)
    train_pair(restored)
    for (name, expected), (restored_name, actual) in zip(trainer.model.named_parameters(), restored.model.named_parameters(), strict=True):
        assert name == restored_name
        assert torch.equal(expected, actual), name
