from llm_local_rl.config import TrainRunConfig, RolloutConfig
from llm_local_rl.driver import TrainingDriver
from llm_local_rl.checkpointing import config_fingerprint


def test_new_defaults_preserve_historical_fingerprint():
    original = {"model_path": "model", "steps": 10}
    current = dict(original, python_optimization_config=None, debate_r23_penalize_word_limit=True)
    assert config_fingerprint(original) == config_fingerprint(current)
    assert config_fingerprint(dict(current, debate_r23_penalize_word_limit=False)) != config_fingerprint(original)
    assert config_fingerprint(dict(current, python_optimization_config="mbpp.json")) != config_fingerprint(original)


def test_resume_retains_unused_adapter_and_optimizer_order(tmp_path):
    driver = object.__new__(TrainingDriver)
    driver.output_dir = tmp_path
    driver.config = TrainRunConfig(model_path="model", output_dir="full", adapter_layout="split",
                                  debate_rounds=1, debate_mock_judge_seed=17,
                                  rollout=RolloutConfig(mode="debate"))
    assert driver._adapter_names() == ("solution",)
    driver._resume_adapter_names = ("solution", "debate")
    assert driver._trainer_config(device="cpu").adapter_names == ("solution", "debate")


def test_default_eras_are_compatible_but_scientific_changes_are_not():
    from llm_local_rl.checkpointing import compatible_config_fingerprints
    older = {"model_path": "model", "steps": 10}
    deployed = dict(older, python_optimization_config=None, debate_r23_penalize_word_limit=True)
    hashes = compatible_config_fingerprints(deployed)
    assert config_fingerprint(older, normalize_new_defaults=False) in hashes
    assert config_fingerprint(deployed, normalize_new_defaults=False) in hashes
    assert config_fingerprint(dict(deployed, steps=11), normalize_new_defaults=False) not in hashes
    assert config_fingerprint(dict(deployed, debate_r23_penalize_word_limit=False), normalize_new_defaults=False) not in hashes
