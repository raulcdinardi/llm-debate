import os

import pytest


@pytest.mark.integration
def test_training_replay_parity_placeholder() -> None:
    model_path = os.environ.get("LLM_LOCAL_RL_MODEL")
    if model_path is None:
        pytest.skip("Set LLM_LOCAL_RL_MODEL to run training replay parity integration tests.")
    # The real implementation should replay the same explicit-mask batch through:
    # - the training path under test
    # - an independent reference forward/backward path
    # and compare grads up to epsilon.
    assert os.path.exists(model_path)
