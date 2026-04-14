import os

import pytest


@pytest.mark.integration
def test_sampling_parity_placeholder() -> None:
    model_path = os.environ.get("LLM_LOCAL_RL_MODEL")
    if model_path is None:
        pytest.skip("Set LLM_LOCAL_RL_MODEL to run sampling parity integration tests.")
    # The real implementation should compare:
    # - pipeline generation through this repo
    # - an independent direct engine path
    # using the same prompt and fixed completion token ids
    assert os.path.exists(model_path)
