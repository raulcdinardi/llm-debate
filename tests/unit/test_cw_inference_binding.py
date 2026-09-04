import json

import pytest

from llm_local_rl.judge_harness import (
    harness_fingerprint, validate_judge_harness_manifest, write_judge_harness_manifest,
)


def test_exact_inference_binding_preserves_training_provenance(tmp_path):
    harness = "constitution_single_token_v1"
    path = write_judge_harness_manifest(adapter_dir=tmp_path, harness_id=harness, max_rounds=3)
    payload = json.loads(path.read_text())
    original_hash = payload["harness_fingerprint"]
    with pytest.raises(ValueError):
        validate_judge_harness_manifest(adapter_dir=tmp_path, harness_id=harness, max_rounds=4)
    payload["inference_bindings"] = [{"max_rounds": 4, "harness_fingerprint": harness_fingerprint(harness, max_rounds=4)}]
    path.write_text(json.dumps(payload))
    result = validate_judge_harness_manifest(adapter_dir=tmp_path, harness_id=harness, max_rounds=4)
    assert result["max_rounds"] == 3
    assert result["harness_fingerprint"] == original_hash
    with pytest.raises(ValueError):
        validate_judge_harness_manifest(adapter_dir=tmp_path, harness_id=harness, max_rounds=5)
    payload["harness_fingerprint"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training harness"):
        validate_judge_harness_manifest(adapter_dir=tmp_path, harness_id=harness, max_rounds=4)
