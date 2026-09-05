#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any


MODEL_REVISIONS = {
    "Qwen/Qwen3.5-4B-Base": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "Qwen/Qwen3.5-0.8B-Base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
}
EXPECTED_PACKAGE_VERSIONS = {
    "sglang": "0.5.15.post1",
    "torch": "2.11.0+cu129",
    "transformers": "5.12.1",
    "huggingface-hub": "1.23.0",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "distro": "1.9.0",
    "pytest": "9.1.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify immutable Phase-0 inputs, pin model snapshots, and create the fresh zero-B solution adapter."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--debate-adapter", required=True)
    parser.add_argument("--judge-08b-adapter", required=True)
    parser.add_argument("--judge-4b-adapter", required=True)
    parser.add_argument("--solution-adapter", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-image-ref", required=True)
    parser.add_argument("--expected-image-digest", required=True)
    parser.add_argument("--debate-adapter-sha256", required=True)
    parser.add_argument("--judge-08b-adapter-sha256", required=True)
    parser.add_argument("--judge-4b-adapter-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGE_VERSIONS}
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in EXPECTED_PACKAGE_VERSIONS.items()
        if observed[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen package-version mismatch: {mismatches}")
    return observed


def verify_adapter(path: Path, expected_sha256: str, expected_base_model: str) -> dict[str, Any]:
    weights_path = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Adapter is incomplete: {path}")
    observed_sha256 = sha256_file(weights_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"Adapter hash mismatch for {path}: expected {expected_sha256}, observed {observed_sha256}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != expected_base_model:
        raise RuntimeError(
            f"Adapter base mismatch for {path}: expected {expected_base_model!r}, "
            f"observed {config.get('base_model_name_or_path')!r}"
        )
    if int(config.get("r", -1)) != 32:
        raise RuntimeError(f"Adapter rank mismatch for {path}: {config.get('r')!r}")
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    if set(config.get("target_modules", [])) != expected_targets:
        raise RuntimeError(f"Adapter target-module mismatch for {path}")
    return {
        "path": str(path),
        "adapter_model_sha256": observed_sha256,
        "adapter_config_sha256": sha256_file(config_path),
        "base_model": expected_base_model,
        "rank": 32,
        "target_modules": sorted(expected_targets),
    }


def create_solution_adapter(source: Path, destination: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    if destination.exists():
        raise FileExistsError(f"Fresh solution destination already exists: {destination}")
    destination.mkdir(parents=True)
    weights = load_file(source / "adapter_model.safetensors", device="cpu")
    torch.manual_seed(2026071401)
    fresh = {}
    a_names: list[str] = []
    b_names: list[str] = []
    for name, tensor in sorted(weights.items()):
        if ".lora_A." in name:
            value = torch.empty_like(tensor)
            torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5))
            fresh[name] = value
            a_names.append(name)
        elif ".lora_B." in name:
            fresh[name] = torch.zeros_like(tensor)
            b_names.append(name)
        else:
            raise RuntimeError(f"Unexpected tensor in debate adapter: {name}")
    if not a_names or len(a_names) != len(b_names):
        raise RuntimeError(f"Malformed LoRA tensor set: A={len(a_names)} B={len(b_names)}")
    if not all(int(torch.count_nonzero(fresh[name]).item()) > 0 for name in a_names):
        raise RuntimeError("At least one freshly initialized LoRA A tensor is all-zero")
    if not all(int(torch.count_nonzero(fresh[name]).item()) == 0 for name in b_names):
        raise RuntimeError("At least one freshly initialized LoRA B tensor is non-zero")
    shutil.copy2(source / "adapter_config.json", destination / "adapter_config.json")
    save_file(fresh, destination / "adapter_model.safetensors")
    return {
        "path": str(destination),
        "seed": 2026071401,
        "a_initialization": "torch.nn.init.kaiming_uniform_(a=sqrt(5))",
        "b_initialization": "zeros_like",
        "a_tensor_count": len(a_names),
        "b_tensor_count": len(b_names),
        "all_a_nonzero": True,
        "all_b_zero": True,
        "adapter_model_sha256": sha256_file(destination / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(destination / "adapter_config.json"),
    }


def pin_models(model_cache_dir: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    model_cache_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for repo_id, revision in MODEL_REVISIONS.items():
        local_name = repo_id.rsplit("/", 1)[1].lower()
        local_dir = model_cache_dir / local_name
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
        )
        config_path = Path(resolved) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Pinned model snapshot has no config.json: {resolved}")
        output[repo_id] = {
            "revision": revision,
            "local_path": str(Path(resolved).resolve()),
            "config_sha256": sha256_file(config_path),
        }
    return output


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    embedded_commit = os.environ.get("LLM_DEBATE_SOURCE_COMMIT", "")
    if embedded_commit != args.expected_source_commit:
        raise RuntimeError(
            f"Embedded source commit mismatch: expected {args.expected_source_commit}, observed {embedded_commit!r}"
        )
    if not args.expected_image_digest.startswith("sha256:"):
        raise ValueError("--expected-image-digest must be a sha256 registry digest")

    versions = package_versions()
    debate = verify_adapter(
        Path(args.debate_adapter), args.debate_adapter_sha256, "Qwen/Qwen3.5-4B-Base"
    )
    judge_08b = verify_adapter(
        Path(args.judge_08b_adapter), args.judge_08b_adapter_sha256, "Qwen/Qwen3.5-0.8B-Base"
    )
    judge_4b = verify_adapter(
        Path(args.judge_4b_adapter), args.judge_4b_adapter_sha256, "Qwen/Qwen3.5-4B-Base"
    )
    solution = create_solution_adapter(Path(args.debate_adapter), Path(args.solution_adapter))
    models = pin_models(Path(args.model_cache_dir))

    provenance = {
        "schema": "cw_judge_signal_phase0_provenance_v1",
        "source": {
            "repository": "https://github.com/raulcdinardi/llm-debate",
            "commit": embedded_commit,
            "dirty_allowed": False,
        },
        "image": {
            "ref": args.expected_image_ref,
            "digest": args.expected_image_digest,
            "launch_contract": "image is invoked by immutable ref@digest",
        },
        "package_versions": versions,
        "models": models,
        "adapters": {
            "debate": debate,
            "judge_08b": judge_08b,
            "judge_4b": judge_4b,
            "solution_zero_b": solution,
        },
        "phase1_forbidden": True,
    }
    write_json(output_dir / "provenance.json", provenance)
    print(json.dumps({"event": "phase0_inputs_prepared", "phase1_forbidden": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
