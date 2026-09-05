from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any


CHECKPOINT_SCHEMA = "llm_local_rl_exact_resume_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        import numpy as np

        np.random.set_state(state["numpy"])


def _inventory(root: Path) -> dict[str, dict[str, int | str]]:
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"checkpoint_manifest.json", "READY"}:
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return files


def save_exact_resume_checkpoint(
    *,
    root: str | Path,
    step: int,
    run_config: dict[str, Any],
    adapter_dirs: dict[str, str],
    trainer: Any,
) -> Path:
    """Atomically publish adapters + optimizer + RNG for the next step."""
    import torch

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{step:06d}"
    if (final / "READY").exists():
        return final
    if final.exists():
        quarantine = root / f"step_{step:06d}.incomplete-{os.getpid()}"
        os.replace(final, quarantine)
    temporary = root / f".step_{step:06d}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    copied_adapters: dict[str, str] = {}
    for name, source_text in sorted(adapter_dirs.items()):
        source = Path(source_text)
        if not source.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {source}")
        destination = temporary / "adapters" / name
        shutil.copytree(source, destination)
        copied_adapters[name] = str(Path("adapters") / name)
    torch.save(trainer.training_state_dict(), temporary / "trainer_state.pt")
    torch.save(capture_rng_state(), temporary / "rng_state.pt")
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "completed_step": step,
        "next_step": step + 1,
        "run_config_sha256": config_fingerprint(run_config),
        "adapter_dirs": copied_adapters,
    }
    manifest["files"] = _inventory(temporary)
    (temporary / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    (temporary / "READY").write_text("ready\n")
    os.replace(temporary, final)
    return final


def validate_exact_resume_checkpoint(
    path: str | Path, *, run_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    path = Path(path)
    if not (path / "READY").is_file():
        raise ValueError(f"Exact-resume checkpoint is not READY: {path}")
    manifest = json.loads((path / "checkpoint_manifest.json").read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported checkpoint schema: {manifest.get('schema')!r}")
    if run_config is not None and manifest["run_config_sha256"] != config_fingerprint(run_config):
        raise ValueError("Exact-resume checkpoint run configuration fingerprint mismatch.")
    for relative, expected in manifest["files"].items():
        candidate = path / relative
        if not candidate.is_file():
            raise ValueError(f"Checkpoint file missing: {relative}")
        if candidate.stat().st_size != expected["bytes"] or _sha256(candidate) != expected["sha256"]:
            raise ValueError(f"Checkpoint file integrity mismatch: {relative}")
    return manifest


def load_exact_resume_checkpoint(*, path: str | Path, trainer: Any, run_config: dict[str, Any]) -> dict[str, Any]:
    import torch

    path = Path(path)
    manifest = validate_exact_resume_checkpoint(path, run_config=run_config)
    trainer.load_training_state_dict(torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False))
    restore_rng_state(torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False))
    return manifest


def checkpoint_adapter_dirs(path: str | Path) -> dict[str, str]:
    path = Path(path)
    manifest = validate_exact_resume_checkpoint(path)
    return {name: str(path / relative) for name, relative in manifest["adapter_dirs"].items()}
