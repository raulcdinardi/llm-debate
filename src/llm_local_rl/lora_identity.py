from __future__ import annotations

from pathlib import Path

AdapterIdentity = tuple[str, tuple[tuple[str, int, int], ...]]


def _adapter_weight_paths(adapter_dir: Path) -> tuple[Path, ...]:
    weights = tuple(sorted(adapter_dir.glob("*.safetensors"))) + tuple(sorted(adapter_dir.glob("*.bin")))
    if not weights:
        raise FileNotFoundError(f"missing adapter weight file under {adapter_dir}")
    return weights


def _adapter_file_stat(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path), int(stat.st_size), int(stat.st_mtime_ns))


def adapter_identity(adapter_path: str) -> AdapterIdentity:
    adapter_dir = Path(adapter_path).expanduser().resolve()
    metadata = [adapter_dir / "adapter_config.json"]
    harness_manifest = adapter_dir / "judge_harness.json"
    if harness_manifest.is_file():
        metadata.append(harness_manifest)
    tracked_files = tuple(metadata) + _adapter_weight_paths(adapter_dir)
    return (str(adapter_dir), tuple(_adapter_file_stat(path) for path in tracked_files))
