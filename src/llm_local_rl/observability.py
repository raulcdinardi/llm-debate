from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Queue
import re
import threading
import time
from typing import Any
import shutil


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_step_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Flatten the stable scalar subset; full arrays/text remain in artifacts."""
    out: dict[str, float] = {}
    for key in ("mean_reward", "mean_parse_success", "reward_std", "reward_min", "reward_max"):
        if _numeric(record.get(key)):
            out[f"rollout/{key}"] = float(record[key])
    for key, value in record.get("mean_reward_metrics", {}).items():
        if _numeric(value):
            out[f"reward_component/{key}"] = float(value)
    for adapter, metrics in record.get("train_metrics", {}).items():
        for key, value in metrics.items():
            if _numeric(value):
                out[f"train/{adapter}/{key}"] = float(value)
    for key, value in record.get("rollout_metrics", {}).items():
        if _numeric(value):
            out[f"rollout/{key}"] = float(value)
    return out


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool = True
    project: str = "llm-local-rl"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    mode: str = "online"
    upload_artifacts: bool = True
    rollout_shard_steps: int = 10
    table_samples_per_shard: int = 32


class RunObservability:
    """Local-canonical observability with fail-open W&B synchronization."""

    def __init__(self, *, output_dir: str | Path, config: dict[str, Any], settings: WandbSettings):
        self.output_dir = Path(output_dir)
        self.settings = settings
        self.state_dir = self.output_dir / "observability"
        self.shard_dir = self.state_dir / "rollout_shards"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.failures_path = self.state_dir / "wandb_failures.jsonl"
        self.run = None
        self._wandb = None
        self._queue: Queue[tuple[str, Path, dict[str, Any]] | None] = Queue()
        self._worker: threading.Thread | None = None
        if settings.enabled:
            self._start_wandb(config=config)

    def _failure(self, operation: str, exc: BaseException) -> None:
        with self.failures_path.open("a") as handle:
            handle.write(json.dumps({"operation": operation, "error_type": type(exc).__name__, "error": str(exc)}) + "\n")

    def _start_wandb(self, *, config: dict[str, Any]) -> None:
        try:
            import wandb

            run_id_path = self.state_dir / "wandb_run_id.txt"
            run_id = run_id_path.read_text().strip() if run_id_path.exists() else wandb.util.generate_id()
            run_id_path.write_text(run_id + "\n")
            self._wandb = wandb
            self.run = wandb.init(
                project=self.settings.project,
                entity=self.settings.entity,
                group=self.settings.group,
                name=self.settings.name,
                id=run_id,
                resume="allow",
                mode=self.settings.mode,
                dir=str(self.state_dir),
                config=config,
            )
            (self.state_dir / "wandb_run_url.txt").write_text(
                str(self.run.url or "") + "\n"
            )
            self._worker = threading.Thread(target=self._artifact_worker, name="wandb-artifacts", daemon=True)
            self._worker.start()
        except BaseException as exc:
            self._failure("init", exc)
            self.run = None

    def log_step(self, record: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.log(flatten_step_metrics(record), step=int(record["step"]), commit=True)
        except BaseException as exc:
            self._failure("log_step", exc)

    def _pending_path(self, start: int, end: int) -> Path:
        return self.shard_dir / f"steps_{start:06d}_{end:06d}.pending.jsonl"

    def record_rollouts(self, record: dict[str, Any], *, final_step: bool = False) -> Path | None:
        step = int(record["step"])
        width = max(1, int(self.settings.rollout_shard_steps))
        start = ((step - 1) // width) * width + 1
        end = start + width - 1
        pending = self._pending_path(start, end)
        seen: set[int] = set()
        if pending.exists():
            for line in pending.read_text().splitlines():
                if line.strip():
                    seen.add(int(json.loads(line)["step"]))
        if step not in seen:
            with pending.open("a") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        if step != end and not final_step:
            return None
        actual_end = step
        final = self.shard_dir / f"steps_{start:06d}_{actual_end:06d}.jsonl.gz"
        temporary = final.with_suffix(final.suffix + ".tmp")
        with pending.open("rb") as source, temporary.open("wb") as compressed:
            with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=6, mtime=0) as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
        if final.exists():
            if _sha256(final) != _sha256(temporary):
                temporary.unlink()
                raise ValueError(f"Refusing to overwrite an immutable rollout shard: {final}")
            temporary.unlink()
        else:
            os.replace(temporary, final)
        self._log_rollout_table(pending=pending, step=actual_end)
        pending.unlink()
        self._update_shard_manifest(path=final, start=start, end=actual_end)
        self._enqueue_artifact("rollouts", final, {"start_step": start, "end_step": actual_end})
        return final

    def _update_shard_manifest(self, *, path: Path, start: int, end: int) -> None:
        manifest_path = self.state_dir / "rollout_shard_manifest.json"
        payload = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": "rollout_shards_v1", "shards": []}
        entry = {
            "path": str(path.relative_to(self.output_dir)),
            "start_step": start,
            "end_step": end,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        prior = [item for item in payload["shards"] if item["path"] != entry["path"]]
        payload["shards"] = sorted([*prior, entry], key=lambda item: (item["start_step"], item["end_step"]))
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, manifest_path)

    def _log_rollout_table(self, *, pending: Path, step: int) -> None:
        if self.run is None or self._wandb is None or self.settings.table_samples_per_shard == 0:
            return
        candidates: list[tuple[str, list[Any]]] = []
        for line in pending.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for index, sample in enumerate(record.get("sample_records", [])):
                sample_id = str(sample.get("instance_id", sample.get("question", f"sample-{index}")))
                key = hashlib.sha256(f"{record['step']}:{index}:{sample_id}".encode()).hexdigest()
                reward = sample.get("reward")
                if reward is None:
                    trajectories = [sample.get("trajectory_a", {}), sample.get("trajectory_b", {})]
                    values = [traj.get("task_reward") for traj in trajectories if isinstance(traj.get("task_reward"), (int, float))]
                    reward = sum(values) / len(values) if values else None
                candidates.append((key, [int(record["step"]), sample_id, reward, json.dumps(sample, ensure_ascii=False)]))
        rows = [row for _, row in sorted(candidates)[: self.settings.table_samples_per_shard]]
        if not rows:
            return
        try:
            table = self._wandb.Table(columns=["step", "sample_id", "reward", "sample_json"], data=rows)
            self.run.log({f"rollouts/table_ending_{step:06d}": table}, step=step, commit=False)
        except BaseException as exc:
            self._failure("log_rollout_table", exc)

    def log_artifact(self, *, kind: str, path: str | Path, metadata: dict[str, Any]) -> None:
        self._enqueue_artifact(kind, Path(path), metadata)

    def _enqueue_artifact(self, kind: str, path: Path, metadata: dict[str, Any]) -> None:
        if self.run is not None and self.settings.upload_artifacts:
            self._queue.put((kind, path, metadata))

    def _artifact_worker(self) -> None:
        assert self._wandb is not None and self.run is not None
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                kind, path, metadata = item
                artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{self.run.id}-{kind}-{path.name}")[:128]
                artifact = self._wandb.Artifact(
                    name=artifact_name, type=kind, metadata=metadata
                )
                if path.is_dir():
                    artifact.add_dir(str(path))
                else:
                    artifact.add_file(str(path))
                self.run.log_artifact(artifact)
            except BaseException as exc:
                self._failure("log_artifact", exc)
            finally:
                self._queue.task_done()

    def finish(self) -> None:
        if self._worker is not None:
            deadline = time.monotonic() + 1800.0
            while self._queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.5)
            if self._queue.unfinished_tasks:
                self._failure(
                    "finish_artifact_timeout",
                    TimeoutError(
                        f"W&B artifact queue retained {self._queue.unfinished_tasks} unfinished tasks"
                    ),
                )
                return
            self._queue.put(None)
            self._worker.join(timeout=30.0)
            if self._worker.is_alive():
                self._failure(
                    "finish_artifact_timeout",
                    TimeoutError(f"W&B artifact worker still active with {self._queue.qsize()} queued items"),
                )
                return
        if self.run is not None:
            try:
                self.run.finish()
                (self.state_dir / "wandb_finished").write_text("finished\n")
            except BaseException as exc:
                self._failure("finish", exc)


def rollback_rollout_shards(*, output_dir: str | Path, completed_step: int) -> None:
    """Quarantine rollout shards newer than an exact-resume boundary."""
    shard_dir = Path(output_dir) / "observability" / "rollout_shards"
    if not shard_dir.exists():
        return
    quarantine = shard_dir / f"superseded_after_step_{completed_step:06d}"
    for path in sorted(shard_dir.iterdir()):
        if not path.is_file() or not path.name.startswith("steps_"):
            continue
        parts = path.name.split("_")
        try:
            start = int(parts[1])
            end = int(parts[2].split(".")[0])
        except (IndexError, ValueError):
            continue
        if start > completed_step:
            quarantine.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), quarantine / path.name)
        elif end > completed_step:
            if path.name.endswith(".pending.jsonl"):
                rows = [
                    line for line in path.read_text().splitlines()
                    if line.strip() and int(json.loads(line)["step"]) <= completed_step
                ]
                if rows:
                    path.write_text("\n".join(rows) + "\n")
                    continue
            if path.exists():
                quarantine.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), quarantine / path.name)
    manifest_path = shard_dir.parent / "rollout_shard_manifest.json"
    entries = []
    for path in sorted(shard_dir.glob("steps_*.jsonl.gz")):
        parts = path.name.split("_")
        start = int(parts[1])
        end = int(parts[2].split(".")[0])
        entries.append(
            {
                "path": str(path.relative_to(Path(output_dir))),
                "start_step": start,
                "end_step": end,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema": "rollout_shards_v1", "shards": entries}, indent=2, sort_keys=True))
    os.replace(temporary, manifest_path)
