from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_csv(*args: str) -> list[list[str]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows: list[list[str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append([part.strip() for part in line.split(",")])
    return rows


def _gpu_samples() -> list[dict]:
    rows = _query_csv(
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    out = []
    for row in rows:
        if len(row) < 7:
            continue
        out.append(
            {
                "index": int(row[0]),
                "name": row[1],
                "memory_used_mib": float(row[2]),
                "memory_total_mib": float(row[3]),
                "utilization_gpu_pct": float(row[4]),
                "power_draw_w": None if row[5] == "[Not Supported]" else float(row[5]),
                "temperature_c": float(row[6]),
            }
        )
    return out


def _gpu_process_samples() -> list[dict]:
    rows = _query_csv(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    out = []
    for row in rows:
        if len(row) < 4:
            continue
        try:
            pid = int(row[1])
            used_memory_mib = float(row[3])
        except ValueError:
            continue
        out.append(
            {
                "gpu_uuid": row[0],
                "pid": pid,
                "process_name": row[2],
                "used_memory_mib": used_memory_mib,
            }
        )
    return out


@dataclass
class ResourceMonitor:
    output_path: Path
    interval_s: float = 5.0
    enabled: bool = True
    _stage: str = "startup"
    _stage_metadata: dict = field(default_factory=dict)
    _started_at: float = field(default_factory=time.monotonic)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_event("monitor_start")
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self.record_event("monitor_stop")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s + 1.0))

    @contextmanager
    def stage(self, name: str, **metadata: object) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        previous_stage, previous_metadata = self.set_stage(name, **metadata)
        self.record_event("stage_start", stage=name, metadata=metadata)
        try:
            yield
        finally:
            self.record_event("stage_end", stage=name, metadata=metadata)
            self.set_stage(previous_stage, **previous_metadata)

    def set_stage(self, name: str, **metadata: object) -> tuple[str, dict]:
        with self._lock:
            previous_stage = self._stage
            previous_metadata = dict(self._stage_metadata)
            self._stage = name
            self._stage_metadata = dict(metadata)
        return previous_stage, previous_metadata

    def record_event(self, event: str, *, stage: str | None = None, metadata: dict | None = None) -> None:
        self._write_record(event=event, stage_override=stage, metadata_override=metadata)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._write_record(event="sample")

    def _write_record(
        self,
        *,
        event: str,
        stage_override: str | None = None,
        metadata_override: dict | None = None,
    ) -> None:
        with self._lock:
            stage = self._stage
            stage_metadata = dict(self._stage_metadata)
        if stage_override is not None:
            stage = stage_override
        if metadata_override is not None:
            stage_metadata = dict(metadata_override)

        record = {
            "event": event,
            "stage": stage,
            "stage_metadata": stage_metadata,
            "timestamp": _utc_now(),
            "elapsed_s": round(time.monotonic() - self._started_at, 3),
            "gpus": _gpu_samples(),
            "gpu_processes": _gpu_process_samples(),
        }
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
