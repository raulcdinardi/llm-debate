from __future__ import annotations

import json
from pathlib import Path
import tempfile

import llm_local_rl.resource_monitor as resource_monitor
from llm_local_rl.resource_monitor import ResourceMonitor


def test_resource_monitor_records_stage_events_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(resource_monitor, "_gpu_samples", lambda: [])
    monkeypatch.setattr(resource_monitor, "_gpu_process_samples", lambda: [])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "resource_usage.jsonl"
        monitor = ResourceMonitor(output_path=path, interval_s=60.0)
        monitor.start()
        with monitor.stage("rollout_debate", step=3):
            monitor.record_event("sample")
        monitor.stop()

        records = [json.loads(line) for line in path.read_text().splitlines()]
        events = [record["event"] for record in records]
        assert events == ["monitor_start", "stage_start", "sample", "stage_end", "monitor_stop"]
        assert records[2]["stage"] == "rollout_debate"
        assert records[2]["stage_metadata"] == {"step": 3}
        assert records[2]["gpus"] == []
