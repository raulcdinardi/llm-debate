from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VllmWorkerProcess:
    base_model: str
    _proc: subprocess.Popen[str]
    _lock: threading.Lock
    _next_request_id: int
    _seen_lora_names: set[str]

    @classmethod
    def start(cls, *, base_model: str) -> "VllmWorkerProcess":
        env = os.environ.copy()
        local_src = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH", "")
        if local_src not in existing.split(":"):
            env["PYTHONPATH"] = f"{local_src}:{existing}" if existing else local_src

        proc = subprocess.Popen(
            [sys.executable, "-m", "tinker.vllm_worker", "--model", base_model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if proc.stdin is None:
            raise RuntimeError("Failed to open stdin for vLLM worker process.")
        if proc.stdout is None:
            raise RuntimeError("Failed to open stdout for vLLM worker process.")
        return cls(
            base_model=base_model,
            _proc=proc,
            _lock=threading.Lock(),
            _next_request_id=1,
            _seen_lora_names=set(),
        )

    def terminate(self) -> None:
        self._proc.terminate()
        self._proc.wait(timeout=30)

    def _maybe_restart(self) -> None:
        env = os.environ.get("TINKER_LOCAL_VLLM_RESTART_EVERY")
        if env is None:
            return
        every = int(env)
        if every <= 0:
            return
        if len(self._seen_lora_names) < every:
            return
        self.terminate()
        restarted = VllmWorkerProcess.start(base_model=self.base_model)
        self._proc = restarted._proc
        self._next_request_id = restarted._next_request_id
        self._seen_lora_names = restarted._seen_lora_names

    def generate(
        self,
        *,
        prompt_token_ids: list[list[int]],
        temperature: float,
        max_tokens: int,
        top_k: int,
        top_p: float,
        min_p: float,
        seed: int | None,
        stop_token_ids: list[int],
        lora_name: str,
        lora_path: str,
    ) -> list[dict]:
        with self._lock:
            self._maybe_restart()

            self._seen_lora_names.add(lora_name)
            request_id = self._next_request_id
            self._next_request_id += 1

            msg = {
                "type": "generate",
                "request_id": request_id,
                "prompt_token_ids": prompt_token_ids,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "min_p": float(min_p),
                "seed": seed,
                "stop_token_ids": stop_token_ids,
                "lora_name": lora_name,
                "lora_path": lora_path,
            }

            proc = self._proc
            assert proc.stdin is not None
            assert proc.stdout is not None

            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("vLLM worker process exited or produced no output.")
            line = line.strip()
            while line and not line.startswith("{"):
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("vLLM worker process exited while skipping non-JSON output.")
                line = line.strip()
            if not line:
                raise RuntimeError("vLLM worker produced empty output.")
            out = json.loads(line)
            if out["type"] != "generate_result":
                raise RuntimeError("Unexpected response type from vLLM worker.")
            if out["request_id"] != request_id:
                raise RuntimeError("Mismatched request_id from vLLM worker.")
            return out["outputs"]


_WORKER: VllmWorkerProcess | None = None


def get_worker(*, base_model: str) -> VllmWorkerProcess:
    global _WORKER
    if _WORKER is None:
        _WORKER = VllmWorkerProcess.start(base_model=base_model)
        return _WORKER
    if _WORKER.base_model != base_model:
        raise ValueError(f"vLLM worker already initialized with base_model={_WORKER.base_model!r}")
    return _WORKER


def terminate_worker() -> None:
    global _WORKER
    if _WORKER is None:
        return
    _WORKER.terminate()
    _WORKER = None
