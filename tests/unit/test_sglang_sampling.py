from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import threading
from typing import Any

import pytest

from llm_local_rl.behavior_policy import (
    BEHAVIOR_POLICY_LOGPROBS,
    RAW_MODEL_LOGPROBS,
    TEMPERATURE_SCALED_MODEL_LOGPROBS,
    BehaviorPolicySpec,
    validate_sampling_result_contract,
)
from llm_local_rl.model_io_trace import configure_model_io_tracing, reset_model_io_tracing
from llm_local_rl.sglang_sampling import (
    SglangRuntimeConfig,
    SglangSampler,
    _extract_completion_tokens_and_logprobs,
    _top_logprob_rows,
)
from llm_local_rl.types import SamplingRequest


def _make_adapter(path: Path, *, weight_bytes: bytes) -> str:
    path.mkdir()
    (path / "adapter_config.json").write_text('{"peft_type":"LORA"}')
    (path / "adapter_model.safetensors").write_bytes(weight_bytes)
    return str(path)


def _lora_salt(lora_name: str | None) -> int:
    if lora_name is None:
        return 17
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(lora_name))


def _expected_completion(prompt_ids: list[int], *, lora_name: str | None, seed: int | None) -> tuple[list[int], list[float]]:
    seed_salt = 0 if seed is None else int(seed) * 37
    salt = _lora_salt(lora_name) + seed_salt + sum((idx + 3) * tok for idx, tok in enumerate(prompt_ids))
    length = 1 + ((sum(prompt_ids) + salt + len(prompt_ids)) % 4)
    token_ids = [100 + ((salt + pos * 53) % 900) for pos in range(length)]
    logprobs = [
        -float(((token_id + 11) * (pos + 5) + salt) % 1_000_003) / 1_000_003.0
        for pos, token_id in enumerate(token_ids)
    ]
    return token_ids, logprobs


class _SglangHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response = self.server.handle_payload(self.path, payload)  # type: ignore[attr-defined]
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        _ = (format, args)


class _FakeSglangServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _SglangHandler)
        self.requests: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"

    def handle_payload(self, path: str, payload: dict) -> object:
        self.requests.append({"path": path, "payload": payload})
        if path == "/load_lora_adapter":
            return {"loaded": payload["lora_name"]}
        if path == "/unload_lora_adapter":
            return {"unloaded": payload["lora_name"]}
        if path == "/release_memory_occupation":
            return {"released": payload["tags"]}
        if path == "/resume_memory_occupation":
            return {"resumed": payload["tags"]}
        if path != "/generate":
            raise AssertionError(f"Unexpected SGLang fake endpoint {path!r}")

        input_ids = payload["input_ids"]
        if len(input_ids) > 0 and isinstance(input_ids[0], int):
            prompt_batch = [input_ids]
        else:
            prompt_batch = input_ids
        lora_name = payload["lora_path"] if "lora_path" in payload else None
        sampling_params = payload["sampling_params"]
        seed = sampling_params["sampling_seed"] if "sampling_seed" in sampling_params else None

        out = []
        for prompt_ids in prompt_batch:
            token_ids, logprobs = _expected_completion(prompt_ids, lora_name=lora_name, seed=seed)
            out.append(
                {
                    "text": " ".join(f"tok{token_id}" for token_id in token_ids),
                    "meta_info": {
                        "finish_reason": {"type": "length", "length": len(token_ids)},
                        "output_token_logprobs": [
                            [logprob, token_id, None]
                            for token_id, logprob in zip(token_ids, logprobs, strict=True)
                        ],
                    },
                }
            )
        return out if len(out) != 1 else out[0]


@pytest.fixture()
def fake_sglang_server():
    server = _FakeSglangServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _payloads(server: _FakeSglangServer, path: str) -> list[dict]:
    return [entry["payload"] for entry in server.requests if entry["path"] == path]


def test_load_adapter_accepts_only_matching_already_loaded_error(tmp_path: Path) -> None:
    sampler = object.__new__(SglangSampler)
    sampler.runtime = SglangRuntimeConfig()
    sampler._loaded_request_names = set()
    request_name = "judge__sgloraid_1"
    adapter_path = str(tmp_path / "judge")

    def already_loaded(_path: str, _payload: dict) -> object:
        raise RuntimeError(
            "SGLang HTTP 400 for load_lora_adapter: "
            f"LoRA adapter {request_name} with rank 32 failed because it is already loaded"
        )

    sampler._post_json = already_loaded
    sampler._load_adapter(request_name=request_name, adapter_path=adapter_path)
    assert sampler._loaded_request_names == {request_name}

    def incompatible(_path: str, _payload: dict) -> object:
        raise RuntimeError("SGLang HTTP 400 for load_lora_adapter: incompatible target modules")

    sampler._post_json = incompatible
    with pytest.raises(RuntimeError, match="incompatible target modules"):
        sampler._load_adapter(request_name="other", adapter_path=adapter_path)


def test_sglang_sampler_loads_loras_batches_token_id_requests_and_preserves_order(
    tmp_path: Path,
    fake_sglang_server: _FakeSglangServer,
) -> None:
    solution_adapter = _make_adapter(tmp_path / "solution", weight_bytes=b"solution")
    debate_adapter = _make_adapter(tmp_path / "debate", weight_bytes=b"debate")
    sampler = SglangSampler(
        runtime=SglangRuntimeConfig(base_url=fake_sglang_server.base_url, timeout_s=5.0),
        adapter_paths={"solution": solution_adapter, "debate": debate_adapter},
    )
    requests = [
        SamplingRequest(
            adapter_name="solution",
            prompt_token_ids=[11, 12, 13],
            stop_token_ids=[99],
            max_tokens=8,
            temperature=0.7,
            seed=123,
            min_p=0.02,
            top_p=0.95,
            stop_strings=("CONCLUDED",),
            include_stop_str_in_output=True,
        ),
        SamplingRequest(
            adapter_name="debate",
            prompt_token_ids=[21, 22],
            stop_token_ids=[99],
            max_tokens=8,
            temperature=0.7,
            seed=123,
            min_p=0.02,
            top_p=0.95,
            stop_strings=("CONCLUDED",),
            include_stop_str_in_output=True,
        ),
        SamplingRequest(
            adapter_name="solution",
            prompt_token_ids=[31, 32, 33, 34],
            stop_token_ids=[99],
            max_tokens=8,
            temperature=0.7,
            seed=123,
            min_p=0.02,
            top_p=0.95,
            stop_strings=("CONCLUDED",),
            include_stop_str_in_output=True,
        ),
    ]

    results = sampler.sample_many(requests)

    load_payloads = _payloads(fake_sglang_server, "/load_lora_adapter")
    assert [payload["lora_name"] for payload in load_payloads] == [
        "debate__sgloraid_1",
        "solution__sgloraid_2",
    ]
    generate_payloads = _payloads(fake_sglang_server, "/generate")
    assert len(generate_payloads) == 2
    assert all("text" not in payload for payload in generate_payloads)
    assert all(payload["return_logprob"] is True for payload in generate_payloads)
    assert all(payload["return_text_in_logprobs"] is False for payload in generate_payloads)
    assert all(payload["logprob_start_len"] == -1 for payload in generate_payloads)
    assert all(payload["sampling_params"]["sampling_seed"] == 123 for payload in generate_payloads)
    assert all(payload["sampling_params"]["top_p"] == 0.95 for payload in generate_payloads)
    assert all(payload["sampling_params"]["top_k"] == -1 for payload in generate_payloads)
    assert all(payload["sampling_params"]["min_p"] == 0.02 for payload in generate_payloads)
    assert all(payload["sampling_params"]["repetition_penalty"] == 1.0 for payload in generate_payloads)
    assert all(payload["sampling_params"]["stop_token_ids"] == [99] for payload in generate_payloads)
    assert all(payload["sampling_params"]["stop"] == ["CONCLUDED"] for payload in generate_payloads)
    assert all(payload["sampling_params"]["no_stop_trim"] is True for payload in generate_payloads)
    assert sorted(len(payload["input_ids"]) for payload in generate_payloads) == [1, 2]

    for request, result in zip(requests, results, strict=True):
        expected_tokens, expected_logprobs = _expected_completion(
            request.prompt_token_ids,
            lora_name=result.raw["sglang_lora_path"],
            seed=request.seed,
        )
        assert result.adapter_name == request.adapter_name
        assert result.prompt_token_ids == request.prompt_token_ids
        assert result.completion_token_ids == expected_tokens
        assert len(result.completion_logprobs) == len(expected_logprobs)
        for got, expected in zip(result.completion_logprobs, expected_logprobs, strict=True):
            assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
        assert result.raw["sampler_backend"] == "sglang"
        assert result.behavior_policy == BehaviorPolicySpec.from_sampling_request(request)
        assert result.completion_logprob_semantics == TEMPERATURE_SCALED_MODEL_LOGPROBS
        assert result.raw["completion_logprobs"] == TEMPERATURE_SCALED_MODEL_LOGPROBS

    sampler.close()
    unload_payloads = _payloads(fake_sglang_server, "/unload_lora_adapter")
    assert [payload["lora_name"] for payload in unload_payloads] == [
        "debate__sgloraid_1",
        "solution__sgloraid_2",
    ]


def test_sglang_sampler_distinguishes_trainable_behavior_from_greedy_judge_logprobs(
    tmp_path: Path,
    fake_sglang_server: _FakeSglangServer,
) -> None:
    adapter_path = _make_adapter(tmp_path / "solution", weight_bytes=b"policy")
    sampler = SglangSampler(
        runtime=SglangRuntimeConfig(base_url=fake_sglang_server.base_url, timeout_s=5.0),
        adapter_paths={"solution": adapter_path},
    )
    policy_request = SamplingRequest(
        adapter_name="solution",
        prompt_token_ids=[1, 2],
        stop_token_ids=[],
        max_tokens=4,
        temperature=0.8,
    )
    judge_request = SamplingRequest(
        adapter_name="solution",
        prompt_token_ids=[3, 4],
        stop_token_ids=[],
        max_tokens=4,
        temperature=0.0,
    )

    policy_result, judge_result = sampler.sample_many([policy_request, judge_request])

    assert policy_result.completion_logprob_semantics == BEHAVIOR_POLICY_LOGPROBS
    validate_sampling_result_contract(request=policy_request, result=policy_result)
    assert judge_result.completion_logprob_semantics == RAW_MODEL_LOGPROBS
    with pytest.raises(ValueError, match="normalized behavior policy"):
        validate_sampling_result_contract(request=judge_request, result=judge_result)


def test_sglang_sampler_hot_swap_reloads_overwritten_adapter_dir(
    tmp_path: Path,
    fake_sglang_server: _FakeSglangServer,
) -> None:
    adapter_path = Path(_make_adapter(tmp_path / "solution", weight_bytes=b"old"))
    sampler = SglangSampler(
        runtime=SglangRuntimeConfig(base_url=fake_sglang_server.base_url, timeout_s=5.0),
        adapter_paths={"solution": str(adapter_path)},
    )
    old_name = sampler._adapter_request_names["solution"]

    (adapter_path / "adapter_model.safetensors").write_bytes(b"new-larger-adapter-bytes")
    sampler.set_adapter_paths(adapter_paths={"solution": str(adapter_path)})
    new_name = sampler._adapter_request_names["solution"]

    assert old_name == "solution__sgloraid_1"
    assert new_name == "solution__sgloraid_2"
    load_names = [payload["lora_name"] for payload in _payloads(fake_sglang_server, "/load_lora_adapter")]
    unload_names = [payload["lora_name"] for payload in _payloads(fake_sglang_server, "/unload_lora_adapter")]
    assert load_names == ["solution__sgloraid_1", "solution__sgloraid_2"]
    assert unload_names == ["solution__sgloraid_1"]

    sampler.set_adapter_paths(adapter_paths={})
    unload_names = [payload["lora_name"] for payload in _payloads(fake_sglang_server, "/unload_lora_adapter")]
    assert unload_names == ["solution__sgloraid_1", "solution__sgloraid_2"]


def test_sglang_sampler_memory_saver_releases_resumes_and_resumes_before_lora_update(
    tmp_path: Path,
    fake_sglang_server: _FakeSglangServer,
) -> None:
    old_adapter = _make_adapter(tmp_path / "old", weight_bytes=b"old")
    new_adapter = _make_adapter(tmp_path / "new", weight_bytes=b"new")
    sampler = SglangSampler(
        runtime=SglangRuntimeConfig(
            base_url=fake_sglang_server.base_url,
            timeout_s=5.0,
            memory_saver=True,
            memory_saver_tags=("kv_cache",),
        ),
        adapter_paths={"solution": old_adapter},
    )

    sampler.sleep(level=2)
    sampler.sleep(level=2)
    sampler.wake_up(level=2)
    sampler.wake_up(level=2)
    sampler.sleep(level=2)
    sampler.set_adapter_paths(adapter_paths={"solution": new_adapter})

    assert _payloads(fake_sglang_server, "/release_memory_occupation") == [
        {"tags": ["kv_cache"]},
        {"tags": ["kv_cache"]},
    ]
    assert _payloads(fake_sglang_server, "/resume_memory_occupation") == [
        {"tags": ["kv_cache"]},
        {"tags": ["kv_cache"]},
    ]
    paths_after_second_release = [
        entry["path"]
        for entry in fake_sglang_server.requests
        if entry["path"] in {"/resume_memory_occupation", "/load_lora_adapter", "/unload_lora_adapter"}
    ]
    assert paths_after_second_release[-3:] == [
        "/resume_memory_occupation",
        "/load_lora_adapter",
        "/unload_lora_adapter",
    ]


def test_sglang_sampler_requests_top_logprobs_only_when_trace_needs_them(
    tmp_path: Path,
    fake_sglang_server: _FakeSglangServer,
) -> None:
    adapter_path = _make_adapter(tmp_path / "solution", weight_bytes=b"trace")
    reset_model_io_tracing()
    sampler = SglangSampler(
        runtime=SglangRuntimeConfig(base_url=fake_sglang_server.base_url, timeout_s=5.0),
        adapter_paths={"solution": adapter_path},
    )
    sampler.sample_many(
        [
            SamplingRequest(
                adapter_name="solution",
                prompt_token_ids=[1, 2, 3],
                stop_token_ids=[],
                max_tokens=4,
                temperature=0.0,
            )
        ]
    )
    untraced_generate = _payloads(fake_sglang_server, "/generate")[-1]
    assert "top_logprobs_num" not in untraced_generate

    configure_model_io_tracing(
        enabled=True,
        output_dir=tmp_path / "trace",
        tokenizer=None,
        top_logprobs=3,
    )
    sampler.sample_many(
        [
            SamplingRequest(
                adapter_name="solution",
                prompt_token_ids=[4, 5, 6],
                stop_token_ids=[],
                max_tokens=4,
                temperature=0.0,
            )
        ]
    )
    traced_generate = _payloads(fake_sglang_server, "/generate")[-1]
    assert traced_generate["top_logprobs_num"] == 3
    reset_model_io_tracing()


def test_sglang_sampler_reload_previously_unloaded_identity(
    tmp_path: Path,
    fake_sglang_server: _FakeSglangServer,
) -> None:
    adapter_path = _make_adapter(tmp_path / "solution", weight_bytes=b"stable")
    sampler = SglangSampler(
        runtime=SglangRuntimeConfig(base_url=fake_sglang_server.base_url, timeout_s=5.0),
        adapter_paths={"solution": adapter_path},
    )
    sampler.set_adapter_paths(adapter_paths={})
    sampler.set_adapter_paths(adapter_paths={"solution": adapter_path})

    assert [payload["lora_name"] for payload in _payloads(fake_sglang_server, "/load_lora_adapter")] == [
        "solution__sgloraid_1",
        "solution__sgloraid_1",
    ]
    assert [payload["lora_name"] for payload in _payloads(fake_sglang_server, "/unload_lora_adapter")] == [
        "solution__sgloraid_1",
    ]


def test_sglang_logprob_parser_fuzz_preserves_values_with_tiny_error() -> None:
    rng = random.Random(20260622)
    for _case_idx in range(500):
        rows = []
        expected_token_ids = []
        expected_logprobs = []
        for _pos in range(rng.randint(0, 12)):
            token_id = rng.randrange(0, 2**31 - 1)
            logprob = -rng.random() * rng.choice([1e-9, 1e-3, 1.0, 1e3])
            expected_token_ids.append(token_id)
            expected_logprobs.append(logprob)
            if rng.randrange(2) == 0:
                rows.append([logprob, token_id, None])
            else:
                rows.append((logprob, token_id, None))

        got_token_ids, got_logprobs = _extract_completion_tokens_and_logprobs(
            {"output_token_logprobs": rows}
        )

        assert got_token_ids == expected_token_ids
        for got, expected in zip(got_logprobs, expected_logprobs, strict=True):
            assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-15)


def test_sglang_top_logprob_parser_accepts_real_list_rows() -> None:
    rows = _top_logprob_rows(
        {
            "output_top_logprobs": [
                [[-0.12, 51, "A"], [-3.5, 9, "B"], [-0.48, 77, "C"]],
                [[-1.25, 12, None]],
                None,
            ]
        },
        max_alternatives=2,
    )

    assert rows == [
        [
            {"token_id": 51, "logprob": -0.12, "rank": 1},
            {"token_id": 77, "logprob": -0.48, "rank": 2},
        ],
        [{"token_id": 12, "logprob": -1.25, "rank": 1}],
        [],
    ]


def test_sglang_top_logprob_parser_accepts_single_record_rows() -> None:
    rows = _top_logprob_rows(
        {"output_top_logprobs": [[-0.75, 42, None], [-0.2, 43, "x"]]},
        max_alternatives=3,
    )

    assert rows == [
        [{"token_id": 42, "logprob": -0.75, "rank": 1}],
        [{"token_id": 43, "logprob": -0.2, "rank": 1}],
    ]
