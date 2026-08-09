from __future__ import annotations

import json
from pathlib import Path
import tempfile

import llm_local_rl.base_model_judge as base_model_judge
from llm_local_rl.base_model_judge import RemoteBaseJudgeConfig, build_remote_base_judge
from llm_local_rl.model_io_trace import (
    configure_model_io_tracing,
    reset_model_io_tracing,
    trace_context,
)
from llm_local_rl.types import SamplingRequest, SamplingResult, TrainExample
from llm_local_rl.vllm_sampling import _extract_top_token_logprobs


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class FakeLogprob:
    def __init__(self, logprob: float, decoded_token: str) -> None:
        self.logprob = logprob
        self.decoded_token = decoded_token


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_disabled_tracer_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            tracer = configure_model_io_tracing(enabled=False, output_dir=tmpdir, tokenizer=TinyTokenizer())
            tracer.record_custom(phase="x", boundary="test", request={}, response={}, error=None)
            assert not (Path(tmpdir) / "model_io_trace.jsonl").exists()
        finally:
            reset_model_io_tracing()


def test_generation_trace_records_tokens_topk_and_viewer() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            tracer = configure_model_io_tracing(
                enabled=True,
                output_dir=tmpdir,
                tokenizer=TinyTokenizer(),
                top_logprobs=2,
            )
            request = SamplingRequest(
                adapter_name="shared",
                prompt_token_ids=[65, 66],
                stop_token_ids=[10],
                max_tokens=4,
                temperature=0.0,
            )
            result = SamplingResult(
                adapter_name="shared",
                prompt_token_ids=[65, 66],
                completion_token_ids=[67],
                completion_logprobs=[-0.25],
                text="C",
                raw={
                    "finish_reason": "stop",
                    "completion_top_logprobs": [
                        [
                            {"rank": 1, "token_id": 67, "text": "C", "logprob": -0.25, "chosen": True},
                            {"rank": 2, "token_id": 68, "text": "D", "logprob": -1.25, "chosen": False},
                        ]
                    ],
                },
            )
            with trace_context(step=3, round_num=1):
                tracer.record_generation(request=request, result=result, boundary="test.sample_many")

            records = _read_jsonl(Path(tmpdir) / "model_io_trace.jsonl")
            assert (Path(tmpdir) / "index.html").exists()
            assert len(records) == 1
            record = records[0]
            assert record["phase"] == "generation"
            assert record["context"]["step"] == 3
            assert record["request"]["prompt_tokens"][0]["text"] == "A"
            assert record["response"]["completion_tokens"][0]["token_id"] == 67
            assert record["response"]["completion_tokens"][0]["top_logprobs"][1]["token_id"] == 68
            assert record["exactness"]["completion_token_ids"] == "actual_model_output"
        finally:
            reset_model_io_tracing()


def test_vllm_top_logprob_parser_keeps_chosen_token_when_outside_requested_top_k() -> None:
    parsed = _extract_top_token_logprobs(
        token_ids=[7],
        token_logprobs=[
            {
                7: FakeLogprob(-0.3, "chosen"),
                8: FakeLogprob(-0.1, "best"),
                9: FakeLogprob(-1.0, "low"),
            }
        ],
        max_alternatives=1,
    )
    assert parsed[0][0]["token_id"] == 8
    assert parsed[0][0]["rank"] == 1
    assert parsed[0][1]["token_id"] == 7
    assert parsed[0][1]["chosen"] is True


def test_trainer_forward_trace_records_tensors_and_target_topk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            tracer = configure_model_io_tracing(enabled=True, output_dir=tmpdir, tokenizer=TinyTokenizer())
            example = TrainExample(
                adapter_name="shared",
                input_ids=[65, 66],
                target_ids=[66, 67],
                loss_mask=[0, 1],
                behavior_logprob_mask=[0, 1],
                old_logprobs=[0.0, -0.5],
                advantages=[0.0, 1.0],
                metadata={"instance_id": "abc"},
            )
            tensors = {
                "input_ids": [[65, 66]],
                "target_ids": [[66, 67]],
                "attention_mask": [[1, 1]],
                "loss_mask": [[False, True]],
                "behavior_logprob_mask": [[False, True]],
                "old_logprobs": [[0.0, -0.5]],
                "advantages": [[0.0, 1.0]],
            }
            tracer.record_trainer_forward(
                phase="trainer_forward",
                boundary="test.train_batch",
                adapter_name="shared",
                batch=[example],
                tensors=tensors,
                minibatch_start=4,
                token_logprobs=[[-0.9, -0.8]],
                top_token_ids=[[[66, 67], [67, 68]]],
                top_logprobs=[[[-0.1, -1.1], [-0.2, -1.2]]],
            )

            record = _read_jsonl(Path(tmpdir) / "model_io_trace.jsonl")[0]
            row = record["request"]["rows"][0]
            assert row["example_index"] == 4
            assert row["behavior_logprob_mask"] == [False, True]
            assert row["metadata"]["instance_id"] == "abc"
            assert row["input_tokens"][0]["text"] == "A"
            assert row["target_top_logprobs"][1][0]["token_id"] == 67
            assert record["exactness"]["input_ids"] == "actual_model_input_tensor"
        finally:
            reset_model_io_tracing()


def test_external_judge_trace_records_prompt_body_and_raw_response(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"raw_text": "<VERDICT>A</VERDICT>"}'

    def fake_urlopen(req, timeout):
        _ = (req, timeout)
        return FakeResponse()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            configure_model_io_tracing(enabled=True, output_dir=tmpdir, tokenizer=TinyTokenizer())
            monkeypatch.setattr(base_model_judge.request, "urlopen", fake_urlopen)
            judge = build_remote_base_judge(RemoteBaseJudgeConfig(url="http://judge.test"))

            verdict, raw_text = judge("Question", "Rules", "A1", "B1", "A2", "B2", "A3", "B3")

            assert verdict == "A"
            assert raw_text == "<VERDICT>A</VERDICT>"
            record = _read_jsonl(Path(tmpdir) / "model_io_trace.jsonl")[0]
            assert record["phase"] == "external_judge"
            assert record["request"]["body"]["prompt_text"].startswith("System:")
            assert record["response"]["verdict"] == "A"
            assert record["exactness"]["token_ids"] == "local_visualization_only"
        finally:
            reset_model_io_tracing()


def test_external_judge_batches_prompts_and_preserves_response_order(monkeypatch) -> None:
    seen_body: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"results":[{"raw_text":"<VERDICT>B</VERDICT> second"},{"raw_text":"<VERDICT>A</VERDICT> first"}]}'

    def fake_urlopen(req, timeout):
        _ = timeout
        seen_body.update(json.loads(req.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(base_model_judge.request, "urlopen", fake_urlopen)
    judge = build_remote_base_judge(RemoteBaseJudgeConfig(url="http://judge.test"))
    common = ("Rules", "A1", "B1", "A2", "B2", "A3", "B3")

    results = judge.judge_many([("Question 1", *common), ("Question 2", *common)])

    assert results == [
        ("B", "<VERDICT>B</VERDICT> second"),
        ("A", "<VERDICT>A</VERDICT> first"),
    ]
    assert len(seen_body["prompt_texts"]) == 2
    assert "Question 1" in seen_body["prompt_texts"][0]
    assert "Question 2" in seen_body["prompt_texts"][1]
