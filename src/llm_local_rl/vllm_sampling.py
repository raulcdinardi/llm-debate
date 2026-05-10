from __future__ import annotations

from dataclasses import dataclass
import os

from llm_local_rl.model_io_trace import get_model_io_tracer, get_trace_top_logprobs
from llm_local_rl.types import SamplingRequest, SamplingResult


def _logprob_value(entry: object) -> float:
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    return float(entry)


def _decoded_token_value(entry: object) -> str | None:
    for attr in ("decoded_token", "token"):
        if hasattr(entry, attr):
            value = getattr(entry, attr)
            if value is not None:
                return str(value)
    return None


def _extract_token_logprobs(*, token_ids: list[int], token_logprobs: object) -> list[float]:
    if token_logprobs is None:
        raise RuntimeError("vLLM did not return per-token logprobs.")
    if not isinstance(token_logprobs, list):
        raise TypeError(f"Unexpected token_logprobs type: {type(token_logprobs).__name__}")
    if len(token_logprobs) != len(token_ids):
        raise ValueError("Token/logprob length mismatch.")

    out: list[float] = []
    for token_id, row in zip(token_ids, token_logprobs, strict=True):
        if not isinstance(row, dict):
            raise TypeError("Expected vLLM per-token logprob rows to be dicts.")
        entry = row.get(int(token_id))
        if entry is None:
            raise RuntimeError("Missing generated token id in vLLM logprob row.")
        out.append(_logprob_value(entry))
    return out


def _extract_top_token_logprobs(
    *,
    token_ids: list[int],
    token_logprobs: object,
    max_alternatives: int,
) -> list[list[dict]]:
    if max_alternatives <= 0:
        return []
    if token_logprobs is None:
        return [[] for _ in token_ids]
    if not isinstance(token_logprobs, list):
        raise TypeError(f"Unexpected token_logprobs type: {type(token_logprobs).__name__}")
    if len(token_logprobs) != len(token_ids):
        raise ValueError("Token/logprob length mismatch.")

    out: list[list[dict]] = []
    for chosen_token_id, row in zip(token_ids, token_logprobs, strict=True):
        if not isinstance(row, dict):
            raise TypeError("Expected vLLM per-token logprob rows to be dicts.")
        ranked = []
        for token_id, entry in row.items():
            token_id_int = int(token_id)
            ranked.append(
                {
                    "token_id": token_id_int,
                    "text": _decoded_token_value(entry),
                    "logprob": _logprob_value(entry),
                    "chosen": token_id_int == int(chosen_token_id),
                }
            )
        ranked.sort(key=lambda item: item["logprob"], reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        selected = ranked[:max_alternatives]
        if not any(item["chosen"] for item in selected):
            chosen = next((item for item in ranked if item["chosen"]), None)
            if chosen is not None:
                selected.append(chosen)
        out.append(selected)
    return out


@dataclass(frozen=True)
class VllmRuntimeConfig:
    model_path: str
    gpu_memory_utilization: float = 0.55
    max_model_len: int = 64
    enforce_eager: bool = True
    max_lora_rank: int = 32
    max_loras: int = 4


def _import_vllm_symbols():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    return LLM, SamplingParams, LoRARequest


class VllmSampler:
    def __init__(
        self,
        *,
        runtime: VllmRuntimeConfig,
        adapter_paths: dict[str, str] | None = None,
    ) -> None:
        LLM, _SamplingParams, _LoRARequest = _import_vllm_symbols()
        _ = (_SamplingParams, _LoRARequest)

        self.runtime = runtime
        self.adapter_paths = {} if adapter_paths is None else dict(adapter_paths)
        self._adapter_ids = {name: idx + 1 for idx, name in enumerate(sorted(self.adapter_paths))}
        self._llm = LLM(
            model=runtime.model_path,
            enable_lora=bool(self.adapter_paths),
            max_lora_rank=runtime.max_lora_rank,
            max_loras=runtime.max_loras,
            gpu_memory_utilization=runtime.gpu_memory_utilization,
            max_model_len=runtime.max_model_len,
            enforce_eager=runtime.enforce_eager,
        )
        self._sleep_level: int | None = None

    @property
    def llm(self):
        return self._llm

    def set_adapter_paths(self, *, adapter_paths: dict[str, str]) -> None:
        self.adapter_paths = dict(adapter_paths)
        self._adapter_ids = {name: idx + 1 for idx, name in enumerate(sorted(self.adapter_paths))}

    def sleep(self, *, level: int = 1) -> None:
        self._llm.sleep(level=level)
        self._sleep_level = level

    def wake_up(self) -> None:
        if self._sleep_level is not None:
            self._llm.wake_up()
            self._sleep_level = None

    def close(self) -> None:
        llm_engine = getattr(self._llm, "llm_engine", None)
        if llm_engine is None:
            return
        engine_core = getattr(llm_engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        elif hasattr(llm_engine, "model_executor") and hasattr(llm_engine.model_executor, "shutdown"):
            llm_engine.model_executor.shutdown()

    def _build_lora_request(self, *, adapter_name: str):
        if adapter_name not in self.adapter_paths:
            return None
        _LLM, _SamplingParams, LoRARequest = _import_vllm_symbols()
        _ = (_LLM, _SamplingParams)

        return LoRARequest(
            adapter_name,
            self._adapter_ids[adapter_name],
            self.adapter_paths[adapter_name],
        )

    def sample(self, request: SamplingRequest) -> SamplingResult:
        return self.sample_many([request])[0]

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        if len(requests) == 0:
            return []
        _LLM, SamplingParams, _LoRARequest = _import_vllm_symbols()
        _ = (_LLM, _LoRARequest)
        trace_top_logprobs = get_trace_top_logprobs()
        grouped: dict[tuple[str, float, float, int, tuple[int, ...], int | None], list[tuple[int, SamplingRequest]]] = {}
        for idx, request in enumerate(requests):
            key = (
                request.adapter_name,
                float(request.temperature),
                float(request.min_p),
                int(request.max_tokens),
                tuple(int(tok) for tok in request.stop_token_ids),
                request.seed,
            )
            grouped.setdefault(key, []).append((idx, request))

        results: list[SamplingResult | None] = [None] * len(requests)
        for (adapter_name, temperature, min_p, max_tokens, stop_token_ids, seed), grouped_requests in grouped.items():
            sampling_params = SamplingParams(
                temperature=temperature,
                min_p=min_p,
                max_tokens=max_tokens,
                stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
                logprobs=max(1, trace_top_logprobs),
                seed=seed,
            )
            outputs = self._llm.generate(
                [{"prompt_token_ids": req.prompt_token_ids} for _, req in grouped_requests],
                sampling_params=sampling_params,
                lora_request=self._build_lora_request(adapter_name=adapter_name),
                use_tqdm=False,
            )
            if len(outputs) != len(grouped_requests):
                raise RuntimeError("Expected one output per request.")
            for output, (request_idx, request) in zip(outputs, grouped_requests, strict=True):
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected exactly one output sequence.")
                seq = output.outputs[0]
                completion_token_ids = list(seq.token_ids)
                completion_logprobs = _extract_token_logprobs(
                    token_ids=completion_token_ids,
                    token_logprobs=seq.logprobs,
                )
                completion_top_logprobs = _extract_top_token_logprobs(
                    token_ids=completion_token_ids,
                    token_logprobs=seq.logprobs,
                    max_alternatives=trace_top_logprobs,
                )
                text = getattr(seq, "text", None) or ""
                raw = {"finish_reason": getattr(seq, "finish_reason", None)}
                if trace_top_logprobs > 0:
                    raw["completion_top_logprobs"] = completion_top_logprobs
                result = SamplingResult(
                    adapter_name=request.adapter_name,
                    prompt_token_ids=list(request.prompt_token_ids),
                    completion_token_ids=completion_token_ids,
                    completion_logprobs=completion_logprobs,
                    text=text,
                    raw=raw,
                )
                get_model_io_tracer().record_generation(
                    request=request,
                    result=result,
                    boundary="llm_local_rl.vllm_sampling.VllmSampler.sample_many",
                )
                results[request_idx] = result
        if any(result is None for result in results):
            raise AssertionError("All sample_many results must be populated.")
        return [result for result in results if result is not None]


def direct_vllm_sample(
    *,
    llm,
    request: SamplingRequest,
    adapter_path: str | None,
    adapter_id: int,
) -> SamplingResult:
    _LLM, SamplingParams, LoRARequest = _import_vllm_symbols()
    _ = _LLM
    trace_top_logprobs = get_trace_top_logprobs()

    sampling_params = SamplingParams(
        temperature=float(request.temperature),
        min_p=float(request.min_p),
        max_tokens=int(request.max_tokens),
        stop_token_ids=list(request.stop_token_ids) if request.stop_token_ids else None,
        logprobs=max(1, trace_top_logprobs),
        seed=request.seed,
    )
    lora_request = None
    if adapter_path is not None:
        lora_request = LoRARequest(request.adapter_name, int(adapter_id), adapter_path)
    outputs = llm.generate(
        [{"prompt_token_ids": request.prompt_token_ids}],
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("Expected exactly one output sequence.")
    seq = outputs[0].outputs[0]
    completion_token_ids = list(seq.token_ids)
    completion_logprobs = _extract_token_logprobs(
        token_ids=completion_token_ids,
        token_logprobs=seq.logprobs,
    )
    completion_top_logprobs = _extract_top_token_logprobs(
        token_ids=completion_token_ids,
        token_logprobs=seq.logprobs,
        max_alternatives=trace_top_logprobs,
    )
    text = getattr(seq, "text", None)
    if text is None:
        text = ""
    raw = {"finish_reason": getattr(seq, "finish_reason", None)}
    if trace_top_logprobs > 0:
        raw["completion_top_logprobs"] = completion_top_logprobs
    result = SamplingResult(
        adapter_name=request.adapter_name,
        prompt_token_ids=list(request.prompt_token_ids),
        completion_token_ids=completion_token_ids,
        completion_logprobs=completion_logprobs,
        text=text,
        raw=raw,
    )
    get_model_io_tracer().record_generation(
        request=request,
        result=result,
        boundary="llm_local_rl.vllm_sampling.direct_vllm_sample",
    )
    return result
