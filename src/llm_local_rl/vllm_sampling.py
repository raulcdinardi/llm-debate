from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import warnings

from llm_local_rl.lora_identity import AdapterIdentity, adapter_identity
from llm_local_rl.model_io_trace import get_model_io_tracer, get_trace_top_logprobs
from llm_local_rl.types import SamplingRequest, SamplingResult

_RAW_LOGPROBS_LEGACY_MAX_VERSION = (0, 9, 99)
_WARNED_RAW_LOGPROB_SAMPLING_POLICY = False


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
    max_num_seqs: int | None = None
    enforce_eager: bool = True
    enable_sleep_mode: bool = False
    max_lora_rank: int = 32
    max_loras: int = 4


def _import_vllm_symbols():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    return LLM, SamplingParams, LoRARequest


def _vllm_version() -> str:
    import vllm

    return str(getattr(vllm, "__version__", ""))


def _parse_version_tuple(version: str) -> tuple[int, int, int] | None:
    parts = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    if not parts:
        return None
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _raw_logprobs_mode_kwargs(SamplingParams) -> dict[str, str]:
    try:
        sampling_params_parameters = inspect.signature(SamplingParams).parameters
    except (TypeError, ValueError):
        sampling_params_parameters = {}
    if "logprobs_mode" in sampling_params_parameters:
        return {"logprobs_mode": "raw_logprobs"}

    version = _vllm_version()
    parsed_version = _parse_version_tuple(version)
    # Older vLLM releases did not expose logprobs_mode; this path assumes
    # their completion logprobs are raw model logprobs. Modern versions should
    # expose logprobs_mode, so fail loudly if that contract is not visible.
    if parsed_version is None or parsed_version > _RAW_LOGPROBS_LEGACY_MAX_VERSION:
        raise RuntimeError(
            "vLLM SamplingParams does not expose logprobs_mode, so raw completion logprob semantics "
            f"cannot be pinned for vllm.__version__={version!r}. Upgrade vLLM or verify the API contract."
        )
    return {}


def _warn_if_raw_logprobs_sampling_policy_differs(
    *,
    temperature: float,
    min_p: float,
    logprobs: int,
) -> None:
    global _WARNED_RAW_LOGPROB_SAMPLING_POLICY
    if _WARNED_RAW_LOGPROB_SAMPLING_POLICY or logprobs <= 0:
        return
    if float(temperature) == 1.0 and float(min_p) <= 0.0:
        return
    _WARNED_RAW_LOGPROB_SAMPLING_POLICY = True
    warnings.warn(
        "vLLM sampling is requesting raw model logprobs while temperature != 1.0 or min_p > 0. "
        "The sampled tokens come from the transformed behavior policy, but stored logprobs are raw model logprobs.",
        RuntimeWarning,
        stacklevel=3,
    )


def _build_sampling_params(
    SamplingParams,
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    max_tokens: int,
    stop_token_ids: tuple[int, ...] | list[int],
    seed: int | None,
    trace_top_logprobs: int,
):
    logprobs = max(1, trace_top_logprobs)
    _warn_if_raw_logprobs_sampling_policy_differs(
        temperature=temperature,
        min_p=min_p,
        logprobs=logprobs,
    )
    kwargs = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "max_tokens": int(max_tokens),
        "stop_token_ids": list(stop_token_ids) if stop_token_ids else None,
        "logprobs": logprobs,
        "seed": seed,
    }
    kwargs.update(_raw_logprobs_mode_kwargs(SamplingParams))
    return SamplingParams(**kwargs)


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
        initial_adapter_paths = {} if adapter_paths is None else dict(adapter_paths)
        self.adapter_paths: dict[str, str] = {}
        self._adapter_ids: dict[str, int] = {}
        self._adapter_request_names: dict[str, str] = {}
        self._adapter_id_by_identity: dict[AdapterIdentity, int] = {}
        self._next_adapter_id = 1
        self.set_adapter_paths(adapter_paths=initial_adapter_paths)
        llm_kwargs = {
            "model": runtime.model_path,
            "enable_lora": bool(self.adapter_paths),
            "max_lora_rank": runtime.max_lora_rank,
            "max_loras": runtime.max_loras,
            "gpu_memory_utilization": runtime.gpu_memory_utilization,
            "max_model_len": runtime.max_model_len,
            "enforce_eager": runtime.enforce_eager,
        }
        if runtime.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = runtime.max_num_seqs
        if runtime.enable_sleep_mode:
            llm_kwargs["enable_sleep_mode"] = True
        self._llm = LLM(**llm_kwargs)
        self._sleep_level: int | None = None

    @property
    def llm(self):
        return self._llm

    def set_adapter_paths(self, *, adapter_paths: dict[str, str]) -> None:
        self.adapter_paths = dict(adapter_paths)
        self._adapter_ids = {}
        self._adapter_request_names = {}
        for name in sorted(self.adapter_paths):
            identity = adapter_identity(self.adapter_paths[name])
            if identity not in self._adapter_id_by_identity:
                self._adapter_id_by_identity[identity] = self._next_adapter_id
                self._next_adapter_id += 1
            adapter_id = self._adapter_id_by_identity[identity]
            self._adapter_ids[name] = adapter_id
            self._adapter_request_names[name] = f"{name}__loraid_{adapter_id}"

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
            self._adapter_request_names[adapter_name],
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
        grouped: dict[tuple[str, float, float, float, int, tuple[int, ...], int | None], list[tuple[int, SamplingRequest]]] = {}
        for idx, request in enumerate(requests):
            if request.stop_strings:
                raise NotImplementedError("String stops are pinned to the SGLang sampler backend.")
            key = (
                request.adapter_name,
                float(request.temperature),
                float(request.min_p),
                float(request.top_p),
                int(request.max_tokens),
                tuple(int(tok) for tok in request.stop_token_ids),
                request.seed,
            )
            grouped.setdefault(key, []).append((idx, request))

        results: list[SamplingResult | None] = [None] * len(requests)
        for (adapter_name, temperature, min_p, top_p, max_tokens, stop_token_ids, seed), grouped_requests in grouped.items():
            sampling_params = _build_sampling_params(
                SamplingParams,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                seed=seed,
                trace_top_logprobs=trace_top_logprobs,
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
                raw = {
                    "finish_reason": getattr(seq, "finish_reason", None),
                    "completion_logprobs": "raw_model_logprobs",
                }
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
    if request.stop_strings:
        raise NotImplementedError("String stops are pinned to the SGLang sampler backend.")
    _LLM, SamplingParams, LoRARequest = _import_vllm_symbols()
    _ = _LLM
    trace_top_logprobs = get_trace_top_logprobs()

    sampling_params = _build_sampling_params(
        SamplingParams,
        temperature=float(request.temperature),
        top_p=float(request.top_p),
        min_p=float(request.min_p),
        max_tokens=int(request.max_tokens),
        stop_token_ids=request.stop_token_ids,
        seed=request.seed,
        trace_top_logprobs=trace_top_logprobs,
    )
    lora_request = None
    if adapter_path is not None:
        lora_request = LoRARequest(f"{request.adapter_name}__loraid_{int(adapter_id)}", int(adapter_id), adapter_path)
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
    raw = {
        "finish_reason": getattr(seq, "finish_reason", None),
        "completion_logprobs": "raw_model_logprobs",
    }
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
