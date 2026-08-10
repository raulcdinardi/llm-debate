from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.error import HTTPError
from urllib import request as urlrequest

from llm_local_rl.behavior_policy import (
    BEHAVIOR_POLICY_LOGPROBS,
    RAW_MODEL_LOGPROBS,
    TEMPERATURE_SCALED_MODEL_LOGPROBS,
    BehaviorPolicySpec,
    behavior_policy_contract_record,
)
from llm_local_rl.lora_identity import AdapterIdentity, adapter_identity
from llm_local_rl.model_io_trace import get_model_io_tracer, get_trace_top_logprobs
from llm_local_rl.types import SamplingRequest, SamplingResult


def _extract_logprob_row(row: object) -> tuple[int, float]:
    if not isinstance(row, (list, tuple)):
        raise TypeError(f"Unexpected SGLang output_token_logprobs row type: {type(row).__name__}")
    if len(row) < 2:
        raise ValueError("SGLang output_token_logprobs row must contain at least logprob and token id.")
    return int(row[1]), float(row[0])


def _extract_top_logprob_entry(entry: object) -> tuple[int, float]:
    if isinstance(entry, dict):
        if "token_id" in entry and "logprob" in entry:
            return int(entry["token_id"]), float(entry["logprob"])
        if len(entry) != 1:
            raise ValueError("SGLang output_top_logprobs dict entries must be token/logprob records or one-item maps.")
        token_id, logprob = next(iter(entry.items()))
        return int(token_id), float(logprob)

    if not isinstance(entry, (list, tuple)):
        raise TypeError(f"Unexpected SGLang output_top_logprobs entry type: {type(entry).__name__}")
    if len(entry) < 2:
        raise ValueError("SGLang output_top_logprobs entries must contain at least logprob and token id.")
    return int(entry[1]), float(entry[0])


def _is_sglang_logprob_record(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (float, int))
    )


def _extract_completion_tokens_and_logprobs(meta_info: dict) -> tuple[list[int], list[float]]:
    rows = meta_info["output_token_logprobs"]
    if not isinstance(rows, list):
        raise TypeError("SGLang meta_info['output_token_logprobs'] must be a list.")

    token_ids: list[int] = []
    logprobs: list[float] = []
    for row in rows:
        token_id, logprob = _extract_logprob_row(row)
        token_ids.append(token_id)
        logprobs.append(logprob)
    return token_ids, logprobs


def _top_logprob_rows(meta_info: dict, *, max_alternatives: int) -> list[list[dict]]:
    if max_alternatives <= 0:
        return []
    if "output_top_logprobs" not in meta_info:
        return []

    rows = meta_info["output_top_logprobs"]
    if not isinstance(rows, list):
        raise TypeError("SGLang meta_info['output_top_logprobs'] must be a list when present.")
    out: list[list[dict]] = []
    for row in rows:
        if row is None:
            out.append([])
        elif isinstance(row, dict):
            ranked = [
                {"token_id": int(token_id), "logprob": float(logprob), "rank": rank}
                for rank, (token_id, logprob) in enumerate(
                    sorted(row.items(), key=lambda item: float(item[1]), reverse=True)[:max_alternatives],
                    start=1,
                )
            ]
            out.append(ranked)
        elif _is_sglang_logprob_record(row):
            token_id, logprob = _extract_top_logprob_entry(row)
            out.append([{"token_id": token_id, "logprob": logprob, "rank": 1}])
        elif isinstance(row, (list, tuple)):
            parsed = [_extract_top_logprob_entry(entry) for entry in row]
            ranked = [
                {"token_id": token_id, "logprob": logprob, "rank": rank}
                for rank, (token_id, logprob) in enumerate(
                    sorted(parsed, key=lambda item: item[1], reverse=True)[:max_alternatives],
                    start=1,
                )
            ]
            out.append(ranked)
        else:
            raise TypeError(f"Unexpected SGLang output_top_logprobs row type: {type(row).__name__}")
    return out


def _split_generation_response(response: object, *, expected_count: int) -> list[dict]:
    if isinstance(response, list):
        if len(response) != expected_count:
            raise RuntimeError("SGLang batch response length mismatch.")
        for item in response:
            if not isinstance(item, dict):
                raise TypeError("SGLang batch response entries must be dictionaries.")
        return response

    if not isinstance(response, dict):
        raise TypeError(f"Unexpected SGLang response type: {type(response).__name__}")
    if expected_count == 1:
        return [response]

    if "text" not in response or "meta_info" not in response:
        raise RuntimeError("SGLang batched response must be a list of records or a dict with text/meta_info lists.")
    texts = response["text"]
    meta_infos = response["meta_info"]
    if not isinstance(texts, list) or not isinstance(meta_infos, list):
        raise TypeError("SGLang dict batched response must contain text and meta_info lists.")
    if len(texts) != expected_count or len(meta_infos) != expected_count:
        raise RuntimeError("SGLang dict batched response length mismatch.")
    return [{"text": text, "meta_info": meta_info} for text, meta_info in zip(texts, meta_infos, strict=True)]


@dataclass(frozen=True)
class SglangRuntimeConfig:
    base_url: str = "http://127.0.0.1:30000"
    timeout_s: float = 600.0
    pin_loras: bool = False
    unload_stale_adapters: bool = True
    memory_saver: bool = False
    memory_saver_tags: tuple[str, ...] = ("kv_cache",)
    return_original_logprobs: bool = False


class SglangSampler:
    def __init__(
        self,
        *,
        runtime: SglangRuntimeConfig,
        adapter_paths: dict[str, str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.adapter_paths: dict[str, str] = {}
        self._adapter_ids: dict[str, int] = {}
        self._adapter_request_names: dict[str, str] = {}
        self._adapter_id_by_identity: dict[AdapterIdentity, int] = {}
        self._adapter_request_name_by_identity: dict[AdapterIdentity, str] = {}
        self._loaded_request_names: set[str] = set()
        self._next_adapter_id = 1
        self._memory_released = False
        self.set_adapter_paths(adapter_paths={} if adapter_paths is None else dict(adapter_paths))

    def _endpoint(self, path: str) -> str:
        return f"{self.runtime.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _post_json(self, path: str, payload: dict) -> object:
        request = urlrequest.Request(
            self._endpoint(path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=self.runtime.timeout_s) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"SGLang HTTP {exc.code} for {path}: {body}"
            ) from exc
        if body == "":
            return None
        return json.loads(body)

    def set_adapter_paths(self, *, adapter_paths: dict[str, str]) -> None:
        if self._memory_released:
            self.wake_up(level=2)
        self.adapter_paths = dict(adapter_paths)
        self._adapter_ids = {}
        self._adapter_request_names = {}
        active_identities: set[AdapterIdentity] = set()

        for logical_name in sorted(self.adapter_paths):
            identity = adapter_identity(self.adapter_paths[logical_name])
            active_identities.add(identity)
            if identity not in self._adapter_id_by_identity:
                adapter_id = self._next_adapter_id
                self._next_adapter_id += 1
                request_name = f"{logical_name}__sgloraid_{adapter_id}"
                self._adapter_id_by_identity[identity] = adapter_id
                self._adapter_request_name_by_identity[identity] = request_name
            adapter_id = self._adapter_id_by_identity[identity]
            request_name = self._adapter_request_name_by_identity[identity]
            if request_name not in self._loaded_request_names:
                self._load_adapter(request_name=request_name, adapter_path=self.adapter_paths[logical_name])
            self._adapter_ids[logical_name] = adapter_id
            self._adapter_request_names[logical_name] = request_name

        if self.runtime.unload_stale_adapters:
            active_request_names = {
                self._adapter_request_name_by_identity[identity]
                for identity in active_identities
            }
            stale_request_names = sorted(self._loaded_request_names - active_request_names)
            for request_name in stale_request_names:
                self._unload_adapter(request_name=request_name)

    def unload_adapters(self, *, adapter_names: set[str]) -> None:
        for logical_name in sorted(adapter_names):
            request_name = self._adapter_request_names.get(logical_name)
            if request_name in self._loaded_request_names:
                self._unload_adapter(request_name=request_name)

    def _load_adapter(self, *, request_name: str, adapter_path: str) -> None:
        payload = {
            "lora_name": request_name,
            "lora_path": adapter_path,
            "pinned": self.runtime.pin_loras,
        }
        try:
            self._post_json("load_lora_adapter", payload)
        except RuntimeError as exc:
            message = str(exc)
            if not (
                f"LoRA adapter {request_name}" in message
                and "because it is already loaded" in message
            ):
                raise
        self._loaded_request_names.add(request_name)

    def _unload_adapter(self, *, request_name: str) -> None:
        self._post_json("unload_lora_adapter", {"lora_name": request_name})
        self._loaded_request_names.remove(request_name)

    def wake_up(self, *, level: int = 1) -> None:
        if not self.runtime.memory_saver:
            return
        if not self._memory_released:
            return
        self._post_json(
            "resume_memory_occupation",
            {"tags": list(self.runtime.memory_saver_tags)},
        )
        self._memory_released = False

    def sleep(self, *, level: int = 1) -> None:
        if not self.runtime.memory_saver:
            return
        if self._memory_released:
            return
        self._post_json(
            "release_memory_occupation",
            {"tags": list(self.runtime.memory_saver_tags)},
        )
        self._memory_released = True

    def close(self) -> None:
        if self._memory_released:
            self.wake_up(level=2)
        for request_name in sorted(self._loaded_request_names):
            self._unload_adapter(request_name=request_name)

    def sample(self, request: SamplingRequest) -> SamplingResult:
        return self.sample_many([request])[0]

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        if len(requests) == 0:
            return []

        trace_top_logprobs = get_trace_top_logprobs()
        grouped: dict[
            tuple[str, float, float, float, int, tuple[int, ...], tuple[str, ...], bool, int | None],
            list[tuple[int, SamplingRequest]],
        ] = {}
        for idx, request in enumerate(requests):
            if (
                request.top_k != -1
                or request.presence_penalty != 0.0
                or request.repetition_penalty != 1.0
            ):
                raise NotImplementedError(
                    "top_k, presence_penalty, and repetition_penalty request overrides "
                    "are currently implemented only by the vLLM sampler."
                )
            key = (
                request.adapter_name,
                float(request.temperature),
                float(request.min_p),
                float(request.top_p),
                int(request.max_tokens),
                tuple(int(tok) for tok in request.stop_token_ids),
                tuple(str(stop) for stop in request.stop_strings),
                bool(request.include_stop_str_in_output),
                request.seed,
            )
            grouped.setdefault(key, []).append((idx, request))

        results: list[SamplingResult | None] = [None] * len(requests)
        for (
            adapter_name,
            temperature,
            min_p,
            top_p,
            max_tokens,
            stop_token_ids,
            stop_strings,
            include_stop_str_in_output,
            seed,
        ), grouped_requests in grouped.items():
            sampling_params = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": -1,
                "min_p": min_p,
                "repetition_penalty": 1.0,
                "stop_token_ids": list(stop_token_ids),
            }
            if stop_strings:
                sampling_params["stop"] = list(stop_strings)
                sampling_params["no_stop_trim"] = include_stop_str_in_output
            if seed is not None:
                sampling_params["sampling_seed"] = int(seed)
            payload = {
                "input_ids": [request.prompt_token_ids for _, request in grouped_requests],
                "sampling_params": sampling_params,
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": -1,
            }
            if trace_top_logprobs > 0:
                payload["top_logprobs_num"] = trace_top_logprobs
            if adapter_name in self._adapter_request_names:
                payload["lora_path"] = self._adapter_request_names[adapter_name]

            response = self._post_json("generate", payload)
            response_items = _split_generation_response(response, expected_count=len(grouped_requests))
            for response_item, (request_idx, request) in zip(response_items, grouped_requests, strict=True):
                meta_info = response_item["meta_info"]
                if not isinstance(meta_info, dict):
                    raise TypeError("SGLang response meta_info must be a dictionary.")
                completion_token_ids, completion_logprobs = _extract_completion_tokens_and_logprobs(meta_info)
                text = response_item["text"] if "text" in response_item else ""
                if not isinstance(text, str):
                    raise TypeError("SGLang response text must be a string.")
                behavior_policy = BehaviorPolicySpec.from_sampling_request(request)
                if self.runtime.return_original_logprobs or behavior_policy.temperature == 0.0:
                    logprob_semantics = RAW_MODEL_LOGPROBS
                elif (
                    behavior_policy.top_p != 1.0
                    or behavior_policy.top_k != -1
                    or behavior_policy.min_p != 0.0
                    or behavior_policy.repetition_penalty != 1.0
                ):
                    # SGLang 0.5.15 returns temperature-scaled logprobs before
                    # top-k/top-p/min-p filtering in its standard sampler path.
                    logprob_semantics = TEMPERATURE_SCALED_MODEL_LOGPROBS
                else:
                    logprob_semantics = BEHAVIOR_POLICY_LOGPROBS
                contract = behavior_policy_contract_record(
                    policy=behavior_policy,
                    backend="sglang",
                    backend_mode="standard_sampler",
                    return_original_logprobs=self.runtime.return_original_logprobs,
                    semantics=logprob_semantics,
                )
                raw = {
                    "sampler_backend": "sglang",
                    "completion_logprobs": logprob_semantics,
                    "behavior_policy_contract": contract,
                }
                if "finish_reason" in meta_info:
                    raw["finish_reason"] = meta_info["finish_reason"]
                if adapter_name in self._adapter_request_names:
                    raw["sglang_lora_path"] = self._adapter_request_names[adapter_name]
                completion_top_logprobs = _top_logprob_rows(meta_info, max_alternatives=trace_top_logprobs)
                if len(completion_top_logprobs) > 0:
                    raw["completion_top_logprobs"] = completion_top_logprobs
                result = SamplingResult(
                    adapter_name=request.adapter_name,
                    prompt_token_ids=list(request.prompt_token_ids),
                    completion_token_ids=completion_token_ids,
                    completion_logprobs=completion_logprobs,
                    text=text,
                    behavior_policy=behavior_policy,
                    completion_logprob_semantics=logprob_semantics,
                    raw=raw,
                )
                get_model_io_tracer().record_generation(
                    request=request,
                    result=result,
                    boundary="llm_local_rl.sglang_sampling.SglangSampler.sample_many",
                )
                results[request_idx] = result

        if any(result is None for result in results):
            raise AssertionError("All SGLang sampler requests must produce results.")
        return [result for result in results if result is not None]
