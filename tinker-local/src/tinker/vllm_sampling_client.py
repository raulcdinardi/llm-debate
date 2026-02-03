from __future__ import annotations

from dataclasses import dataclass

import os

from ._vllm_process import get_worker, terminate_worker
from .model_input import ModelInput
from .types import SampleResponse, SampleSequence, SamplingParams


def _require_stop_token_ids(stop: object) -> list[int]:
    if stop is None:
        return []
    if isinstance(stop, list) and (len(stop) == 0 or isinstance(stop[0], int)):
        for x in stop:
            if not isinstance(x, int):
                raise TypeError("SamplingParams.stop must be list[int] in vLLM mode.")
        return [int(x) for x in stop]
    raise TypeError("SamplingParams.stop must be list[int] (token IDs) in vLLM mode.")


@dataclass(frozen=True)
class VllmSamplingClient:
    base_model: str
    lora_name: str
    lora_path: str

    async def sample_token_batch_async(
        self,
        *,
        prompt_tokens_list: list[list[int]],
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        if len(prompt_tokens_list) == 0:
            return []

        stop_token_ids = _require_stop_token_ids(sampling_params.stop)
        top_k = int(sampling_params.top_k)
        top_p = float(sampling_params.top_p)
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0,1], got {top_p}")
        if top_k < -1:
            raise ValueError(f"top_k must be -1 (no limit) or >= 0, got {top_k}")
        min_p = float(sampling_params.min_p)
        if not (0.0 <= min_p <= 1.0):
            raise ValueError(f"min_p must be in [0,1], got {min_p}")

        worker = get_worker(base_model=self.base_model)
        outputs = worker.generate(
            prompt_token_ids=prompt_tokens_list,
            temperature=float(sampling_params.temperature),
            max_tokens=int(sampling_params.max_tokens or 256),
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            seed=(None if sampling_params.seed is None else int(sampling_params.seed)),
            stop_token_ids=stop_token_ids,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
        )
        if len(outputs) != len(prompt_tokens_list):
            raise RuntimeError("vLLM worker returned unexpected number of outputs.")
        if os.environ.get("TINKER_LOCAL_VLLM_TERMINATE_AFTER") == "1":
            terminate_worker()

        responses: list[SampleResponse] = []
        for out in outputs:
            toks = out["token_ids"]
            lps = out["logprobs"]
            if not isinstance(toks, list) or not isinstance(lps, list):
                raise TypeError("vLLM worker output format mismatch.")
            responses.append(SampleResponse(sequences=[SampleSequence(tokens=toks, logprobs=lps)]))

        return responses

    async def sample_async(
        self,
        *,
        prompt: ModelInput,
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> SampleResponse:
        if num_samples != 1:
            raise NotImplementedError("tinker-local vLLM sampling only supports num_samples=1.")
        res = await self.sample_token_batch_async(
            prompt_tokens_list=[prompt.to_ints()],
            sampling_params=sampling_params,
        )
        if len(res) != 1:
            raise RuntimeError("Expected exactly 1 response.")
        return res[0]
