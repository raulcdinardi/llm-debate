from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import GenerationConfig

from llm_local_rl.behavior_policy import (
    BEHAVIOR_POLICY_LOGPROBS,
    RAW_MODEL_LOGPROBS,
    TEMPERATURE_SCALED_MODEL_LOGPROBS,
    BehaviorPolicySpec,
    behavior_policy_contract_record,
)
from llm_local_rl.model_io_trace import get_model_io_tracer
from llm_local_rl.trainer import MultiAdapterTrainer
from llm_local_rl.types import AdapterName, SamplingRequest, SamplingResult


@dataclass
class TrainerTransformersSampler:
    """Rollout sampler that reuses the training LoRA model via Transformers generate()."""

    trainer: MultiAdapterTrainer
    tokenizer: Any
    adapter_paths: dict[str, str] = field(default_factory=dict)
    _active_adapter: AdapterName | None = None

    def set_adapter_paths(self, *, adapter_paths: dict[str, str]) -> None:
        self.adapter_paths = dict(adapter_paths)

    def wake_up(self, *, level: int = 1) -> None:
        _ = level
        self.trainer.wake_up()

    def sleep(self, *, level: int = 1) -> None:
        _ = level

    def close(self) -> None:
        return None

    def sample(self, request: SamplingRequest) -> SamplingResult:
        return self.sample_many([request])[0]

    def sample_many(self, requests: list[SamplingRequest]) -> list[SamplingResult]:
        if len(requests) == 0:
            return []

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
        pad_token_id = int(self.tokenizer.pad_token_id)

        for (adapter_name, temperature, min_p, top_p, max_tokens, stop_token_ids, seed), grouped_requests in grouped.items():
            self._set_active_adapter(adapter_name=str(adapter_name))
            self.trainer.model.eval()

            stop_token_id = None
            if len(stop_token_ids) == 1:
                stop_token_id = int(stop_token_ids[0])
            elif len(stop_token_ids) > 1:
                raise NotImplementedError(
                    "TrainerTransformersSampler currently supports at most one stop token id."
                )

            do_sample = temperature > 0.0
            gen_config = GenerationConfig(
                do_sample=do_sample,
                temperature=(max(1e-6, temperature) if do_sample else 1.0),
                top_p=top_p,
                top_k=0,
                min_p=min_p,
                repetition_penalty=1.0,
                pad_token_id=pad_token_id,
                eos_token_id=stop_token_id,
            )

            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

            batch_indices = [idx for idx, _ in grouped_requests]
            prompt_token_lists = [request.prompt_token_ids for _, request in grouped_requests]
            max_prompt_len = max(len(tokens) for tokens in prompt_token_lists)
            device = torch.device(self.trainer.current_device)

            input_ids = torch.full(
                (len(grouped_requests), max_prompt_len),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(grouped_requests), max_prompt_len),
                dtype=torch.long,
                device=device,
            )
            for row_idx, prompt_token_ids in enumerate(prompt_token_lists):
                n = len(prompt_token_ids)
                start = max_prompt_len - n
                input_ids[row_idx, start:max_prompt_len] = torch.tensor(
                    prompt_token_ids,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask[row_idx, start:max_prompt_len] = 1

            with torch.no_grad():
                out = self.trainer.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

            sequences = out.sequences
            generated_len = int(sequences.shape[1]) - max_prompt_len
            generated_attention_mask = torch.ones(
                (len(grouped_requests), generated_len),
                dtype=attention_mask.dtype,
                device=device,
            )
            full_attention_mask = torch.cat([attention_mask, generated_attention_mask], dim=1)
            with torch.no_grad():
                raw_outputs = self.trainer.model(input_ids=sequences, attention_mask=full_attention_mask)

            batch_size = len(grouped_requests)
            for row_idx in range(batch_size):
                completion_token_ids, completion_logprobs = self._completion_tokens_and_behavior_logprobs(
                    logits=raw_outputs.logits[row_idx],
                    sequence=sequences[row_idx],
                    completion_start=max_prompt_len,
                    generated_len=generated_len,
                    stop_token_id=stop_token_id,
                    pad_token_id=pad_token_id,
                    behavior_temperature=temperature,
                )

                request = grouped_requests[row_idx][1]
                behavior_policy = BehaviorPolicySpec.from_sampling_request(request)
                if behavior_policy.exact_trainer_reconstruction_supported():
                    logprob_semantics = BEHAVIOR_POLICY_LOGPROBS
                elif behavior_policy.temperature == 0.0:
                    logprob_semantics = RAW_MODEL_LOGPROBS
                else:
                    # This explicit forward pass reproduces temperature but
                    # does not renormalize top-p/min-p truncation.
                    logprob_semantics = TEMPERATURE_SCALED_MODEL_LOGPROBS
                text = self.tokenizer.decode(completion_token_ids, skip_special_tokens=True)
                result = SamplingResult(
                    adapter_name=request.adapter_name,
                    prompt_token_ids=list(request.prompt_token_ids),
                    completion_token_ids=completion_token_ids,
                    completion_logprobs=completion_logprobs,
                    text=text,
                    behavior_policy=behavior_policy,
                    completion_logprob_semantics=logprob_semantics,
                    raw={
                        "sampler_backend": "transformers",
                        "completion_logprobs": logprob_semantics,
                        "behavior_policy_contract": behavior_policy_contract_record(
                            policy=behavior_policy,
                            backend="transformers",
                            backend_mode="generate_then_full_forward",
                            return_original_logprobs=False,
                            semantics=logprob_semantics,
                            scoring_dtype="float32",
                        ),
                    },
                )
                get_model_io_tracer().record_generation(
                    request=request,
                    result=result,
                    boundary="llm_local_rl.transformers_sampling.TrainerTransformersSampler.sample_many",
                )
                results[batch_indices[row_idx]] = result

            del out, sequences, raw_outputs

        if any(result is None for result in results):
            raise AssertionError("All Transformers sampler requests must produce results.")
        return [result for result in results if result is not None]

    @staticmethod
    def _completion_tokens_and_behavior_logprobs(
        *,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        completion_start: int,
        generated_len: int,
        stop_token_id: int | None,
        pad_token_id: int,
        behavior_temperature: float,
    ) -> tuple[list[int], list[float]]:
        completion_token_ids: list[int] = []
        logprob_positions: list[int] = []
        for step_idx in range(generated_len):
            token_position = completion_start + step_idx
            next_token = int(sequence[token_position].item())
            if stop_token_id is not None and next_token == stop_token_id:
                break
            if next_token == pad_token_id:
                break
            completion_token_ids.append(next_token)
            logprob_positions.append(token_position - 1)

        completion_logprobs: list[float] = []
        if len(logprob_positions) == 0:
            return completion_token_ids, completion_logprobs

        positions = torch.tensor(logprob_positions, dtype=torch.long, device=logits.device)
        target_ids = torch.tensor(completion_token_ids, dtype=torch.long, device=logits.device)
        # Only completion token logprobs are consumed by training. Chunking avoids the
        # full [batch, seq, vocab] float log-softmax allocation that OOMs on long R3 prompts.
        chunk_size = 16
        for start in range(0, len(logprob_positions), chunk_size):
            end = start + chunk_size
            scoring_temperature = (
                float(behavior_temperature)
                if float(behavior_temperature) > 0.0
                else 1.0
            )
            logits_chunk = logits.index_select(0, positions[start:end]).float() / scoring_temperature
            target_logits = logits_chunk.gather(dim=-1, index=target_ids[start:end, None]).squeeze(-1).float()
            normalizers = torch.logsumexp(logits_chunk, dim=-1)
            completion_logprobs.extend((target_logits - normalizers).detach().cpu().tolist())

        return completion_token_ids, [float(logprob) for logprob in completion_logprobs]

    def _set_active_adapter(self, *, adapter_name: str) -> None:
        adapter: AdapterName = adapter_name  # type: ignore[assignment]
        if self._active_adapter == adapter:
            return
        self.trainer.set_adapter(adapter)
        self._active_adapter = adapter
