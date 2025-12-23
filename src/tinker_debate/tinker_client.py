"""Tinker API client for debate training (async).

Implements:
- parallelizable sampling (submit all requests first, then await futures)
- forward_backward for policy gradient-style losses (importance sampling)

This is the API boundary; avoid silent fallbacks here.
Uses async Tinker API throughout for better performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import tinker
from tinker import types as tinker_types

from tinker_cookbook.renderers import Qwen3InstructRenderer

LossFn = Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]


def serialize_response(response: Any, sequence_idx: int = 0) -> dict:
    """Serialize a Tinker SampleResponse to a dict with ALL fields.

    Captures everything the API returns for debugging/logging.
    """
    raw: dict[str, Any] = {
        "response_type": type(response).__name__,
        "response_dir": [a for a in dir(response) if not a.startswith("_")],
    }

    # Capture all non-callable attributes from response
    for attr in dir(response):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(response, attr)
            if callable(val):
                continue
            # Handle sequences specially
            if attr == "sequences":
                raw["sequences_count"] = len(val) if val else 0
            else:
                raw[f"response_{attr}"] = _serialize_value(val)
        except Exception as e:
            raw[f"response_{attr}_error"] = str(e)

    # Capture the specific sequence we used
    if hasattr(response, "sequences") and response.sequences:
        seq = response.sequences[sequence_idx]
        raw["sequence_type"] = type(seq).__name__
        raw["sequence_dir"] = [a for a in dir(seq) if not a.startswith("_")]

        for attr in dir(seq):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(seq, attr)
                if callable(val):
                    continue
                raw[f"seq_{attr}"] = _serialize_value(val)
            except Exception as e:
                raw[f"seq_{attr}_error"] = str(e)

    return raw


def _serialize_value(val: Any) -> Any:
    """Serialize a value for JSON, handling common types. Keeps everything."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_value(x) for x in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if hasattr(val, "__dict__"):
        return {"_type": type(val).__name__, "_dict": _serialize_value(val.__dict__)}
    # Fallback: convert to string
    return {"_type": type(val).__name__, "_str": str(val)}


@dataclass
class TinkerDebateClient:
    """Combined sampling and training client using async Tinker API."""

    service: tinker.ServiceClient
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient
    tokenizer: Any  # HuggingFace tokenizer
    renderer: Qwen3InstructRenderer
    model_name: str

    @classmethod
    async def create(
        cls, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    ) -> "TinkerDebateClient":
        service = tinker.ServiceClient()
        training_client = await service.create_lora_training_client_async(base_model=model_name)
        tokenizer = training_client.get_tokenizer()
        renderer = Qwen3InstructRenderer(tokenizer)
        sampling_client = await training_client.save_weights_and_get_sampling_client_async("debate_init")

        return cls(
            service=service,
            training_client=training_client,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            renderer=renderer,
            model_name=model_name,
        )

    async def generate(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float = 0.8,
    ) -> list[tuple[str, list[int], list[int], list[float], dict] | None]:
        """Generate completions (async, parallel).

        Returns list of:
          (completion_text, prompt_tokens, completion_tokens, completion_logprobs, raw_response)

        raw_response contains ALL fields returned by the API for debugging.

        NOTE: we require completion logprobs for RL-style training; if the server
        doesn't return them, we raise (no silent fallback).
        """
        import asyncio

        stop_strings = self.renderer.get_stop_sequences()
        stop = stop_strings if (stop_strings and isinstance(stop_strings[0], str)) else None

        prompt_tokens_list: list[list[int]] = []
        sample_coros = []

        # Prepare all requests
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tokens_list.append(prompt_tokens)

            model_input = tinker.ModelInput.from_ints(prompt_tokens)
            sampling_params = tinker.SamplingParams(
                temperature=temperature,
                stop=stop,
            )
            if max_tokens is not None:
                sampling_params.max_tokens = max_tokens

            # Don't await yet - just create the coroutine
            sample_coros.append(
                self.sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
            )

        # Run requests sequentially with delay to avoid rate limiting (429)
        responses = []
        for i, coro in enumerate(sample_coros):
            try:
                resp = await coro
                responses.append(resp)
            except Exception as e:
                print(f"\n{'!'*60}")
                print(f"!!! SAMPLING REQUEST FAILED: {type(e).__name__}: {e}")
                print(f"!!! Prompt {i+1}/{len(sample_coros)} will be skipped")
                print(f"{'!'*60}\n")
                responses.append(None)
            if i < len(sample_coros) - 1:
                await asyncio.sleep(1.0)

        results: list[tuple[str, list[int], list[int], list[float], dict] | None] = []

        for prompt_tokens, resp in zip(prompt_tokens_list, responses):
            if resp is None:
                results.append(None)
                continue

            seq = resp.sequences[0]

            completion_tokens = list(seq.tokens)
            if seq.logprobs is None:
                raise RuntimeError(
                    "Tinker sampling did not return completion logprobs (seq.logprobs is None). "
                    "This codebase requires completion logprobs for importance-sampling / PPO-style training."
                )
            completion_logprobs = list(seq.logprobs)

            completion_text = self.tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )

            # Serialize raw response for logging
            raw_response = serialize_response(resp, sequence_idx=0)

            results.append(
                (completion_text, prompt_tokens, completion_tokens, completion_logprobs, raw_response)
            )

        return results

    # Back-compat alias
    async def generate_with_logprobs(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.8,
    ) -> list[tuple[str, list[int], list[int], list[float], dict]]:
        return await self.generate(prompts, max_tokens=max_tokens, temperature=temperature)

    async def forward_backward(
        self,
        prompt_tokens_batch: list[list[int]],
        completion_tokens_batch: list[list[int]],
        completion_logprobs_batch: list[list[float]],
        completion_advantages_batch: list[list[float]],
        loss_fn: LossFn = "importance_sampling",
    ) -> dict:
        """Compute forward-backward pass with policy gradient loss (async).

        Note: if you're doing a full training step, prefer `train_step()` which overlaps
        the forward_backward and optim_step requests (see Tinker docs on clock cycles).
        """

        import torch

        data: list[tinker_types.Datum] = []
        sampling_logprobs_list: list[list[float]] = []

        for prompt_toks, comp_toks, comp_lps, comp_advs in zip(
            prompt_tokens_batch,
            completion_tokens_batch,
            completion_logprobs_batch,
            completion_advantages_batch,
        ):
            if not (
                len(comp_toks) == len(comp_lps) == len(comp_advs)
            ):
                raise ValueError(
                    f"Length mismatch: tokens={len(comp_toks)}, logprobs={len(comp_lps)}, advantages={len(comp_advs)}"
                )

            full_tokens = prompt_toks + comp_toks
            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]

            prompt_len = len(prompt_toks)

            adv_tensor = [0.0] * (prompt_len - 1) + list(comp_advs)
            sampling_logprobs = [0.0] * (prompt_len - 1) + list(comp_lps)
            sampling_logprobs_list.append(sampling_logprobs)

            datum = tinker_types.Datum(
                model_input=tinker.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker_types.TensorData.from_torch(
                        torch.tensor(target_tokens)
                    ),
                    "logprobs": tinker_types.TensorData.from_torch(
                        torch.tensor(sampling_logprobs, dtype=torch.float32)
                    ),
                    "advantages": tinker_types.TensorData.from_torch(
                        torch.tensor(adv_tensor, dtype=torch.float32)
                    ),
                },
            )
            data.append(datum)

        fwd_bwd_future = await self.training_client.forward_backward_async(data, loss_fn)
        fwd_bwd_out = await fwd_bwd_future

        # Compute approximate loss from returned logprobs
        total_loss = 0.0
        num_trained_tokens = 0

        for i, (out, old_lps, comp_advs) in enumerate(
            zip(fwd_bwd_out.loss_fn_outputs, sampling_logprobs_list, completion_advantages_batch)
        ):
            if "logprobs" not in out:
                raise RuntimeError(
                    "forward_backward output missing 'logprobs' – cannot compute importance-sampling loss."
                )

            new_lps = out["logprobs"].data

            prompt_len = len(prompt_tokens_batch[i])
            completion_start = prompt_len - 1

            for j in range(completion_start, len(new_lps)):
                old_lp = old_lps[j]
                new_lp = new_lps[j]
                adv = comp_advs[j - completion_start]

                if adv > 0:
                    ratio = torch.exp(torch.tensor(new_lp - old_lp))
                    total_loss += float((-ratio * adv).item())
                    num_trained_tokens += 1

        # If the server returns a loss metric, prefer it (matches server-side definition).
        metrics = getattr(fwd_bwd_out, "metrics", {}) or {}
        loss_from_metrics = None
        if isinstance(metrics, dict):
            loss_from_metrics = metrics.get("loss:sum")

        return {
            "loss": float(loss_from_metrics) if loss_from_metrics is not None else float(total_loss),
            "num_tokens": sum(len(d.model_input.to_ints()) for d in data),
            "num_trained_tokens": num_trained_tokens,
            "metrics": metrics,
        }

    async def optim_step(self, learning_rate: float = 1e-5) -> None:
        fut = await self.training_client.optim_step_async(
            tinker_types.AdamParams(learning_rate=learning_rate)
        )
        await fut

    async def train_step(
        self,
        *,
        prompt_tokens_batch: list[list[int]],
        completion_tokens_batch: list[list[int]],
        completion_logprobs_batch: list[list[float]],
        completion_advantages_batch: list[list[float]],
        learning_rate: float,
        loss_fn: LossFn = "importance_sampling",
    ) -> dict:
        """One training step with overlapped forward_backward + optim_step (async).

        This follows the pattern recommended in Tinker docs (submit both operations
        before blocking on results) to avoid missing clock cycles.
        """

        import torch

        data: list[tinker_types.Datum] = []
        sampling_logprobs_list: list[list[float]] = []

        for prompt_toks, comp_toks, comp_lps, comp_advs in zip(
            prompt_tokens_batch,
            completion_tokens_batch,
            completion_logprobs_batch,
            completion_advantages_batch,
        ):
            if not (len(comp_toks) == len(comp_lps) == len(comp_advs)):
                raise ValueError(
                    f"Length mismatch: tokens={len(comp_toks)}, logprobs={len(comp_lps)}, advantages={len(comp_advs)}"
                )

            full_tokens = prompt_toks + comp_toks
            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]

            prompt_len = len(prompt_toks)
            adv_tensor = [0.0] * (prompt_len - 1) + list(comp_advs)
            sampling_logprobs = [0.0] * (prompt_len - 1) + list(comp_lps)
            sampling_logprobs_list.append(sampling_logprobs)

            data.append(
                tinker_types.Datum(
                    model_input=tinker.ModelInput.from_ints(input_tokens),
                    loss_fn_inputs={
                        "target_tokens": tinker_types.TensorData.from_torch(
                            torch.tensor(target_tokens)
                        ),
                        "logprobs": tinker_types.TensorData.from_torch(
                            torch.tensor(sampling_logprobs, dtype=torch.float32)
                        ),
                        "advantages": tinker_types.TensorData.from_torch(
                            torch.tensor(adv_tensor, dtype=torch.float32)
                        ),
                    },
                )
            )

        # Submit both requests before waiting (overlap) - async double-await pattern.
        fwd_bwd_future = await self.training_client.forward_backward_async(data, loss_fn)
        optim_future = await self.training_client.optim_step_async(
            tinker_types.AdamParams(learning_rate=learning_rate)
        )

        fwd_bwd_out = await fwd_bwd_future
        _optim_out = await optim_future

        # Prefer loss computed server-side if available.
        metrics = getattr(fwd_bwd_out, "metrics", {}) or {}
        loss_from_metrics = None
        if isinstance(metrics, dict):
            loss_from_metrics = metrics.get("loss:sum")

        # Fallback compute (client-side) if metrics doesn't include a loss.
        total_loss = 0.0
        num_trained_tokens = 0

        for i, (out, old_lps, comp_advs) in enumerate(
            zip(fwd_bwd_out.loss_fn_outputs, sampling_logprobs_list, completion_advantages_batch)
        ):
            if "logprobs" not in out:
                raise RuntimeError(
                    "forward_backward output missing 'logprobs' – cannot compute importance-sampling loss."
                )

            new_lps = out["logprobs"].data
            prompt_len = len(prompt_tokens_batch[i])
            completion_start = prompt_len - 1

            for j in range(completion_start, len(new_lps)):
                old_lp = old_lps[j]
                new_lp = new_lps[j]
                adv = comp_advs[j - completion_start]
                if adv > 0:
                    ratio = torch.exp(torch.tensor(new_lp - old_lp))
                    total_loss += float((-ratio * adv).item())
                    num_trained_tokens += 1

        return {
            "loss": float(loss_from_metrics) if loss_from_metrics is not None else float(total_loss),
            "num_tokens": sum(len(d.model_input.to_ints()) for d in data),
            "num_trained_tokens": num_trained_tokens,
            "metrics": metrics,
        }

    async def sync_weights(self, name: str = "step") -> None:
        self.sampling_client = await self.training_client.save_weights_and_get_sampling_client_async(name)
