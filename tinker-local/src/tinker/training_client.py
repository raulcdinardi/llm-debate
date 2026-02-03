from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from ._async import LocalFuture
from ._transformers_backend import build_optimizer, load_lora_model_and_tokenizer
from .local_config import LocalConfig
from .model_input import ModelInput
from .sampling_client import SamplingClient
from .types import AdamParams, Datum, ForwardBackwardResult, LossFn, TensorData


def _get_device() -> str:
    import torch

    forced = os.environ.get("TINKER_LOCAL_DEVICE")
    if forced is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return forced


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mem_debug(prefix: str, device: str) -> None:
    if os.environ.get("TINKER_MEM_DEBUG") != "1":
        return
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        print(f"[mem] {prefix} device={device}", flush=True)
        return
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    print(
        f"[mem] {prefix} alloc={allocated/1e9:.2f}G reserved={reserved/1e9:.2f}G "
        f"free={free/1e9:.2f}G total={total/1e9:.2f}G",
        flush=True,
    )


@dataclass
class TrainingClient:
    _base_model: str
    _model: object
    _tokenizer: object
    _device: str
    _optimizer: object
    _pending_backward: "asyncio.Task[ForwardBackwardResult] | None" = None

    @classmethod
    def create(cls, *, base_model: str, rank: int) -> "TrainingClient":
        config = LocalConfig.from_env()
        _set_seed(config.seed)
        model, tokenizer = load_lora_model_and_tokenizer(model_name=base_model, rank=rank, config=config)
        optimizer = build_optimizer(model, learning_rate=1e-5)
        return cls(_base_model=base_model, _model=model, _tokenizer=tokenizer, _device=config.device, _optimizer=optimizer)

    def get_tokenizer(self):
        return self._tokenizer

    async def save_weights_and_get_sampling_client_async(self, name: str) -> SamplingClient:
        ckpt_dir = os.environ.get("TINKER_LOCAL_CHECKPOINT_DIR")
        if ckpt_dir is None:
            ckpt_dir = ".tinker-local/checkpoints"
        ckpt_root = Path(ckpt_dir)
        out_dir = ckpt_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        model = self._model
        model.save_pretrained(str(out_dir))

        sampler = os.environ.get("TINKER_LOCAL_SAMPLER", "hf")
        if sampler == "hf":
            return SamplingClient(_model=self._model, _tokenizer=self._tokenizer, _device=self._device)
        if sampler == "vllm":
            from .vllm_sampling_client import VllmSamplingClient

            return VllmSamplingClient(
                base_model=self._base_model,
                lora_name=name,
                lora_path=str(out_dir),
            )
        raise ValueError(f"Unknown TINKER_LOCAL_SAMPLER={sampler!r} (expected 'hf' or 'vllm').")

    async def forward_backward_async(self, data: list[Datum], loss_fn: LossFn) -> LocalFuture[ForwardBackwardResult]:
        if loss_fn in ("ppo", "cispo", "dro"):
            print(
                f"[warn] tinker-local maps loss_fn={loss_fn!r} -> 'importance_sampling' (PPO/CISPO/DRO not implemented).",
                flush=True,
            )
            loss_fn = "importance_sampling"
        if loss_fn not in ("importance_sampling", "cross_entropy"):
            raise NotImplementedError(
                f"tinker-local currently only implements loss_fn='importance_sampling' and 'cross_entropy', got {loss_fn!r}."
            )

        async def _run() -> ForwardBackwardResult:
            if loss_fn == "importance_sampling":
                return self._forward_backward_importance_sampling(data)
            if loss_fn == "cross_entropy":
                return self._forward_backward_cross_entropy(data)
            raise AssertionError(f"Unreachable loss_fn={loss_fn!r}")

        task = asyncio.create_task(_run())
        self._pending_backward = task
        return LocalFuture(task)

    async def score_token_logprobs_async(
        self,
        *,
        input_tokens_list: list[list[int]],
        token_id: int,
    ) -> LocalFuture[list[list[float]]]:
        async def _run() -> list[list[float]]:
            return self._score_token_logprobs(input_tokens_list=input_tokens_list, token_id=int(token_id))

        task = asyncio.create_task(_run())
        return LocalFuture(task)

    def _score_token_logprobs(
        self,
        *,
        input_tokens_list: list[list[int]],
        token_id: int,
    ) -> list[list[float]]:
        import torch

        if len(input_tokens_list) == 0:
            return []

        model = self._model
        model.eval()

        tokenizer = self._tokenizer
        pad_token_id = tokenizer.pad_token_id
        assert pad_token_id is not None

        lengths = [len(x) for x in input_tokens_list]
        max_len = max(lengths)

        input_ids = torch.full(
            (len(input_tokens_list), max_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros((len(input_tokens_list), max_len), dtype=torch.long, device=self._device)

        for i, toks in enumerate(input_tokens_list):
            n = len(toks)
            input_ids[i, :n] = torch.tensor(toks, dtype=torch.long, device=self._device)
            attention_mask[i, :n] = 1

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            logprobs = torch.log_softmax(logits, dim=-1)[..., int(token_id)]

        results: list[list[float]] = []
        for i, n in enumerate(lengths):
            results.append(list(logprobs[i, :n].detach().cpu().tolist()))

        return results

    def _forward_backward_importance_sampling(self, data: list[Datum]) -> ForwardBackwardResult:
        if len(data) == 0:
            return ForwardBackwardResult(loss_fn_outputs=[], metrics={"loss:sum": 0.0})

        accum_env = os.environ.get("TINKER_LOCAL_GRAD_ACCUM_STEPS")
        if accum_env is None:
            grad_accum_steps = 1
        else:
            grad_accum_steps = int(accum_env)
        if grad_accum_steps <= 0:
            raise ValueError(f"TINKER_LOCAL_GRAD_ACCUM_STEPS must be > 0, got {grad_accum_steps}")
        if len(data) % grad_accum_steps != 0:
            raise ValueError(
                f"TINKER_LOCAL_GRAD_ACCUM_STEPS={grad_accum_steps} must divide batch size {len(data)}"
            )
        microbatch_size = len(data) // grad_accum_steps

        loss_fn_outputs: list[dict[str, TensorData]] = []
        loss_sum = 0.0
        for start in range(0, len(data), microbatch_size):
            out = self._forward_backward_importance_sampling_one_batch(data[start : start + microbatch_size])
            loss_fn_outputs.extend(out.loss_fn_outputs)
            loss_sum += float(out.metrics["loss:sum"])

        return ForwardBackwardResult(loss_fn_outputs=loss_fn_outputs, metrics={"loss:sum": loss_sum})

    def _forward_backward_cross_entropy(self, data: list[Datum]) -> ForwardBackwardResult:
        if len(data) == 0:
            return ForwardBackwardResult(loss_fn_outputs=[], metrics={"loss:sum": 0.0})

        accum_env = os.environ.get("TINKER_LOCAL_GRAD_ACCUM_STEPS")
        if accum_env is None:
            grad_accum_steps = 1
        else:
            grad_accum_steps = int(accum_env)
        if grad_accum_steps <= 0:
            raise ValueError(f"TINKER_LOCAL_GRAD_ACCUM_STEPS must be > 0, got {grad_accum_steps}")
        if len(data) % grad_accum_steps != 0:
            raise ValueError(
                f"TINKER_LOCAL_GRAD_ACCUM_STEPS={grad_accum_steps} must divide batch size {len(data)}"
            )
        microbatch_size = len(data) // grad_accum_steps

        loss_fn_outputs: list[dict[str, TensorData]] = []
        loss_sum = 0.0
        for start in range(0, len(data), microbatch_size):
            out = self._forward_backward_cross_entropy_one_batch(data[start : start + microbatch_size])
            loss_fn_outputs.extend(out.loss_fn_outputs)
            loss_sum += float(out.metrics["loss:sum"])

        return ForwardBackwardResult(loss_fn_outputs=loss_fn_outputs, metrics={"loss:sum": loss_sum})

    def _forward_backward_cross_entropy_one_batch(self, data: list[Datum]) -> ForwardBackwardResult:
        import torch

        model = self._model
        model.train()

        if len(data) == 0:
            loss_total = torch.tensor(0.0, device=self._device)
            loss_total.backward()
            return ForwardBackwardResult(loss_fn_outputs=[], metrics={"loss:sum": 0.0})

        tokenizer = self._tokenizer
        pad_token_id = tokenizer.pad_token_id
        assert pad_token_id is not None

        input_tokens_list = [d.model_input.to_ints() for d in data]
        lengths = [len(x) for x in input_tokens_list]
        max_len = max(lengths)
        if os.environ.get("TINKER_MEM_DEBUG") == "1":
            mean_len = sum(lengths) / len(lengths)
            print(
                f"[mem] ce_batch size={len(data)} min_len={min(lengths)} "
                f"max_len={max_len} mean_len={mean_len:.1f}",
                flush=True,
            )
        _mem_debug("ce_before_forward", self._device)

        input_ids = torch.full(
            (len(data), max_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros((len(data), max_len), dtype=torch.long, device=self._device)

        target_tokens_padded = torch.full(
            (len(data), max_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )

        for i, (d, inp) in enumerate(zip(data, input_tokens_list)):
            n = len(inp)
            input_ids[i, :n] = torch.tensor(inp, dtype=torch.long, device=self._device)
            attention_mask[i, :n] = 1

            target_td = d.loss_fn_inputs["target_tokens"].tensor
            assert isinstance(target_td, torch.Tensor)

            target = target_td.to(device=self._device, dtype=torch.long).view(-1)
            assert len(target) == n
            target_tokens_padded[i, :n] = target

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]
        logprobs_all = torch.log_softmax(logits, dim=-1).gather(
            dim=-1, index=target_tokens_padded.unsqueeze(-1)
        ).squeeze(-1)

        valid = attention_mask.to(dtype=torch.bool)
        if not torch.any(valid):
            raise RuntimeError("No valid tokens in batch.")
        loss_total = torch.sum(-logprobs_all[valid])
        loss_total.backward()
        _mem_debug("ce_after_backward", self._device)

        loss_fn_outputs: list[dict[str, TensorData]] = []
        for i, n in enumerate(lengths):
            loss_fn_outputs.append({"logprobs": TensorData.from_torch(logprobs_all[i, :n])})

        return ForwardBackwardResult(
            loss_fn_outputs=loss_fn_outputs,
            metrics={"loss:sum": float(loss_total.detach().cpu().item())},
        )

    def _forward_backward_importance_sampling_one_batch(self, data: list[Datum]) -> ForwardBackwardResult:
        import torch

        model = self._model
        model.train()

        if len(data) == 0:
            loss_total = torch.tensor(0.0, device=self._device)
            loss_total.backward()
            return ForwardBackwardResult(loss_fn_outputs=[], metrics={"loss:sum": 0.0})

        tokenizer = self._tokenizer
        pad_token_id = tokenizer.pad_token_id
        assert pad_token_id is not None

        input_tokens_list = [d.model_input.to_ints() for d in data]
        lengths = [len(x) for x in input_tokens_list]
        max_len = max(lengths)
        if os.environ.get("TINKER_MEM_DEBUG") == "1":
            mean_len = sum(lengths) / len(lengths)
            print(
                f"[mem] is_batch size={len(data)} min_len={min(lengths)} "
                f"max_len={max_len} mean_len={mean_len:.1f}",
                flush=True,
            )
        _mem_debug("is_before_forward", self._device)

        input_ids = torch.full(
            (len(data), max_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros((len(data), max_len), dtype=torch.long, device=self._device)

        target_tokens_padded = torch.full(
            (len(data), max_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )
        old_logprobs_padded = torch.zeros((len(data), max_len), dtype=torch.float32, device=self._device)
        advantages_padded = torch.zeros((len(data), max_len), dtype=torch.float32, device=self._device)

        for i, (d, inp) in enumerate(zip(data, input_tokens_list)):
            n = len(inp)
            input_ids[i, :n] = torch.tensor(inp, dtype=torch.long, device=self._device)
            attention_mask[i, :n] = 1

            target_td = d.loss_fn_inputs["target_tokens"].tensor
            old_td = d.loss_fn_inputs["logprobs"].tensor
            adv_td = d.loss_fn_inputs["advantages"].tensor

            assert isinstance(target_td, torch.Tensor)
            assert isinstance(old_td, torch.Tensor)
            assert isinstance(adv_td, torch.Tensor)

            target = target_td.to(device=self._device, dtype=torch.long).view(-1)
            old = old_td.to(device=self._device, dtype=torch.float32).view(-1)
            adv = adv_td.to(device=self._device, dtype=torch.float32).view(-1)

            assert len(target) == n
            assert len(old) == n
            assert len(adv) == n

            target_tokens_padded[i, :n] = target
            old_logprobs_padded[i, :n] = old
            advantages_padded[i, :n] = adv

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]
        new_logprobs_all = torch.log_softmax(logits, dim=-1).gather(
            dim=-1, index=target_tokens_padded.unsqueeze(-1)
        ).squeeze(-1)

        valid = attention_mask.to(dtype=torch.bool)
        nonzero_advantages = advantages_padded != 0
        mask = valid & nonzero_advantages
        if not torch.any(mask):
            loss_total = torch.tensor(0.0, device=self._device)
            loss_fn_outputs: list[dict[str, TensorData]] = []
            for i, n in enumerate(lengths):
                loss_fn_outputs.append({"logprobs": TensorData.from_torch(new_logprobs_all[i, :n])})
            return ForwardBackwardResult(
                loss_fn_outputs=loss_fn_outputs,
                metrics={"loss:sum": float(loss_total.detach().cpu().item())},
            )

        ratio = torch.exp(new_logprobs_all[mask] - old_logprobs_padded[mask])
        loss_total = torch.sum(-ratio * advantages_padded[mask])

        loss_total.backward()
        _mem_debug("is_after_backward", self._device)

        loss_fn_outputs: list[dict[str, TensorData]] = []
        for i, n in enumerate(lengths):
            loss_fn_outputs.append({"logprobs": TensorData.from_torch(new_logprobs_all[i, :n])})

        return ForwardBackwardResult(
            loss_fn_outputs=loss_fn_outputs,
            metrics={"loss:sum": float(loss_total.detach().cpu().item())},
        )

    async def optim_step_async(self, adam_params: AdamParams) -> LocalFuture[None]:
        async def _run() -> None:
            # Ensure the most recent backward has completed before stepping.
            if self._pending_backward is not None:
                await self._pending_backward

            lr = float(adam_params.learning_rate)
            for group in self._optimizer.param_groups:
                group["lr"] = lr

            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

        return LocalFuture.from_awaitable(_run())
