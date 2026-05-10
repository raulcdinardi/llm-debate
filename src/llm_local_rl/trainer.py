from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_local_rl.model_io_trace import (
    get_model_io_tracer,
    get_trace_top_logprobs,
    is_model_io_tracing_enabled,
)
from llm_local_rl.types import AdapterName, TrainExample


@dataclass(frozen=True)
class TrainerConfig:
    base_model_path: str
    adapter_names: tuple[AdapterName, ...] = ("shared",)
    lora_rank: int = 32
    learning_rate: float = 1e-4
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    ppo_clip_epsilon: float = 0.2
    train_minibatch_size: int = 0
    train_max_tokens: int = 0


def _resolve_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype={name!r}")


def _pad_batch(
    *,
    batch: list[TrainExample],
    pad_token_id: int,
    device: str,
    max_tokens: int = 0,
) -> dict[str, torch.Tensor]:
    def _start(example: TrainExample) -> int:
        if max_tokens <= 0 or len(example.input_ids) <= max_tokens:
            return 0
        return len(example.input_ids) - max_tokens

    max_len = max(len(example.input_ids) - _start(example) for example in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    target_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    old_logprobs = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    advantages = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

    for row_idx, example in enumerate(batch):
        start = _start(example)
        n = len(example.input_ids) - start
        if not (
            len(example.input_ids)
            == len(example.target_ids)
            == len(example.loss_mask)
            == len(example.old_logprobs)
            == len(example.advantages)
        ):
            raise ValueError("TrainExample fields must all have equal length.")
        input_ids[row_idx, :n] = torch.tensor(example.input_ids[start:], dtype=torch.long, device=device)
        target_ids[row_idx, :n] = torch.tensor(example.target_ids[start:], dtype=torch.long, device=device)
        attention_mask[row_idx, :n] = 1
        loss_mask[row_idx, :n] = torch.tensor(example.loss_mask[start:], dtype=torch.bool, device=device)
        old_logprobs[row_idx, :n] = torch.tensor(example.old_logprobs[start:], dtype=torch.float32, device=device)
        advantages[row_idx, :n] = torch.tensor(example.advantages[start:], dtype=torch.float32, device=device)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
    }


class MultiAdapterTrainer:
    def __init__(self, *, config: TrainerConfig) -> None:
        self.config = config
        self.compute_device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.current_device = "cpu"
        self.tokenizer = self._load_tokenizer(base_model_path=config.base_model_path)
        self.model = self._build_new_model()
        self.optimizer = self._build_optimizer()

    @classmethod
    def from_saved_adapters(
        cls,
        *,
        config: TrainerConfig,
        adapter_dirs: dict[AdapterName, str],
    ) -> "MultiAdapterTrainer":
        trainer = object.__new__(cls)
        trainer.config = config
        trainer.compute_device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        trainer.current_device = "cpu"
        trainer.tokenizer = trainer._load_tokenizer(base_model_path=config.base_model_path)
        first_name = config.adapter_names[0]
        first_dir = adapter_dirs[first_name]

        base_model = trainer._build_base_model()

        model = PeftModel.from_pretrained(
            base_model,
            first_dir,
            adapter_name=first_name,
            is_trainable=True,
        )
        for adapter_name in config.adapter_names[1:]:
            model.load_adapter(adapter_dirs[adapter_name], adapter_name=adapter_name, is_trainable=True)
        model.set_adapter(first_name)
        model.train()
        model.to(torch.device("cpu"))
        trainer.model = model
        trainer.optimizer = trainer._build_optimizer()
        return trainer

    @staticmethod
    def _load_tokenizer(*, base_model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id or eos_token_id.")
        return tokenizer

    def _build_base_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            torch_dtype=_resolve_dtype(self.config.torch_dtype),
        )
        base_model.config.use_cache = False
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        base_model.to(torch.device("cpu"))
        return base_model

    def _build_new_model(self):
        base_model = self._build_base_model()
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=max(8, self.config.lora_rank * 2),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(self.config.target_modules),
        )
        model = get_peft_model(base_model, lora_config, adapter_name=self.config.adapter_names[0])
        for adapter_name in self.config.adapter_names[1:]:
            model.add_adapter(adapter_name, lora_config)
        model.set_adapter(self.config.adapter_names[0])
        model.train()
        model.to(torch.device("cpu"))
        return model

    def _build_optimizer(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=self.config.learning_rate)

    def _move_optimizer_state(self, *, device: str) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(torch.device(device))

    def wake_up(self) -> None:
        if self.current_device == self.compute_device:
            return
        self.model.to(torch.device(self.compute_device))
        self._move_optimizer_state(device=self.compute_device)
        self.current_device = self.compute_device

    def sleep(self) -> None:
        if self.current_device == "cpu":
            return
        self.model.to(torch.device("cpu"))
        self._move_optimizer_state(device="cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.current_device = "cpu"

    def set_adapter(self, adapter_name: AdapterName) -> None:
        self.model.set_adapter(adapter_name)

    def compute_logprobs(self, *, adapter_name: AdapterName, batch: list[TrainExample]) -> list[list[float]]:
        self.wake_up()
        self.set_adapter(adapter_name)
        self.model.eval()
        tensors = _pad_batch(
            batch=batch,
            pad_token_id=int(self.tokenizer.pad_token_id),
            device=self.current_device,
            max_tokens=self.config.train_max_tokens,
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            )
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
            token_logprobs = log_probs.gather(
                dim=-1,
                index=tensors["target_ids"].unsqueeze(-1),
            ).squeeze(-1)
            if is_model_io_tracing_enabled():
                trace_top_k = get_trace_top_logprobs()
                top_values = None
                top_indices = None
                if trace_top_k > 0:
                    top_values, top_indices = torch.topk(
                        log_probs,
                        k=min(trace_top_k, int(log_probs.shape[-1])),
                        dim=-1,
                    )
                get_model_io_tracer().record_trainer_forward(
                    phase="trainer_logprobs",
                    boundary="llm_local_rl.trainer.MultiAdapterTrainer.compute_logprobs",
                    adapter_name=adapter_name,
                    batch=batch,
                    tensors=tensors,
                    minibatch_start=0,
                    token_logprobs=token_logprobs,
                    top_token_ids=top_indices,
                    top_logprobs=top_values,
                )

        out: list[list[float]] = []
        for row_idx, example in enumerate(batch):
            n = len(example.input_ids)
            out.append(token_logprobs[row_idx, :n].detach().cpu().tolist())
        self.model.train()
        return out

    def train_batch(self, *, adapter_name: AdapterName, batch: list[TrainExample]) -> dict[str, float]:
        if len(batch) == 0:
            return {"loss": 0.0, "num_examples": 0.0, "num_trained_tokens": 0.0}

        self.wake_up()
        self.set_adapter(adapter_name)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        minibatch_size = self.config.train_minibatch_size if self.config.train_minibatch_size > 0 else len(batch)
        total_loss_value = 0.0
        total_trained_tokens = 0
        approx_kl_numerator = 0.0
        for start_idx in range(0, len(batch), minibatch_size):
            minibatch = batch[start_idx : start_idx + minibatch_size]
            tensors = _pad_batch(
                batch=minibatch,
                pad_token_id=int(self.tokenizer.pad_token_id),
                device=self.current_device,
                max_tokens=self.config.train_max_tokens,
            )
            outputs = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            )
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
            token_logprobs = log_probs.gather(
                dim=-1,
                index=tensors["target_ids"].unsqueeze(-1),
            ).squeeze(-1)
            if is_model_io_tracing_enabled():
                trace_top_k = get_trace_top_logprobs()
                top_values = None
                top_indices = None
                if trace_top_k > 0:
                    top_values, top_indices = torch.topk(
                        log_probs,
                        k=min(trace_top_k, int(log_probs.shape[-1])),
                        dim=-1,
                    )
                get_model_io_tracer().record_trainer_forward(
                    phase="trainer_forward",
                    boundary="llm_local_rl.trainer.MultiAdapterTrainer.train_batch",
                    adapter_name=adapter_name,
                    batch=minibatch,
                    tensors=tensors,
                    minibatch_start=start_idx,
                    token_logprobs=token_logprobs,
                    top_token_ids=top_indices,
                    top_logprobs=top_values,
                )

            trained_positions = tensors["loss_mask"] & (tensors["advantages"] != 0.0)
            if not torch.any(trained_positions):
                continue

            ratio = torch.exp(token_logprobs[trained_positions] - tensors["old_logprobs"][trained_positions])
            clipped_ratio = torch.clamp(
                ratio,
                min=1.0 - self.config.ppo_clip_epsilon,
                max=1.0 + self.config.ppo_clip_epsilon,
            )
            advantages = tensors["advantages"][trained_positions]
            objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = torch.sum(-objective)
            loss.backward()
            trained_tokens = int(trained_positions.sum().item())
            total_trained_tokens += trained_tokens
            total_loss_value += float(loss.detach().cpu().item())
            approx_kl_numerator += float(
                torch.sum(tensors["old_logprobs"][trained_positions] - token_logprobs[trained_positions])
                .detach()
                .cpu()
                .item()
            )

        if total_trained_tokens == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": 0.0, "num_examples": float(len(batch)), "num_trained_tokens": 0.0}

        self.optimizer.step()

        return {
            "loss": total_loss_value,
            "num_examples": float(len(batch)),
            "num_trained_tokens": float(total_trained_tokens),
            "approx_kl": approx_kl_numerator / total_trained_tokens,
        }

    def save_adapter(self, *, adapter_name: AdapterName, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.set_adapter(adapter_name)
        self.model.save_pretrained(str(output_path), selected_adapters=[adapter_name])
        if adapter_name == "default":
            return str(output_path)
        return str(output_path / adapter_name)

    def load_adapter(self, *, adapter_name: AdapterName, adapter_dir: str) -> None:
        if adapter_name in self.model.peft_config:
            if not hasattr(self.model, "delete_adapter"):
                raise NotImplementedError("This PEFT runtime cannot replace an already-loaded adapter in place.")
            if self.model.active_adapter == adapter_name:
                alternatives = [name for name in self.model.peft_config if name != adapter_name]
                if alternatives:
                    self.model.set_adapter(alternatives[0])
            self.model.delete_adapter(adapter_name)
        self.model.load_adapter(adapter_dir, adapter_name=adapter_name, is_trainable=True)

    def adapter_parameter_snapshot(self, *, adapter_name: AdapterName) -> dict[str, torch.Tensor]:
        self.set_adapter(adapter_name)
        out: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if adapter_name not in name:
                continue
            out[name] = param.detach().cpu().float().clone()
        return out
