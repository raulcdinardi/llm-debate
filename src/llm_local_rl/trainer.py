from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_local_rl.behavior_policy import BehaviorPolicySpec
from llm_local_rl.memory_trace import (
    MemoryTraceRecorder,
    current_alloc_bytes,
    maybe_create_recorder,
    now_seconds,
    peak_alloc_bytes,
    reset_peak,
    reserved_bytes,
)
from llm_local_rl.model_io_trace import (
    get_model_io_tracer,
    get_trace_top_logprobs,
    is_model_io_tracing_enabled,
)
from llm_local_rl.on_policy_logprobs import check_on_policy_logprobs
from llm_local_rl.types import AdapterName, TrainExample


TRAIN_LOGPROB_BACKEND_FULL_LOGITS = "full_logits"
TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD = "selective_lm_head"
TRAIN_LOGPROB_BACKENDS = (TRAIN_LOGPROB_BACKEND_FULL_LOGITS, TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD)


class BehaviorPolicyLogprobMismatchError(RuntimeError):
    """Raised before PPO ratio/backward when zero-update parity fails."""


@dataclass(frozen=True)
class TrainerConfig:
    base_model_path: str
    adapter_names: tuple[AdapterName, ...] = ("shared",)
    lora_rank: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    target_parameters: tuple[str, ...] = ()
    ppo_clip_epsilon: float = 0.2
    train_minibatch_size: int = 0
    train_max_tokens: int = 0
    train_length_bucket_batches: bool = False
    train_logprob_backend: str = TRAIN_LOGPROB_BACKEND_FULL_LOGITS
    compile_train_logprob_helper: bool = False
    gradient_checkpointing: bool = True
    on_policy_logprob_check: bool = True
    on_policy_logprob_warn_only: bool = False
    on_policy_logprob_abs_tol: float = 1e-3
    on_policy_logprob_warning_path: str | None = None
    on_policy_logprob_max_records_per_batch: int = 8
    behavior_policy: BehaviorPolicySpec = field(default_factory=BehaviorPolicySpec)

    def __post_init__(self) -> None:
        if self.train_logprob_backend not in TRAIN_LOGPROB_BACKENDS:
            raise ValueError(
                f"Unsupported train_logprob_backend={self.train_logprob_backend!r}; "
                f"expected one of {TRAIN_LOGPROB_BACKENDS!r}."
            )
        self.behavior_policy.assert_exact_trainer_reconstruction_supported()
        if not self.on_policy_logprob_check:
            raise ValueError(
                "PPO training requires the fail-closed on-policy logprob check; "
                "on_policy_logprob_check=False is not allowed."
            )


def _resolve_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype={name!r}")


def _validate_example_lengths(example: TrainExample) -> None:
    if not (
        len(example.input_ids)
        == len(example.target_ids)
        == len(example.loss_mask)
        == len(example.behavior_logprob_mask)
        == len(example.old_logprobs)
        == len(example.advantages)
    ):
        raise ValueError("TrainExample fields must all have equal length.")
    if any(value not in (0, 1) for value in example.loss_mask):
        raise ValueError("TrainExample loss_mask must contain only 0 or 1.")
    if any(value not in (0, 1) for value in example.behavior_logprob_mask):
        raise ValueError("TrainExample behavior_logprob_mask must contain only 0 or 1.")
    if any(
        has_behavior_logprob and not has_loss
        for has_behavior_logprob, has_loss in zip(
            example.behavior_logprob_mask,
            example.loss_mask,
            strict=True,
        )
    ):
        raise ValueError("behavior_logprob_mask must be a subset of loss_mask.")
    if any(
        advantage != 0.0 and not has_behavior_logprob
        for advantage, has_behavior_logprob in zip(
            example.advantages,
            example.behavior_logprob_mask,
            strict=True,
        )
    ):
        raise ValueError("Every nonzero-advantage token must have a behavior-policy logprob.")


def _is_overlength(*, example: TrainExample, max_tokens: int = 0) -> bool:
    return max_tokens > 0 and len(example.input_ids) > max_tokens


def _drop_overlength_examples(
    *,
    batch: list[TrainExample],
    max_tokens: int = 0,
) -> tuple[list[TrainExample], int]:
    if max_tokens <= 0:
        return list(batch), 0
    kept = [example for example in batch if not _is_overlength(example=example, max_tokens=max_tokens)]
    return kept, len(batch) - len(kept)


def _pad_batch(
    *,
    batch: list[TrainExample],
    pad_token_id: int,
    device: str,
    max_tokens: int = 0,
) -> dict[str, torch.Tensor]:
    for example in batch:
        _validate_example_lengths(example)
        if _is_overlength(example=example, max_tokens=max_tokens):
            raise ValueError(
                "Over-length TrainExample reached _pad_batch; drop over-length samples before padding "
                f"(len={len(example.input_ids)}, max_tokens={max_tokens})."
            )

    max_len = max(len(example.input_ids) for example in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    target_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    behavior_logprob_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    old_logprobs = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    advantages = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

    for row_idx, example in enumerate(batch):
        n = len(example.input_ids)
        input_ids[row_idx, :n] = torch.tensor(example.input_ids, dtype=torch.long, device=device)
        target_ids[row_idx, :n] = torch.tensor(example.target_ids, dtype=torch.long, device=device)
        attention_mask[row_idx, :n] = 1
        loss_mask[row_idx, :n] = torch.tensor(example.loss_mask, dtype=torch.bool, device=device)
        behavior_logprob_mask[row_idx, :n] = torch.tensor(
            example.behavior_logprob_mask,
            dtype=torch.bool,
            device=device,
        )
        old_logprobs[row_idx, :n] = torch.tensor(example.old_logprobs, dtype=torch.float32, device=device)
        advantages[row_idx, :n] = torch.tensor(example.advantages, dtype=torch.float32, device=device)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "behavior_logprob_mask": behavior_logprob_mask,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
    }


_TARGET_LOGPROB_POSITIONS_PER_CHUNK = 2048
_COMPILED_LM_HEAD_LOGITS_NO_BIAS = None


def _target_token_logprobs(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    behavior_temperature: float = 1.0,
    max_positions_per_chunk: int = _TARGET_LOGPROB_POSITIONS_PER_CHUNK,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError(f"target_ids shape {tuple(target_ids.shape)} does not match logits prefix {tuple(logits.shape[:2])}")
    if max_positions_per_chunk <= 0:
        raise ValueError("max_positions_per_chunk must be positive")
    if not math.isfinite(float(behavior_temperature)) or float(behavior_temperature) <= 0.0:
        raise ValueError(
            f"behavior_temperature must be finite and positive, got {behavior_temperature!r}."
        )
    vocab_size = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab_size)
    flat_target_ids = target_ids.reshape(-1)
    chunks = []
    for start in range(0, int(flat_target_ids.numel()), max_positions_per_chunk):
        end = min(start + max_positions_per_chunk, int(flat_target_ids.numel()))
        chunks.append(
            -F.cross_entropy(
                flat_logits[start:end].float() / float(behavior_temperature),
                flat_target_ids[start:end],
                reduction="none",
            )
        )
    return torch.cat(chunks, dim=0).reshape(target_ids.shape)


def _lm_head_logits_no_bias(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden_states.matmul(weight.t())


def _compiled_lm_head_logits_no_bias(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global _COMPILED_LM_HEAD_LOGITS_NO_BIAS
    if _COMPILED_LM_HEAD_LOGITS_NO_BIAS is None:
        if not hasattr(torch, "compile"):
            raise ValueError("compile_train_logprob_helper=True requires torch.compile.")
        _COMPILED_LM_HEAD_LOGITS_NO_BIAS = torch.compile(_lm_head_logits_no_bias, dynamic=True)
    return _COMPILED_LM_HEAD_LOGITS_NO_BIAS(hidden_states, weight)


def _lm_head_logits(
    *,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    compile_helper: bool,
) -> torch.Tensor:
    if compile_helper:
        if bias is not None:
            raise ValueError("compile_train_logprob_helper=True currently supports bias-free lm_head only.")
        return _compiled_lm_head_logits_no_bias(hidden_states, weight)
    logits = _lm_head_logits_no_bias(hidden_states, weight)
    if bias is not None:
        logits = logits + bias
    return logits


def _selected_lm_head_token_logprobs(
    *,
    hidden_states: torch.Tensor,
    lm_head,
    target_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    entropy_positions: torch.Tensor | None = None,
    behavior_temperature: float = 1.0,
    max_positions_per_chunk: int = _TARGET_LOGPROB_POSITIONS_PER_CHUNK,
    compile_helper: bool = False,
) -> tuple[torch.Tensor, float]:
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [batch, seq, hidden], got {tuple(hidden_states.shape)}")
    if target_ids.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"target_ids shape {tuple(target_ids.shape)} does not match hidden prefix {tuple(hidden_states.shape[:2])}"
        )
    if selected_positions.shape != target_ids.shape:
        raise ValueError(
            f"selected_positions shape {tuple(selected_positions.shape)} does not match target_ids {tuple(target_ids.shape)}"
        )
    if selected_positions.dtype != torch.bool:
        raise ValueError(f"selected_positions must be bool, got {selected_positions.dtype}")
    if entropy_positions is not None:
        if entropy_positions.shape != selected_positions.shape:
            raise ValueError(
                f"entropy_positions shape {tuple(entropy_positions.shape)} does not match "
                f"selected_positions {tuple(selected_positions.shape)}"
            )
        if entropy_positions.dtype != torch.bool:
            raise ValueError(f"entropy_positions must be bool, got {entropy_positions.dtype}")
        if bool((entropy_positions & ~selected_positions).any().detach().cpu().item()):
            raise ValueError("entropy_positions must be a subset of selected_positions.")
    if max_positions_per_chunk <= 0:
        raise ValueError("max_positions_per_chunk must be positive")
    if not math.isfinite(float(behavior_temperature)) or float(behavior_temperature) <= 0.0:
        raise ValueError(
            f"behavior_temperature must be finite and positive, got {behavior_temperature!r}."
        )
    if not hasattr(lm_head, "weight"):
        raise ValueError("lm_head must expose a weight tensor.")

    selected_target_ids = target_ids[selected_positions]
    if int(selected_target_ids.numel()) == 0:
        return hidden_states.new_empty((0,), dtype=torch.float32), 0.0

    selected_hidden_states = hidden_states[selected_positions]
    selected_entropy_positions = (
        torch.ones_like(selected_target_ids, dtype=torch.bool)
        if entropy_positions is None
        else entropy_positions[selected_positions]
    )
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    if int(weight.shape[1]) != int(selected_hidden_states.shape[-1]):
        raise ValueError(
            f"lm_head weight hidden dim {int(weight.shape[1])} does not match hidden dim "
            f"{int(selected_hidden_states.shape[-1])}."
        )

    logprob_chunks = []
    entropy_sum = 0.0
    for start in range(0, int(selected_target_ids.numel()), max_positions_per_chunk):
        end = min(start + max_positions_per_chunk, int(selected_target_ids.numel()))
        logits = _lm_head_logits(
            hidden_states=selected_hidden_states[start:end],
            weight=weight,
            bias=bias,
            compile_helper=compile_helper,
        )
        logits_float = logits.float() / float(behavior_temperature)
        logprob_chunks.append(
            -F.cross_entropy(
                logits_float,
                selected_target_ids[start:end],
                reduction="none",
            )
        )
        with torch.no_grad():
            log_probs = torch.log_softmax(logits_float.detach(), dim=-1)
            entropy_values = -(log_probs.exp() * log_probs).sum(dim=-1)
            entropy_sum += float(
                entropy_values[selected_entropy_positions[start:end]].sum().detach().cpu().item()
            )

    return torch.cat(logprob_chunks, dim=0), entropy_sum


def _effective_train_length(*, example: TrainExample, max_tokens: int = 0) -> int:
    if max_tokens <= 0 or len(example.input_ids) <= max_tokens:
        return len(example.input_ids)
    return max_tokens


def _truncated_row_length(*, example: TrainExample, max_tokens: int = 0) -> int:
    if _is_overlength(example=example, max_tokens=max_tokens):
        raise ValueError("Over-length examples should be dropped before logprob row construction.")
    return len(example.input_ids)


def _current_logprob_rows_from_token_logprobs(
    *,
    token_logprobs: torch.Tensor,
    batch: list[TrainExample],
    max_tokens: int = 0,
) -> list[list[float]]:
    token_logprobs_cpu = token_logprobs.detach().float().cpu()
    current_logprob_rows = []
    for row_idx, example in enumerate(batch):
        row_len = _truncated_row_length(example=example, max_tokens=max_tokens)
        current_logprob_rows.append(token_logprobs_cpu[row_idx, :row_len].tolist())
    return current_logprob_rows


def _current_logprob_rows_from_selected_logprobs(
    *,
    selected_logprobs: torch.Tensor,
    selected_positions: torch.Tensor,
    batch: list[TrainExample],
    max_tokens: int = 0,
) -> list[list[float]]:
    selected_logprobs_cpu = selected_logprobs.detach().float().cpu().tolist()
    selected_positions_cpu = selected_positions.detach().cpu()
    current_logprob_rows: list[list[float]] = []
    selected_idx = 0
    for row_idx, example in enumerate(batch):
        row_len = _truncated_row_length(example=example, max_tokens=max_tokens)
        row = [0.0] * row_len
        active_positions = torch.nonzero(
            selected_positions_cpu[row_idx, :row_len],
            as_tuple=False,
        ).flatten().tolist()
        for position in active_positions:
            row[int(position)] = float(selected_logprobs_cpu[selected_idx])
            selected_idx += 1
        current_logprob_rows.append(row)
    assert selected_idx == len(selected_logprobs_cpu)
    return current_logprob_rows


def _order_batch_for_minibatching(
    *,
    batch: list[TrainExample],
    max_tokens: int = 0,
    length_bucket_batches: bool = False,
) -> list[TrainExample]:
    if not length_bucket_batches:
        return list(batch)
    return sorted(batch, key=lambda example: _effective_train_length(example=example, max_tokens=max_tokens))


def _patch_weight_converter_compat() -> None:
    import inspect

    try:
        import peft.utils.transformers_weight_conversion as weight_conversion
    except ModuleNotFoundError as exc:
        if exc.name == "peft.utils.transformers_weight_conversion":
            return
        raise

    converter_cls = weight_conversion.WeightConverter
    signature = inspect.signature(converter_cls.__init__)
    if "distributed_operation" in signature.parameters:
        return
    if getattr(converter_cls, "_llm_local_rl_accepts_peft_ops", False):
        return

    original_init = converter_cls.__init__

    def patched_init(
        self,
        source_patterns,
        target_patterns,
        operations,
        distributed_operation=None,
        quantization_operation=None,
    ):
        original_init(
            self,
            source_patterns=source_patterns,
            target_patterns=target_patterns,
            operations=operations,
        )
        self.distributed_operation = distributed_operation
        self.quantization_operation = quantization_operation

    converter_cls.__init__ = patched_init
    converter_cls._llm_local_rl_accepts_peft_ops = True


def _is_configured_adapter_parameter(*, parameter_name: str, adapter_names: tuple[AdapterName, ...]) -> bool:
    name_parts = parameter_name.split(".")
    return any(adapter_name in name_parts for adapter_name in adapter_names)


class MultiAdapterTrainer:
    def __init__(self, *, config: TrainerConfig) -> None:
        self.config = config
        self.compute_device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.current_device = "cpu"
        self.tokenizer = self._load_tokenizer(base_model_path=config.base_model_path)
        self.saved_adapter_dirs: dict[AdapterName, str] = {}
        self.single_target_parameter_adapter_mode = False
        self.loaded_adapter_name: AdapterName | None = None
        self.reference_adapter_names: dict[str, str] = {}
        _patch_weight_converter_compat()
        self.model = self._build_new_model()
        self.optimizer = self._build_optimizer()
        self._mem_rec: MemoryTraceRecorder | None = maybe_create_recorder()

    def _mem_trace_active(self) -> bool:
        return self._mem_rec is not None and self.compute_device == "cuda"

    def _write_on_policy_logprob_records(self, *, records: list[dict]) -> None:
        if self.config.on_policy_logprob_warning_path is None:
            return
        path = Path(self.config.on_policy_logprob_warning_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def _causal_lm_for_selective_lm_head(self):
        causal_lm = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        if not hasattr(causal_lm, "model"):
            raise ValueError(
                "train_logprob_backend='selective_lm_head' requires a Hugging Face causal LM with a .model backbone."
            )
        if not hasattr(causal_lm, "get_output_embeddings"):
            raise ValueError(
                "train_logprob_backend='selective_lm_head' requires get_output_embeddings() for lm_head access."
            )
        lm_head = causal_lm.get_output_embeddings()
        if lm_head is None:
            raise ValueError("train_logprob_backend='selective_lm_head' could not resolve lm_head.")
        return causal_lm

    def _selective_lm_head_hidden_states(self, *, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        causal_lm = self._causal_lm_for_selective_lm_head()
        outputs = causal_lm.model(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if hidden_states.shape[:2] != tensors["input_ids"].shape:
            raise ValueError(
                f"Backbone hidden prefix {tuple(hidden_states.shape[:2])} does not match input_ids "
                f"{tuple(tensors['input_ids'].shape)}."
            )
        return hidden_states

    def _selective_lm_head_logprobs(
        self,
        *,
        tensors: dict[str, torch.Tensor],
        selected_positions: torch.Tensor,
        entropy_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        hidden_states = self._selective_lm_head_hidden_states(tensors=tensors)
        lm_head = self._causal_lm_for_selective_lm_head().get_output_embeddings()
        return _selected_lm_head_token_logprobs(
            hidden_states=hidden_states,
            lm_head=lm_head,
            target_ids=tensors["target_ids"],
            selected_positions=selected_positions,
            entropy_positions=entropy_positions,
            behavior_temperature=self.config.behavior_policy.temperature,
            compile_helper=self.config.compile_train_logprob_helper,
        )

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
        trainer.saved_adapter_dirs = dict(adapter_dirs)
        trainer.single_target_parameter_adapter_mode = _uses_target_parameter_adapters(adapter_dirs=adapter_dirs)
        trainer.loaded_adapter_name = None
        trainer.reference_adapter_names = {}
        _patch_weight_converter_compat()
        first_name = config.adapter_names[0]
        first_dir = adapter_dirs[first_name]

        base_model = trainer._build_base_model()

        model = PeftModel.from_pretrained(
            base_model,
            first_dir,
            adapter_name=first_name,
            is_trainable=True,
        )
        if not trainer.single_target_parameter_adapter_mode:
            for adapter_name in config.adapter_names[1:]:
                model.load_adapter(adapter_dirs[adapter_name], adapter_name=adapter_name, is_trainable=True)
        model.set_adapter(first_name)
        model.train()
        model.to(torch.device("cpu"))
        trainer.model = model
        trainer.loaded_adapter_name = first_name
        trainer.optimizer = trainer._build_optimizer()
        trainer._mem_rec: MemoryTraceRecorder | None = maybe_create_recorder()
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
        if self.config.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
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
            target_parameters=list(self.config.target_parameters),
        )
        model = get_peft_model(base_model, lora_config, adapter_name=self.config.adapter_names[0])
        for adapter_name in self.config.adapter_names[1:]:
            model.add_adapter(adapter_name, lora_config)
        model.set_adapter(self.config.adapter_names[0])
        model.train()
        model.to(torch.device("cpu"))
        return model

    def _build_optimizer(self):
        params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad
            or _is_configured_adapter_parameter(
                parameter_name=name,
                adapter_names=self.config.adapter_names,
            )
        ]
        return torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _move_optimizer_state(self, *, device: str) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(torch.device(device))

    def _selected_layer_optimizer_metrics(self) -> dict[str, float]:
        """Cheap scalar diagnostics for representative trainable LoRA layers."""
        named = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        layer_ids = sorted(
            {
                int(match.group(1))
                for name, _ in named
                if (match := re.search(r"(?:layers|blocks)\.(\d+)\.", name)) is not None
            }
        )
        selected: dict[int, str] = {}
        if layer_ids:
            last = layer_ids[-1]
            targets = ((0.0, "first"), (0.25, "depth25"), (0.50, "depth50"), (0.75, "depth75"), (1.0, "final"))
            for fraction, label in targets:
                selected[min(layer_ids, key=lambda value: abs(value - fraction * last))] = label

        accum: dict[str, dict[str, float]] = {}
        for name, param in named:
            lower = name.lower()
            group = None
            if "embed" in lower:
                group = "embedding"
            elif "lm_head" in lower:
                group = "lm_head"
            else:
                match = re.search(r"(?:layers|blocks)\.(\d+)\.", name)
                if match is not None:
                    group = selected.get(int(match.group(1)))
            if group is None:
                continue
            bucket = accum.setdefault(
                group,
                {"grad_sq": 0.0, "grad_max": 0.0, "param_sq": 0.0, "m1_sq": 0.0, "m2_sq": 0.0},
            )
            bucket["param_sq"] += float(param.detach().float().square().sum().cpu().item())
            if param.grad is not None:
                grad = param.grad.detach().float()
                bucket["grad_sq"] += float(grad.square().sum().cpu().item())
                bucket["grad_max"] = max(bucket["grad_max"], float(grad.abs().max().cpu().item()))
            state = self.optimizer.state.get(param, {})
            if torch.is_tensor(state.get("exp_avg")):
                bucket["m1_sq"] += float(state["exp_avg"].detach().float().square().sum().cpu().item())
            if torch.is_tensor(state.get("exp_avg_sq")):
                bucket["m2_sq"] += float(state["exp_avg_sq"].detach().float().square().sum().cpu().item())
        out: dict[str, float] = {}
        lr = float(self.optimizer.param_groups[0]["lr"])
        for group, values in accum.items():
            prefix = f"selected_layer/{group}"
            param_norm = math.sqrt(values["param_sq"])
            m1_norm = math.sqrt(values["m1_sq"])
            out[f"{prefix}/grad_norm_after_global_clip"] = math.sqrt(values["grad_sq"])
            out[f"{prefix}/grad_max_abs_after_global_clip"] = values["grad_max"]
            out[f"{prefix}/adam_first_moment_norm"] = m1_norm
            out[f"{prefix}/adam_second_moment_norm"] = math.sqrt(values["m2_sq"])
            out[f"{prefix}/relative_update_proxy"] = lr * m1_norm / max(param_norm, 1e-30)
        return out

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
        if self.single_target_parameter_adapter_mode:
            self._load_single_target_parameter_adapter(adapter_name=adapter_name)
        self.model.set_adapter(adapter_name)

    def _load_single_target_parameter_adapter(self, *, adapter_name: AdapterName) -> None:
        if self.loaded_adapter_name == adapter_name:
            return
        if self.loaded_adapter_name is not None and self.loaded_adapter_name in self.model.peft_config:
            self.model.delete_adapter(self.loaded_adapter_name)
        self.model.load_adapter(self.saved_adapter_dirs[adapter_name], adapter_name=adapter_name, is_trainable=True)
        self.model.set_adapter(adapter_name)
        self.loaded_adapter_name = adapter_name
        self.optimizer = self._build_optimizer()

    def load_reference_adapters(self, *, adapter_dirs: dict[str, str]) -> None:
        """Load frozen initialization adapters for periodic sampled-token KL."""
        if self.single_target_parameter_adapter_mode:
            self.reference_adapter_names = {}
            return
        for logical_name, adapter_dir in sorted(adapter_dirs.items()):
            reference_name = f"reference__{logical_name}"
            if reference_name not in self.model.peft_config:
                self.model.load_adapter(adapter_dir, adapter_name=reference_name, is_trainable=False)
            self.reference_adapter_names[logical_name] = reference_name

    def compute_logprobs(self, *, adapter_name: AdapterName, batch: list[TrainExample]) -> list[list[float]]:
        self.wake_up()
        self.set_adapter(adapter_name)
        self.model.eval()
        if (
            self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD
            and is_model_io_tracing_enabled()
        ):
            raise ValueError(
                "train_logprob_backend='selective_lm_head' does not support model I/O tracing; "
                "disable trace_model_io before calling compute_logprobs."
            )
        tensors = _pad_batch(
            batch=batch,
            pad_token_id=int(self.tokenizer.pad_token_id),
            device=self.current_device,
            max_tokens=self.config.train_max_tokens,
        )
        with torch.no_grad():
            if self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_FULL_LOGITS:
                outputs = self.model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                )
                token_logprobs = _target_token_logprobs(
                    logits=outputs.logits,
                    target_ids=tensors["target_ids"],
                    behavior_temperature=self.config.behavior_policy.temperature,
                )
                if is_model_io_tracing_enabled():
                    trace_top_k = get_trace_top_logprobs()
                    top_values = None
                    top_indices = None
                    if trace_top_k > 0:
                        log_probs = torch.log_softmax(
                            outputs.logits.float() / float(self.config.behavior_policy.temperature),
                            dim=-1,
                        )
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
                out = _current_logprob_rows_from_token_logprobs(
                    token_logprobs=token_logprobs,
                    batch=batch,
                    max_tokens=self.config.train_max_tokens,
                )
            elif self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD:
                completion_positions = tensors["behavior_logprob_mask"]
                completion_tokens = int(completion_positions.sum().detach().cpu().item())
                if completion_tokens > 0:
                    selected_logprobs, _ = self._selective_lm_head_logprobs(
                        tensors=tensors,
                        selected_positions=completion_positions,
                    )
                else:
                    selected_logprobs = tensors["old_logprobs"].new_empty((0,), dtype=torch.float32)
                out = _current_logprob_rows_from_selected_logprobs(
                    selected_logprobs=selected_logprobs,
                    selected_positions=completion_positions,
                    batch=batch,
                    max_tokens=self.config.train_max_tokens,
                )
            else:
                raise ValueError(f"Unsupported train_logprob_backend={self.config.train_logprob_backend!r}.")
        self.model.train()
        return out

    def train_batch(
        self,
        *,
        adapter_name: AdapterName,
        batch: list[TrainExample],
        measure_reference_kl: bool = False,
    ) -> dict[str, object]:
        if len(batch) == 0:
            return {
                "loss": 0.0,
                "loss_per_trained_token": 0.0,
                "num_examples": 0.0,
                "num_input_examples": 0.0,
                "num_dropped_overlength": 0.0,
                "num_trained_tokens": 0.0,
                "train_logprob_backend": self.config.train_logprob_backend,
                "train_logprob_backend_is_selective_lm_head": float(
                    self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD
                ),
            }
        if (
            self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD
            and is_model_io_tracing_enabled()
        ):
            raise ValueError(
                "train_logprob_backend='selective_lm_head' does not support model I/O tracing; "
                "disable trace_model_io for this training run."
            )

        trainable_batch, num_dropped_overlength = _drop_overlength_examples(
            batch=batch,
            max_tokens=self.config.train_max_tokens,
        )
        if len(trainable_batch) == 0:
            raise ValueError(
                "All train samples exceed train_max_tokens; "
                f"dropped {num_dropped_overlength} of {len(batch)} examples "
                f"(train_max_tokens={self.config.train_max_tokens})."
            )

        self.wake_up()
        reference_rows_by_example: dict[int, list[float]] = {}
        reference_name = self.reference_adapter_names.get(adapter_name)
        if measure_reference_kl and reference_name is not None:
            reference_rows = self.compute_logprobs(adapter_name=reference_name, batch=trainable_batch)
            reference_rows_by_example = {
                id(example): row for example, row in zip(trainable_batch, reference_rows, strict=True)
            }
        self.set_adapter(adapter_name)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        minibatch_size = (
            self.config.train_minibatch_size if self.config.train_minibatch_size > 0 else len(trainable_batch)
        )
        ordered_batch = _order_batch_for_minibatching(
            batch=trainable_batch,
            max_tokens=self.config.train_max_tokens,
            length_bucket_batches=self.config.train_length_bucket_batches and minibatch_size < len(trainable_batch),
        )
        normalization_sample_count = len(ordered_batch)
        total_loss_value = 0.0
        total_trained_tokens = 0
        approx_kl_numerator = 0.0
        total_forward_input_tokens = 0
        total_padded_input_tokens = 0
        total_lm_head_positions = 0
        total_minibatches = 0
        on_policy_checked_tokens = 0
        on_policy_trained_tokens_checked = 0
        on_policy_zero_advantage_loss_mask_tokens_checked = 0
        on_policy_injected_loss_mask_tokens_skipped = 0
        on_policy_violations = 0
        on_policy_trained_token_violations = 0
        on_policy_sum_abs_diff = 0.0
        on_policy_max_abs_diff = 0.0
        on_policy_trained_sum_abs_diff = 0.0
        on_policy_trained_max_abs_diff = 0.0
        ratio_values: list[float] = []
        delta_logp_values: list[float] = []
        advantage_values: list[float] = []
        reference_delta_logp_values: list[float] = []
        clip_count = 0
        clip_high_count = 0
        clip_low_count = 0
        positive_adv_clip_count = 0
        negative_adv_clip_count = 0
        positive_adv_count = 0
        negative_adv_count = 0
        entropy_sum = 0.0
        mem_step_idx = -1
        if self._mem_trace_active():
            mem_step_idx = self._mem_rec.next_step()
        for start_idx in range(0, len(ordered_batch), minibatch_size):
            minibatch = ordered_batch[start_idx : start_idx + minibatch_size]
            if self._mem_trace_active():
                reset_peak(self.compute_device)
                _mem_t_start = now_seconds()
                _mem_alloc_before_mb = current_alloc_bytes(self.compute_device)
                self._mem_rec.maybe_capture_weights_baseline(_mem_alloc_before_mb)
            else:
                _mem_t_start = 0.0
                _mem_alloc_before_mb = 0
            tensors = _pad_batch(
                batch=minibatch,
                pad_token_id=int(self.tokenizer.pad_token_id),
                device=self.current_device,
                max_tokens=self.config.train_max_tokens,
            )
            total_minibatches += 1
            total_forward_input_tokens += int(tensors["attention_mask"].sum().detach().cpu().item())
            total_padded_input_tokens += int(tensors["input_ids"].numel())
            trained_positions = tensors["loss_mask"] & (tensors["advantages"] != 0.0)
            invalid_trained_positions = trained_positions & ~tensors["behavior_logprob_mask"]
            if bool(invalid_trained_positions.any().detach().cpu().item()):
                self.optimizer.zero_grad(set_to_none=True)
                raise ValueError(
                    "A nonzero-advantage token is missing a behavior-policy logprob; "
                    "refusing to compute a PPO ratio."
                )
            trained_tokens = int(trained_positions.sum().detach().cpu().item())

            _mem_alloc_after_forward_full = ""
            _mem_peak_after_forward_full = ""
            _mem_alloc_after_backbone_sel = ""
            _mem_peak_after_backbone_sel = ""
            _mem_alloc_after_lm_head_sel = ""
            _mem_peak_after_lm_head_sel = ""
            _mem_elapsed_forward = 0.0
            _mem_elapsed_lm_head = 0.0

            if self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_FULL_LOGITS:
                if self._mem_trace_active():
                    reset_peak(self.compute_device)
                    _mem_t_forward = now_seconds()
                outputs = self.model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                )
                total_lm_head_positions += int(tensors["input_ids"].numel())
                token_logprobs = _target_token_logprobs(
                    logits=outputs.logits,
                    target_ids=tensors["target_ids"],
                    behavior_temperature=self.config.behavior_policy.temperature,
                )
                selected_logprobs = token_logprobs[trained_positions]
                selected_entropy_sum = 0.0
                if trained_tokens > 0:
                    with torch.no_grad():
                        trained_log_probs = torch.log_softmax(
                            outputs.logits[trained_positions].float()
                            / float(self.config.behavior_policy.temperature),
                            dim=-1,
                        )
                        selected_entropy_sum = float(
                            (-(trained_log_probs.exp() * trained_log_probs).sum(dim=-1)).sum().detach().cpu().item()
                        )
                if self._mem_trace_active():
                    _mem_alloc_after_forward_full = current_alloc_bytes(self.compute_device)
                    _mem_peak_after_forward_full = peak_alloc_bytes(self.compute_device)
                    _mem_elapsed_forward = now_seconds() - _mem_t_forward
                if self.config.on_policy_logprob_check:
                    current_logprob_rows = _current_logprob_rows_from_token_logprobs(
                        token_logprobs=token_logprobs,
                        batch=minibatch,
                        max_tokens=self.config.train_max_tokens,
                    )
                if is_model_io_tracing_enabled():
                    trace_top_k = get_trace_top_logprobs()
                    top_values = None
                    top_indices = None
                    if trace_top_k > 0:
                        log_probs = torch.log_softmax(
                            outputs.logits.float() / float(self.config.behavior_policy.temperature),
                            dim=-1,
                        )
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
            elif self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD:
                # Config validation makes the fail-closed parity gate mandatory, so
                # selective scoring always covers every sampled behavior-policy token.
                selected_positions = tensors["behavior_logprob_mask"]
                selected_position_count = int(selected_positions.sum().detach().cpu().item())
                total_lm_head_positions += selected_position_count
                if selected_position_count > 0:
                    if self._mem_trace_active():
                        reset_peak(self.compute_device)
                        _mem_t_forward = now_seconds()
                    hidden_states = self._selective_lm_head_hidden_states(tensors=tensors)
                    if self._mem_trace_active():
                        _mem_alloc_after_backbone_sel = current_alloc_bytes(self.compute_device)
                        _mem_peak_after_backbone_sel = peak_alloc_bytes(self.compute_device)
                        _mem_elapsed_forward = now_seconds() - _mem_t_forward
                        reset_peak(self.compute_device)
                        _mem_t_lm_head = now_seconds()
                    lm_head = self._causal_lm_for_selective_lm_head().get_output_embeddings()
                    scored_logprobs, selected_entropy_sum = _selected_lm_head_token_logprobs(
                        hidden_states=hidden_states,
                        lm_head=lm_head,
                        target_ids=tensors["target_ids"],
                        selected_positions=selected_positions,
                        entropy_positions=trained_positions,
                        behavior_temperature=self.config.behavior_policy.temperature,
                        compile_helper=self.config.compile_train_logprob_helper,
                    )
                    trained_within_selected = trained_positions[selected_positions]
                    selected_logprobs = scored_logprobs[trained_within_selected]
                    if self._mem_trace_active():
                        _mem_alloc_after_lm_head_sel = current_alloc_bytes(self.compute_device)
                        _mem_peak_after_lm_head_sel = peak_alloc_bytes(self.compute_device)
                        _mem_elapsed_lm_head = now_seconds() - _mem_t_lm_head
                else:
                    scored_logprobs = tensors["old_logprobs"].new_empty((0,), dtype=torch.float32)
                    selected_logprobs = scored_logprobs
                    selected_entropy_sum = 0.0
                if self.config.on_policy_logprob_check:
                    current_logprob_rows = _current_logprob_rows_from_selected_logprobs(
                        selected_logprobs=scored_logprobs,
                        selected_positions=selected_positions,
                        batch=minibatch,
                        max_tokens=self.config.train_max_tokens,
                    )
            else:
                raise ValueError(f"Unsupported train_logprob_backend={self.config.train_logprob_backend!r}.")

            if self.config.on_policy_logprob_check:
                check_result = check_on_policy_logprobs(
                    adapter_name=adapter_name,
                    examples=minibatch,
                    current_logprob_rows=current_logprob_rows,
                    tokenizer=self.tokenizer,
                    abs_tol=self.config.on_policy_logprob_abs_tol,
                    max_tokens=self.config.train_max_tokens,
                    max_records=self.config.on_policy_logprob_max_records_per_batch,
                    minibatch_start=start_idx,
                )
                on_policy_checked_tokens += check_result.num_checked_tokens
                on_policy_trained_tokens_checked += check_result.num_trained_tokens_checked
                on_policy_zero_advantage_loss_mask_tokens_checked += (
                    check_result.num_zero_advantage_loss_mask_tokens_checked
                )
                on_policy_injected_loss_mask_tokens_skipped += (
                    check_result.num_injected_loss_mask_tokens_skipped
                )
                on_policy_violations += check_result.num_violations
                on_policy_trained_token_violations += check_result.num_trained_token_violations
                on_policy_sum_abs_diff += check_result.sum_abs_logprob_diff
                on_policy_max_abs_diff = max(on_policy_max_abs_diff, check_result.max_abs_logprob_diff)
                on_policy_trained_sum_abs_diff += check_result.trained_token_sum_abs_logprob_diff
                on_policy_trained_max_abs_diff = max(
                    on_policy_trained_max_abs_diff,
                    check_result.trained_token_max_abs_logprob_diff,
                )
                if check_result.records:
                    self._write_on_policy_logprob_records(records=check_result.records)
                if check_result.num_violations > 0:
                    event = {
                        "event": "behavior_policy_logprob_contract_violation",
                        "adapter_name": adapter_name,
                        "minibatch_start": start_idx,
                        "completion_tokens_checked": check_result.num_checked_tokens,
                        "trained_tokens_checked": check_result.num_trained_tokens_checked,
                        "zero_advantage_loss_mask_tokens_checked": (
                            check_result.num_zero_advantage_loss_mask_tokens_checked
                        ),
                        "injected_loss_mask_tokens_skipped": (
                            check_result.num_injected_loss_mask_tokens_skipped
                        ),
                        "violations": check_result.num_violations,
                        "trained_token_violations": check_result.num_trained_token_violations,
                        "max_abs_diff": check_result.max_abs_logprob_diff,
                        "trained_token_max_abs_diff": check_result.trained_token_max_abs_logprob_diff,
                        "mean_abs_diff": (
                            check_result.sum_abs_logprob_diff / check_result.num_checked_tokens
                            if check_result.num_checked_tokens > 0
                            else 0.0
                        ),
                        "trained_token_mean_abs_diff": (
                            check_result.trained_token_sum_abs_logprob_diff
                            / check_result.num_trained_tokens_checked
                            if check_result.num_trained_tokens_checked > 0
                            else 0.0
                        ),
                        "first_offending_token": check_result.first_offending_token,
                        "first_offending_trained_token": check_result.first_offending_trained_token,
                        "report_path": self.config.on_policy_logprob_warning_path,
                        "behavior_policy": self.config.behavior_policy.to_dict(),
                        "gate_mode": (
                            "warn_only" if self.config.on_policy_logprob_warn_only else "fail_closed"
                        ),
                    }
                    print(json.dumps(event, sort_keys=True), flush=True)
                    if not self.config.on_policy_logprob_warn_only:
                        self.optimizer.zero_grad(set_to_none=True)
                        raise BehaviorPolicyLogprobMismatchError(
                            "Behavior-policy logprob parity failed before PPO ratio/backward: "
                            f"adapter={adapter_name!r}, minibatch_start={start_idx}, "
                            f"violations={check_result.num_violations}/"
                            f"{check_result.num_checked_tokens}, "
                            f"max_abs_diff={check_result.max_abs_logprob_diff:.9g}, "
                            f"abs_tol={self.config.on_policy_logprob_abs_tol:.9g}."
                        )
            if trained_tokens == 0:
                continue

            old_logprobs = tensors["old_logprobs"][trained_positions]
            if reference_rows_by_example:
                reference_examples = [
                    replace(example, old_logprobs=reference_rows_by_example[id(example)])
                    for example in minibatch
                ]
                reference_tensors = _pad_batch(
                    batch=reference_examples,
                    pad_token_id=int(self.tokenizer.pad_token_id),
                    device=self.current_device,
                    max_tokens=self.config.train_max_tokens,
                )
                reference_selected = reference_tensors["old_logprobs"][trained_positions]
                reference_delta_logp_values.extend(
                    (selected_logprobs.detach().float() - reference_selected.detach().float()).cpu().tolist()
                )
            nonfinite_logprobs = int((~torch.isfinite(selected_logprobs)).sum().detach().cpu().item())
            nonfinite_old_logprobs = int((~torch.isfinite(old_logprobs)).sum().detach().cpu().item())
            if nonfinite_logprobs or nonfinite_old_logprobs:
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Non-finite policy logprobs before PPO backward: "
                    f"current={nonfinite_logprobs}, old={nonfinite_old_logprobs}."
                )
            ratio = torch.exp(selected_logprobs - old_logprobs)
            clipped_ratio = torch.clamp(
                ratio,
                min=1.0 - self.config.ppo_clip_epsilon,
                max=1.0 + self.config.ppo_clip_epsilon,
            )
            advantages = tensors["advantages"][trained_positions]
            with torch.no_grad():
                ratio_detached = ratio.detach().float()
                ratio_values.extend(ratio_detached.cpu().tolist())
                delta_logp_values.extend((selected_logprobs.detach().float() - old_logprobs.detach().float()).cpu().tolist())
                advantage_values.extend(advantages.detach().float().cpu().tolist())
                clipped_positions = (
                    (ratio_detached < (1.0 - self.config.ppo_clip_epsilon))
                    | (ratio_detached > (1.0 + self.config.ppo_clip_epsilon))
                )
                clip_high_count += int(
                    (ratio_detached > (1.0 + self.config.ppo_clip_epsilon)).sum().detach().cpu().item()
                )
                clip_low_count += int(
                    (ratio_detached < (1.0 - self.config.ppo_clip_epsilon)).sum().detach().cpu().item()
                )
                positive_adv_positions = advantages.detach() > 0.0
                negative_adv_positions = advantages.detach() < 0.0
                clip_count += int(clipped_positions.sum().detach().cpu().item())
                positive_adv_count += int(positive_adv_positions.sum().detach().cpu().item())
                negative_adv_count += int(negative_adv_positions.sum().detach().cpu().item())
                positive_adv_clip_count += int(
                    (clipped_positions & positive_adv_positions).sum().detach().cpu().item()
                )
                negative_adv_clip_count += int(
                    (clipped_positions & negative_adv_positions).sum().detach().cpu().item()
                )
                entropy_sum += selected_entropy_sum
            objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = torch.sum(-objective) / normalization_sample_count
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite PPO loss before backward.")
            if self._mem_trace_active():
                reset_peak(self.compute_device)
                _mem_t_backward = now_seconds()
            loss.backward()
            if self._mem_trace_active():
                _mem_alloc_after_backward = current_alloc_bytes(self.compute_device)
                _mem_peak_after_backward = peak_alloc_bytes(self.compute_device)
                _mem_elapsed_backward = now_seconds() - _mem_t_backward
                seq_len_max = int(tensors["input_ids"].shape[1])
                self._mem_rec.append(
                    {
                        "step": mem_step_idx,
                        "adapter_name": adapter_name,
                        "minibatch_idx": total_minibatches - 1,
                        "seq_len_max": seq_len_max,
                        "num_examples": len(minibatch),
                        "num_trained_tokens": trained_tokens,
                        "num_padded_positions": int(tensors["input_ids"].numel()),
                        "train_logprob_backend": self.config.train_logprob_backend,
                        "alloc_before_mb": _mem_alloc_before_mb,
                        "alloc_after_forward_full_logits": _mem_alloc_after_forward_full,
                        "peak_after_forward_full_logits": _mem_peak_after_forward_full,
                        "alloc_after_backbone_selective": _mem_alloc_after_backbone_sel,
                        "peak_after_backbone_selective": _mem_peak_after_backbone_sel,
                        "alloc_after_lm_head_selective": _mem_alloc_after_lm_head_sel,
                        "peak_after_lm_head_selective": _mem_peak_after_lm_head_sel,
                        "alloc_after_backward": _mem_alloc_after_backward,
                        "peak_after_backward": _mem_peak_after_backward,
                        "wall_clock_s": now_seconds() - _mem_t_start,
                        "phase_elapsed_forward_s": _mem_elapsed_forward,
                        "phase_elapsed_lm_head_s": _mem_elapsed_lm_head,
                        "phase_elapsed_backward_s": _mem_elapsed_backward,
                    }
                )
            total_trained_tokens += trained_tokens
            total_loss_value += float(loss.detach().cpu().item())
            approx_kl_numerator += float(
                torch.sum(old_logprobs - selected_logprobs)
                .detach()
                .cpu()
                .item()
            )

        if total_trained_tokens == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "loss": 0.0,
                "loss_per_trained_token": 0.0,
                "num_examples": float(len(ordered_batch)),
                "num_input_examples": float(len(batch)),
                "num_dropped_overlength": float(num_dropped_overlength),
                "num_trained_tokens": 0.0,
                "on_policy_logprob_checked_tokens": float(on_policy_checked_tokens),
                "completion_tokens_checked": float(on_policy_checked_tokens),
                "trained_tokens_checked": float(on_policy_trained_tokens_checked),
                "zero_advantage_loss_mask_tokens_checked": float(
                    on_policy_zero_advantage_loss_mask_tokens_checked
                ),
                "zero_advantage_loss_mask_tokens_skipped": 0.0,
                "injected_loss_mask_tokens_skipped": float(
                    on_policy_injected_loss_mask_tokens_skipped
                ),
                "trained_token_mean_abs_diff": (
                    on_policy_trained_sum_abs_diff / on_policy_trained_tokens_checked
                    if on_policy_trained_tokens_checked > 0
                    else 0.0
                ),
                "trained_token_max_abs_diff": float(on_policy_trained_max_abs_diff),
                "on_policy_logprob_trained_tokens_checked": float(
                    on_policy_trained_tokens_checked
                ),
                "on_policy_logprob_trained_token_violations": float(
                    on_policy_trained_token_violations
                ),
                "on_policy_logprob_zero_advantage_loss_mask_tokens_checked": float(
                    on_policy_zero_advantage_loss_mask_tokens_checked
                ),
                "on_policy_logprob_zero_advantage_loss_mask_tokens_skipped": 0.0,
                "on_policy_logprob_injected_loss_mask_tokens_skipped": float(
                    on_policy_injected_loss_mask_tokens_skipped
                ),
                "on_policy_logprob_trained_token_mean_abs_diff": (
                    on_policy_trained_sum_abs_diff / on_policy_trained_tokens_checked
                    if on_policy_trained_tokens_checked > 0
                    else 0.0
                ),
                "on_policy_logprob_trained_token_max_abs_diff": float(
                    on_policy_trained_max_abs_diff
                ),
                "on_policy_logprob_violations": float(on_policy_violations),
                "on_policy_logprob_mean_abs_diff": (
                    on_policy_sum_abs_diff / on_policy_checked_tokens
                    if on_policy_checked_tokens > 0
                    else 0.0
                ),
                "on_policy_logprob_max_abs_diff": float(on_policy_max_abs_diff),
                "ratio_mean": 0.0,
                "ratio_p95": 0.0,
                "ratio_p99": 0.0,
                "clipfrac": 0.0,
                "clipfrac_positive_advantage": 0.0,
                "clipfrac_negative_advantage": 0.0,
                "entropy": 0.0,
                "grad_norm": 0.0,
                "weight_decay": float(self.config.weight_decay),
                "max_grad_norm": float(self.config.max_grad_norm),
                "num_forward_input_tokens": float(total_forward_input_tokens),
                "num_padded_input_tokens": float(total_padded_input_tokens),
                "num_train_minibatches": float(total_minibatches),
                "train_logprob_backend": self.config.train_logprob_backend,
                "train_logprob_backend_is_selective_lm_head": float(
                    self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD
                ),
                "compile_train_logprob_helper": float(self.config.compile_train_logprob_helper),
                "lm_head_positions": float(total_lm_head_positions),
                "lm_head_positions_avoided": float(total_padded_input_tokens - total_lm_head_positions),
                "lm_head_position_fraction": (
                    total_lm_head_positions / total_padded_input_tokens if total_padded_input_tokens > 0 else 0.0
                ),
            }

        grad_params = [param for param in self.model.parameters() if param.requires_grad and param.grad is not None]
        nonfinite_gradient_count = sum(
            int((~torch.isfinite(param.grad)).sum().detach().cpu().item()) for param in grad_params
        )
        if nonfinite_gradient_count:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite gradients before optimizer step: {nonfinite_gradient_count}."
            )
        grad_max_abs = max(
            (float(param.grad.detach().abs().max().cpu().item()) for param in grad_params),
            default=0.0,
        )
        if grad_params:
            clip_limit = self.config.max_grad_norm if self.config.max_grad_norm > 0.0 else math.inf
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(grad_params, max_norm=clip_limit).detach().cpu().item()
            )
        else:
            grad_norm = 0.0
        if self._mem_trace_active():
            reset_peak(self.compute_device)
        self.optimizer.step()
        selected_layer_metrics = self._selected_layer_optimizer_metrics()
        if self._mem_trace_active():
            alloc_after_optim = current_alloc_bytes(self.compute_device)
            peak_after_optim = peak_alloc_bytes(self.compute_device)
            self._mem_rec.maybe_capture_optim_state_baseline(alloc_after_optim)
            # Append one "optim-step summary row" tagged minibatch_idx=-1
            # per train_batch. Mirrors nvidia-smi peak-after-step intent and
            # lets the plot script draw the optimizer/state baseline band.
            self._mem_rec.append(
                {
                    "step": mem_step_idx,
                    "adapter_name": adapter_name,
                    "minibatch_idx": -1,
                    "seq_len_max": 0,
                    "num_examples": 0,
                    "num_trained_tokens": total_trained_tokens,
                    "num_padded_positions": total_padded_input_tokens,
                    "train_logprob_backend": self.config.train_logprob_backend,
                    "alloc_after_optim_step": alloc_after_optim,
                    "peak_during_optim_step": peak_after_optim,
                    "reserved_bytes_end": reserved_bytes(self.compute_device),
                    "wall_clock_s": 0.0,
                }
            )

        sorted_ratios = sorted(ratio_values)
        sorted_delta_logp_abs = sorted(abs(value) for value in delta_logp_values)
        sorted_sampled_old_kl = sorted(-value for value in delta_logp_values)
        sorted_advantages = sorted(advantage_values)
        sorted_reference_delta = sorted(reference_delta_logp_values)

        def _percentile(sorted_values: list[float], q: float) -> float:
            if not sorted_values:
                return 0.0
            idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
            return float(sorted_values[idx])

        return {
            "loss": total_loss_value,
            "loss_per_trained_token": total_loss_value / total_trained_tokens,
            "num_examples": float(len(ordered_batch)),
            "num_input_examples": float(len(batch)),
            "num_dropped_overlength": float(num_dropped_overlength),
            "num_trained_tokens": float(total_trained_tokens),
            "approx_kl": approx_kl_numerator / total_trained_tokens,
            "ppo_sampled_approx_kl": (
                sum((ratio - 1.0) - math.log(max(ratio, 1e-30)) for ratio in ratio_values)
                / len(ratio_values)
                if ratio_values else 0.0
            ),
            "ratio_mean": float(sum(ratio_values) / len(ratio_values)) if ratio_values else 0.0,
            "ratio_p01": _percentile(sorted_ratios, 0.01),
            "ratio_p05": _percentile(sorted_ratios, 0.05),
            "ratio_p95": _percentile(sorted_ratios, 0.95),
            "ratio_p99": _percentile(sorted_ratios, 0.99),
            "ratio_p999": _percentile(sorted_ratios, 0.999),
            "ratio_max": max(ratio_values) if ratio_values else 0.0,
            "clipfrac": clip_count / total_trained_tokens,
            "clipfrac_high": clip_high_count / total_trained_tokens,
            "clipfrac_low": clip_low_count / total_trained_tokens,
            "clipfrac_positive_advantage": (
                positive_adv_clip_count / positive_adv_count if positive_adv_count > 0 else 0.0
            ),
            "clipfrac_negative_advantage": (
                negative_adv_clip_count / negative_adv_count if negative_adv_count > 0 else 0.0
            ),
            "entropy": entropy_sum / total_trained_tokens,
            "grad_norm": grad_norm,
            "grad_was_clipped": float(self.config.max_grad_norm > 0.0 and grad_norm > self.config.max_grad_norm),
            "grad_max_abs": grad_max_abs,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "nonfinite_loss_count": 0.0,
            "nonfinite_logprob_count": 0.0,
            "nonfinite_gradient_count": 0.0,
            "delta_logp_mean_abs": (
                sum(abs(value) for value in delta_logp_values) / len(delta_logp_values)
                if delta_logp_values else 0.0
            ),
            "delta_logp_abs_p95": _percentile(sorted_delta_logp_abs, 0.95),
            "delta_logp_abs_p99": _percentile(sorted_delta_logp_abs, 0.99),
            "delta_logp_abs_p999": _percentile(sorted_delta_logp_abs, 0.999),
            "delta_logp_max_abs": max((abs(value) for value in delta_logp_values), default=0.0),
            "sampled_old_policy_kl_p50": _percentile(sorted_sampled_old_kl, 0.50),
            "sampled_old_policy_kl_p95": _percentile(sorted_sampled_old_kl, 0.95),
            "sampled_old_policy_kl_p99": _percentile(sorted_sampled_old_kl, 0.99),
            "sampled_old_policy_kl_p999": _percentile(sorted_sampled_old_kl, 0.999),
            "sampled_old_policy_kl_max": max(sorted_sampled_old_kl, default=0.0),
            "reference_sampled_kl_mean": (
                mean(reference_delta_logp_values) if reference_delta_logp_values else 0.0
            ),
            "reference_sampled_delta_logp_max_abs": max(
                (abs(value) for value in reference_delta_logp_values), default=0.0
            ),
            "reference_sampled_kl_max": max(reference_delta_logp_values, default=0.0),
            "reference_sampled_delta_logp_p95": _percentile(sorted_reference_delta, 0.95),
            "reference_kl_measured": float(bool(reference_delta_logp_values)),
            "advantage_mean": mean(advantage_values) if advantage_values else 0.0,
            "advantage_std": pstdev(advantage_values) if len(advantage_values) > 1 else 0.0,
            "advantage_max_abs": max((abs(value) for value in advantage_values), default=0.0),
            "advantage_fraction_positive": (
                sum(value > 0.0 for value in advantage_values) / len(advantage_values)
                if advantage_values else 0.0
            ),
            "advantage_fraction_negative": (
                sum(value < 0.0 for value in advantage_values) / len(advantage_values)
                if advantage_values else 0.0
            ),
            "advantage_p01": _percentile(sorted_advantages, 0.01),
            "advantage_p05": _percentile(sorted_advantages, 0.05),
            "advantage_p95": _percentile(sorted_advantages, 0.95),
            "advantage_p99": _percentile(sorted_advantages, 0.99),
            "advantage_p999": _percentile(sorted_advantages, 0.999),
            "weight_decay": float(self.config.weight_decay),
            "max_grad_norm": float(self.config.max_grad_norm),
            "num_forward_input_tokens": float(total_forward_input_tokens),
            "num_padded_input_tokens": float(total_padded_input_tokens),
            "num_train_minibatches": float(total_minibatches),
            "train_logprob_backend": self.config.train_logprob_backend,
            "train_logprob_backend_is_selective_lm_head": float(
                self.config.train_logprob_backend == TRAIN_LOGPROB_BACKEND_SELECTIVE_LM_HEAD
            ),
            "compile_train_logprob_helper": float(self.config.compile_train_logprob_helper),
            "lm_head_positions": float(total_lm_head_positions),
            "lm_head_positions_avoided": float(total_padded_input_tokens - total_lm_head_positions),
            "lm_head_position_fraction": (
                total_lm_head_positions / total_padded_input_tokens if total_padded_input_tokens > 0 else 0.0
            ),
            "on_policy_logprob_checked_tokens": float(on_policy_checked_tokens),
            "completion_tokens_checked": float(on_policy_checked_tokens),
            "trained_tokens_checked": float(on_policy_trained_tokens_checked),
            "zero_advantage_loss_mask_tokens_checked": float(
                on_policy_zero_advantage_loss_mask_tokens_checked
            ),
            "zero_advantage_loss_mask_tokens_skipped": 0.0,
            "injected_loss_mask_tokens_skipped": float(
                on_policy_injected_loss_mask_tokens_skipped
            ),
            "trained_token_mean_abs_diff": (
                on_policy_trained_sum_abs_diff / on_policy_trained_tokens_checked
                if on_policy_trained_tokens_checked > 0
                else 0.0
            ),
            "trained_token_max_abs_diff": float(on_policy_trained_max_abs_diff),
            "on_policy_logprob_trained_tokens_checked": float(
                on_policy_trained_tokens_checked
            ),
            "on_policy_logprob_trained_token_violations": float(
                on_policy_trained_token_violations
            ),
            "on_policy_logprob_zero_advantage_loss_mask_tokens_checked": float(
                on_policy_zero_advantage_loss_mask_tokens_checked
            ),
            "on_policy_logprob_zero_advantage_loss_mask_tokens_skipped": 0.0,
            "on_policy_logprob_injected_loss_mask_tokens_skipped": float(
                on_policy_injected_loss_mask_tokens_skipped
            ),
            "on_policy_logprob_trained_token_mean_abs_diff": (
                on_policy_trained_sum_abs_diff / on_policy_trained_tokens_checked
                if on_policy_trained_tokens_checked > 0
                else 0.0
            ),
            "on_policy_logprob_trained_token_max_abs_diff": float(
                on_policy_trained_max_abs_diff
            ),
            "on_policy_logprob_violations": float(on_policy_violations),
            "on_policy_logprob_mean_abs_diff": (
                on_policy_sum_abs_diff / on_policy_checked_tokens
                if on_policy_checked_tokens > 0
                else 0.0
            ),
            "on_policy_logprob_max_abs_diff": float(on_policy_max_abs_diff),
            **selected_layer_metrics,
        }

    def save_adapter(self, *, adapter_name: AdapterName, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.set_adapter(adapter_name)
        self.model.save_pretrained(str(output_path), selected_adapters=[adapter_name])
        if adapter_name == "default":
            return str(output_path)
        saved_path = str(output_path / adapter_name)
        self.saved_adapter_dirs[adapter_name] = saved_path
        return saved_path

    def training_state_dict(self) -> dict[str, object]:
        """Return the non-PEFT state required for an exact optimizer resume."""
        if self.single_target_parameter_adapter_mode:
            raise NotImplementedError(
                "Exact optimizer resume is not yet supported for target-parameter adapters "
                "because switching logical adapters rebuilds the optimizer."
            )
        return {
            "schema": "multi_adapter_trainer_state_v1",
            "optimizer": self.optimizer.state_dict(),
            "adapter_names": list(self.config.adapter_names),
            "learning_rate": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
        }

    def load_training_state_dict(self, state: dict[str, object]) -> None:
        if state.get("schema") != "multi_adapter_trainer_state_v1":
            raise ValueError(f"Unsupported trainer-state schema: {state.get('schema')!r}")
        if tuple(state.get("adapter_names", ())) != tuple(self.config.adapter_names):
            raise ValueError("Trainer-state adapter names do not match the configured adapters.")
        if float(state.get("learning_rate", float("nan"))) != float(self.config.learning_rate):
            raise ValueError("Trainer-state learning rate does not match the run configuration.")
        if float(state.get("weight_decay", float("nan"))) != float(self.config.weight_decay):
            raise ValueError("Trainer-state weight decay does not match the run configuration.")
        self.optimizer.load_state_dict(state["optimizer"])
        self._move_optimizer_state(device=self.current_device)

    def load_adapter(self, *, adapter_name: AdapterName, adapter_dir: str) -> None:
        if self.single_target_parameter_adapter_mode:
            self.saved_adapter_dirs[adapter_name] = adapter_dir
            self._load_single_target_parameter_adapter(adapter_name=adapter_name)
            return
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


def _uses_target_parameter_adapters(*, adapter_dirs: dict[AdapterName, str]) -> bool:
    for adapter_dir in adapter_dirs.values():
        payload = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
        if payload["target_parameters"]:
            return True
    return False
