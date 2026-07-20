from __future__ import annotations

from types import SimpleNamespace

import importlib.util
import math

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("torch and transformers are required for transformers sampler tests", allow_module_level=True)

import torch

from llm_local_rl.transformers_sampling import TrainerTransformersSampler
from llm_local_rl.types import SamplingRequest


class FakeTokenizer:
    pad_token_id = 0

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class FakeModel:
    def __init__(self) -> None:
        self.forward_calls: list[dict[str, torch.Tensor]] = []

    def eval(self) -> None:
        return None

    def generate(self, *, input_ids, attention_mask, max_new_tokens, generation_config, return_dict_in_generate, output_scores):
        _ = (attention_mask, return_dict_in_generate, output_scores)
        assert generation_config.temperature == 0.5
        assert max_new_tokens == 1
        generated = torch.tensor([[3], [4]], dtype=torch.long, device=input_ids.device)
        return SimpleNamespace(sequences=torch.cat([input_ids, generated], dim=1))

    def __call__(self, *, input_ids, attention_mask):
        self.forward_calls.append({"input_ids": input_ids.detach().clone(), "attention_mask": attention_mask.detach().clone()})
        logits = torch.zeros((*input_ids.shape, 6), dtype=torch.float32, device=input_ids.device)
        logits[0, 2] = torch.tensor([0.0, -1.0, -2.0, 3.0, 1.0, -3.0], device=input_ids.device)
        logits[1, 2] = torch.tensor([0.0, 1.0, -1.0, -2.0, 4.0, -3.0], device=input_ids.device)
        return SimpleNamespace(logits=logits)


class FakeTrainer:
    def __init__(self) -> None:
        self.current_device = "cpu"
        self.model = FakeModel()
        self.active_adapters: list[str] = []

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapters.append(adapter_name)

    def wake_up(self) -> None:
        return None


def test_transformers_sampler_records_raw_model_logprobs_not_temperature_scores(monkeypatch) -> None:
    trainer = FakeTrainer()
    sampler = TrainerTransformersSampler(trainer=trainer, tokenizer=FakeTokenizer())
    requests = [
        SamplingRequest(
            adapter_name="debate",
            prompt_token_ids=[1, 2, 5],
            stop_token_ids=[],
            max_tokens=1,
            temperature=0.5,
        ),
        SamplingRequest(
            adapter_name="debate",
            prompt_token_ids=[7, 8, 9],
            stop_token_ids=[],
            max_tokens=1,
            temperature=0.5,
        ),
    ]
    expected_first = torch.log_softmax(torch.tensor([0.0, -1.0, -2.0, 3.0, 1.0, -3.0]), dim=-1)[3].item()
    expected_second = torch.log_softmax(torch.tensor([0.0, 1.0, -1.0, -2.0, 4.0, -3.0]), dim=-1)[4].item()

    def fail_full_log_softmax(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("sampler must not materialize full log_softmax over batch x seq x vocab")

    monkeypatch.setattr(torch, "log_softmax", fail_full_log_softmax)

    results = sampler.sample_many(requests)

    assert math.isclose(results[0].completion_logprobs[0], expected_first, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(results[1].completion_logprobs[0], expected_second, rel_tol=0.0, abs_tol=1e-6)
    assert results[0].raw["completion_logprobs"] == "raw_model_logprobs"
    assert results[1].raw["completion_logprobs"] == "raw_model_logprobs"
    assert trainer.model.forward_calls[0]["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
