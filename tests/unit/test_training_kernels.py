"""CPU configuration/dispatch tests plus real Triton interpreter or CUDA parity.

TRITON_INTERPRET=1 executes kernel arithmetic on CPU; it is NOT a GPU benchmark.
Normal pytest runs exercise CUDA kernels when a device is present.
"""
import functools
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llm_local_rl.config import TrainRunConfig
from llm_local_rl.checkpointing import config_fingerprint
from llm_local_rl.trainer import TrainerConfig, _selected_lm_head_token_logprobs
from llm_local_rl.training_kernels import _resolve_hf_fallback, verify_fla_gdn
from scripts.run_train import parse_args


INTERPRETER = os.environ.get("TRITON_INTERPRET") == "1"
HAS_CUDA = torch.cuda.is_available()
KERNEL_DEVICE = "cpu" if INTERPRETER else "cuda"
kernel_test = pytest.mark.skipif(not (INTERPRETER or HAS_CUDA), reason="requires CUDA or explicit Triton interpreter")


def test_kernel_config_cli_roundtrip_and_invalid_combinations():
    args = parse_args(["--model-path", "/model", "--output-dir", "/output",
                       "--train-logprob-backend", "selective_lm_head",
                       "--train-lm-head-kernel", "triton", "--train-gdn-backend", "fla"])
    config = TrainRunConfig(model_path=args.model_path, output_dir=args.output_dir,
                            train_logprob_backend=args.train_logprob_backend,
                            train_lm_head_kernel=args.train_lm_head_kernel,
                            train_gdn_backend=args.train_gdn_backend)
    assert TrainRunConfig.from_dict(config.to_dict()) == config
    legacy = config.to_dict()
    del legacy["train_lm_head_kernel"], legacy["train_gdn_backend"]
    restored = TrainRunConfig.from_dict(legacy)
    assert restored.train_lm_head_kernel == "torch"
    assert restored.train_gdn_backend == "auto"
    assert config_fingerprint(legacy) == config_fingerprint(restored.to_dict())
    assert config_fingerprint(config.to_dict()) != config_fingerprint(legacy)
    for kwargs in ({"train_lm_head_kernel": "invalid"}, {"train_gdn_backend": "invalid"},
                   {"train_lm_head_kernel": "triton"},
                   {"train_lm_head_kernel": "triton", "train_logprob_backend": "selective_lm_head",
                    "compile_train_logprob_helper": True}):
        with pytest.raises(ValueError):
            TrainerConfig(base_model_path="/model", **kwargs)
        with pytest.raises(ValueError):
            TrainRunConfig(model_path="/model", output_dir="/output", **kwargs)


def test_explicit_triton_rejects_cpu():
    from llm_local_rl.fused_lm_head import require_cuda
    with pytest.raises(ValueError, match="requires CUDA"):
        require_cuda(torch.zeros(2, 4))


def test_allowed_token_target_rejected_before_kernel(monkeypatch):
    from llm_local_rl.fused_lm_head import selected_logprobs
    monkeypatch.setattr("llm_local_rl.fused_lm_head.require_cuda", lambda tensor: None)
    with pytest.raises(ValueError, match="escaped"):
        selected_logprobs(hidden=torch.zeros(1, 4), weight=torch.zeros(7, 4), bias=None,
                          targets=torch.tensor([5]), entropy_positions=torch.tensor([True]),
                          temperature=1.0, allowed_tokens=[(0, 1)], chunk_size=2)


def _hf_wrapper(fallback, implementation):
    @functools.wraps(fallback)
    def wrapped(*args, **kwargs):
        return implementation(*args, **kwargs)
    return wrapped


@pytest.mark.parametrize("new_hf", [False, True])
def test_gdn_requires_actual_dispatch_not_just_installed_package(monkeypatch, new_hf):
    def chunk(): pass
    def conv(): pass
    def fallback(): pass
    fla = ModuleType("fla.ops.gated_delta_rule")
    fla.chunk_gated_delta_rule = chunk
    causal = ModuleType("causal_conv1d")
    causal.causal_conv1d_fn = conv
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fla)
    monkeypatch.setitem(sys.modules, "causal_conv1d", causal)
    monkeypatch.setattr("importlib.metadata.version", lambda package: "test-only")
    layer = type("Qwen3_5GatedDeltaNet", (), {})()
    model = SimpleNamespace(modules=lambda: [layer])
    if new_hf:
        module = SimpleNamespace(torch_chunk_gated_delta_rule=_hf_wrapper(fallback, chunk),
                                 causal_conv1d_fn=_hf_wrapper(fallback, conv))
        monkeypatch.setattr("llm_local_rl.training_kernels.inspect.getmodule", lambda cls: module)
        assert _resolve_hf_fallback(module.torch_chunk_gated_delta_rule) is chunk
    else:
        layer.chunk_gated_delta_rule, layer.causal_conv1d_fn = chunk, conv
    assert verify_fla_gdn(model)["gdn_layers"] == 1
    if new_hf:
        module.torch_chunk_gated_delta_rule = _hf_wrapper(fallback, fallback)
    else:
        layer.chunk_gated_delta_rule = fallback
    with pytest.raises(RuntimeError, match="refusing Torch fallback"):
        verify_fla_gdn(model)
    with pytest.raises(ValueError, match="requires Qwen3.5"):
        verify_fla_gdn(SimpleNamespace(modules=lambda: []))


@kernel_test
@pytest.mark.parametrize("vocab", [31, 8199, 248320])
@pytest.mark.parametrize("temperature", [0.8, 1.0, 1.7])
def test_fused_reductions_full_vocab_tail_temperature_signed_vjp(vocab, temperature):
    pytest.importorskip("triton")
    from llm_local_rl.triton_logprobs import logprobs_entropy
    torch.manual_seed(51)
    logits = (torch.randn(3, vocab, device=KERNEL_DEVICE) * 3 + 100).requires_grad_()
    targets = torch.tensor([0, vocab - 1, vocab // 2], device=KERNEL_DEVICE)
    actual, entropy = logprobs_entropy(logits, targets, temperature)
    weights = torch.tensor([1.3, -0.7, 0.0], device=KERNEL_DEVICE)
    grad, = torch.autograd.grad(actual, logits, weights)
    # CPU FP32 log_softmax can accumulate ~5e-5 normalization error at 248k
    # vocabulary. Compare reductions to float64 after reference FP32 scaling.
    ref = (logits.float() / temperature).double().log_softmax(-1)
    expected = ref.gather(1, targets[:, None]).squeeze(1)
    ref_grad, = torch.autograd.grad(expected, logits, weights)
    torch.testing.assert_close(actual, expected.float(), atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(entropy, -(ref.exp() * ref).sum(-1).float(), atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(grad, ref_grad, atol=3e-6, rtol=3e-5)
    assert not entropy.requires_grad


@kernel_test
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("restricted", ["none", "mixed", "all"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_head_gradients_masks_restrictions_and_saved_memory(monkeypatch, frozen, bias, restricted, dtype):
    pytest.importorskip("triton")
    if INTERPRETER and dtype == torch.bfloat16:
        pytest.skip("Triton interpreter does not support BF16; run on CUDA")
    if INTERPRETER:
        monkeypatch.setattr("llm_local_rl.fused_lm_head.require_cuda", lambda tensor: None)
    torch.manual_seed(14)
    hidden = torch.randn(2, 5, 8, dtype=dtype, device=KERNEL_DEVICE, requires_grad=True)
    head = torch.nn.Linear(8, 37, bias=bias, dtype=dtype, device=KERNEL_DEVICE).requires_grad_(not frozen)
    selected = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 1, 1]], device=KERNEL_DEVICE, dtype=torch.bool)
    entropy_mask = selected.clone()
    entropy_mask[0, 0] = False
    targets = torch.arange(10, device=KERNEL_DEVICE).reshape(2, 5) % 2
    allowed = {"none": None, "all": [(1, 0)] * 7,
               "mixed": [(), (1, 0), (), (0, 1), (), (1, 0), ()]}[restricted]
    kwargs = dict(hidden_states=hidden, lm_head=head, target_ids=targets,
                  selected_positions=selected, entropy_positions=entropy_mask,
                  behavior_temperature=0.8, allowed_token_ids_by_selected_position=allowed,
                  max_positions_per_chunk=3)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t) or t), lambda t: t):
        actual, entropy = _selected_lm_head_token_logprobs(**kwargs, kernel="triton")
    assert not any(t.ndim == 2 and t.shape[-1] == 37 for t in saved)
    params = (hidden,) if frozen else ((hidden, head.weight, head.bias) if bias else (hidden, head.weight))
    weights = torch.linspace(-0.7, 1.2, 7, device=KERNEL_DEVICE)
    grads = torch.autograd.grad(actual, params, weights)
    expected, expected_entropy = _selected_lm_head_token_logprobs(**kwargs)
    ref_grads = torch.autograd.grad(expected, params, weights)
    # Changing restricted GEMM shape can change BF16 rounding on CUDA.
    atol, rtol = (0.02, 0.02) if dtype == torch.bfloat16 else (2e-6, 2e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert entropy == pytest.approx(expected_entropy, abs=0.02 if dtype == torch.bfloat16 else 4e-6)
    for actual_grad, reference_grad in zip(grads, ref_grads):
        torch.testing.assert_close(actual_grad, reference_grad, atol=atol, rtol=rtol)
    with torch.no_grad():
        values, _ = _selected_lm_head_token_logprobs(**kwargs, kernel="triton")
    assert not values.requires_grad
    torch.testing.assert_close(values, actual)


@pytest.mark.skipif(not HAS_CUDA or INTERPRETER, reason="real CUDA bf16 test")
def test_cuda_bf16_signed_vjp():
    from llm_local_rl.triton_logprobs import logprobs_entropy
    torch.manual_seed(2)
    logits = torch.randn(5, 248320, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.tensor([0, 8191, 8192, 248319, 100], device="cuda")
    weights = torch.tensor([1.0, -0.7, 0.3, 0.0, 2.0], device="cuda")
    actual, ent = logprobs_entropy(logits, targets, 0.8)
    ref = (logits.float() / 0.8).log_softmax(-1)
    expected = ref.gather(1, targets[:, None]).squeeze(1)
    ga, = torch.autograd.grad(actual, logits, weights)
    ge, = torch.autograd.grad(expected, logits, weights)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(ent, -(ref.exp() * ref).sum(-1), atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(ga, ge, atol=2e-7, rtol=0.01)


@pytest.mark.skipif(not HAS_CUDA or INTERPRETER, reason="real CUDA FLA backward test")
def test_cuda_fla_gdn_gradients_match_torch():
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    import inspect
    reference = inspect.unwrap(hf.torch_chunk_gated_delta_rule)
    torch.manual_seed(41)
    q, k, v = [torch.randn(2, 67, 4, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True) for _ in range(3)]
    g = (-torch.rand(2, 67, 4, device="cuda") * 0.1).requires_grad_()
    beta = torch.rand(2, 67, 4, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    args = dict(g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    actual, _ = fla.chunk_gated_delta_rule(q, k, v, **args)
    expected, _ = reference(q, k, v, **args)
    upstream = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (q, k, v, g, beta), upstream)
    ge = torch.autograd.grad(expected, (q, k, v, g, beta), upstream)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=0.03, rtol=0.04)
