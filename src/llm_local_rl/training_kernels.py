"""Explicit training kernel selection; no silent fallback for requested kernels."""

import inspect


def validate_training_kernels(*, train_gdn_backend, train_lm_head_kernel,
                              train_logprob_backend, compile_train_logprob_helper):
    if train_gdn_backend not in ("auto", "fla"):
        raise ValueError("train_gdn_backend must be 'auto' or 'fla'")
    if train_lm_head_kernel not in ("torch", "triton"):
        raise ValueError("train_lm_head_kernel must be 'torch' or 'triton'")
    if train_lm_head_kernel == "triton":
        if train_logprob_backend != "selective_lm_head":
            raise ValueError("triton LM-head kernel requires train_logprob_backend='selective_lm_head'")
        if compile_train_logprob_helper:
            raise ValueError("triton LM-head kernel cannot use compile_train_logprob_helper")


def _resolve_hf_fallback(function):
    # Transformers 5.16 captures the selected optional kernel in a closure;
    # __wrapped__ alone points to the Torch fallback, not the actual callee.
    while inspect.isfunction(function):
        nonlocals = inspect.getclosurevars(function).nonlocals
        if "implementation" in nonlocals:
            return nonlocals["implementation"]
        if not hasattr(function, "__wrapped__"):
            break
        function = function.__wrapped__
    return function


def verify_fla_gdn(model):
    """Require HF's existing FLA + causal-conv dispatch on every Qwen3.5 GDN.

    Install optional dependencies before starting Python. Do not monkey-patch
    HF globals or replace parameters, which would complicate LoRA/checkpoints.
    """
    from importlib.metadata import version
    from causal_conv1d import causal_conv1d_fn
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    layers = [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
    if not layers:
        raise ValueError("train_gdn_backend='fla' requires Qwen3.5 GatedDeltaNet layers")
    for layer in layers:
        if hasattr(layer, "chunk_gated_delta_rule"):
            chunk, conv = layer.chunk_gated_delta_rule, layer.causal_conv1d_fn
        else:
            module = inspect.getmodule(type(layer))
            chunk = _resolve_hf_fallback(module.torch_chunk_gated_delta_rule)
            conv = _resolve_hf_fallback(module.causal_conv1d_fn)
        if chunk is not chunk_gated_delta_rule or conv is not causal_conv1d_fn:
            raise RuntimeError("Requested FLA GDN/causal-conv kernels are not active; refusing Torch fallback")
    return {
        "gdn_backend": "fla", "gdn_layers": len(layers),
        "chunk_kernel": f"{chunk.__module__}.{chunk.__name__}",
        "conv_kernel": f"{conv.__module__}.{conv.__name__}",
        "flash_linear_attention_version": version("flash-linear-attention"),
        "causal_conv1d_version": version("causal-conv1d"),
    }
