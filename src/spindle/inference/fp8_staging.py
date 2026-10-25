"""Restore canonical BF16 loader parameters in isolated online-FP8 staging views.

SGLang's online quantizer replaces parameters, drops their loader callbacks,
and transposes dense weights. Reloading must restore the pre-quantization shape,
dtype, parameter class and callbacks before loading another BF16 publication.
Only the isolated CPU shadow invokes restore; live serving weights stay intact.
"""

import os

import torch


def online_block_quantization():
    return os.environ.get("SGLANG_FP8_BLOCK_ONLINE") == "1"


def quantize_blocks(weight, block=128):
    """E4M3 weights with FP32 dequantization scales per 128 x 128 block."""
    height, width = weight.shape[-2:]
    if height % block or width % block:
        raise ValueError("online FP8 weights must align with their quantization blocks")
    values = weight.float().reshape(
        *weight.shape[:-2], height // block, block, width // block, block
    )
    scales = values.abs().amax(dim=(-3, -1)).clamp_min(1e-12) / 448.0
    quantized = (values / scales.unsqueeze(-1).unsqueeze(-3)).clamp(-448, 448)
    return quantized.to(torch.float8_e4m3fn).reshape(weight.shape), scales


def quantize_block_weights(layer, names):
    for name in names:
        parameter = layer._parameters[name]
        quantized, scales = quantize_blocks(parameter.data)
        parameter.data = quantized
        scale = torch.nn.Parameter(scales, requires_grad=False)
        scale.format_ue8m0 = False
        layer._parameters[name + "_scale_inv"] = scale


def remember_loader_state(layer, names):
    if hasattr(layer, "_online_fp8_loader_state"):
        return
    layer._online_fp8_loader_state = {
        name: (
            type(parameter),
            tuple(parameter.shape),
            parameter.dtype,
            dict(vars(parameter)),
        )
        for name in names
        if (parameter := layer._parameters[name]) is not None
    }


def restore_loader_state(layer):
    for name, (
        parameter_type,
        shape,
        dtype,
        attributes,
    ) in layer._online_fp8_loader_state.items():
        parameter = parameter_type._make_subclass(
            parameter_type, torch.empty(shape, dtype=dtype, device="cpu"), False
        )
        parameter.__dict__.update(attributes)
        layer._parameters[name] = parameter
