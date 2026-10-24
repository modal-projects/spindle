"""Isolate CUDA low-rank batch-size rounding without loading model weights."""

import json
from itertools import product

import modal

from spindle.providers.modal.torchtitan_image import image

app = modal.App("spindle-torchtitan-lora-validation")

with image.imports():
    import torch
    from spindle.backends.torchtitan_runtime.lora import lora_bmm


@app.function(image=image, gpu="H200", cpu=8, memory=16384, timeout=600)
def probe():
    torch.manual_seed(42)

    def mm(left, right, pad):
        return lora_bmm(left, right) if pad else torch.bmm(left, right)

    results = []
    shapes = [
        (2560, 9216, True),
        (9216, 2560, False),
        (2560, 8192, False),
        (2560, 1024, False),
        (4096, 2560, False),
        (5120, 17408, True),
        (17408, 5120, False),
    ]
    for dtype, (inputs, outputs, fused) in product(
        (torch.bfloat16, torch.float32), shapes
    ):
        x = torch.randn(8, 1024, inputs, device="cuda", dtype=torch.bfloat16).to(dtype)
        branches = 2 if fused else 1
        a = (torch.randn(8, branches, 32, inputs, device="cuda") * 0.01).to(dtype)[:, 0]
        b = (torch.randn(8, branches, outputs, 32, device="cuda") * 0.0001).to(dtype)[
            :, 0
        ]
        for pad in (False, True):
            serial = mm(
                mm(x[:1], a[:1].transpose(1, 2), pad), b[:1].transpose(1, 2), pad
            )
            for count in (2, 4, 8):
                packed = mm(
                    mm(x[:count], a[:count].transpose(1, 2), pad),
                    b[:count].transpose(1, 2),
                    pad,
                )[:1]
                if pad:
                    torch.testing.assert_close(packed, serial, atol=0, rtol=0)
                results.append(
                    {
                        "dtype": str(dtype),
                        "shape": [inputs, outputs, fused],
                        "pad_singleton": pad,
                        "batch": count,
                        "max_absolute": (packed - serial).abs().max().item(),
                        "unequal_fraction": (packed != serial).float().mean().item(),
                    }
                )
    return results


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2))
