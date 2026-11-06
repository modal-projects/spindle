"""GPU reference check for online NVFP4 with nonzero pre-SwiGLU LoRA deltas.

Run with ``modal run -e kailash-dev -m mixed_samplers.validate_nvfp4_lora``.
The reference dequantizes exactly the same weights. Residual error includes
NVFP4 activation quantization inside both GEMMs, measured separately from the
adapter's effect. This is a kernel integration check, not an RL stability result.
"""

import json
from pathlib import Path

import modal

from spindle.providers.modal.rollout_image import image

app = modal.App("spindle-mimo-nvfp4-lora-validation")


@app.function(
    image=image,
    gpu="B200",
    cpu=16,
    memory=131072,
    timeout=3600,
    max_containers=1,
    env={"MAX_JOBS": "2"},
)
def check():
    import torch
    from flashinfer import SfLayout, nvfp4_quantize
    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp4_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    torch.manual_seed(42)
    device = "cuda"
    experts = 8
    hidden = 2048
    inter = 512
    topk = 2
    w13 = (
        torch.randn(experts, 2 * inter, hidden, device=device, dtype=torch.bfloat16)
        * 0.01
    )
    w2 = torch.randn(experts, hidden, inter, device=device, dtype=torch.bfloat16) * 0.02

    def quantize(weights):
        packed = []
        scales = []
        decode = []
        restored = []
        levels = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=device,
        )
        for weight in weights:
            scale = weight.abs().amax().float() / (448 * 6)
            q, s = nvfp4_quantize(
                weight.contiguous(),
                1 / scale,
                sfLayout=SfLayout.layout_linear,
                backend="cute-dsl",
            )
            q = q.reshape(weight.shape[0], weight.shape[1] // 2).view(torch.uint8)
            s = s.view(torch.float8_e4m3fn).reshape(
                weight.shape[0], weight.shape[1] // 16
            )
            indices = torch.stack([q & 15, q >> 4], dim=-1).flatten(-2).long()
            restored.append(
                levels[indices] * s.float().repeat_interleave(16, dim=-1) * scale
            )
            packed.append(q)
            scales.append(s)
            decode.append(scale)
        return (
            torch.stack(packed),
            torch.stack(scales),
            torch.stack(decode),
            torch.stack(restored),
        )

    q13, s13, d13, r13 = quantize(w13)
    q2, s2, d2, r2 = quantize(w2)
    rows = []
    # SGLang's TRTLLM loader stores [up, gate], then applies the gated shuffle.
    # The LoRA delta remains [gate, up], matching exported adapter tensors.
    torch.manual_seed(123)
    p13, ps13, p2, ps2 = prepare_static_weights_for_trtllm_fp4_moe(
        torch.cat(q13.chunk(2, dim=1)[::-1], dim=1),
        q2,
        torch.cat(s13.chunk(2, dim=1)[::-1], dim=1),
        s2,
        hidden,
        inter,
        experts,
        is_gated=True,
    )
    for n in [1, 17, 128]:
        x = torch.randn(n, hidden, device=device, dtype=torch.bfloat16) * 0.5
        ids = torch.stack(
            [
                torch.arange(n, device=device) % experts,
                (torch.arange(n, device=device) + 3) % experts,
            ],
            dim=-1,
        ).int()
        weights = torch.tensor([0.3, 0.7], device=device).expand(n, 2).contiguous()
        packed = fused_pack_topk(ids, weights)
        outputs = []
        refs = []
        for strength in [0, 0.2]:
            delta = (
                torch.randn(n, topk, 2 * inter, device=device, dtype=torch.bfloat16)
                * strength
            )
            activation = torch.empty(
                n, topk, inter, device=device, dtype=torch.bfloat16
            )
            out = trtllm_fp4_block_scale_routed_moe_lora(
                topk_ids=packed,
                routing_bias=None,
                hidden_states=x,
                hidden_states_scale=None,
                gemm1_weights=p13,
                gemm1_weights_scale=ps13.view(torch.float8_e4m3fn),
                gemm2_weights=p2,
                gemm2_weights_scale=ps2.view(torch.float8_e4m3fn),
                output1_scales_scalar=d13,
                output1_scales_gate_scalar=d13,
                output2_scales_scalar=d2,
                gate_up_lora_delta=delta,
                activation_lora_input=activation,
                num_experts=experts,
                top_k=topk,
                intermediate_size=inter,
                local_expert_offset=0,
                local_num_experts=experts,
                routed_scaling_factor=1.0,
                routing_method_type=2,
            )
            torch.cuda.synchronize()
            projection = (
                torch.einsum("nh,nkih->nki", x.float(), r13[ids.long()]) + delta.float()
            )
            gate, up = projection.chunk(2, dim=-1)
            ref_act = torch.nn.functional.silu(gate) * up
            ref = torch.einsum("nki,nkhi,nk->nh", ref_act, r2[ids.long()], weights)

            def error(actual, reference):
                return float(
                    (actual.float() - reference).square().mean().sqrt()
                    / reference.square().mean().sqrt()
                )

            row = {
                "tokens": n,
                "delta": strength,
                "activation_nrmse": error(activation, ref_act),
                "output_nrmse": error(out, ref),
            }
            print(json.dumps(row), flush=True)
            assert torch.isfinite(out).all() and torch.isfinite(activation).all()
            assert row["activation_nrmse"] < 0.2, row
            assert row["output_nrmse"] < 0.25, row
            rows.append(row)
            outputs.append(out.float())
            refs.append(ref)
        change = error(outputs[1] - outputs[0], refs[1] - refs[0])
        assert change < 0.25, change
        rows.append({"tokens": n, "adapter_effect_nrmse": change})
    return {"passed": True, "gpu": torch.cuda.get_device_name(), "checks": rows}


@app.local_entrypoint()
def main(output: str = "nvfp4-lora-validation.json"):
    result = check.remote()
    Path(output).write_text(json.dumps(result, indent=2))
