"""One base projection and batched low-rank updates for independent clients."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AdapterBatch:
    # Set once per microbatch; kept unchanged through checkpoint recomputation.
    slots: tuple[str, ...] = ()


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, batch: AdapterBatch, max_rank: int):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.batch = batch
        self.max_rank = max_rank
        self.num_linears = base.weight.shape[0] if base.weight.ndim == 3 else 1
        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()
        self.scales: dict[str, float] = {}

    def add_adapter(
        self, slot: str, rank: int, alpha: float, generator: torch.Generator
    ):
        # FP32 master parameters preserve small Adam updates. Only the operands
        # of the low-rank GEMMs are cast to the activation dtype.
        leading = (self.num_linears,) if self.num_linears > 1 else ()
        a = torch.zeros(
            (*leading, self.max_rank, self.base.in_features),
            device=self.base.weight.device,
            dtype=torch.float32,
        )
        b = torch.zeros(
            (*leading, self.base.out_features, self.max_rank),
            device=self.base.weight.device,
            dtype=torch.float32,
        )
        # Initialize each logical projection independently, including fused MLPs.
        for matrix in a.unbind(0) if self.num_linears > 1 else (a,):
            nn.init.kaiming_uniform_(matrix[:rank], a=math.sqrt(5), generator=generator)
        self.a[slot] = nn.Parameter(a)
        self.b[slot] = nn.Parameter(b)
        self.scales[slot] = alpha / rank

    def remove_adapter(self, slot: str):
        del self.a[slot], self.b[slot], self.scales[slot]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.batch.slots:
            return output
        slots = self.batch.slots
        # TorchTitan flattens equal-length padded rows into one token axis.
        # Select once per sequence, not once per token (which would expand
        # adapter weights by sequence length).
        x = inputs.reshape(len(slots), -1, inputs.shape[-1])
        a = torch.stack([self.a[slot] for slot in slots]).to(inputs.dtype)
        b = torch.stack([self.b[slot] for slot in slots]).to(inputs.dtype)
        scales = inputs.new_tensor([self.scales[slot] for slot in slots])[:, None, None]
        if self.num_linears == 1:
            delta = torch.bmm(torch.bmm(x, a.transpose(1, 2)), b.transpose(1, 2))
            return output + (delta * scales).reshape_as(output)
        updates = [
            torch.bmm(torch.bmm(x, a[:, i].transpose(1, 2)), b[:, i].transpose(1, 2))
            * scales
            for i in range(self.num_linears)
        ]
        return output + torch.stack(updates, dim=-2).reshape_as(output)

    def peft_tensors(self, name, slot, rank):
        if self.num_linears == 1:
            return {
                f"{name}.lora_A.weight": self.a[slot][:rank],
                f"{name}.lora_B.weight": self.b[slot][:, :rank],
            }
        result = {}
        for i, target in enumerate(("gate_proj", "up_proj")):
            prefix = name.removesuffix("gate_up_proj") + target
            result[f"{prefix}.lora_A.weight"] = self.a[slot][i, :rank]
            result[f"{prefix}.lora_B.weight"] = self.b[slot][i, :, :rank]
        return result


def install_adapters(model: nn.Module, target_modules, max_rank: int):
    """Return projections keyed by their exact PEFT/HF names.

    GDN fused projections are deliberately excluded: exporting three separate
    low-rank matrices as a single rank-r PEFT matrix is not equivalent.
    """
    mapping = {
        "attention.wq": "self_attn.q_proj",
        "attention.wk": "self_attn.k_proj",
        "attention.wv": "self_attn.v_proj",
        "attention.wo": "self_attn.o_proj",
        "feed_forward.w1": "mlp.gate_proj",
        "feed_forward.w3": "mlp.up_proj",
        "feed_forward.w13": "mlp.gate_up_proj",
        "feed_forward.w2": "mlp.down_proj",
    }
    model.requires_grad_(False)
    batch = AdapterBatch()
    projections = {}
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if len(parts) != 4 or parts[0] != "layers":
            continue
        hf_suffix = mapping.get(".".join(parts[2:]))
        if hf_suffix == "mlp.gate_up_proj":
            requested = {"gate_proj", "up_proj"} & set(target_modules)
            if not requested:
                continue
            if len(requested) != 2:
                raise ValueError(
                    "TorchTitan fused MLP requires gate_proj and up_proj together"
                )
        elif hf_suffix is None or hf_suffix.split(".")[-1] not in target_modules:
            continue
        if not isinstance(module, nn.Linear) or module.weight.ndim not in (2, 3):
            raise ValueError(f"unsupported TorchTitan projection: {name}")
        adapter = MultiLoRALinear(module, batch, max_rank)
        model.get_submodule(".".join(parts[:-1]))._modules[parts[-1]] = adapter
        projections[f"base_model.model.model.layers.{parts[1]}.{hf_suffix}"] = adapter
    if not projections:
        raise ValueError("no TorchTitan LoRA projections matched the deployment")
    return batch, projections


def selected_logprobs(hidden, weight, targets):
    logits = F.linear(hidden, weight).float()
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(
        logits, dim=-1
    )
