"""Remove per-layer host syncs from the multi-LoRA MoE training path.

A host sync blocks the Python thread until the GPU has drained its queue, so the
GPU then idles while the host launches the next kernels. The pinned
Megatron-Bridge and Megatron-LM add several per transformer layer and
micro-batch (twice that with full activation recompute):

* Megatron-Bridge's multi-LoRA layers expand per-adapter values to per-token
  values with ``torch.repeat_interleave`` on device-side counts; without an
  ``output_size``, each call reads the output length back to the host. This
  happens six times per gpt-oss layer.
* The MoE slot routing copies host-side expert counts to the GPU with a
  blocking copy and counts rows with ``torch.bincount``, which reads its
  maximum back to the host.
* Megatron-LM's weighted quick-GeGLU builds a scalar offset tensor from a
  Python float on every call, a blocking host-to-device copy.

The patched functions keep the upstream math and change only how sizes and
constants reach the GPU:

* expansion sizes come from the host-side token counts that
  ``set_tokens_per_adapter_slot`` already caches for every micro-batch;
* the per-token ``alpha / rank`` scaling is skipped when it is exactly 1.0 for
  every adapter present (the default, ``lora_alpha == rank``);
* when a micro-batch holds a single adapter, only that adapter's weights enter
  the computation. Idle slots get no all-zero gradient to accumulate (Miles
  reduces multi-LoRA gradients synchronously after the backward pass, so a slot
  without a gradient in a micro-batch needs none), and when experts are not
  split across ranks the expert layers also skip the per-adapter sort and
  unsort: the dispatcher's expert-major row order is already sorted by adapter;
* host values reach the GPU through pinned, asynchronous copies, and the GeGLU
  offset tensor is created once per value, dtype and device.

Outputs and the gradients accumulated into Megatron's gradient buffers match the
upstream functions bit for bit. Each patch is
installed only when the upstream code is the version it mirrors: Megatron-Bridge
at ``BRIDGE_COMMIT`` and a GeGLU function whose source hashes to
``GEGLU_SOURCE_SHA256``. Set ``SPINDLE_HOST_SYNC_PATCHES=0`` to keep the
upstream code.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os
from functools import wraps
from importlib.metadata import PackageNotFoundError, distribution

import torch
import torch.nn.functional as F

ENV_FLAG = "SPINDLE_HOST_SYNC_PATCHES"
BRIDGE_COMMIT = "8cd3466d14d2337c8492827b3712482c2b3e4866"
GEGLU_SOURCE_SHA256 = "7a23f78fb6827f0e2fe3157c679fa735727373c81f3555f950650930c0824727"
_INSTALLED = "_spindle_host_sync_free"


def install() -> dict[str, bool]:
    """Install every applicable patch; return which ones are active."""
    if os.environ.get(ENV_FLAG, "1") == "0":
        _log("disabled", reason=f"{ENV_FLAG}=0")
        return {"multi_lora": False, "geglu": False}
    return {"multi_lora": install_multi_lora(), "geglu": install_geglu()}


def bridge_commit() -> str | None:
    try:
        direct_url = distribution("megatron-bridge").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def install_multi_lora(layers=None) -> bool:
    """Patch ``megatron.bridge.peft.multi_lora_layers`` (or ``layers``)."""
    if layers is None:
        commit = bridge_commit()
        if commit != BRIDGE_COMMIT:
            _log(
                "skipped",
                patch="multi_lora",
                reason=f"megatron-bridge {commit} != {BRIDGE_COMMIT}",
            )
            return False
        from megatron.bridge.peft import multi_lora_layers as layers
    if getattr(layers, _INSTALLED, False):
        return True

    dense, expert = layers.MultiLoRALinear, layers.MultiLoRAGroupedExpertLinear
    init_slot, clear_slot = dense.init_adapter_slot, dense.clear_adapter_slot

    @wraps(init_slot)
    def init_adapter_slot(self, idx, rank, alpha):
        init_slot(self, idx, rank, alpha)
        scaling = self.alpha_values[idx] / self.rank_values[idx]
        _unit_scaling(self)[idx] = bool(scaling.item() == 1.0)

    @wraps(clear_slot)
    def clear_adapter_slot(self, idx):
        clear_slot(self, idx)
        _unit_scaling(self).pop(idx, None)

    dense.init_adapter_slot = init_adapter_slot
    dense.clear_adapter_slot = clear_adapter_slot
    dense.forward = _dense_forward(layers)
    expert.forward = _expert_forward()
    # install_moe_slot_routing looks this up when the model is built.
    layers._make_slot_routing_hook = _routing_hook_factory(layers)
    setattr(layers, _INSTALLED, True)
    _log("installed", patch="multi_lora", bridge=BRIDGE_COMMIT[:8])
    return True


def install_geglu(fused=None, experts=None) -> bool:
    """Cache the GeGLU offset tensor in ``megatron.core``'s weighted quick-GeGLU."""
    if fused is None:
        try:
            from megatron.core.fusions import fused_bias_geglu as fused
            from megatron.core.transformer.moe import experts
        except ImportError as exc:
            _log("skipped", patch="geglu", reason=type(exc).__name__)
            return False
    original = fused.weighted_bias_quick_geglu_impl
    if getattr(original, _INSTALLED, False):
        return True
    digest = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if digest != GEGLU_SOURCE_SHA256:
        _log("skipped", patch="geglu", reason=f"source sha256 {digest[:12]} differs")
        return False
    patched = _geglu(fused)
    fused.weighted_bias_quick_geglu_impl = patched
    if (
        experts is not None
        and getattr(experts, "weighted_bias_quick_geglu_impl", None) is original
    ):
        experts.weighted_bias_quick_geglu_impl = patched
    _log("installed", patch="geglu")
    return True


def _log(action: str, **fields) -> None:
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"spindle_host_sync_patches action={action} {rendered}", flush=True)


def _unit_scaling(module) -> dict[int, bool]:
    """Slot -> whether ``alpha / rank`` is exactly 1.0 in the layer's dtype."""
    return module.__dict__.setdefault("_spindle_unit_scaling", {})


def _skips_scaling(module, active_slots) -> bool:
    cache = _unit_scaling(module)
    return active_slots is not None and all(
        cache.get(slot, False) for slot in active_slots
    )


def _active_slots(token_splits) -> tuple[int, ...]:
    return tuple(slot for slot, count in enumerate(token_splits) if count > 0)


def _to_device(values, dtype, device) -> torch.Tensor:
    """Host values to a device tensor without blocking the host."""
    host = torch.tensor(values, dtype=dtype)
    if torch.device(device).type != "cuda":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)


def _dense_forward(layers):
    """``MultiLoRALinear.forward`` using host-side sizes; mirrors the pinned upstream."""
    parallel_state = layers.parallel_state

    def forward(self, x, *args, **kwargs):
        linear_output, bias, layernorm_output = self.base_linear_forward(
            x, *args, **kwargs
        )
        if not self._adapter_enabled:
            return linear_output, bias

        tokens_per_adapter = self.tokens_per_adapter
        token_splits = self.tokens_per_adapter_splits
        assert tokens_per_adapter is not None and token_splits is not None
        x = layernorm_output.contiguous()
        if not self.disable_sequence_parallel_comm and not self.input_is_parallel:
            x = layers.gather_from_sequence_parallel_region(x)
        x_flat = x.reshape(-1, x.shape[-1])

        total = self.tokens_per_adapter_total
        if self.replicate_adapter and total is not None and x_flat.shape[0] != total:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if x_flat.shape[0] * tp_size != total:
                raise RuntimeError(
                    f"{self.base_linear_name}: adapter token spans cover {total} tokens but the "
                    f"base linear received {x_flat.shape[0]} rows, which is not the full batch "
                    f"or its 1/{tp_size} sequence-parallel shard."
                )
            start = parallel_state.get_tensor_model_parallel_rank() * x_flat.shape[0]
            token_splits = layers._narrow_token_counts_to_window(
                token_splits, start, x_flat.shape[0]
            )
            tokens_per_adapter = tokens_per_adapter.new_tensor(token_splits)

        active = _active_slots(token_splits)
        single = active[0] if len(active) == 1 else None
        offsets = (
            None
            if single is not None
            else tokens_per_adapter.cumsum(dim=0, dtype=torch.int32)
        )

        if single is not None and not self._external_output_reduce:
            # The GEMM upstream's per-slot path runs on this slot's span, which is
            # all of the input; the other slots' weights stay out of the graph.
            stacked_A = self.adapters[single].linear_in.weight
            stacked_B = self.adapters[single].linear_out.weight

            def project(inputs, weight):
                return F.linear(inputs, weight)

        else:
            stacked_A = torch.stack([a.linear_in.weight for a in self.adapters])
            stacked_B = torch.stack([a.linear_out.weight for a in self.adapters])

            def project(inputs, stacked):
                if single is not None:
                    return F.linear(inputs, stacked[single])
                return layers._dense_multi_lora_mm(
                    inputs, stacked, token_splits=token_splits, offsets=offsets
                )

        mid = project(x_flat, stacked_A)

        if self._external_output_reduce:
            stacked_B = layers.gather_from_sequence_parallel_region(
                stacked_B.movedim(1, 0).contiguous(), tensor_parallel_output_grad=True
            ).movedim(0, 1)
        elif not self.replicate_adapter:
            if self.input_is_parallel:
                mid = layers.reduce_from_tensor_model_parallel_region(mid)
            else:
                mid = layers.gather_from_tensor_model_parallel_region(mid)

        out = project(mid, stacked_B)

        if not _skips_scaling(self, active):
            scaling = self.alpha_values / self.rank_values
            per_token_scaling = torch.repeat_interleave(
                scaling, tokens_per_adapter, output_size=sum(token_splits)
            ).unsqueeze(-1)
            out = out * per_token_scaling

        if self._gather_output and not self._external_output_reduce:
            out = layers.gather_from_tensor_model_parallel_region(out)
        if not self.disable_sequence_parallel_comm and self.input_is_parallel:
            if self.use_a2a:
                out = layers.all2all_hp2sp(out)
            else:
                out = layers.scatter_to_sequence_parallel_region(out)
        return linear_output + out.reshape(linear_output.shape), bias

    return forward


def _expert_forward():
    """``MultiLoRAGroupedExpertLinear.forward`` that honors the routing fast paths."""

    def forward(self, x, *args, **kwargs):
        linear_output, bias, layernorm_output = self.base_linear_forward(
            x, *args, **kwargs
        )
        if not self._adapter_enabled:
            return linear_output, bias

        routing = self.expert_slot_routing
        if routing is None:
            raise RuntimeError(
                f"{self.base_linear_name}: no expert slot routing for this forward. "
                f"install_moe_slot_routing(model) must run after MultiLoRA is applied, and "
                f"set_tokens_per_adapter_slot(model, counts) before every forward."
            )
        x_flat = layernorm_output.reshape(-1, layernorm_output.shape[-1])
        if routing.num_tokens == 0:
            # Keeps every slot in the autograd graph so Megatron's grad buckets complete.
            zero_term = (
                sum(
                    (a.linear_in.weight.sum() + a.linear_out.weight.sum())
                    for a in self.adapters
                )
                * 0.0
            )
            return linear_output + zero_term, bias
        if x_flat.shape[0] != routing.num_tokens:
            raise RuntimeError(
                f"{self.base_linear_name}: base layer received {x_flat.shape[0]} rows but the "
                f"slot routing was built for {routing.num_tokens}."
            )

        if routing.single_slot is not None:
            # One group per local expert of the only slot with tokens.
            grouped_A = self.adapters[routing.single_slot].linear_in.weight
            grouped_B = self.adapters[routing.single_slot].linear_out.weight
        else:
            stacked_A = torch.stack([a.linear_in.weight for a in self.adapters])
            stacked_B = torch.stack([a.linear_out.weight for a in self.adapters])
            num_groups = stacked_A.shape[0] * stacked_A.shape[1]
            grouped_A = stacked_A.reshape(num_groups, *stacked_A.shape[2:])
            grouped_B = stacked_B.reshape(num_groups, *stacked_B.shape[2:])

        identity = routing.sort_idx is None
        x_sorted = x_flat if identity else x_flat.index_select(0, routing.sort_idx)
        mid = torch._grouped_mm(
            x_sorted, grouped_A.transpose(-2, -1), routing.group_offsets
        )
        out = torch._grouped_mm(mid, grouped_B.transpose(-2, -1), routing.group_offsets)

        if not _skips_scaling(self, routing.active_slots):
            scaling = self.alpha_values / self.rank_values
            out = out * torch.repeat_interleave(
                scaling, routing.slot_token_counts, output_size=routing.num_tokens
            ).unsqueeze(-1)
        if not identity:
            out = out.index_select(0, routing.inverse_idx)
        return linear_output + out.reshape(linear_output.shape), bias

    return forward


def _routing_hook_factory(layers):
    """Replacement for ``_make_slot_routing_hook`` that builds routing from host counts."""

    def make_hook(moe_layer, expert_layers):
        def hook(module, args):
            if not any(layer._adapter_enabled for layer in expert_layers):
                return None
            if len(args) < 2:
                raise RuntimeError(
                    f"Expected the experts module to be called as (hidden_states, tokens_per_expert, "
                    f"...); got {len(args)} positional argument(s)."
                )
            hidden_states, tokens_per_expert = args[0], args[1]
            reference = expert_layers[0]
            if (
                reference.tokens_per_adapter is None
                or reference.tokens_per_adapter_splits is None
            ):
                raise RuntimeError(
                    "set_tokens_per_adapter_slot(model, adapter_token_counts) must run before every "
                    "forward when MoE experts carry multi-LoRA adapters."
                )
            routing = build_routing(
                layers,
                moe_layer.token_dispatcher,
                tokens_per_expert,
                reference,
                hidden_states.device,
            )
            for layer in expert_layers:
                layer.expert_slot_routing = routing
            return None

        return hook

    return make_hook


class Routing:
    """``ExpertSlotRouting`` plus the adapters present on this rank.

    ``sort_idx is None`` means the rows are already in (slot, expert) order.
    ``active_slots`` is None when rows may come from other ranks' micro-batches.
    With ``single_slot`` set, every row belongs to that slot and ``group_offsets``
    has one entry per local expert instead of one per (slot, expert).
    """

    __slots__ = (
        "sort_idx",
        "inverse_idx",
        "group_offsets",
        "slot_token_counts",
        "num_tokens",
        "active_slots",
        "single_slot",
    )

    def __init__(
        self,
        *,
        sort_idx,
        inverse_idx,
        group_offsets,
        slot_token_counts,
        num_tokens,
        active_slots,
        single_slot=None,
    ):
        self.single_slot = single_slot
        self.sort_idx = sort_idx
        self.inverse_idx = inverse_idx
        self.group_offsets = group_offsets
        self.slot_token_counts = slot_token_counts
        self.num_tokens = num_tokens
        self.active_slots = active_slots


def build_routing(layers, dispatcher, tokens_per_expert, reference, device) -> Routing:
    """Per-(slot, expert) segmentation of one MoE layer's rows without host syncs.

    Mirrors ``_build_expert_slot_routing`` in the pinned Megatron-Bridge.
    """
    n_adapters, num_local_experts = reference.n_adapters, reference.num_local_experts
    token_splits = reference.tokens_per_adapter_splits
    per_expert_host = None
    if not (isinstance(tokens_per_expert, torch.Tensor) and tokens_per_expert.is_cuda):
        per_expert_host = [
            int(count) for count in torch.as_tensor(tokens_per_expert).tolist()
        ]
    # With no expert parallelism every row comes from this rank's micro-batch.
    local_rows = (
        getattr(dispatcher, "ep_size", 1) == 1
        and getattr(dispatcher, "tp_size", 1) == 1
    )
    active = _active_slots(token_splits) if local_rows else None

    if active is not None and len(active) == 1 and per_expert_host is not None:
        slot = active[0]
        num_tokens = sum(per_expert_host)
        offsets = list(itertools.accumulate(per_expert_host))
        slot_counts = [0] * n_adapters
        slot_counts[slot] = num_tokens
        return Routing(
            sort_idx=None,
            inverse_idx=None,
            group_offsets=_to_device(offsets, torch.int32, device),
            slot_token_counts=_to_device(slot_counts, torch.long, device),
            num_tokens=num_tokens,
            active_slots=active,
            single_slot=slot,
        )

    total = sum(token_splits)
    slot_ids = torch.repeat_interleave(
        torch.arange(n_adapters, device=device, dtype=torch.int32),
        reference.tokens_per_adapter.to(device=device),
        output_size=total,
    )
    local_tokens = int(dispatcher.hidden_shape_before_permute[0])
    if total != local_tokens:
        tp_size = layers.parallel_state.get_tensor_model_parallel_world_size()
        if total != local_tokens * tp_size:
            raise RuntimeError(
                f"Cannot map {total} adapter-slot token ids onto {local_tokens} "
                f"local MoE tokens with tensor_model_parallel_size={tp_size}."
            )
        tp_rank = layers.parallel_state.get_tensor_model_parallel_rank()
        slot_ids = slot_ids.narrow(0, tp_rank * local_tokens, local_tokens)

    slot_ids = layers._co_permute_slot_ids(dispatcher, slot_ids, num_local_experts)
    num_tokens = int(slot_ids.shape[0])
    if per_expert_host is not None:
        if sum(per_expert_host) != num_tokens:
            raise RuntimeError(
                f"Dispatched token count mismatch: tokens_per_expert sums to {sum(per_expert_host)} "
                f"but the co-permuted slot ids cover {num_tokens} tokens."
            )
        per_expert = _to_device(per_expert_host, torch.long, device)
    else:
        per_expert = tokens_per_expert.to(device=device, dtype=torch.long)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_local_experts, device=device, dtype=torch.long),
        per_expert,
        output_size=num_tokens,
    )

    keys = slot_ids.long() * num_local_experts + expert_ids
    sort_idx = torch.argsort(keys, stable=True)
    inverse_idx = torch.empty_like(sort_idx)
    inverse_idx.scatter_(0, sort_idx, torch.arange(num_tokens, device=device))
    counts = torch.zeros(
        n_adapters * num_local_experts, dtype=torch.long, device=device
    )
    counts.scatter_add_(0, keys, torch.ones_like(keys))
    return Routing(
        sort_idx=sort_idx,
        inverse_idx=inverse_idx,
        group_offsets=counts.cumsum(dim=0, dtype=torch.int32),
        slot_token_counts=counts.view(n_adapters, num_local_experts).sum(dim=1),
        num_tokens=num_tokens,
        active_slots=active,
    )


def _geglu(fused):
    """Megatron-LM's ``weighted_bias_quick_geglu_impl`` with a cached offset tensor."""
    offsets: dict[tuple, torch.Tensor] = {}

    def weighted_bias_quick_geglu_impl(
        input, bias, weights, fp8_input_store=False, linear_offset=0.0, clamp_value=None
    ):
        ori_shape = input.shape
        assert len(ori_shape) in [2, 3]
        if clamp_value is not None:
            x_glu, x_linear = input.chunk(2, -1)
            input = torch.cat(
                (
                    x_glu.clamp(min=None, max=clamp_value),
                    x_linear.clamp(min=-clamp_value, max=clamp_value),
                ),
                -1,
            )
        input = input.view(-1, ori_shape[-1])
        key = (float(linear_offset), input.dtype, input.device)
        offset = offsets.get(key)
        if offset is None:
            offset = offsets[key] = torch.tensor(
                linear_offset, dtype=input.dtype, device=input.device
            )
        if bias is not None:
            output = fused.WeightedBiasQuickGeGLUFunction.apply(
                input, bias, weights, fp8_input_store, offset
            )
        else:
            output = fused.WeightedQuickGeGLUFunction.apply(
                input, weights, fp8_input_store, offset
            )
        return (
            output
            if len(ori_shape) == 2
            else output.view(ori_shape[0], ori_shape[1], -1)
        )

    setattr(weighted_bias_quick_geglu_impl, _INSTALLED, True)
    return weighted_bias_quick_geglu_impl
