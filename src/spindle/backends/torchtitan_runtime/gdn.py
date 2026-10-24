"""Use the same GDN kernels for single and multiple training sequences."""

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention.gdn import InnerGDN


class PackedInnerGDN(InnerGDN):
    @dataclass(kw_only=True, slots=True)
    class Config(InnerGDN.Config):
        pass

    @staticmethod
    def sequence_offsets(inputs, offsets):
        if offsets is not None:
            return offsets
        return torch.tensor(
            [0, inputs.shape[0]], dtype=torch.int32, device=inputs.device
        )

    def short_convolution(
        self, inputs, weight, initial_state=None, *, cu_seqlens, routing
    ):
        # Upstream drops offsets for a single document and uses separate BF16
        # conv/SILU ops. Packed documents use a fused kernel, with different
        # rounding. Keep the fused path even when only one client is ready.
        return super().short_convolution(
            inputs,
            weight,
            initial_state,
            cu_seqlens=self.sequence_offsets(inputs, cu_seqlens),
            routing=routing,
        )

    def gdn_core(self, query, key, value, gate, beta, *, cu_seqlens, routing):
        return super().gdn_core(
            query,
            key,
            value,
            gate,
            beta,
            cu_seqlens=self.sequence_offsets(query, cu_seqlens),
            routing=routing,
        )


def configure_packed_attention(model_config):
    for layer in model_config.layers:
        inner = layer.attention.inner_attention
        if isinstance(inner, InnerGDN.Config):
            layer.attention.inner_attention = PackedInnerGDN.Config(
                conv_kernel_size=inner.conv_kernel_size,
                kernel=inner.kernel,
                param_init=inner.param_init,
                sharding_config=inner.sharding_config,
            )
