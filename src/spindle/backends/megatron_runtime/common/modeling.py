from __future__ import annotations

from dataclasses import replace

import torch
from megatron.bridge import AutoBridge
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.transformer.enums import AttnBackend

from .config import EngineModelConfig
from .settings import distributed_settings, optimizer_settings, provider_settings


def model_provider(config: EngineModelConfig):
    dtype = parameter_dtype(config)
    bridge = AutoBridge.from_hf_pretrained(
        config.hf_checkpoint,
        trust_remote_code=True,
    )
    provider = bridge.to_megatron_provider()
    settings = provider_settings(config, dtype)
    if provider.moe_token_dispatcher_type == "allgather":
        settings.setdefault("moe_token_dispatcher_type", "alltoall")
    configured = replace(
        provider,
        **{**settings, "attention_backend": AttnBackend[config.attention_backend]},
    )
    # Bridge attaches weight loading as a hook outside the dataclass fields.
    # Replacing the provider must preserve it before constructing the model.
    if provider.pre_wrap_hook is not None:
        configured.register_pre_wrap_hook(provider.pre_wrap_hook)
    if provider.post_wrap_hook is not None:
        configured.register_post_wrap_hook(provider.post_wrap_hook)
    return bridge, configured, dtype


def parameter_dtype(config: EngineModelConfig):
    return (
        torch.bfloat16
        if config.bf16
        else torch.float16
        if config.fp16
        else torch.float32
    )


def distributed_model(
    provider,
    config: EngineModelConfig,
    *,
    distributed_optimizer: bool,
):
    return provider.provide_distributed_model(
        ddp_config=DistributedDataParallelConfig(
            **distributed_settings(config, distributed_optimizer)
        ),
        bf16=config.bf16,
        fp16=config.fp16,
    )


def optimizer_config(
    config: EngineModelConfig,
    dtype,
    *,
    distributed_optimizer: bool,
):
    return MCoreOptimizerConfig(
        **optimizer_settings(config, dtype, distributed_optimizer)
    )
