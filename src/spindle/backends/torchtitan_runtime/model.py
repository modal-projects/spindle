"""Build the pinned native TorchTitan decoder and stream HF checkpoint shards."""

import json
from pathlib import Path

from safetensors import safe_open
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config as qwen35_config
from torchtitan.models.qwen3_5.model import Qwen35Attention
from torchtitan.models.qwen3_8 import build_model_config as qwen38_config

from .lora import install_adapters


def build_model(config, device):
    factory = {"qwen3_5": qwen35_config, "qwen3_8": qwen38_config}[config.model_family]
    model_config = factory(
        config.model_flavor,
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        seq_len=config.max_context_length,
        attn_backend="varlen",
    )
    # Spindle's training API supplies text tokens. Skip the unused vision tower.
    model_config.vision_encoder = None
    validate_hf_config(model_config, Path(config.hf_checkpoint))
    with torch.device("meta"):
        model = model_config.build()
    model.to(dtype=torch.bfloat16)
    model.to_empty(device=device)
    model.init_weights(buffer_device=device)
    load_hf_weights(model, model_config, Path(config.hf_checkpoint))
    model._skip_lm_head = True
    batch, projections = install_adapters(
        model, config.target_modules, config.max_lora_rank
    )
    for name, layer in list(model.layers.items()):
        if config.compile_layers:
            layer = torch.compile(layer, dynamic=True)
        if config.activation_checkpointing:
            layer = checkpoint_wrapper(layer, preserve_rng_state=False)
        model.layers[name] = layer
    return model, batch, projections


def validate_hf_config(model_config, path):
    checkpoint = json.loads((path / "config.json").read_text())
    text = checkpoint.get("text_config", checkpoint)
    full = next(
        layer.attention
        for layer in model_config.layers
        if isinstance(layer.attention, Qwen35Attention.Config)
    )
    linear = next(
        layer.attention
        for layer in model_config.layers
        if not isinstance(layer.attention, Qwen35Attention.Config)
    )
    expected = {
        "hidden_size": model_config.dim,
        "vocab_size": model_config.vocab_size,
        "num_hidden_layers": len(model_config.layers),
        "intermediate_size": model_config.layers[0].feed_forward.w13.out_features,
        "num_attention_heads": full.n_heads,
        "num_key_value_heads": full.n_kv_heads,
        "head_dim": full.head_dim,
        "tie_word_embeddings": model_config.enable_weight_tying,
        "linear_key_head_dim": linear.key_head_dim,
        "linear_value_head_dim": linear.value_head_dim,
        "linear_num_key_heads": linear.in_proj_q.out_features // linear.key_head_dim,
        "linear_num_value_heads": linear.in_proj_v.out_features
        // linear.value_head_dim,
        "layer_types": [
            "full_attention"
            if isinstance(layer.attention, Qwen35Attention.Config)
            else "linear_attention"
            for layer in model_config.layers
        ],
    }
    mismatches = {
        key: (text.get(key), value)
        for key, value in expected.items()
        if text.get(key) != value
    }
    rope = text.get("rope_parameters", {})
    if (
        rope.get("rope_theta") != full.rope.theta
        or rope.get("mrope_section") != full.rope.mrope_section
    ):
        mismatches["rope_parameters"] = (rope, "TorchTitan flavor RoPE")
    if mismatches:
        raise ValueError(f"HF config does not match TorchTitan flavor: {mismatches}")


def load_hf_weights(model, model_config, path: Path):
    # This loader owns shard I/O; the adapter only converts tensor names/layouts.
    adapter = model.state_dict_adapter_cls(model_config, None)
    expected = dict(model.named_parameters())
    loaded = set()
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(
            path / "model.safetensors", framework="pt", device="cpu"
        ) as source:
            weight_map = {key: "model.safetensors" for key in source.keys()}
    shards = sorted(set(weight_map.values()))
    with torch.no_grad():
        for shard in shards:
            with safe_open(path / shard, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if key.startswith("model.visual."):
                        continue
                    if key.endswith(".mlp.up_proj.weight"):
                        # Load the pair together even when it spans HF shards.
                        continue
                    normalized = key.replace("model.language_model.", "model.", 1)
                    # TorchTitan's converter expects a complete checkpoint and
                    # infers a tied head when lm_head is absent. Supply the
                    # existing head while converting one tensor at a time.
                    tensors = {
                        "lm_head.weight": model.lm_head.weight,
                        normalized: source.get_tensor(key),
                    }
                    if key.endswith(".mlp.gate_proj.weight"):
                        up_key = key.removesuffix("gate_proj.weight") + "up_proj.weight"
                        with safe_open(
                            path / weight_map[up_key], framework="pt", device="cpu"
                        ) as up_source:
                            tensors[
                                up_key.replace("model.language_model.", "model.", 1)
                            ] = up_source.get_tensor(up_key)
                    converted = adapter.from_hf(tensors)
                    for name, weight in converted.items():
                        if name == "lm_head.weight" and normalized != "lm_head.weight":
                            continue
                        if name not in expected:
                            raise ValueError(f"unexpected checkpoint parameter: {name}")
                        parameter = expected[name]
                        if parameter.shape != weight.shape:
                            raise ValueError(
                                f"checkpoint shape mismatch for {name}: {weight.shape} != {parameter.shape}"
                            )
                        parameter.copy_(weight)
                        loaded.add(name)
    if model_config.enable_weight_tying and "lm_head.weight" not in loaded:
        loaded.add("lm_head.weight")
    missing = set(expected) - loaded
    if missing:
        raise ValueError(f"checkpoint is missing parameters: {sorted(missing)}")
