"""Settings consumed by the TorchTitan multi-adapter trainer."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TorchTitanConfig:
    hf_checkpoint: str
    model_family: str = "qwen3_8"
    model_flavor: str = "27B"
    max_context_length: int = 16384
    max_lora_slots: int = 8
    max_lora_rank: int = 32
    lora_alpha: float = 32.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    max_batch_tokens: int = 16384
    max_batch_rows: int = 8
    loss_chunk_size: int = 128
    activation_checkpointing: bool = True
    compile_layers: bool = False

    def __post_init__(self):
        if self.model_family not in {"qwen3_5", "qwen3_8"}:
            raise ValueError("TorchTitan supports qwen3_5 and qwen3_8 dense models")
        if self.model_flavor not in {"debugmodel", "0.8B", "2B", "4B", "9B", "27B"}:
            raise ValueError("unsupported dense TorchTitan model flavor")
        if self.model_family == "qwen3_8" and self.model_flavor not in {
            "debugmodel",
            "27B",
        }:
            raise ValueError("qwen3_8 supports debugmodel and 27B")
        for name, value in (
            ("max_context_length", self.max_context_length),
            ("max_lora_slots", self.max_lora_slots),
            ("max_lora_rank", self.max_lora_rank),
            ("max_batch_tokens", self.max_batch_tokens),
            ("max_batch_rows", self.max_batch_rows),
            ("loss_chunk_size", self.loss_chunk_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be finite and positive")
        supported = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        if not self.target_modules or set(self.target_modules) - supported:
            raise ValueError(
                "target_modules must name attention or dense MLP projections"
            )
