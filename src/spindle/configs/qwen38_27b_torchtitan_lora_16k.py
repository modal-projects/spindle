"""Experimental native TorchTitan multi-LoRA trainer; validate GPU capacity before use."""

from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen38-27b-torchtitan-lora-16k"
    model = "Qwen/Qwen3.8-27B"
    backend = "torchtitan"
    trainer_gpu = "H200"
    trainer_memory_mib = 131072
    trainer_max_clients_per_instance = 8
    torchtitan_cfg = {
        "model_family": "qwen3_8",
        "model_flavor": "27B",
        "max_lora_slots": 8,
        "max_lora_rank": 32,
        "lora_alpha": 32,
        "max_batch_tokens": 16384,
        "max_batch_rows": 8,
        "activation_checkpointing": True,
        "compile_layers": False,
    }
    inference_gpu = "H200"
    sglang_cfg = {"max_loaded_loras": 64, "max_loras_per_batch": 8}


config = Config()
