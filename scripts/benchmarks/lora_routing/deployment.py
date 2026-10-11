"""16 GPUs per arm: eight H100 trainer GPUs plus eight H200 samplers."""

import os
from spindle.configs.qwen38_27b_lora_16k import Config as Parent

MODE = os.environ.get("LORA_ROUTING_MODE", "naive")


class Config(Parent):
    name = f"routing27-{MODE}-1011b"
    max_context_length = 8192
    trainer_gpu = "H100"
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 16
    inference_gpu = "H200"
    inference_gpus_per_node = 1
    inference_min_replicas = 8
    inference_max_replicas = 8
    inference_memory_mib = 65536
    inference_target_concurrency = 64
    inference_lora_routing = MODE
    session_idle_timeout_s = 3600
    pool_idle_timeout_s = 3600
    overrides = {
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.max_lora_slots": 16,
        "miles_cfg.max_tokens_per_gpu": 8192,
        "platform.frontend": f"spindle-routing27-{MODE}-1011b",
        "platform.modal.environment": "lilo-deploy",
        "platform.modal.region": "us-west",
        "platform.secrets.api": "spindle-swe-staggered-api",
        "platform.secrets.sampler_proxy": "lilo-proxy",
        "platform.storage.assets": "lilo-model-assets",
        "platform.storage.checkpoints": "routing27-checkpoints-1011",
        "platform.storage.bulletin": "routing27-bulletin-1011",
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 4096,
        "sglang_cfg.max_loras_per_batch": 8,
    }


config = Config()
