"""Edit this recipe to compare GPU layouts; keep config.json fixed."""

from spindle.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "benchmark-27b"
    max_context_length = 32768
    trainer_region = "us"
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    inference_gpu = "H200"
    inference_gpus_per_node = 1
    inference_min_replicas = 8
    inference_max_replicas = 8
    session_idle_timeout_s = 86400
    pool_idle_timeout_s = 86400
    overrides = {
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 8,
        "miles_cfg.max_tokens_per_gpu": 18432,
        "platform.frontend": "spindle-benchmark-27b",
        "platform.modal.region": "us-west",
        "platform.storage.checkpoints": "benchmark-27b-checkpoints",
        "platform.storage.bulletin": "benchmark-27b-bulletin",
        "sglang_cfg.tp_size": 1,
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_queued_requests": 128,
    }


config = Config()
