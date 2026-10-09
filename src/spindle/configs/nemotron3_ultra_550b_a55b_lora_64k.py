from spindle.configs.nemotron3_nano_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron3-ultra-550b-a55b-lora-64k"
    model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    trainer_gpu = "B300"
    trainer_nodes = 2
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 524288
    trainer_max_clients_per_instance = 4
    # FP8 leaves room for two resident adapters and 128 requests of Mamba state.
    inference_gpu = "B300"
    inference_gpus_per_node = 4
    inference_max_replicas = 2
    inference_startup_timeout_s = 3600
    inference_memory_mib = 1048576
    inference_cpu = 32
    overrides = {
        "miles_cfg.max_lora_slots": 4,
        "trainer_env.TORCHINDUCTOR_CACHE_DIR": "/tmp/spindle-inductor",
        "sglang_cfg.max_loras_per_batch": 2,
        # Cached adapter tensors must not keep the snapshot volume mapped.
        "sglang_cfg.weight_loader_disable_mmap": True,
        "sglang_cfg.model_loader_extra_config": {"num_threads": 2},
        "sglang_cfg.max_loaded_loras": 4,
        "miles_cfg.model_type": "nemotron-3-ultra-550b-a55b",
        "miles_cfg.expert_model_parallel_size": 16,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "sglang_cfg.quantization": "fp8",
        "sglang_cfg.tp_size": 4,
        "sglang_cfg.ep_size": 4,
        "sglang_cfg.mem_fraction_static": 0.9,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_mamba_cache_size": 640,
        "inference_target_concurrency": 64,
    }


config = Config()
