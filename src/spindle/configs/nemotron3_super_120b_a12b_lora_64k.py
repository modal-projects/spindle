from spindle.configs.nemotron3_nano_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron3-super-120b-a12b-lora-64k"
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_cpu = 32
    trainer_memory_mib = 524288
    trainer_max_clients_per_instance = 4
    inference_gpus_per_node = 4
    inference_max_replicas = 2
    inference_memory_mib = 262144
    overrides = {
        "miles_cfg.max_lora_slots": 4,
        "sglang_cfg.max_loras_per_batch": 4,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "sglang_cfg.max_loaded_loras": 4,
        # Five state slots per request; the default cache capped this model at 38.
        "sglang_cfg.max_mamba_cache_size": 640,
        "miles_cfg.model_type": "nemotron-3-super-120b-a12b",
        "miles_cfg.expert_model_parallel_size": 8,
        "sglang_cfg.tp_size": 4,
        "sglang_cfg.ep_size": 4,
    }


config = Config()
