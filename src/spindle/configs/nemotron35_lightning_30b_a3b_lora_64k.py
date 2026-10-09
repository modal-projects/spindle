from spindle.configs.nemotron3_nano_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-64k"
    model = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    trainer_gpus_per_node = 2
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_clients_per_instance = 4
    overrides = {
        "miles_cfg.expert_model_parallel_size": 2,
        "miles_cfg.max_lora_slots": 4,
        "sglang_cfg.max_loras_per_batch": 4,
        "sglang_cfg.max_loaded_loras": 32,
    }


config = Config()
