from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen3-8b-lora-32k"
    model = "Qwen/Qwen3-8B"
    max_context_length = 32768
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_cpu = 32
    trainer_memory_mib = 131072
    trainer_max_clients_per_instance = 8
    miles_cfg = {
        "model_type": "qwen3-8B",
        "tensor_model_parallel_size": 2,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "max_tokens_per_gpu": 32768,
        "max_lora_slots": 8,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    inference_gpu = "H100"
    inference_gpus_per_node = 2
    inference_max_replicas = 4
    inference_target_concurrency = 64
    sglang_cfg = {
        "tp_size": 2,
        "mem_fraction_static": 0.8,
        "max_running_requests": 128,
        "max_queued_requests": 32,
        "max_loaded_loras": 64,
        "max_loras_per_batch": 8,
        "schedule_policy": "lpm",
    }


config = Config()
