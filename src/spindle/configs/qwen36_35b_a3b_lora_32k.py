from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "qwen36-35b-a3b-lora-32k"
    model = "Qwen/Qwen3.6-35B-A3B"
    max_context_length = 32768
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    miles_cfg = {
        "model_type": "qwen3.6-35B-A3B_lora",
        "tensor_model_parallel_size": 2,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
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
        "max_lora_slots": 6,
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
    trainer_max_clients_per_instance = 6
    inference_gpu = "H200"
    inference_gpus_per_node = 2
    inference_target_concurrency = 64
    sglang_cfg = {
        "tp_size": 2,
        "ep_size": 1,
        "mem_fraction_static": 0.85,
        "max_running_requests": 128,
        "max_queued_requests": 32,
        "max_loaded_loras": 256,
        "max_loras_per_batch": 8,
    }


config = Config()
