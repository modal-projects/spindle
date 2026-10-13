from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "gpt-oss-120b-lora-32k"
    model = "openai/gpt-oss-120b"
    max_context_length = 32768
    trainer_gpu = "H200"
    trainer_gpus_per_node = 4
    trainer_max_clients_per_instance = 6
    trainer_cpu = 32
    trainer_memory_mib = 524288
    miles_cfg = {
        # Both GPT-OSS sizes share this Miles preset; override the 120B dimensions.
        "model_type": "gpt-oss-20b",
        "tensor_model_parallel_size": 4,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
            "lm_head",
        ],
        "max_tokens_per_gpu": 32768,
        "max_lora_slots": 6,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            "num_layers": 36,
            "num_experts": 128,
            # Packed sink attention requires cuDNN >= 9.18 (Miles v0.1.0 has 9.22).
            "qkv_format": "thd",
            "attention_backend": "fused",
            "moe_permute_fusion": False,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    }
    trainer_env = {
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "EXPERT_LORA_COMPAT": "1",
    }
    inference_gpu = "H200"
    # Marlin serves the original 120B MXFP4 weights on one GPU.
    inference_gpus_per_node = 1
    inference_max_replicas = 4
    inference_memory_mib = 262144
    inference_target_concurrency = 64
    sglang_cfg = {
        "tp_size": 1,
        "dtype": "bfloat16",
        "mem_fraction_static": 0.9,
        "max_running_requests": 128,
        "max_queued_requests": 32,
        "max_loaded_loras": 16,
        "max_loras_per_batch": 6,
        "schedule_policy": "lpm",
        "moe_runner_backend": "marlin",
    }


config = Config()
