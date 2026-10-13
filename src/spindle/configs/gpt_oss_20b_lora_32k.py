from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "gpt-oss-20b-lora-32k"
    model = "openai/gpt-oss-20b"
    model_weights = "lmsys/gpt-oss-20b-bf16"
    max_context_length = 32768
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_max_clients_per_instance = 8
    miles_cfg = {
        "model_type": "gpt-oss-20b",
        "tensor_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
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
        "max_lora_slots": 8,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
            # Packed sink attention requires cuDNN >= 9.18 (Miles v0.1.0 has 9.22).
            "qkv_format": "thd",
            "attention_backend": "fused",
            "moe_permute_fusion": False,
            "log_probs_chunk_size": 4096,
            # Split each batch across data-parallel ranks by token count rather than
            # round-robin, so no rank waits on a slower one at the gradient sync.
            "balance_data": True,
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
    inference_gpus_per_node = 4
    sglang_cfg = {
        "tp_size": 4,
        "dtype": "bfloat16",
        "mem_fraction_static": 0.7,
        "max_running_requests": 32,
        "max_queued_requests": 8,
        "max_loaded_loras": 64,
        "max_loras_per_batch": 8,
        "schedule_policy": "lpm",
        "moe_runner_backend": "triton",
    }


config = Config()
