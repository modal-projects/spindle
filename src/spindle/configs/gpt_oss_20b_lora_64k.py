from spindle.configuration import BaseConfig


class Config(BaseConfig):
    name = "gpt-oss-20b-lora-64k"
    model = "openai/gpt-oss-20b"
    # SGLang LoRA MoE needs unquantized experts; this is the MXFP4 release upcast to BF16.
    model_weights = "lmsys/gpt-oss-20b-bf16"
    max_context_length = 65536
    trainer_gpu = "H200"
    trainer_gpus_per_node = 8
    trainer_max_clients_per_instance = 8
    miles_cfg = {
        "model_type": "gpt-oss-20b",
        "tensor_model_parallel_size": 8,
        "expert_model_parallel_size": 8,
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
        "max_tokens_per_gpu": 65536,
        "max_lora_slots": 8,
        "max_lora_rank": 32,
        "default_lora_alpha": 32,
        "cli_options": {
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
        # Expert-LoRA checkpoint-load workaround; see miles_runtime/expert_lora_compat.py.
        "EXPERT_LORA_COMPAT": "1",
    }
    inference_gpu = "H200"
    inference_gpus_per_node = 4
    # Start the whole pool while the trainer loads and hold it through training
    # steps, so no update's rollouts wait for replicas to cold-start.
    inference_training_min_replicas = 8
    # Agentic RL keeps hundreds of turns in flight (64 groups x 8 rollouts here). A
    # TP4 H200 replica has KV room for far more than 32 of them, and requests over the
    # running + queued limit are rejected with 503 and retried after a backoff.
    inference_target_concurrency = 64
    sglang_cfg = {
        "tp_size": 4,
        "dtype": "bfloat16",
        "mem_fraction_static": 0.7,
        "max_running_requests": 128,
        "max_queued_requests": 128,
        "max_loaded_loras": 64,
        "max_loras_per_batch": 8,
        "schedule_policy": "lpm",
        "moe_runner_backend": "triton",
        # Multi-turn prompts are long and outputs short, so prompt chunks would
        # otherwise run as separate steps that pause every in-flight generation.
        "enable_mixed_chunk": True,
    }


config = Config()
