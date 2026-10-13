from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-4b-lora-32k"
    model = "Qwen/Qwen3.5-4B"
    max_context_length = 32768
    # Replicate the small model across four GPUs for batched multi-tenant training.
    trainer_gpus_per_node = 4
    trainer_cpu = 32
    trainer_memory_mib = 131072
    trainer_max_clients_per_instance = 8
    inference_gpu = "H100"
    inference_target_concurrency = 32
    overrides = {
        "miles_cfg.model_type": "qwen3.5-4B",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 8,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.cli_options.log_probs_chunk_size": 4096,
        "miles_cfg.cli_options.balance_data": True,
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.mem_fraction_static": 0.85,
    }


config = Config()
