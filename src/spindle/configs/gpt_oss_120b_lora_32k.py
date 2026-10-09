from spindle.configs.gpt_oss_20b_lora_64k import Config as Parent


class Config(Parent):
    name = "gpt-oss-120b-lora-32k"
    model = "openai/gpt-oss-120b"
    max_context_length = 32768
    trainer_gpus_per_node = 4
    trainer_max_clients_per_instance = 6
    trainer_cpu = 32
    trainer_memory_mib = 524288
    inference_max_replicas = 4
    inference_memory_mib = 262144
    overrides = {
        # Both GPT-OSS sizes share the architecture; 120B has more layers/experts.
        "miles_cfg.cli_options.num_layers": 36,
        "miles_cfg.cli_options.num_experts": 128,
        "miles_cfg.tensor_model_parallel_size": 4,
        "miles_cfg.expert_model_parallel_size": 4,
        "miles_cfg.max_lora_slots": 6,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 32,
        "sglang_cfg.max_loras_per_batch": 6,
        "sglang_cfg.max_loaded_loras": 16,
        "sglang_cfg.mem_fraction_static": 0.9,
        "inference_target_concurrency": 64,
    }


config = Config()
