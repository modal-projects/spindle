from spindle.configs.deepseek_v31_lora_32k import Config as Parent


class Config(Parent):
    name = "kimi-k26-lora-32k"
    model = "moonshotai/Kimi-K2.6"
    trainer_gpu = "H200"
    trainer_nodes = 3
    trainer_max_clients_per_instance = 2
    inference_gpu = "B300"
    inference_gpus_per_node = 4
    inference_memory_mib = 1048576
    overrides = {
        # Shared-volume generated Python had null bytes during native compilation.
        "trainer_env.TORCHINDUCTOR_CACHE_DIR": "/tmp/spindle-inductor",
        "miles_cfg.model_type": "kimi-k25",
        "miles_cfg.tensor_model_parallel_size": 4,
        "miles_cfg.expert_model_parallel_size": 24,
        "miles_cfg.max_lora_slots": 2,
        "miles_cfg.max_tokens_per_gpu": 8192,
        "sglang_cfg.quantization": "compressed-tensors",
        "sglang_cfg.tp_size": 4,
        "sglang_cfg.ep_size": 4,
        "sglang_cfg.moe_runner_backend": "marlin",
        # The token-ID API does not need Kimi's custom vision processor or tokenizer.
        "sglang_cfg.enable_multimodal": False,
        "sglang_cfg.skip_tokenizer_init": True,
        "sglang_cfg.max_loaded_loras": 4,
        # Two GPU adapter slots leave more KV space for long rollouts.
        "sglang_cfg.max_loras_per_batch": 2,
        "sglang_cfg.mem_fraction_static": 0.9,
        "sglang_cfg.max_running_requests": 128,
        "inference_target_concurrency": 64,
    }


config = Config()
