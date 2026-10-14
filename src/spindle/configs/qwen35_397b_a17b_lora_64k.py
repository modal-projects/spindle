from spindle.configs.qwen36_35b_a3b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen35-397b-a17b-lora-64k"
    model = "Qwen/Qwen3.5-397B-A17B"
    max_context_length = 65536
    trainer_gpu = "H200"
    trainer_nodes = 2
    trainer_cpu = 32
    trainer_memory_mib = 524288
    trainer_max_clients_per_instance = 2
    inference_gpu = "B300"
    inference_gpus_per_node = 4
    inference_cpu = 32
    inference_memory_mib = 786432
    inference_max_replicas = 2
    # Large eager weight loads can exceed the default twenty-minute startup window.
    inference_startup_timeout_s = 3600
    overrides = {
        # Keep generated Python modules on local disk after a shared-cache read failure.
        "trainer_env.TORCHINDUCTOR_CACHE_DIR": "/tmp/spindle-inductor",
        # Cached adapter tensors must not keep the snapshot volume mapped.
        "sglang_cfg.weight_loader_disable_mmap": True,
        "sglang_cfg.model_loader_extra_config": {"num_threads": 2},
        "miles_cfg.model_type": "qwen3.5-35B-A3B_lora",
        "miles_cfg.expert_model_parallel_size": 16,
        "miles_cfg.max_lora_slots": 2,
        "miles_cfg.max_tokens_per_gpu": 16384,
        "miles_cfg.cli_options.num_layers": 60,
        "miles_cfg.cli_options.hidden_size": 4096,
        "miles_cfg.cli_options.ffn_hidden_size": 1024,
        "miles_cfg.cli_options.num_attention_heads": 32,
        "miles_cfg.cli_options.num_experts": 512,
        "miles_cfg.cli_options.moe_ffn_hidden_size": 1024,
        "miles_cfg.cli_options.moe_shared_expert_intermediate_size": 1024,
        "miles_cfg.cli_options.moe_router_topk": 10,
        "miles_cfg.cli_options.moe_layer_freq": 1,
        "sglang_cfg.tp_size": 4,
        # Shard expert adapter buffers instead of replicating their outer factors.
        "sglang_cfg.ep_size": 4,
        "sglang_cfg.moe_runner_backend": "triton",
        "sglang_cfg.mem_fraction_static": 0.9,
        "sglang_cfg.max_running_requests": 96,
        "sglang_cfg.max_mamba_cache_size": 480,
        "sglang_cfg.max_loaded_loras": 4,
        "sglang_cfg.max_loras_per_batch": 2,
        "sglang_cfg.lora_strict_loading": True,
        "inference_target_concurrency": 48,
    }


config = Config()
