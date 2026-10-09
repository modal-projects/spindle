from spindle.configs.deepseek_v31_lora_32k import Config as Parent


class Config(Parent):
    name = "glm53-lora-256k"
    model = "zai-org/GLM-5.3"
    max_context_length = 262144
    trainer_gpu = "B300"
    trainer_nodes = 2
    trainer_max_clients_per_instance = 4
    # Eight B300s passed mixed 256K/8K sampling without cache retractions.
    inference_gpu = "B300"
    inference_gpus_per_node = 8
    inference_memory_mib = 1048576
    inference_target_concurrency = 64
    overrides = {
        "trainer_env.TORCHINDUCTOR_CACHE_DIR": "/tmp/spindle-inductor",
        "miles_cfg.tensor_model_parallel_size": 4,
        "miles_cfg.expert_model_parallel_size": 16,
        "miles_cfg.max_lora_slots": 4,
        # Reuse the 78-layer dimensions; Bridge reads the model's HF configuration.
        "miles_cfg.model_type": "glm5.2-744B-A40B_lora",
        "miles_cfg.context_parallel_size": 4,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.align_sequences_to_parallel_layout": True,
        "miles_cfg.cli_options.dsa_attention_backend": "tilelang",
        "miles_cfg.cli_options.apply_rope_fusion": False,
        "miles_cfg.cli_options.cp_comm_type": ["allgather"],
        "sglang_cfg.attention_backend": "dsa",
        "sglang_cfg.tp_size": 8,
        "sglang_cfg.ep_size": 8,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 512,
        "sglang_cfg.dsa_prefill_backend": "tilelang",
        "sglang_cfg.dsa_decode_backend": "tilelang",
        "sglang_cfg.moe_runner_backend": "triton",
        "sglang_cfg.kv_cache_dtype": "bfloat16",
        "sglang_cfg.mem_fraction_static": 0.9,
        "sglang_cfg.chunked_prefill_size": 4096,
    }


config = Config()
