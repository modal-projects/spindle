"""SWE-gym 256K recipe: four B300 trainer GPUs and eight B300 samplers."""

from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-instruct-lora-256k"
    max_context_length = 262144
    trainer_gpu = "B300"
    trainer_gpus_per_node = 4
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    trainer_cpu = 32
    trainer_memory_mib = 262144
    inference_gpu = "B300"
    inference_gpus_per_node = 1
    inference_min_replicas = 8
    inference_max_replicas = 8
    inference_target_concurrency = 32
    inference_cpu = 16
    inference_memory_mib = 65536
    inference_startup_timeout_s = 1800
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.context_parallel_size": 4,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.max_lora_slots": 8,
        "miles_cfg.align_sequences_to_parallel_layout": True,
        "miles_cfg.cli_options.cp_comm_type": ["p2p"],
        "trainer_env.TORCHINDUCTOR_CACHE_DIR": "/tmp/swe-inductor",
        "sglang_cfg.mem_fraction_static": 0.85,
        "sglang_cfg.lora_backend": "csgmv",
        "inference_env.SGLANG_CSGMV_SPLIT_K": "1",
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_queued_requests": 128,
        "sglang_cfg.max_loaded_loras": 32,
        "sglang_cfg.max_loras_per_batch": 16,
        "sglang_cfg.chunked_prefill_size": 8192,
    }


config = Config()
