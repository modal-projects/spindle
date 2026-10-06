from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "request-path-before-20261006fit"
    trainer_gpu = "H100"
    trainer_gpus_per_node = 4
    trainer_cpu = 32
    trainer_memory_mib = 262144
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 8
    max_context_length = 32768
    # Exact archived rollouts are replayed, so inference is configured but idle.
    inference_gpus_per_node = 2
    inference_min_replicas = 0
    inference_max_replicas = 8
    session_idle_timeout_s = 3600
    pool_idle_timeout_s = 3600
    overrides = {
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_lora_slots": 8,
        # Fits H100 memory; longest replay sequence is 16,670 tokens.
        "miles_cfg.max_tokens_per_gpu": 20480,
        "platform.frontend": "spindle-request-path-before-20261006fit",
        "platform.modal.environment": "lilo-deploy",
        "platform.modal.region": "us-west",
        "platform.secrets.api": "spindle-swe-staggered-api",
        "platform.secrets.sampler_proxy": "lilo-proxy",
        "platform.storage.assets": "lilo-model-assets",
        "platform.storage.checkpoints": "spindle-request-path-20261006fit-checkpoints",
        "platform.storage.bulletin": "spindle-request-path-20261006fit-bulletin",
        "sglang_cfg.tp_size": 2,
        "sglang_cfg.max_running_requests": 128,
        "sglang_cfg.max_queued_requests": 256,
    }


config = Config()
