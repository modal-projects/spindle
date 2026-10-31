from spindle.configs.qwen35_2b_lora_32k import Config as Parent


class Config(Parent):
    name = "runtime-add-smoke-2b"
    max_context_length = 4096
    trainer_gpus_per_node = 1
    trainer_cpu = 8
    trainer_memory_mib = 32768
    trainer_max_clients_per_instance = 1
    inference_gpu = "H100"
    inference_gpus_per_node = 1
    inference_min_replicas = 0
    platform = {
        **Parent.platform,
        "requires_proxy_auth": True,
    }
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_tokens_per_gpu": 4096,
    }


config = Config()
