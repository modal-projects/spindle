from spindle.configs.deepseek_v31_lora_32k import Config as Parent


class Config(Parent):
    name = "deepseek-v31-lora-32k-b300"
    trainer_gpu = "B300"
    trainer_max_clients_per_instance = 6
    inference_target_concurrency = 96
    overrides = {
        "trainer_env.TORCHINDUCTOR_CACHE_DIR": "/tmp/spindle-inductor",
        "miles_cfg.max_lora_slots": 6,
        "sglang_cfg.max_loras_per_batch": 6,
        "sglang_cfg.max_loaded_loras": 8,
        "sglang_cfg.max_running_requests": 192,
        "sglang_cfg.max_queued_requests": 512,
    }


config = Config()
