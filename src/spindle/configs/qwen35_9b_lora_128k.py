from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-128k"
    max_context_length = 131_072
    trainer_gpu = "H200"
    trainer_max_clients_per_instance = 8
    inference_min_replicas = 8
    inference_max_replicas = 8
    overrides = {
        "miles_cfg.max_tokens_per_gpu": 131_072,
        "miles_cfg.max_lora_slots": 8,
    }


config = Config()
