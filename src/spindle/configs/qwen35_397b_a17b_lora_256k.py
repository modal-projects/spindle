from spindle.configs.qwen35_397b_a17b_lora_64k import Config as Parent


class Config(Parent):
    name = "qwen35-397b-a17b-lora-256k"
    max_context_length = 262144
    trainer_gpu = "B300"
    trainer_nodes = 2
    overrides = {
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 8,
        "miles_cfg.expert_model_parallel_size": 16,
        "miles_cfg.align_sequences_to_parallel_layout": True,
    }


config = Config()
