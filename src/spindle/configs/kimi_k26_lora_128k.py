from spindle.configs.kimi_k26_lora_32k import Config as Parent


class Config(Parent):
    name = "kimi-k26-lora-128k"
    max_context_length = 131072
    trainer_nodes = 4
    overrides = {
        "miles_cfg.expert_model_parallel_size": 32,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 4,
        "miles_cfg.align_sequences_to_parallel_layout": True,
    }


config = Config()
