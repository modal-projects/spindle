from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron35-lightning-30b-a3b-lora-256k"
    max_context_length = 262144
    trainer_gpus_per_node = 8
    overrides = {
        "miles_cfg.expert_model_parallel_size": 8,
        "miles_cfg.context_parallel_size": 4,
        "miles_cfg.align_sequences_to_parallel_layout": True,
    }


config = Config()
