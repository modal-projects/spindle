from spindle.configs.nemotron3_ultra_550b_a55b_lora_64k import Config as Parent


class Config(Parent):
    name = "nemotron3-ultra-550b-a55b-lora-256k"
    max_context_length = 262144
    overrides = {
        "miles_cfg.context_parallel_size": 8,
        "miles_cfg.align_sequences_to_parallel_layout": True,
    }


config = Config()
