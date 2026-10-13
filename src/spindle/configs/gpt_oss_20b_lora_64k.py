from spindle.configs.gpt_oss_20b_lora_32k import Config as Parent


class Config(Parent):
    name = "gpt-oss-20b-lora-64k"
    overrides = {
        "max_context_length": 65536,
        "miles_cfg.tensor_model_parallel_size": 2,
    }


config = Config()
