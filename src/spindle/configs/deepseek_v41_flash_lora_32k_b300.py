from spindle.configs.deepseek_v41_flash_lora_32k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash LoRA with 32K context on B300, where a 32K row fits at TP2."""

    name = "deepseek-v41-flash-lora-32k-b300"
    trainer_gpu = "B300"
    inference_gpu = "B300"
    overrides = {"miles_cfg.tensor_model_parallel_size": 2}


config = Config()
