from spindle.configs.deepseek_v41_flash_lora_32k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash LoRA with 64K context.

    A 64K row fits at TP8 on B200 and B300 but not on H200, so the trainer runs
    on Blackwell only.
    """

    name = "deepseek-v41-flash-lora-64k"
    max_context_length = 65536
    trainer_gpu = ["B300", "B200"]
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 8,
        "miles_cfg.dynamic_batch_size": False,
        "miles_cfg.cli_options.qkv_format": "bshd",
    }


config = Config()
