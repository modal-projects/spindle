from spindle.configs.deepseek_v41_flash_lora_32k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash LoRA with 128K context."""

    name = "deepseek-v41-flash-lora-128k"
    max_context_length = 131072
    trainer_gpu = "B300"
    overrides = {
        "miles_cfg.context_parallel_size": 2,
        "miles_cfg.cli_options.allgather_cp": True,
        # Adapters under all-gather context parallelism need one row per micro-batch.
        "miles_cfg.dynamic_batch_size": False,
        "miles_cfg.cli_options.qkv_format": "bshd",
        "sglang_cfg.mem_fraction_static": 0.85,
    }


config = Config()
