from spindle.configs.deepseek_v41_flash_lora_8k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash LoRA with 32K context."""

    name = "deepseek-v41-flash-lora-32k"
    max_context_length = 32768
    overrides = {
        # TP4 runs out of memory past about 24K tokens on H200.
        "miles_cfg.tensor_model_parallel_size": 8,
        # Megatron pads the vocabulary to a multiple of this times TP; 32 x 8 keeps the
        # checkpoint's 129,280 rows unpadded.
        "miles_cfg.cli_options.make_vocab_size_divisible_by": 32,
    }


config = Config()
