from spindle.configs.deepseek_v41_flash_lora_8k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash LoRA with 32K context."""

    name = "deepseek-v41-flash-lora-32k"
    max_context_length = 32768
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 4,
        "miles_cfg.cli_options.make_vocab_size_divisible_by": 32,
        "sglang_cfg.speculative_algorithm": "DSPARK",
        "sglang_cfg.speculative_dspark_block_size": 5,
        "sglang_cfg.cuda_graph_max_bs_decode": 64,
    }


config = Config()
