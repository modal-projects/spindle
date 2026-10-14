from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen35-2b-lora-32k"
    model = "Qwen/Qwen3.5-2B"
    # Miles has no 2B preset; keep the shared Qwen3.5 spec and tied embeddings.
    # Linear-attention dimensions are loaded from the checkpoint's text_config.
    overrides = {
        "miles_cfg.cli_options.num_layers": 24,
        "miles_cfg.cli_options.hidden_size": 2048,
        "miles_cfg.cli_options.ffn_hidden_size": 6144,
        "miles_cfg.cli_options.num_attention_heads": 8,
        "miles_cfg.cli_options.num_query_groups": 2,
    }


config = Config()
