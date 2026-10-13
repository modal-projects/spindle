from spindle.configs.qwen35_4b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-2b-lora-16k"
    model = "Qwen/Qwen3.5-2B"
    trainer_gpus_per_node = 1
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.cli_options.num_layers": 24,
        "miles_cfg.cli_options.hidden_size": 2048,
        "miles_cfg.cli_options.ffn_hidden_size": 6144,
        "miles_cfg.cli_options.num_attention_heads": 8,
        "miles_cfg.cli_options.num_query_groups": 2,
    }


config = Config()
