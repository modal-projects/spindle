from spindle.configs.gpt_oss_120b_lora_32k import Config as Parent


class Config(Parent):
    name = "gpt-oss-120b-lora-128k"
    max_context_length = 131072
    trainer_gpus_per_node = 8
    overrides = {
        "miles_cfg.context_parallel_size": 2,
        "miles_cfg.expert_model_parallel_size": 8,
        # Packed learnable attention sinks require all-to-all context parallelism.
        "miles_cfg.cli_options.cp_comm_type": ["a2a"],
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.align_sequences_to_parallel_layout": True,
    }


config = Config()
