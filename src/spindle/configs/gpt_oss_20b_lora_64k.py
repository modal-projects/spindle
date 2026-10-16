from spindle.configs.gpt_oss_20b_lora_32k import Config as Parent


class Config(Parent):
    name = "gpt-oss-20b-lora-64k"
    overrides = {
        "max_context_length": 65536,
        "miles_cfg.tensor_model_parallel_size": 2,
        "inference_min_replicas": 8,
        "inference_target_concurrency": 64,
        "sglang_cfg.max_running_requests": 128,
        # Multi-turn prompts are long and outputs short, so prompt chunks would
        # otherwise run as separate steps that pause every in-flight generation.
        "sglang_cfg.enable_mixed_chunk": True,
    }


config = Config()
