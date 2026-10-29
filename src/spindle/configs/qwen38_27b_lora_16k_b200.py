from spindle.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-16k-b200"
    inference_gpu = "B200"
    overrides = {
        "sglang_cfg.mamba_full_memory_ratio": 0.4,
        "sglang_cfg.attention_backend": "trtllm_mha",
        "sglang_cfg.page_size": 64,
    }


config = Config()
