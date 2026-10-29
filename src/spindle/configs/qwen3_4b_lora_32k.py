from spindle.configs.qwen3_8b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen3-4b-lora-32k"
    model = "Qwen/Qwen3-4B"
    # The 4B model fits one GPU, so the four trainer GPUs each hold a replica.
    inference_gpus_per_node = 1
    overrides = {
        "miles_cfg.model_type": "qwen3-4B",
        "miles_cfg.tensor_model_parallel_size": 1,
        "sglang_cfg.tp_size": 1,
    }


config = Config()
