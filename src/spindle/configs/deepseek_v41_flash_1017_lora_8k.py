from spindle.configs.deepseek_v41_flash_lora_8k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash-1017 LoRA on attention and the shared expert."""

    name = "deepseek-v41-flash-1017-lora-8k"
    model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    # An exact BF16 export of the 1017 release, in the same layout as the parent's weights.
    model_weights = "EnvLoop/DeepSeek-V4.1-Flash-1017-bf16"
    weights_revision = "04bad68f50984a6e84ed4a7b436d3d75168a97b8"
    overrides = {
        "inference_gpu_overrides.H200.model_weights": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "inference_gpu_overrides.H200.weights_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
    }


config = Config()
