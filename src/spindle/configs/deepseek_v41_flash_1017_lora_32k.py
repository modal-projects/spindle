from spindle.configs.deepseek_v41_flash_1017_lora_8k import Config as Weights
from spindle.configs.deepseek_v41_flash_lora_32k import Config as Parent


class Config(Parent):
    """DeepSeek-V4.1-Flash-1017 LoRA with 32K context."""

    name = "deepseek-v41-flash-1017-lora-32k"
    model = Weights.model
    weights_revision = Weights.weights_revision


config = Config()
