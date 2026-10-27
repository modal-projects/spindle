from spindle.configs.qwen3_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "shared-qwen3-4b-lora-32k"
    overrides = {
        "platform.frontend": "spindle-shared",
        "platform.secrets.api": "spindle-shared-api",
        "platform.storage.checkpoints": "spindle-shared-checkpoints",
    }


config = Config()
