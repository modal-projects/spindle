from spindle.configs.qwen35_2b_lora_32k import Config as Parent


class Config(Parent):
    name = "shared-qwen35-2b-lora-32k"
    overrides = {
        "platform.frontend": "spindle-shared",
        "platform.secrets.api": "spindle-shared-api",
        "platform.storage.checkpoints": "spindle-shared-checkpoints",
    }


config = Config()
