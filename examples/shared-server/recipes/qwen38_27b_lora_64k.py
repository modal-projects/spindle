from spindle.configs.qwen38_27b_lora_64k import Config as Parent


class Config(Parent):
    name = "shared-qwen38-27b-lora-64k"
    overrides = {
        "platform.frontend": "spindle-shared",
        "platform.secrets.api": "spindle-shared-api",
        "platform.storage.checkpoints": "spindle-shared-checkpoints",
    }


config = Config()
