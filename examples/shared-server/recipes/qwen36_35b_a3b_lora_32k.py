from spindle.configs.qwen36_35b_a3b_lora_32k import Config as Parent


class Config(Parent):
    name = "shared-qwen36-35b-a3b-lora-32k"
    overrides = {
        "platform.frontend": "spindle-shared",
        "platform.secrets.api": "spindle-shared-api",
        "platform.storage.checkpoints": "spindle-shared-checkpoints",
    }


config = Config()
