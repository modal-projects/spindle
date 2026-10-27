from spindle.configs.deepseek_v41_flash_lora_32k import Config as Parent


class Config(Parent):
    name = "shared-deepseek-v41-flash-lora-32k"
    overrides = {
        "platform.frontend": "spindle-shared",
        "platform.secrets.api": "spindle-shared-api",
        "platform.storage.checkpoints": "spindle-shared-checkpoints",
    }


config = Config()
