from spindle.configs.deepseek_v41_flash_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-deepseek-v41-flash-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
    }


config = Config()
