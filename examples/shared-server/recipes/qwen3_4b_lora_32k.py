from spindle.configs.qwen3_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-qwen3-4b-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
    }


config = Config()
