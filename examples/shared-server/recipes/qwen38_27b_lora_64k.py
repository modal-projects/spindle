from spindle.configs.qwen38_27b_lora_64k import Config as Parent


class Config(Parent):
    name = "prod-qwen38-27b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
    }


config = Config()
