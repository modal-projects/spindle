from spindle.configs.nemotron3_nano_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "prod-nemotron3-nano-30b-a3b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
    }


config = Config()
