from spindle.configs.gpt_oss_20b_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-gpt-oss-20b-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.requires_proxy_auth": True,
        "platform.modal.environment": "spindle",
    }


config = Config()
