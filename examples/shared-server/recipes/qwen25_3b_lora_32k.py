from spindle.configs.qwen25_3b_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-qwen25-3b-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
