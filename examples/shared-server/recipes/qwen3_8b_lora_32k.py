from spindle.configs.qwen3_8b_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-qwen3-8b-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
