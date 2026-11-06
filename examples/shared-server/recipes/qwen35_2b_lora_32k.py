from spindle.configs.qwen35_2b_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-qwen35-2b-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
