from spindle.configs.deepseek_v31_lora_32k import Config as Parent


class Config(Parent):
    name = "prod-deepseek-v31-lora-32k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
