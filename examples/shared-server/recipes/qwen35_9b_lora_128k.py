from spindle.configs.qwen35_9b_lora_128k import Config as Parent


class Config(Parent):
    name = "prod-qwen35-9b-lora-128k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
