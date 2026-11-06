from spindle.configs.qwen35_397b_a17b_lora_64k import Config as Parent


class Config(Parent):
    name = "prod-qwen35-397b-a17b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
