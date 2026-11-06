from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "prod-nemotron35-lightning-30b-a3b-lora-64k"
    overrides = {
        "platform.frontend": "prod",
        "platform.url_label": "prod",
        "platform.modal.environment": "spindle",
        "trainer_max_instances": 4,
    }


config = Config()
