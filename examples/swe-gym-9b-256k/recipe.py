"""9B SWE-Gym recipe: four B300 trainer GPUs and eight B300 samplers.

The compute layout matches ``qwen35_9b_instruct_lora_256k``. This subclass
gives the run its own frontend and leaves the trainer and sampler pool up for
a day of idle time, which is what the October 9 supervisor deployed.
"""

from spindle.configs.qwen35_9b_instruct_lora_256k import Config as Parent


class Config(Parent):
    name = "swe-gym-9b-256k"
    session_idle_timeout_s = 86400
    pool_idle_timeout_s = 86400
    overrides = {
        "platform.frontend": "spindle-swe-gym-9b-256k",
        "platform.storage.checkpoints": "swe-gym-9b-256k-checkpoints",
        "platform.storage.bulletin": "swe-gym-9b-256k-bulletin",
    }


config = Config()
