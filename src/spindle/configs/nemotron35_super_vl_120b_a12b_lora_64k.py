from spindle.configs.nemotron3_super_120b_a12b_lora_64k import Config as Parent


# Nemotron 3.5 Super VL wraps the Nemotron 3 Super 120B-A12B language model with a
# vision encoder. Training is text-only: LoRA adapters train the language model and
# the sampler serves the vision encoder frozen from the base checkpoint.
class Config(Parent):
    name = "nemotron35-super-vl-120b-a12b-lora-64k"
    model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"


config = Config()
