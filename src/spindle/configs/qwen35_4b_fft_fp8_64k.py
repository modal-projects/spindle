from spindle.configs.qwen35_4b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen35-4b-fft-fp8-64k"
    # The trainer still publishes BF16. The sampler requantizes on load.
    overrides = {"sglang_cfg.quantization": "fp8"}


config = Config()
