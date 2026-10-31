# Quantized sampler

`qwen35_4b_fft_fp8_64k` is the Qwen3.5-4B full-parameter recipe with one FP8
sampler. The trainer publishes BF16. Full-parameter pools stage those
publications on CPU (`weight_update_staging="cpu"`) and SGLang requantizes them
when `sglang_cfg.quantization` is set. There is no quantized trainer.

`fp8` and `nvfp4_online` are both SGLang quantization modes and pass through
the same way. A LoRA pool quantizes the frozen base when the server loads it
and keeps publishing BF16 adapters, so it does not stage every update on CPU.

Deploy the recipe, then run [score centering](../score-centering/README.md)
against that server.

```bash
uv run spindle config init qwen35_4b_fft_fp8_64k
```
