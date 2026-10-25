"""Allow online NVFP4 with the pinned SGLang LoRA-capable TRTLLM runner.

The existing loader and kernels share the gated [up, gate] weight layout.
No alternate quantized layout or adapter quantization is needed.
"""

PATCH = r"""--- a/python/sglang/srt/arg_groups/overrides.py
+++ b/python/sglang/srt/arg_groups/overrides.py
@@ -2335,11 +2335,12 @@
         elif moe_runner_backend not in [
             "flashinfer_trtllm",
             "flashinfer_trtllm_routed",
+            "experimental_sgl_trtllm",
         ]:
             raise ValueError(
                 "--quantization nvfp4_online supports only "
                 "--moe-runner-backend flashinfer_trtllm or "
-                "flashinfer_trtllm_routed."
+                "flashinfer_trtllm_routed or experimental_sgl_trtllm."
             )
     # Ascend runs MXFP8 MoE on the Ascend runner; every backend selected below is
     # CUDA/ROCm-only. Forcing one here would not merely pick the wrong runner:
"""
