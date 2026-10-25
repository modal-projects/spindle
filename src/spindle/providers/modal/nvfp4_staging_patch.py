"""Restore isolated NVFP4 loader callbacks for the pinned SGLang runtime."""

PATCH = r'''--- a/python/sglang/srt/layers/quantization/nvfp4_online.py
+++ b/python/sglang/srt/layers/quantization/nvfp4_online.py
@@ -172,6 +172,11 @@
                 "FlashInfer A2A or DeepEP low_latency."
             )
 
+    def restore_weights_before_cpu_staging(self, layer: torch.nn.Module) -> None:
+        from spindle.inference.nvfp4_staging import restore_nvfp4_loaders
+
+        restore_nvfp4_loaders(layer)
+
     @staticmethod
     def _quantize_weight_nvfp4(
         weight: torch.Tensor,
@@ -235,7 +240,10 @@
             fp8_fp4_max = e4m3_max * 6.0
             weight_scale_2 = torch.where(
                 weight_amax > 0,
-                weight_amax / fp8_fp4_max,
+                # CUDA scalar division multiplies by a rounded reciprocal.
+                # Use that arithmetic on CPU too so staged decode scales
+                # reproduce the startup GPU weights byte for byte.
+                weight_amax * (1.0 / fp8_fp4_max),
                 torch.ones_like(weight_amax),
             )
         else:
@@ -332,6 +340,7 @@
         the full checkpoint. FP8 checkpoints stream weight and scale tensors
         separately, so those pairs are staged until both sides have arrived.
         """
+        layer._nvfp4_staging_source_loader = original_weight_loader
         pending_w13_weights = {}
         pending_w13_lock = threading.Lock()
         pending_fp8_weights = {}
'''
