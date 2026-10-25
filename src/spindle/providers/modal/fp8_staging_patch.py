"""Online FP8 quantization and loader restoration for pinned SGLang."""

PATCH = r"""--- a/python/sglang/srt/layers/quantization/fp8.py
+++ b/python/sglang/srt/layers/quantization/fp8.py
@@ -11,6 +11,10 @@
 import torch.nn.functional as F
 from torch.nn import Module
 from torch.nn.parameter import Parameter
+
+from spindle.inference.fp8_staging import (
+    online_block_quantization, quantize_block_weights, remember_loader_state, restore_loader_state,
+)
 
 from sglang.kernels.ops.quantization.fp8_kernel import (
     fp8_dtype,
@@ -278,6 +282,9 @@
                 weight_block_size = [1, 32]
             elif weight_block_size != [1, 32]:
                 raise ValueError("MXFP8 requires weight_block_size=[1, 32].")
+        if (not is_checkpoint_fp8_serialized and not use_mxfp8
+                and weight_block_size is None and online_block_quantization()):
+            weight_block_size = [128, 128]
         self.weight_block_size = weight_block_size
 
     def get_name(self) -> str:
@@ -651,6 +658,12 @@
         if self.block_quant and not self.use_mxfp8:
             restore_ue8m0_scale_checkpoint_layout(layer.weight_scale_inv)
 
+    def restore_weights_before_cpu_staging(self, layer: Module) -> None:
+        if hasattr(layer, "_online_fp8_loader_state"):
+            restore_loader_state(layer)
+        else:
+            self.restore_weights_before_loading(layer)
+
     def weight_staging_postprocess_device(self, layer: Module) -> str | None:
         if _is_cpu:
             return None
@@ -861,6 +874,10 @@
         layer.input_scale = None
 
     def process_weights_after_loading(self, layer: Module) -> None:
+        if not self.is_checkpoint_fp8_serialized and not self.use_mxfp8:
+            remember_loader_state(layer, ("weight",))
+            if self.block_quant:
+                quantize_block_weights(layer, ("weight",))
         if self.block_quant:
             self.process_weights_after_loading_block_quant(layer)
         else:
@@ -1441,6 +1458,12 @@
             restore_ue8m0_scale_checkpoint_layout(layer.w13_weight_scale_inv)
             restore_ue8m0_scale_checkpoint_layout(layer.w2_weight_scale_inv)
 
+    def restore_weights_before_cpu_staging(self, layer: Module) -> None:
+        if hasattr(layer, "_online_fp8_loader_state"):
+            restore_loader_state(layer)
+        else:
+            self.restore_weights_before_loading(layer)
+
     def weight_staging_postprocess_device(self, layer: Module) -> str | None:
         if _is_cpu or self.is_fp4_expert:
             return None
@@ -2055,6 +2078,10 @@
             align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)
 
     def process_weights_after_loading(self, layer: Module) -> None:
+        if not self.quant_config.is_checkpoint_fp8_serialized and not self.use_mxfp8:
+            remember_loader_state(layer, ("w13_weight", "w2_weight"))
+            if self.block_quant:
+                quantize_block_weights(layer, ("w13_weight", "w2_weight"))
         if _is_hip and _use_hip_int4:
             self.process_weights_hip_int4(layer)
 
"""
