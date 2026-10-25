"""Support-probability capture for the pinned SGLang runtime."""

PATCH = r"""--- a/python/sglang/srt/managers/io_struct.py
+++ b/python/sglang/srt/managers/io_struct.py
@@ -211,6 +211,7 @@
     # If return logprobs, the token ids to return logprob for.
     token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
     # Whether to return output-token sampling support and renormalized logprobs.
+    sampling_logprobs_mode: Union[List[str], str] = "sampled"
     return_sampling_mask: Optional[Union[List[bool], bool]] = None
     # Whether to detokenize tokens in text in the returned logprobs.
     return_text_in_logprobs: bool = False
@@ -646,6 +647,10 @@
             self.return_sampling_mask, False, "return_sampling_mask"
         )
 
+        self.sampling_logprobs_mode = normalize_param(
+            self.sampling_logprobs_mode, "sampled", "sampling_logprobs_mode"
+        )
+
         # Handle token_ids_logprob specially due to its nested structure
         if not self.token_ids_logprob:  # covers both None and []
             self.token_ids_logprob = [None] * num
@@ -778,6 +783,7 @@
             top_logprobs_num=self.top_logprobs_num[i],
             token_ids_logprob=self.token_ids_logprob[i],
             return_sampling_mask=self.return_sampling_mask[i],
+            sampling_logprobs_mode=self.sampling_logprobs_mode[i],
             return_text_in_logprobs=self.return_text_in_logprobs,
             return_flat_raw_top_logprobs=self.return_flat_raw_top_logprobs,
             return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,
@@ -867,6 +873,7 @@
     stream: bool
     # Whether to return sparse output-token support from top-k/top-p/min-p sampling.
     return_sampling_mask: bool = False
+    sampling_logprobs_mode: str = "sampled"
     # Assemble prompt top logprobs as flat arrays scheduler-side (see
     # GenerateReqInput.return_flat_raw_top_logprobs). The b64 flag stays
     # tokenizer-manager-side: the scheduler ships arrays either way.
--- a/python/sglang/srt/managers/tokenizer_manager.py
+++ b/python/sglang/srt/managers/tokenizer_manager.py
@@ -1342,6 +1342,14 @@
         sampling_params.normalize(self.tokenizer)
         sampling_params.verify(self.model_config.vocab_size)
 
+        if isinstance(obj, GenerateReqInput) and obj.sampling_logprobs_mode == "support":
+            if not obj.return_sampling_mask:
+                raise ValueError("support probabilities require return_sampling_mask")
+            if (self.server_args.sampling_backend != "pytorch"
+                    or self.server_args.speculative_algorithm
+                    or self.server_args.disaggregation_mode != "null"):
+                raise ValueError("exact support probabilities require non-speculative, colocated pytorch sampling")
+
         # Build return object
         if isinstance(obj, GenerateReqInput):
             session_params = (
@@ -1366,6 +1374,7 @@
                 top_logprobs_num=obj.top_logprobs_num,
                 token_ids_logprob=obj.token_ids_logprob,
                 return_sampling_mask=obj.return_sampling_mask,
+                sampling_logprobs_mode=obj.sampling_logprobs_mode,
                 return_flat_raw_top_logprobs=obj.return_flat_raw_top_logprobs,
                 stream=obj.stream,
                 rid=obj.rid,
--- a/python/sglang/srt/managers/scheduler.py
+++ b/python/sglang/srt/managers/scheduler.py
@@ -2374,6 +2374,7 @@
                 top_logprobs_num=recv_req.top_logprobs_num,
                 token_ids_logprob=recv_req.token_ids_logprob,
                 return_sampling_mask=recv_req.return_sampling_mask,
+                sampling_logprobs_mode=recv_req.sampling_logprobs_mode,
                 return_flat_raw_top_logprobs=recv_req.return_flat_raw_top_logprobs,
                 stream=recv_req.stream,
                 lora_id=recv_req.lora_id,
--- a/python/sglang/srt/managers/schedule_batch.py
+++ b/python/sglang/srt/managers/schedule_batch.py
@@ -804,6 +804,7 @@
         dllm_config: Optional[DllmConfig] = None,
         token_ids_logprob: List[int] = None,
         return_sampling_mask: bool = False,
+        sampling_logprobs_mode: str = "sampled",
         return_flat_raw_top_logprobs: bool = False,
         stream: bool = False,
         origin_input_ids_unpadded: Optional[array[int]] = None,
@@ -1030,6 +1031,9 @@
         self.temp_scaled_logprobs = False
         self.top_p_normalized_logprobs = False
         self.return_sampling_mask = return_sampling_mask
+        if sampling_logprobs_mode not in {"sampled", "support"}:
+            raise ValueError("sampling_logprobs_mode must be sampled or support")
+        self.sampling_logprobs_mode = sampling_logprobs_mode
         self.return_flat_raw_top_logprobs = return_flat_raw_top_logprobs
 
         # Logprobs (return values)
--- a/python/sglang/srt/sampling/sampling_batch_info.py
+++ b/python/sglang/srt/sampling/sampling_batch_info.py
@@ -74,6 +74,7 @@
     sampling_seed: Optional[torch.Tensor] = None
 
     # Per-request flag for returning sparse sampling support metadata.
+    sampling_logprobs_modes: Optional[List[str]] = None
     return_sampling_masks: Optional[List[bool]] = None
     sampling_mask_max_top_k: int = 0
 
@@ -215,6 +216,7 @@
             device=device,
             logit_bias=logit_bias,
             return_sampling_masks=return_sampling_masks,
+            sampling_logprobs_modes=[r.sampling_logprobs_mode for r in reqs],
             sampling_mask_max_top_k=sampling_mask_max_top_k,
         )
         ret.adjusted_from_schedule_batch(batch, vocab_size)
@@ -324,6 +326,8 @@
                 self.return_sampling_masks[i] for i in keep_indices
             ]
 
+        if self.sampling_logprobs_modes is not None:
+            self.sampling_logprobs_modes = [self.sampling_logprobs_modes[i] for i in keep_indices]
         self.adjusted_filter_batch(keep_indices, keep_indices_device)
 
     def _filter_batch_custom_logit_processor(
@@ -427,6 +431,10 @@
             self.sampling_mask_max_top_k = max(
                 self.sampling_mask_max_top_k, other.sampling_mask_max_top_k
             )
+
+        self.sampling_logprobs_modes = (
+            self.sampling_logprobs_modes or ["sampled"] * self_len
+        ) + (other.sampling_logprobs_modes or ["sampled"] * other_len)
 
         # Note: because the __len()__ operator is defined on the temperatures tensor,
         # please make sure any merge operation with len(self) or len(other) is done before
--- a/python/sglang/srt/layers/sampler.py
+++ b/python/sglang/srt/layers/sampler.py
@@ -377,7 +377,7 @@
         logprobs = [None] * len(return_sampling_masks)
         for i, token in zip(request_indices, tokens):
             masks[i] = [int(token)]
-            logprobs[i] = 0.0
+            logprobs[i] = [0.0] if sampling_info.sampling_logprobs_modes[i] == "support" else 0.0
         logits_output.next_token_sampling_mask_idx = masks
         logits_output.next_token_sampling_logprobs = logprobs
 
@@ -417,6 +417,10 @@
         mask_lengths = effective_keep_mask.sum(dim=-1, dtype=torch.int32)
         effective_lengths = mask_lengths + (~sampled_in_idx).to(torch.int32)
         overflow = (effective_lengths > self.sampling_mask_max_tokens).to(torch.int32)
+        exact_rows = torch.tensor([sampling_info.sampling_logprobs_modes[i] == "support"
+                                   for i in request_indices], device=probs.device)
+        # Never turn a reconstructed support into an action-dependent support.
+        overflow |= (exact_rows & ~(keep_mask & sampled_matches_all).any(dim=-1)).to(torch.int32)
         # Overflow changes scheduler control flow, so all ranks serving the
         # request must agree.
         if dist.is_initialized():
@@ -440,6 +444,9 @@
         flat_ids = probs_idx[flat_rows, flat_cols].to(torch.int32)
         bounded_mask_lengths = torch.where(overflow, 0, mask_lengths)
 
+        support_logprobs_cpu = torch.log(
+            probs_sort[flat_rows, flat_cols].float() / support_mass[flat_rows].float()
+        ).cpu().tolist()
         flat_ids_cpu = flat_ids.cpu().tolist()
         mask_lengths_cpu = bounded_mask_lengths.cpu().tolist()
         sampled_in_idx_cpu = sampled_in_idx.cpu().tolist()
@@ -458,11 +465,14 @@
 
             mask_len = int(mask_lengths_cpu[row])
             row_ids = flat_ids_cpu[cursor : cursor + mask_len]
+            row_logprobs = support_logprobs_cpu[cursor : cursor + mask_len]
             cursor += mask_len
             if not sampled_in_idx_cpu[row]:
                 row_ids.append(int(sampled_tokens_cpu[row]))
+                row_logprobs.append(float(selected_logprobs_cpu[row]))
             masks[i] = row_ids
-            logprobs[i] = float(selected_logprobs_cpu[row])
+            logprobs[i] = (row_logprobs if sampling_info.sampling_logprobs_modes[i] == "support"
+                           else float(selected_logprobs_cpu[row]))
 
         logits_output.next_token_sampling_mask_idx = masks
         logits_output.next_token_sampling_logprobs = logprobs
"""
