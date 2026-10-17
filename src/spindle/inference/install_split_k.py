"""Backport upstream SGLang PR41951 onto pinned 79e38a5 (path relocation only)."""

from pathlib import Path
import subprocess

PATCH = r'''diff --git a/python/sglang/kernels/ops/gemm/chunked_sgmv_expand.py b/python/sglang/kernels/ops/gemm/chunked_sgmv_expand.py
index 8dd7430ec5d1..b091659cb0e3 100644
--- a/python/sglang/kernels/ops/gemm/chunked_sgmv_expand.py
+++ b/python/sglang/kernels/ops/gemm/chunked_sgmv_expand.py
@@ -10,7 +10,16 @@
 
 
 @cached_triton_kernel(
-    lambda _, kwargs: (kwargs["NUM_SLICES"], kwargs["BLOCK_M"], kwargs["OUTPUT_DIM"])
+    lambda _, kwargs: (
+        kwargs["NUM_SLICES"],
+        kwargs["BLOCK_M"],
+        kwargs["OUTPUT_DIM"],
+        kwargs["MAX_RANK"],
+        kwargs["BLOCK_N"],
+        kwargs["BLOCK_K"],
+        kwargs["SLICE_OFFSETS"],
+        kwargs["x"].dtype,
+    )
 )
 @triton.jit(do_not_specialize=["num_segs", "output_stride_0", "output_stride_1"])
 def _chunked_lora_expand_kernel(
@@ -39,6 +48,7 @@ def _chunked_lora_expand_kernel(
     BLOCK_M: tl.constexpr,
     BLOCK_N: tl.constexpr,
     BLOCK_K: tl.constexpr,
+    SLICE_OFFSETS: tl.constexpr,
 ):
     """
     Computes a chunked SGMV for LoRA expand operations.
@@ -63,7 +73,7 @@ def _chunked_lora_expand_kernel(
     w_stride_1: tl.constexpr = MAX_RANK
     w_stride_2: tl.constexpr = 1
 
-    pid_s = tl.program_id(axis=2)
+    pid_s = tl.program_id(axis=1) if len(SLICE_OFFSETS) > 0 else tl.program_id(axis=2)
     if pid_s >= num_segs:
         return
 
@@ -82,9 +92,30 @@ def _chunked_lora_expand_kernel(
     if cur_rank == 0:
         return
 
-    slice_id = tl.program_id(axis=1)
-    slice_start = tl.load(slice_offsets + slice_id)
-    slice_end = tl.load(slice_offsets + slice_id + 1)
+    if len(SLICE_OFFSETS) > 0:
+        # Flatten the ragged slice grid. SLICE_OFFSETS is model metadata, known on CPU.
+        flat_tile = tl.program_id(0)
+        slice_id = tl.full((), 0, tl.int32)
+        slice_start = tl.full((), 0, tl.int32)
+        slice_end = tl.full((), 0, tl.int32)
+        pid_n = tl.full((), 0, tl.int32)
+        tile_start = tl.full((), 0, tl.int32)
+        for s in tl.static_range(NUM_SLICES):
+            tiles = (SLICE_OFFSETS[s + 1] - SLICE_OFFSETS[s] + BLOCK_N - 1) // BLOCK_N
+            belongs = (flat_tile >= tile_start) & (flat_tile < tile_start + tiles)
+            slice_id = tl.where(belongs, s, slice_id)
+            slice_start = tl.where(belongs, SLICE_OFFSETS[s], slice_start)
+            slice_end = tl.where(belongs, SLICE_OFFSETS[s + 1], slice_end)
+            pid_n = tl.where(belongs, flat_tile - tile_start, pid_n)
+            tile_start += tiles
+
+    else:
+        slice_id = tl.program_id(axis=1)
+        slice_start = tl.load(slice_offsets + slice_id)
+        slice_end = tl.load(slice_offsets + slice_id + 1)
+        pid_n = tl.program_id(axis=0)
+        if pid_n * BLOCK_N >= slice_end - slice_start:
+            return
 
     scaling = tl.load(scalings + w_index)
     # Adjust K (rank) according to the specific LoRA adapter
@@ -99,7 +130,6 @@ def _chunked_lora_expand_kernel(
     # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
     # The pointers will be advanced as we move in the K direction
     # and accumulate
-    pid_n = tl.program_id(axis=0)
     n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start
     k_offset = tl.arange(0, BLOCK_K)
 
@@ -153,8 +183,8 @@ def chunked_sgmv_lora_expand_forward(
     slice_offsets: torch.Tensor,
     max_slice_size: int,
     base_output: Optional[torch.Tensor],
+    slice_offsets_cpu: Optional[torch.Tensor] = None,
 ) -> torch.Tensor:
-
     # x: (s, slice_num * r)
     # weights: (num_lora, output_dim, r)
     # slice_offsets: boundaries for different slices in the output dimension
@@ -192,12 +222,20 @@ def chunked_sgmv_lora_expand_forward(
         else num_segments
     )
 
+    # Layer metadata is already on the CPU; never copy device offsets here.
+    offsets = tuple(slice_offsets_cpu.tolist()) if slice_offsets_cpu is not None else ()
     grid = (
         triton.cdiv(max_slice_size, BLOCK_N),
         num_slices,  # number of slices in the input/output
         segment_grid,
     )
 
+    if offsets:
+        grid = (
+            sum(triton.cdiv(hi - lo, BLOCK_N) for lo, hi in zip(offsets, offsets[1:])),
+            segment_grid,
+        )
+
     if base_output is None:
         output = torch.zeros((M, OUTPUT_DIM), device=x.device, dtype=x.dtype)
     else:
@@ -232,6 +270,7 @@ def chunked_sgmv_lora_expand_forward(
         BLOCK_M=BLOCK_M,
         BLOCK_N=BLOCK_N,
         BLOCK_K=BLOCK_K,
+        SLICE_OFFSETS=offsets,
         **extra_kwargs,
     )
 
diff --git a/python/sglang/kernels/ops/gemm/chunked_sgmv_shrink.py b/python/sglang/kernels/ops/gemm/chunked_sgmv_shrink.py
index 8502cfc8b04a..61e1976e9621 100644
--- a/python/sglang/kernels/ops/gemm/chunked_sgmv_shrink.py
+++ b/python/sglang/kernels/ops/gemm/chunked_sgmv_shrink.py
@@ -3,6 +3,7 @@
 import triton.language as tl
 
 from sglang.kernels.ops.gemm.lora_tuning_config import get_lora_shrink_config
+from sglang.srt.environ import envs
 from sglang.srt.lora.utils import LoRABatchInfo
 from sglang.srt.utils import cached_triton_kernel
 
@@ -122,12 +123,158 @@ def _chunked_lora_shrink_kernel(
     tl.store(output_ptr, partial_sum, mask=output_mask)
 
 
+@triton.jit(do_not_specialize=["num_segs", "TOTAL_ROWS"])
+def _chunked_lora_shrink_split_k_kernel(
+    # Pointers to matrices
+    x,
+    weights,
+    output,
+    # Information on sequence lengths,ranks and weight id
+    seg_indptr,
+    weight_indices,
+    lora_ranks,
+    permutation,
+    num_segs,
+    # Meta parameters
+    N: tl.constexpr,  # num_slices * r
+    K: tl.constexpr,  # input_dim
+    NUM_SLICES: tl.constexpr,
+    BLOCK_M: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    BLOCK_K: tl.constexpr,
+    SPLIT_K: tl.constexpr,
+    TOTAL_ROWS,
+):
+    """Write FP32 partials with shape [SPLIT_K, TOTAL_ROWS, N]."""
+    x_stride_1: tl.constexpr = 1
+    x_stride_0: tl.constexpr = K
+
+    w_stride_0: tl.constexpr = N * K
+    w_stride_1: tl.constexpr = K
+    w_stride_2: tl.constexpr = 1
+
+    output_stride_0: tl.constexpr = N
+    output_stride_1: tl.constexpr = 1
+
+    pid_s = tl.program_id(1)
+    if pid_s >= num_segs:
+        return
+
+    pid_n = tl.program_id(0)
+
+    seg_start = tl.load(seg_indptr + pid_s)
+    seg_end = tl.load(seg_indptr + pid_s + 1)
+    if seg_start == seg_end:
+        return
+
+    # Current block computes sequence with batch_id,
+    # which starts from row seg_start of x with length seg_len
+    w_index = tl.load(weight_indices + pid_s)
+    rank = tl.load(lora_ranks + w_index)
+
+    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
+    if rank == 0:
+        return
+
+    # Adjust N dim according to the specific LoRA adapter
+    cur_n = tl.minimum(N, rank * NUM_SLICES)
+
+    # Map logical sequence index to physical index
+    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
+    s_offset_physical = tl.load(
+        permutation + s_offset_logical, mask=s_offset_logical < seg_end, other=0
+    )
+
+    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
+    split_id = tl.program_id(2)
+    steps: tl.constexpr = tl.cdiv(K, BLOCK_K * SPLIT_K)
+    k_offset = tl.arange(0, BLOCK_K) + split_id * steps * BLOCK_K
+    x_ptrs = x + (
+        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
+    )
+    w_ptrs = (weights + w_index * w_stride_0) + (
+        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
+    )
+
+    # Iterate to compute the block in output matrix
+    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
+    for k in range(0, steps):
+        x_tile = tl.load(
+            x_ptrs,
+            mask=(s_offset_logical[:, None] < seg_end)
+            & (k_offset[None, :] < K - k * BLOCK_K),
+            other=0.0,
+        )
+        w_tile = tl.load(
+            w_ptrs,
+            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
+            other=0.0,
+        )
+        partial_sum += tl.dot(x_tile, w_tile)
+
+        x_ptrs += BLOCK_K * x_stride_1
+        w_ptrs += BLOCK_K * w_stride_2
+
+    # Store result to output matrix
+    output_ptr = (
+        output
+        + split_id * TOTAL_ROWS * N
+        + (
+            s_offset_physical[:, None] * output_stride_0
+            + n_offset[None, :] * output_stride_1
+        )
+    )
+    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
+    tl.store(output_ptr, partial_sum, mask=output_mask)
+
+
+@triton.jit(do_not_specialize=["TOTAL_ROWS"])
+def _reduce_shrink_split_k(
+    partials,
+    output,
+    seg_indptr,
+    weight_indices,
+    lora_ranks,
+    permutation,
+    N: tl.constexpr,
+    TOTAL_ROWS,
+    NUM_SLICES: tl.constexpr,
+    SPLIT_K: tl.constexpr,
+    BLOCK_M: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+):
+    segment = tl.program_id(1)
+    start = tl.load(seg_indptr + segment)
+    end = tl.load(seg_indptr + segment + 1)
+    if start == end:
+        return
+    adapter = tl.load(weight_indices + segment)
+    rank = tl.load(lora_ranks + adapter)
+    if rank == 0:
+        return
+    logical_rows = start + tl.arange(0, BLOCK_M)
+    rows = tl.load(permutation + logical_rows, mask=logical_rows < end, other=0)
+    columns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
+    mask = (logical_rows[:, None] < end) & (
+        columns[None, :] < tl.minimum(N, rank * NUM_SLICES)
+    )
+    offsets = rows[:, None] * N + columns[None, :]
+    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
+    for split in tl.static_range(SPLIT_K):
+        acc += tl.load(partials + split * TOTAL_ROWS * N + offsets, mask=mask, other=0)
+    tl.store(output + offsets, acc, mask=mask)
+
+
 def chunked_sgmv_lora_shrink_forward(
     x: torch.Tensor,
     weights: torch.Tensor,
     batch_info: LoRABatchInfo,
     num_slices: int,
+    split_k: int | None = None,
 ) -> torch.Tensor:
+    if split_k is not None and split_k < 1:
+        raise ValueError("split_k must be positive")
+
     # x: (s, input_dim)
     # weights: (num_lora, num_slices * r, input_dim)
     # output: (s, num_slices * r)
@@ -173,7 +320,59 @@ def chunked_sgmv_lora_shrink_forward(
     if "num_stages" in config:
         extra_kwargs["num_stages"] = config["num_stages"]
 
+    # Opt-in policy measured on H200 with BF16 rank-32 adapters. Keep other
+    # shapes and deterministic inference on the original accumulation order.
+    if (
+        split_k is None
+        and envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.get()
+        and MAX_RANK == 32
+        and K in (4096, 12288)
+        and x.dtype == torch.bfloat16
+        and (BLOCK_M, BLOCK_N, BLOCK_K) == (16, 16, 256)
+    ):
+        split_k = 16 if K == 12288 else 8
+    if split_k is None or envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
+        split_k = 1
+
     output = torch.empty((S, N), device=x.device, dtype=x.dtype)
+    if split_k > 1:
+        partials = torch.empty((split_k, S, N), device=x.device, dtype=torch.float32)
+        _chunked_lora_shrink_split_k_kernel[(*grid, split_k)](
+            x=x,
+            weights=weights,
+            output=partials,
+            seg_indptr=batch_info.seg_indptr,
+            weight_indices=batch_info.weight_indices,
+            lora_ranks=batch_info.lora_ranks,
+            permutation=batch_info.permutation,
+            num_segs=segment_grid,
+            N=N,
+            K=K,
+            NUM_SLICES=num_slices,
+            BLOCK_M=BLOCK_M,
+            BLOCK_N=BLOCK_N,
+            BLOCK_K=BLOCK_K,
+            SPLIT_K=split_k,
+            TOTAL_ROWS=S,
+            **extra_kwargs,
+        )
+        # Reduce only live rows/rank columns. Unused adapter slots and graph
+        # padding have no initialized partials and must not be read.
+        _reduce_shrink_split_k[grid](
+            partials,
+            output,
+            batch_info.seg_indptr,
+            batch_info.weight_indices,
+            batch_info.lora_ranks,
+            batch_info.permutation,
+            N=N,
+            TOTAL_ROWS=S,
+            NUM_SLICES=num_slices,
+            SPLIT_K=split_k,
+            BLOCK_M=BLOCK_M,
+            BLOCK_N=BLOCK_N,
+        )
+        return output
     _chunked_lora_shrink_kernel[grid](
         x=x,
         weights=weights,
diff --git a/python/sglang/srt/environ.py b/python/sglang/srt/environ.py
index fc3993f33bf2..2767935cd3e6 100644
--- a/python/sglang/srt/environ.py
+++ b/python/sglang/srt/environ.py
@@ -1120,6 +1120,8 @@ class Envs:
     # Master switch for the experimental TRT-LLM LoRA fast path; when OFF (default) every
     # fine-grained opt switch reads False, keeping non-experimental paths byte-identical.
     SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
+    # Small-batch CSGMV split-K changes rounding. Disabled in deterministic mode.
+    SGLANG_ENABLE_LORA_CSGMV_SPLIT_K = EnvBool(False)
     # SGLang needs to know FlashInfer NVFP4 4over6 config to compute the global scale factor.
     FLASHINFER_NVFP4_4OVER6 = EnvBool(False)
     FLASHINFER_NVFP4_4OVER6_E4M3_USE_256 = EnvBool(False)
diff --git a/python/sglang/srt/lora/backend/chunked_backend.py b/python/sglang/srt/lora/backend/chunked_backend.py
index 64533908efe2..c89f6f7d8bba 100644
--- a/python/sglang/srt/lora/backend/chunked_backend.py
+++ b/python/sglang/srt/lora/backend/chunked_backend.py
@@ -108,6 +108,7 @@ def run_lora_b_sgemm(
             weights=weights,
             batch_info=batch_info,
             slice_offsets=output_offset,
+            slice_offsets_cpu=kwargs.get("output_offset_cpu"),
             max_slice_size=max_slice_size,
             base_output=base_output,
         )
@@ -124,7 +125,6 @@ def run_qkv_lora(
         *args,
         **kwargs,
     ) -> torch.Tensor:
-
         # x: (s, input_dim)
         # qkv_lora_a: (num_lora, n_slices * r, input_dim)
         # qkv_lora_b: (num_lora, total_output_dim, r)
@@ -141,6 +141,7 @@ def run_qkv_lora(
             weights=qkv_lora_b,
             batch_info=self.batch_info,
             slice_offsets=output_offset,
+            slice_offsets_cpu=kwargs.get("output_offset_cpu"),
             max_slice_size=max_qkv_out_dim,
             base_output=base_output,
         )
@@ -156,7 +157,6 @@ def run_gate_up_lora(
         *args,
         **kwargs,
     ) -> torch.Tensor:
-
         # x: (s, input_dim)
         # gate_up_lora_a: (num_lora, 2 * r, input_dim)
         # gate_up_lora_b: (num_lora, 2 * output_dim, r)
@@ -175,6 +175,7 @@ def run_gate_up_lora(
             weights=gate_up_lora_b,
             batch_info=self.batch_info,
             slice_offsets=output_offset,
+            slice_offsets_cpu=kwargs.get("output_offset_cpu"),
             max_slice_size=output_dim,
             base_output=base_output,
         )
@@ -393,7 +394,6 @@ def _prepare_lm_head_batch_info(
         weight_indices: list[int],
         batch_info: LoRABatchInfo,
     ) -> Tuple[Optional[LoRABatchInfo], Optional[List[LoRABatchInfo]]]:
-
         # Precompute lm_head_batch_info for pruned lm_head LoRA
         pruned_lens = get_lm_head_pruned_lens(forward_batch)
         lm_head_batch_info = None
'''


B200_HELPER = '\n\n@functools.lru_cache(maxsize=256)\ndef _spindle_b200_split_k(k, tokens, device):\n    # Calibrated separately from RL on B200, BF16 rank32, chunk16.\n    # Large prefills keep the original reduction; they already fill the GPU.\n    if "B200" not in torch.cuda.get_device_name(device):\n        return 1\n    factor = 16 if k == 17408 else 8\n    logging.getLogger(__name__).info(\n        "SPINDLE_SPLIT_K_DISPATCH k=%s tokens=%s factor=%s", k, tokens, factor\n    )\n    return factor\n'


def extend_b200_dispatch(root):
    path = root / "python/sglang/kernels/ops/gemm/chunked_sgmv_shrink.py"
    source = path.read_text()
    source = "import functools\nimport logging\n" + source
    marker = "\ndef chunked_sgmv_lora_shrink_forward("
    assert source.count(marker) == 1
    source = source.replace(marker, B200_HELPER + marker)
    marker = "        and K in (4096, 12288)"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        "        and (K in (4096, 12288) or (K in (5120, 6144, 17408) and S <= 128))",
    )
    marker = "        split_k = 16 if K == 12288 else 8"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        """        if K in (4096, 12288):
            split_k = 16 if K == 12288 else 8
        elif not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
            split_k = _spindle_b200_split_k(K, S, x.device)""",
    )
    path.write_text(source)


if __name__ == "__main__":
    root = Path("/sgl-workspace/sglang")
    for kind in ("shrink", "expand"):
        Path("/opt/pre_pr_" + kind + ".py").write_bytes(
            (
                root / ("python/sglang/kernels/ops/gemm/chunked_sgmv_" + kind + ".py")
            ).read_bytes()
        )
    subprocess.run(
        ["git", "apply", "--check", "-"], input=PATCH, text=True, cwd=root, check=True
    )
    subprocess.run(["git", "apply", "-"], input=PATCH, text=True, cwd=root, check=True)

    extend_b200_dispatch(root)
