import modal

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.17"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
# MXFP4 expert-LoRA backport: https://github.com/sgl-project/sglang/pull/42940
SGLANG_BRANCH = "codex/gpt-oss-mxfp4-lora-v0.5.17"
SGLANG_REVISION = "9e565a1d757998d4e9ca7d867a33c5e97cfebbdf"

# Reference-lifetime fix for the pinned SGLang revision. git apply fails the build
# if an upstream update changes this context; keep until the fix is upstream.
SGLANG_LORA_LIFETIME_PATCH = (
    "--- a/python/sglang/srt/managers/tokenizer_manager.py\n"
    "+++ b/python/sglang/srt/managers/tokenizer_manager.py\n"
    "@@ -202,6 +202,14 @@\n"
    " \n"
    " \n"
    " @dataclasses.dataclass\n"
    "+class LoRARequestRef:\n"
    '+    """One registry acquisition shared by a logical request and its children."""\n'
    "+\n"
    "+    lora_id: str\n"
    "+    users: int = 1\n"
    "+\n"
    "+\n"
    "+@dataclasses.dataclass\n"
    " class ReqState:\n"
    '     """Store the state a request."""\n'
    " \n"
    "@@ -215,6 +223,7 @@\n"
    "     abort_requested: bool = False\n"
    "     lifecycle_id: object = dataclasses.field(default_factory=object)\n"
    "     dispatched: bool = False\n"
    "+    lora_ref: Optional[LoRARequestRef] = None\n"
    "     last_completion_tokens: int = 1\n"
    "     ttft_observed: bool = False\n"
    " \n"
    "@@ -1645,14 +1654,8 @@\n"
    "             HTTPStatus.INTERNAL_SERVER_ERROR,\n"
    "             CLIENT_CLOSED_REQUEST,\n"
    "         ):\n"
    "-            # Delete the key to prevent resending abort request to the scheduler and\n"
    "-            # to ensure aborted request state is cleaned up.\n"
    "-            if state.obj.rid in self.rid_to_state:\n"
    "-                self._remove_req_state(state.obj.rid)\n"
    "-\n"
    "-            # Mark ongoing LoRA request as finished.\n"
    "-            if self.enable_lora and state.obj.lora_path:\n"
    "-                await self.lora_registry.release(state.obj.lora_id)\n"
    "+            # Request-state removal owns the LoRA release. This consumer may\n"
    "+            # run after the RID has already been reused by a new request.\n"
    "             if not is_stream:\n"
    "                 raise fastapi.HTTPException(\n"
    '                     status_code=finish_reason["status_code"],\n'
    "@@ -2493,10 +2496,6 @@\n"
    "                     )\n"
    " \n"
    "                 self._remove_req_state(rid)\n"
    "-\n"
    "-                # Mark ongoing LoRA request as finished.\n"
    "-                if self.enable_lora and state.obj.lora_path:\n"
    "-                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))\n"
    " \n"
    "             if out_dict is not None:\n"
    "                 state.out_list.append(out_dict)\n"
    "@@ -3384,13 +3383,35 @@\n"
    '                     f"Failed to implicitly load LoRA adapter {lora_path}: {load_result.error_message}"\n'
    "                 )\n"
    " \n"
    "-        # Look up the LoRA ID from the registry and start tracking ongoing LoRA requests.\n"
    "-        obj.lora_id = await self.lora_registry.acquire(obj.lora_path)\n"
    "-        # Propagate lora_id to any sub-objects already cached by __getitem__.\n"
    '-        for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():\n'
    "-            sub_obj.lora_id = (\n"
    "-                obj.lora_id[i] if isinstance(obj.lora_id, list) else obj.lora_id\n"
    "-            )\n"
    "+        # Acquire once per logical request, not once per expanded n-sample slot.\n"
    "+        # Children share the parent's reference until the final child finishes.\n"
    "+        states = [self.rid_to_state[rid] for rid in self._logical_rids(obj)]\n"
    "+        paths = [state.obj.lora_path for state in states]\n"
    "+\n"
    "+        async def acquire_and_attach():\n"
    "+            ids = await self.lora_registry.acquire(paths)\n"
    "+            obj.lora_id = ids[0] if obj.is_single else ids\n"
    "+            for state, lora_id in zip(states, ids):\n"
    "+                state.obj.lora_id = lora_id\n"
    "+                if lora_id is None:\n"
    "+                    continue\n"
    "+                if self.rid_to_state.get(state.obj.rid) is state:\n"
    "+                    state.lora_ref = LoRARequestRef(lora_id)\n"
    "+                else:\n"
    "+                    # Cancellation removed this state while acquisition waited.\n"
    "+                    self._track_lora_task(self.lora_registry.release(lora_id))\n"
    "+\n"
    "+        # Cancellation cannot interrupt a partially acquired batch or leave an\n"
    "+        # acquisition unattached. Cleanup can remove states while this finishes.\n"
    "+        await asyncio.shield(self._track_lora_task(acquire_and_attach()))\n"
    "+\n"
    "+    def _track_lora_task(self, coroutine):\n"
    "+        # Hold strong references to cleanup tasks after HTTP consumers disappear.\n"
    '+        tasks = self.__dict__.setdefault("_lora_tasks", set())\n'
    "+        task = asyncio.create_task(coroutine)\n"
    "+        tasks.add(task)\n"
    "+        task.add_done_callback(tasks.discard)\n"
    "+        return task\n"
    " \n"
    "     @staticmethod\n"
    "     def _logical_rids(obj) -> List[str]:\n"
    "@@ -3424,6 +3445,10 @@\n"
    "             request,\n"
    "             lifecycle_id=logical_state.lifecycle_id,\n"
    "         )\n"
    "+        child_state = self.rid_to_state[obj.rid]\n"
    "+        child_state.lora_ref = logical_state.lora_ref\n"
    "+        if child_state.lora_ref is not None:\n"
    "+            child_state.lora_ref.users += 1\n"
    "         try:\n"
    "             self._register_child_rid(logical_rid, obj.rid)\n"
    "         except BaseException:\n"
    "@@ -3442,6 +3467,12 @@\n"
    "         ):\n"
    "             return None\n"
    "         self.rid_to_state.pop(rid)\n"
    "+        ref, state.lora_ref = state.lora_ref, None\n"
    "+        if ref is not None:\n"
    "+            assert ref.users > 0\n"
    "+            ref.users -= 1\n"
    "+            if ref.users == 0:\n"
    "+                self._track_lora_task(self.lora_registry.release(ref.lora_id))\n"
    "         logical_rid = self.child_rid_to_logical_rid.pop(rid, None)\n"
    "         if logical_rid is not None:\n"
    "             children = self.logical_rid_to_child_rids.get(logical_rid)\n"
    "@@ -3536,9 +3567,9 @@\n"
    "     ):\n"
    '         """Drop all logical and child state owned by *obj*.\n'
    " \n"
    "-        Safe to call after a partial/failed dispatch: only requests known to have\n"
    "-        reached the scheduler are aborted, all owned state is removed, and a later\n"
    "-        output for a discarded RID is ignored by the scheduler-response path.\n"
    "+        Undispatched states release immediately. Dispatched LoRA states stay\n"
    "+        alive until the scheduler acknowledges their abort or normal completion;\n"
    "+        dropping them earlier either leaks the reference or unloads live weights.\n"
    '         """\n'
    "         if lifecycle_ids is None:\n"
    "             lifecycle_ids = {\n"
    "@@ -3577,9 +3608,14 @@\n"
    '                         "Failed to abort request rid=%s",\n'
    "                         target_rid,\n"
    "                     )\n"
    "-            for child_rid in child_rids:\n"
    "-                self._remove_req_state(child_rid, lifecycle_id)\n"
    "-            self._remove_req_state(logical_rid, lifecycle_id)\n"
    "+            for rid in (*child_rids, logical_rid):\n"
    "+                state = self.rid_to_state.get(rid)\n"
    "+                if state is None or state.lifecycle_id is not lifecycle_id:\n"
    "+                    continue\n"
    "+                if state.dispatched and state.lora_ref is not None:\n"
    "+                    state.abort_requested = True\n"
    "+                else:\n"
    "+                    self._remove_req_state(rid, lifecycle_id)\n"
    " \n"
    "     def _should_dispatch_to_encoder(\n"
    "         self, obj: Union[GenerateReqInput, EmbeddingReqInput]\n"
    "--- a/python/sglang/srt/lora/lora_registry.py\n"
    "+++ b/python/sglang/srt/lora/lora_registry.py\n"
    "@@ -142,27 +142,24 @@\n"
    "             self._registry.move_to_end(name)\n"
    "             return lora_ref.lora_id\n"
    " \n"
    "-        if isinstance(lora_name, str):\n"
    "-            async with self._registry_lock.writer_lock:\n"
    "-                lora_id = _lookup(lora_name)\n"
    "-\n"
    "-            await self._counters[lora_id].increment(notify_all=False)\n"
    "-            return lora_id\n"
    "-        elif isinstance(lora_name, list):\n"
    "-            async with self._registry_lock.writer_lock:\n"
    "-                lora_ids = [_lookup(name) for name in lora_name]\n"
    "-\n"
    "-            # Increment the counters only after all IDs are looked up.\n"
    "-            await asyncio.gather(\n"
    "-                *[\n"
    "-                    self._counters[id].increment(notify_all=False)\n"
    "-                    for id in lora_ids\n"
    "-                    if id is not None\n"
    "-                ]\n"
    "-            )\n"
    "-            return lora_ids\n"
    "-        else:\n"
    "+        if not isinstance(lora_name, (str, list)):\n"
    '             raise TypeError("lora_name must be either a string or a list of strings.")\n'
    "+        names = [lora_name] if isinstance(lora_name, str) else lora_name\n"
    "+        async with self._registry_lock.writer_lock:\n"
    "+            # Lookup and pin are atomic with respect to unregister. Otherwise an\n"
    "+            # eviction can observe zero and delete a counter before its increment.\n"
    "+            ids = [_lookup(name) for name in names]\n"
    "+            acquired = []\n"
    "+            try:\n"
    "+                for lora_id in ids:\n"
    "+                    if lora_id is not None:\n"
    "+                        await self._counters[lora_id].increment(notify_all=False)\n"
    "+                        acquired.append(lora_id)\n"
    "+            except BaseException:\n"
    "+                for lora_id in acquired:\n"
    "+                    await self._counters[lora_id].decrement()\n"
    "+                raise\n"
    "+        return ids[0] if isinstance(lora_name, str) else ids\n"
    " \n"
    "     async def release(self, lora_id: Union[str, List[str]]):\n"
    '         """\n'
)

SGLANG_UNGATED_EXPERT_PATCH = (
    "--- a/python/sglang/srt/lora/lora.py\n"
    "+++ b/python/sglang/srt/lora/lora.py\n"
    "@@ -463,7 +463,11 @@\n"
    "         self, weight_names: List[str], weights: Dict[str, torch.Tensor]\n"
    "     ):\n"
    "         for weight_name in weight_names:\n"
    '-            if "gate_proj" in weight_name:\n'
    '+            if ".up_proj." in weight_name and self._is_non_gated_moe_weight(weight_name):\n'
    "+                # Ungated routed experts have only one input projection.\n"
    '+                merged_name = weight_name.replace(".up_proj.", ".gate_up_proj.")\n'
    "+                weights[merged_name] = weights.pop(weight_name)\n"
    '+            elif "gate_proj" in weight_name:\n'
    '                 up_name = weight_name.replace("gate_proj", "up_proj")\n'
    '                 gate_up_name = weight_name.replace("gate_proj", "gate_up_proj")\n'
    "                 # PEFT can ship up_proj in two forms when there's no real\n"
    "--- a/python/sglang/srt/lora/lora_manager.py\n"
    "+++ b/python/sglang/srt/lora/lora_manager.py\n"
    "@@ -723,6 +723,13 @@\n"
    "                 # Otherwise, infer target_modules from adapter configs.\n"
    "                 self.target_modules.update(adapter_target_modules)\n"
    " \n"
    "+        # Ungated dense/shared experts retain an actual up_proj module, while\n"
    "+        # routed experts use the fused gate_up_proj buffer. Preserve both targets.\n"
    '+        if "gate_up_proj" in self.target_modules and any(\n'
    '+            name.endswith(".up_proj") for name, _ in self.base_model.named_modules()\n'
    "+        ):\n"
    '+            self.target_modules.add("up_proj")\n'
    "+\n"
    "         # Fusion folds wk + weights_proj into wk_weights_proj, so the modules\n"
    "         # LoRA wraps are absent and an indexer-targeted adapter is silently dropped.\n"
    "         indexer_targets = self.target_modules & DSA_INDEXER_LORA_NAMES\n"
)

# Normalize the target list only for embedding tensors that require filtering.
# Repeating it for every expert tensor makes large MoE adapter loads quadratic.
SGLANG_TARGET_FILTER_PATCH = (
    "--- a/python/sglang/srt/lora/lora.py\n"
    "+++ b/python/sglang/srt/lora/lora.py\n"
    "@@ -164,10 +164,6 @@\n"
    "     def _process_weight(self, name: str, loaded_weight: torch.Tensor):\n"
    "         from sglang.srt.lora.utils import get_normalized_target_modules\n"
    " \n"
    "-        normalized_target_modules = get_normalized_target_modules(\n"
    "-            self.config.target_modules\n"
    "-        )\n"
    "-\n"
    '         # Remap PEFT "unembed_tokens" key to "lm_head" so the weight is\n'
    "         # recognized and loaded into the correct buffer.\n"
    '         if "unembed_tokens" in name:\n'
    "@@ -177,6 +173,9 @@\n"
    "         if layer_id is not None:\n"
    "             self.layers[layer_id].weights[name] = loaded_weight.cpu()\n"
    '         elif "embed_tokens" in name or "lm_head" in name:\n'
    "+            normalized_target_modules = get_normalized_target_modules(\n"
    "+                self.config.target_modules\n"
    "+            )\n"
    "             # Check if this module is declared in target_modules before loading.\n"
    '             # When normalized_target_modules is {"all"} (e.g. target_modules was\n'
    '             # "all-linear"), we allow loading since the server-level\n'
)

# Marlin must receive the EP map initialized by the token dispatcher.
SGLANG_MARLIN_EP_PATCH = (
    "--- a/python/sglang/srt/lora/layers.py\n"
    "+++ b/python/sglang/srt/lora/layers.py\n"
    "@@ -1166,6 +1166,12 @@\n"
    " \n"
    "         # Use pre-computed quant info (doesn't change so not sure why we need to pass it in every time)\n"
    "         quant_info = self._quant_info\n"
    "+        if self._lora_runner_backend.is_marlin():\n"
    "+            # The dispatcher initializes the EP map lazily on its first dispatch.\n"
    "+            quant_info.expert_map = base_layer.dispatcher.local_expert_mapping\n"
    "+            quant_info.global_num_experts = (\n"
    "+                base_layer.num_experts if quant_info.expert_map is not None else -1\n"
    "+            )\n"
    " \n"
    "         # ===== TO BE REFACTORED ====\n"
    "         if self._lora_runner_backend.is_experimental_sgl_trtllm():\n"
)

# EP sampler ranks need only their local per-expert CPU adapter tensors.
SGLANG_CPU_EXPERT_RETENTION_PATCH = (
    "--- a/python/sglang/srt/lora/mem_pool.py\n"
    "+++ b/python/sglang/srt/lora/mem_pool.py\n"
    "@@ -334,6 +334,23 @@\n"
    "             return global_eid\n"
    "         local = global_eid - self.moe_ep_rank * self._num_experts_local\n"
    "         return local if 0 <= local < self._num_experts_local else None\n"
    "+\n"
    "+    def retain_local_expert_weights(self, adapter: LoRAAdapter) -> None:\n"
    '+        """Keep only this rank\'s per-expert CPU weights after normalization."""\n'
    "+        if not self.moe_use_local_expert_ids or self.experts_shared_outer_loras:\n"
    "+            return\n"
    "+        for layer in adapter.layers:\n"
    "+            for name in list(layer.weights):\n"
    '+                expert = re.search(r"\\.experts\\.(\\d+)\\.", name)\n'
    "+                if expert and self._global_to_local_expert_id(int(expert.group(1))) is None:\n"
    "+                    del layer.weights[name]\n"
    "+                else:\n"
    "+                    # Eager safetensors can expose slices of a whole-file buffer.\n"
    "+                    # A retained local tensor must not keep that global buffer alive.\n"
    "+                    layer.weights[name] = layer.weights[name].clone()\n"
    "+        for weights in (adapter.embedding_layers, adapter.added_tokens_embeddings):\n"
    "+            for name in weights:\n"
    "+                weights[name] = weights[name].clone()\n"
    " \n"
    "     def _iter_local_expert_weights(\n"
    "         self,\n"
    "--- a/python/sglang/srt/lora/lora_manager.py\n"
    "+++ b/python/sglang/srt/lora/lora_manager.py\n"
    "@@ -784,6 +784,8 @@\n"
    "             base_model=self.base_model,\n"
    "         )\n"
    "         lora_adapter.initialize_weights()\n"
    '+        if hasattr(self, "memory_pool"):\n'
    "+            self.memory_pool.retain_local_expert_weights(lora_adapter)\n"
    " \n"
    "         self.loras[lora_ref.lora_id] = lora_adapter\n"
    " \n"
    "@@ -802,6 +804,8 @@\n"
    "             base_model=self.base_model,\n"
    "         )\n"
    "         lora_adapter.initialize_weights_from_tensors(tensors)\n"
    '+        if hasattr(self, "memory_pool"):\n'
    "+            self.memory_pool.retain_local_expert_weights(lora_adapter)\n"
    "         self.loras[lora_ref.lora_id] = lora_adapter\n"
    " \n"
    "     def load_lora_adapter_from_tensors(\n"
    "@@ -874,6 +878,9 @@\n"
    "             strict_loading=self.lora_strict_loading,\n"
    "             enable_lora_overlap_loading=self.enable_lora_overlap_loading,\n"
    "         )\n"
    "+\n"
    "+        for adapter in self.loras.values():\n"
    "+            self.memory_pool.retain_local_expert_weights(adapter)\n"
    " \n"
    "         # Initializing memory pool with base model\n"
    "         self.fetch_new_loras({None})\n"
)

# FP8 expert weights/scales must match gated versus ungated projections.
SGLANG_UNGATED_FP8_PATCH = (
    "--- a/python/sglang/srt/layers/quantization/fp8.py\n"
    "+++ b/python/sglang/srt/layers/quantization/fp8.py\n"
    "@@ -1155,7 +1155,7 @@\n"
    "         w13_up_dim, w2_up_dim, weight_padded = get_moe_weight_sizes(\n"
    "             intermediate_size_per_partition,\n"
    "             is_aiter_moe=_use_aiter,\n"
    "-            is_concat=True,\n"
    "+            is_concat=layer.moe_runner_config.is_gated,\n"
    "             is_packed=False,\n"
    "         )\n"
    " \n"
    "@@ -1306,7 +1306,8 @@\n"
    "             w13_weight_scale = torch.nn.Parameter(\n"
    "                 scale_init(\n"
    "                     num_experts,\n"
    "-                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),\n"
    "+                    (2 if layer.moe_runner_config.is_gated else 1)\n"
    "+                    * ((intermediate_size_per_partition + block_n - 1) // block_n),\n"
    "                     (hidden_size + block_k - 1) // block_k,\n"
    "                     dtype=scale_dtype,\n"
    "                 ),\n"
    "@@ -1330,10 +1331,15 @@\n"
    '             assert quant_config.activation_scheme == "dynamic"\n'
    " \n"
    "         else:\n"
    "-            # Allocate 2 scales for w1 and w3 respectively.\n"
    "+            # Allocate one scale per projection (gate/up or ungated up).\n"
    "             # They will be combined to a single scale after weight loading.\n"
    "             w13_weight_scale = torch.nn.Parameter(\n"
    "-                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False\n"
    "+                torch.ones(\n"
    "+                    num_experts,\n"
    "+                    2 if layer.moe_runner_config.is_gated else 1,\n"
    "+                    dtype=torch.float32,\n"
    "+                ),\n"
    "+                requires_grad=False,\n"
    "             )\n"
    "             w2_weight_scale = torch.nn.Parameter(\n"
    "                 torch.ones(num_experts, dtype=torch.float32), requires_grad=False\n"
    "@@ -2155,7 +2161,7 @@\n"
    "             max_w13_scales = layer.w13_weight_scale.max(dim=1).values\n"
    "             for expert_id in range(layer.num_local_experts):\n"
    "                 start = 0\n"
    "-                for shard_id in range(2):\n"
    "+                for shard_id in range(layer.w13_weight_scale.shape[1]):\n"
    "                     dq_weight = per_tensor_dequantize(\n"
    "                         layer.w13_weight[expert_id][start : start + shard_size, :],\n"
    "                         layer.w13_weight_scale[expert_id][shard_id],\n"
)

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "rm -rf /tmp/stitch-sglang-overlay"
        f" && git clone --filter=blob:none --single-branch --branch {SGLANG_BRANCH}"
        f" {SGLANG_REPOSITORY} /tmp/stitch-sglang-overlay"
        f" && git -C /tmp/stitch-sglang-overlay checkout --detach {SGLANG_REVISION}",
        "cd /tmp/stitch-sglang-overlay && git apply --check - <<'PATCH'\n"
        + SGLANG_LORA_LIFETIME_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_LORA_LIFETIME_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_UNGATED_EXPERT_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_TARGET_FILTER_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_MARLIN_EP_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_CPU_EXPERT_RETENTION_PATCH
        + "PATCH\n",
        "cd /tmp/stitch-sglang-overlay && git apply - <<'PATCH'\n"
        + SGLANG_UNGATED_FP8_PATCH
        + "PATCH\n",
        "rm -rf /sgl-workspace/sglang/python/sglang"
        " && cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && rm -rf /tmp/stitch-sglang-overlay",
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .pip_install("huggingface-hub")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_MODULES_CACHE": "/tmp/huggingface/modules",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
