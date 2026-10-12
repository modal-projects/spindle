import modal

from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.21"
SGLANG_REPOSITORY = "https://github.com/modal-projects/sglang.git"
SGLANG_BRANCH = "stitch-sglang-v0.5.21"
SGLANG_REVISION = "1d420cc80ae48717e553bac60f56330628a8c075"

# Reference-lifetime fix for the pinned SGLang revision. git apply fails the build
# if an upstream update changes this context; keep until the fix is upstream.
SGLANG_LORA_LIFETIME_PATCH = (
    "--- a/python/sglang/srt/managers/tokenizer_manager.py\n"
    "+++ b/python/sglang/srt/managers/tokenizer_manager.py\n"
    "@@ -232,6 +232,14 @@\n"
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
    "@@ -242,6 +250,7 @@\n"
    " \n"
    "     # For performance metrics\n"
    "     time_stats: APIServerReqTimeStats\n"
    "+    lora_ref: Optional[LoRARequestRef] = None\n"
    "     last_completion_tokens: int = 1\n"
    "     ttft_observed: bool = False\n"
    " \n"
    "@@ -1816,14 +1825,10 @@\n"
    "             HTTPStatus.SERVICE_UNAVAILABLE,\n"
    "             HTTPStatus.INTERNAL_SERVER_ERROR,\n"
    "         ):\n"
    "-            # Delete the key to prevent resending abort request to the scheduler and\n"
    "-            # to ensure aborted request state is cleaned up.\n"
    "-            if state.obj.rid in self.rid_to_state:\n"
    "-                del self.rid_to_state[state.obj.rid]\n"
    "-\n"
    "-            # Mark ongoing LoRA request as finished.\n"
    "-            if self.enable_lora and state.obj.lora_path:\n"
    "-                await self.lora_registry.release(state.obj.lora_id)\n"
    "+            # Request-state removal owns the LoRA release. This consumer may\n"
    "+            # run after the RID has already been reused by a new request.\n"
    "+            if self.rid_to_state.get(state.obj.rid) is state:\n"
    "+                self._pop_req_state(state.obj.rid)\n"
    "             if not is_stream:\n"
    "                 raise fastapi.HTTPException(\n"
    '                     status_code=finish_reason["status_code"],\n'
    "@@ -2034,6 +2039,7 @@\n"
    "                 tokenized_obj.sampling_params.max_new_tokens = 0\n"
    "                 tokenized_obj.stream = False\n"
    "                 self._init_req_state(tmp_obj)\n"
    "+                self._share_lora_ref(objs[i].rid, tmp_obj.rid)\n"
    "                 request_rids.add(tmp_obj.rid)\n"
    "                 await self._send_one_request(tokenized_obj)\n"
    "                 await self._wait_one_response(tmp_obj, request).__anext__()\n"
    "@@ -2051,6 +2057,7 @@\n"
    "                         ]\n"
    "                     tokenized_obj.rid = tmp_obj.regenerate_rid()\n"
    "                     self._init_req_state(tmp_obj)\n"
    "+                    self._share_lora_ref(objs[i].rid, tmp_obj.rid)\n"
    "                     request_rids.add(tmp_obj.rid)\n"
    "                     state = self.rid_to_state[tmp_obj.rid]\n"
    "                     tokenized_obj.time_stats = state.time_stats\n"
    "@@ -2061,7 +2068,7 @@\n"
    "                     rids.append(tmp_obj.rid)\n"
    " \n"
    "                 self.rid_to_state[objs[i].rid].time_stats.set_finished_time()\n"
    "-                del self.rid_to_state[objs[i].rid]\n"
    "+                self._pop_req_state(objs[i].rid)\n"
    " \n"
    "         # Wait for all requests\n"
    '         is_stream = hasattr(obj, "stream") and obj.stream\n'
    "@@ -2639,11 +2646,7 @@\n"
    "                         )\n"
    "                     )\n"
    " \n"
    "-                del self.rid_to_state[rid]\n"
    "-\n"
    "-                # Mark ongoing LoRA request as finished.\n"
    "-                if self.enable_lora and state.obj.lora_path:\n"
    "-                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))\n"
    "+                self._pop_req_state(rid)\n"
    " \n"
    "             if out_dict is not None:\n"
    "                 state.out_list.append(out_dict)\n"
    "@@ -3441,7 +3444,7 @@\n"
    "         }\n"
    "         if state.prompt_token_ids is not None:\n"
    '             out["prompt_token_ids"] = state.prompt_token_ids\n'
    "-        del self.rid_to_state[recv_obj.rid]\n"
    "+        self._pop_req_state(recv_obj.rid)\n"
    " \n"
    "         state.out_list.append(out)\n"
    "         state.event.set()\n"
    "@@ -3589,14 +3592,57 @@\n"
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
    '+        single = not hasattr(obj, "is_single") or obj.is_single\n'
    "+        rids = [obj.rid] if single else list(obj.rid)\n"
    "+        states = [self.rid_to_state[rid] for rid in rids]\n"
    "+        paths = [state.obj.lora_path for state in states]\n"
    "+\n"
    "+        async def acquire_and_attach():\n"
    "+            ids = await self.lora_registry.acquire(paths)\n"
    "+            obj.lora_id = ids[0] if single else ids\n"
    "+            for state, lora_id in zip(states, ids):\n"
    "+                state.obj.lora_id = lora_id\n"
    "+                if lora_id is None:\n"
    "+                    continue\n"
    "+                if self.rid_to_state.get(state.obj.rid) is state:\n"
    "+                    state.lora_ref = LoRARequestRef(lora_id)\n"
    "+                else:\n"
    "+                    # Cleanup removed this state while acquisition waited.\n"
    "+                    self._track_lora_task(self.lora_registry.release(lora_id))\n"
    " \n"
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
    "+\n"
    "+    def _share_lora_ref(self, parent_rid: str, child_rid: str) -> None:\n"
    "+        child = self.rid_to_state[child_rid]\n"
    "+        child.lora_ref = self.rid_to_state[parent_rid].lora_ref\n"
    "+        if child.lora_ref is not None:\n"
    "+            child.lora_ref.users += 1\n"
    "+\n"
    "+    def _pop_req_state(self, rid: str) -> Optional[ReqState]:\n"
    '+        """Remove a request state, releasing its LoRA reference exactly once."""\n'
    "+        state = self.rid_to_state.pop(rid, None)\n"
    "+        if state is None:\n"
    "+            return None\n"
    "+        ref, state.lora_ref = state.lora_ref, None\n"
    "+        if ref is not None:\n"
    "+            assert ref.users > 0\n"
    "+            ref.users -= 1\n"
    "+            if ref.users == 0:\n"
    "+                self._track_lora_task(self.lora_registry.release(ref.lora_id))\n"
    "+        return state\n"
    "+\n"
    "     def _init_req_state(\n"
    "         self,\n"
    "         obj: Union[GenerateReqInput, EmbeddingReqInput],\n"
    "@@ -3659,7 +3705,7 @@\n"
    '                             "Failed to abort request %s during cleanup", rid\n'
    "                         )\n"
    "                 else:\n"
    "-                    del self.rid_to_state[rid]\n"
    "+                    self._pop_req_state(rid)\n"
    "             dispatch_ready = self.encoder_dispatch_ready.pop(rid, None)\n"
    "             if dispatch_ready is not None:\n"
    "                 dispatch_ready.set()\n"
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

# Backport until the pinned SGLang revision includes this upstream fix:
# https://github.com/sgl-project/sglang/pull/43916
SGLANG_UNGATED_EXPERT_PATCH = (
    "--- a/python/sglang/srt/lora/lora.py\n"
    "+++ b/python/sglang/srt/lora/lora.py\n"
    "@@ -463,7 +463,14 @@\n"
    "         self, weight_names: List[str], weights: Dict[str, torch.Tensor]\n"
    "     ):\n"
    "         for weight_name in weight_names:\n"
    '-            if "gate_proj" in weight_name:\n'
    '+            if ".up_proj." in weight_name and self._is_non_gated_moe_weight(weight_name):\n'
    '+                gate_name = weight_name.replace(".up_proj.", ".gate_proj.")\n'
    "+                if weight_name in weights and gate_name not in weights:\n"
    "+                    # Non-gated experts can export their single input projection\n"
    "+                    # as up_proj. The fused MoE buffer still calls it gate_up_proj.\n"
    '+                    merged_name = weight_name.replace(".up_proj.", ".gate_up_proj.")\n'
    "+                    weights[merged_name] = weights.pop(weight_name)\n"
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
    "+        # Normalization maps up_proj to gate_up_proj for fused MoE weights.\n"
    "+        # Keep standalone dense/shared-expert up_proj modules eligible too.\n"
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
        "cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
        " && rm -rf /tmp/stitch-sglang-overlay",
        'python -c "import sglang.srt.mem_cache.rust_tree_core.mem_cache"',
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
