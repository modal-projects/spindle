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
SGLANG_REVISION = "ffbaf2cf907e050015242a60a2eb2955b3dfe46f"

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
        # Overlay in place: the image's compiled Rust extensions live inside the
        # package, and the fork only adds or edits Python sources.
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
