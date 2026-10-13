"""CPU-only snapshots for the pinned SGLang scheduler.

Installed as sglang.srt.spindle_routing_observer by the rollout image. The
scheduler owns all inspected state. No tensor contents are read, no prefix
match mutates the engine tree, and a bounded writer handles file I/O separately.
Enable explicitly with SGLANG_SPINDLE_ROUTING_OBSERVATIONS=1.
"""

from array import array
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

import xxhash


ENABLED = os.environ.get("SGLANG_SPINDLE_ROUTING_OBSERVATIONS") == "1"
PATH = Path("/tmp/spindle-routing-work.json")
pending = Queue(maxsize=1)
writer = None
last_snapshot = 0.0
sequence = 0
hashes = {}


def device_prefixes(cache, adapter_names):
    """Hash reusable device boundaries, including hybrid-state requirements.

    No salted, bigram, host-only or unsupported cache state receives credit.
    A snapshot replaces the previous set, so evictions need no event replay.
    """
    kind = type(cache).__name__
    if kind == "UnifiedRadixCache":
        if cache.is_swa_enabled:
            return None
        core = cache.tree_core
        if type(core).__name__ != "UnifiedTreeCore":
            return None
        root = core.root_node
        validators = [
            c.create_match_validator(match_device_only=True)
            for c in cache.components.values()
        ]

        def valid(node):
            return all(check(node) for check in validators)

        def resident(node):
            return not node.evicted

        hybrid = cache.is_mamba_enabled
    elif kind == "MambaRadixCache":
        root = cache.root_node

        def valid(node):
            return node.value is not None and node.mamba_value is not None

        def resident(node):
            return node.value is not None

        hybrid = True
    elif kind == "RadixCache":
        root = cache.root_node

        def valid(node):
            return node.value is not None

        resident = valid
        hybrid = False
    else:
        return None
    result = []
    fresh_hashes = {}
    stack = [(n, None, 0) for n in root.children.values()]
    while stack:
        node, parent_digest, start = stack.pop()
        if not resident(node) or node.key.is_bigram:
            continue
        adapter = adapter_names.get(node.key.extra_key)
        if adapter is None:
            continue
        previous = hashes.get(node.id)
        if previous is not None and previous[0] is node.key and previous[1] == start:
            digest, boundaries = previous[2], previous[3]
        else:
            digest = (
                parent_digest.copy()
                if parent_digest is not None
                else xxhash.xxh3_128((adapter + "\0").encode())
            )
            tokens = array("q", node.key.raw_token_ids())
            offset = 0
            boundaries = []
            for end in range((start // 64 + 1) * 64, start + len(tokens) + 1, 64):
                take = end - start
                digest.update(tokens[offset:take].tobytes())
                boundaries.append((digest.hexdigest(), end))
                offset = take
            digest.update(tokens[offset:].tobytes())
        end = start + len(node.key)
        fresh_hashes[node.id] = (node.key, start, digest, boundaries)
        if valid(node):
            for key, length in boundaries:
                if not hybrid or length == end:
                    result.append((node.last_access_time, key))
        stack.extend((child, digest, end) for child in node.children.values())
    hashes.clear()
    hashes.update(fresh_hashes)
    # Cap transport and frontend memory, dropping the least recently used hints.
    return [key for _, key in sorted(result, reverse=True)[:8192]]


def write_snapshots():
    while True:
        state = pending.get()
        encoded = json.dumps(state, separators=(",", ":"))
        PATH.with_suffix(".tmp").write_text(encoded)
        os.replace(PATH.with_suffix(".tmp"), PATH)


def snapshot(scheduler, running_batch, last_batch):
    global last_snapshot, writer, sequence
    if (
        not ENABLED
        or scheduler.ps.tp_rank != 0
        or scheduler.ps.dp_size != 1
        or scheduler.ps.pp_size != 1
    ):
        return
    now = time.monotonic()
    if now - last_snapshot < 1:
        return
    last_snapshot = now
    if not scheduler.enable_lora:
        return
    manager = scheduler.tp_worker.model_runner.lora_manager
    names = {uid: ref.lora_name for uid, ref in manager.lora_refs.items()}
    prefixes = device_prefixes(scheduler.tree_cache, names)
    if prefixes is None:
        return
    requests = {r.rid: r for r in scheduler.waiting_queue}
    for batch in (running_batch, last_batch):
        if batch is not None:
            requests.update((r.rid, r) for r in batch.reqs)
    if scheduler.chunked_req is not None:
        requests[scheduler.chunked_req.rid] = scheduler.chunked_req
    rows = []
    for req in requests.values():
        if req.finished():
            continue
        decoded = len(req.output_ids)
        rows.append(
            dict(
                rid=req.rid,
                adapter=names.get(req.lora_id, "base"),
                prefill_tokens=0
                if decoded
                else max(0, len(req.origin_input_ids) - len(req.prefix_indices)),
                decode_tokens=max(0, req.sampling_params.max_new_tokens - decoded),
            )
        )
    sequence += 1
    state = dict(
        timestamp=time.time(),
        sequence=sequence,
        incarnation=os.getpid(),
        prefixes=prefixes,
        requests=rows,
        cpu_adapters=[names[uid] for uid in manager.loras if uid in names],
        gpu_adapters=[
            names[uid] for uid in manager.memory_pool.uid_to_buffer_id if uid in names
        ],
        snapshot_seconds=time.monotonic() - now,
        cache_kind=type(scheduler.tree_cache).__name__,
    )
    if writer is None:
        writer = threading.Thread(target=write_snapshots, daemon=True)
        writer.start()
    try:
        pending.put_nowait(state)
    except Full:
        # Never wait for the writer on the scheduling path.
        try:
            pending.get_nowait()
        except Empty:
            pass
        try:
            pending.put_nowait(state)
        except Full:
            pass


def install(sglang_root):
    target = sglang_root / "srt/managers/scheduler.py"
    source = target.read_text()
    old = "        self.process_pending_chunked_abort()\n"
    assert source.count(old) == 1, "pinned SGLang routing hook changed"
    source = source.replace(
        "import logging\n",
        "import logging\nfrom sglang.srt import spindle_routing_observer\n",
        1,
    )
    source = source.replace(
        old,
        "        spindle_routing_observer.snapshot(self, running_batch, last_batch)\n"
        + old,
    )
    target.write_text(source)
    (sglang_root / "srt/spindle_routing_observer.py").write_text(
        Path(__file__).read_text()
    )


if __name__ == "__main__":
    install(Path("/sgl-workspace/sglang/python/sglang"))
