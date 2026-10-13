"""Benchmark-only counters for the pinned SGLang image; no CUDA synchronization.

Applied to a derived image, never to the production rollout image. The patch
fails closed if the pinned engine or Spindle sidecar changes.
"""

import argparse
from pathlib import Path


METRICS_MODULE = '''"""CPU bookkeeping for the routing benchmark (TP=1, no speculative decode)."""
import json
import os
import time
from collections import Counter

counts = Counter()
seen_requests = set()
residents = []
last_write = 0.0

def flush(force=False):
    global last_write
    now = time.monotonic()
    if not force and now - last_write < 0.5:
        return
    last_write = now
    state = dict(counts, resident_adapters=residents, timestamp=time.time())
    path = "/tmp/spindle-routing-metrics.json"
    with open(path + ".tmp", "w") as out:
        json.dump(state, out)
    os.replace(path + ".tmp", path)

def requests(reqs, resident_ids):
    changed = False
    for req in reqs:
        if req.lora_id is None or req.rid in seen_requests:
            continue
        seen_requests.add(req.rid)
        counts["request_lora_hits" if req.lora_id in resident_ids else "request_lora_misses"] += 1
        changed = True
    flush(force=changed)

def loaded(before, after, refs, seconds):
    global residents
    counts["gpu_loads"] += len((after - before) - {None})
    counts["gpu_evictions"] += len((before - after) - {None})
    if (after - before) - {None}:
        counts["load_host_seconds"] += seconds
    residents = sorted(refs[uid].lora_name for uid in after if uid is not None)
    flush(force=before != after)
'''


def replace(path, old, new):
    source = path.read_text()
    assert source.count(old) == 1, (str(path), old[:100], source.count(old))
    path.write_text(source.replace(old, new))


def instrument(sglang_root, sidecar):
    lora = sglang_root / "srt/lora"
    (lora / "routing_benchmark.py").write_text(METRICS_MODULE)
    manager = lora / "lora_manager.py"
    replace(
        manager,
        "import logging\n",
        "import logging\nimport time\nfrom sglang.srt.lora import routing_benchmark\n",
    )
    replace(
        manager,
        "        self.memory_pool.prepare_lora_batch(\n",
        "        benchmark_before = set(self.memory_pool.uid_to_buffer_id)\n        benchmark_started = time.perf_counter()\n        self.memory_pool.prepare_lora_batch(\n",
    )
    replace(
        manager,
        "        if new_uids:\n",
        "        routing_benchmark.loaded(benchmark_before, set(self.memory_pool.uid_to_buffer_id), self.lora_refs, time.perf_counter() - benchmark_started)\n        if new_uids:\n",
    )
    scheduler = sglang_root / "srt/managers/scheduler.py"
    replace(
        scheduler,
        "import logging\n",
        "import logging\nfrom sglang.srt.lora import routing_benchmark\n",
    )
    replace(
        scheduler,
        '        """Run a batch."""\n',
        '        """Run a batch."""\n        if self.enable_lora and batch.forward_mode.is_extend():\n            routing_benchmark.requests(batch.reqs, self.tp_worker.model_runner.lora_manager.memory_pool.uid_to_buffer_id)\n',
    )
    replace(sidecar, "import contextlib\n", "import contextlib\nimport json\n")
    replace(
        sidecar,
        '    @app.get("/routing-state")\n',
        """    @app.get("/benchmark-state")
    async def benchmark_state():
        path = Path("/tmp/spindle-routing-metrics.json")
        counters = json.loads(path.read_text()) if path.exists() else {}
        response = await app.state.client.get("/v1/loads?include=core")
        response.raise_for_status()
        return {**await routing_state(), "gpu_lora": counters, "engine": response.json()}

    @app.post("/benchmark-flush-cache")
    async def benchmark_flush_cache():
        response = await app.state.client.post("/flush_cache")
        app.state.prefixes.clear()
        return Response(content=response.content, status_code=response.status_code)

    @app.get("/routing-state")
""",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    args = parser.parse_args()
    instrument(args.sglang_root, args.sidecar)
