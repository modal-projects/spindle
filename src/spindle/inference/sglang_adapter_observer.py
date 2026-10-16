"""Publish a bounded, CPU-only adapter catalog from SGLang's owning scheduler.

No tensor/cache inspection or file I/O occurs on the scheduler path. The hook
is inactive unless model placement is enabled, and never blocks on its writer.
"""

import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time


ENABLED = os.environ.get("SGLANG_SPINDLE_MODEL_ROUTING") == "1"
PATH = Path("/tmp/spindle-adapter-catalog.json")
pending = Queue(maxsize=1)
writer = None
last_snapshot = 0.0


def write_snapshots():
    while True:
        state = pending.get()
        PATH.with_suffix(".tmp").write_text(json.dumps(state, separators=(",", ":")))
        os.replace(PATH.with_suffix(".tmp"), PATH)


def snapshot(scheduler, parallel):
    global last_snapshot, writer
    if (
        not ENABLED
        or parallel.tp_rank != 0
        or parallel.dp_size != 1
        or parallel.pp_size != 1
        or not scheduler.enable_lora
    ):
        return
    now = time.monotonic()
    if now - last_snapshot < 1:
        return
    last_snapshot = now
    manager = scheduler.tp_worker.model_runner.lora_manager
    state = dict(
        timestamp=time.time(),
        cpu_adapters=[
            manager.lora_refs[uid].lora_name
            for uid in manager.loras
            if uid in manager.lora_refs
        ],
    )
    if writer is None:
        writer = threading.Thread(target=write_snapshots, daemon=True)
        writer.start()
    try:
        pending.put_nowait(state)
    except Full:
        try:
            pending.get_nowait()
        except Empty:
            pass
        try:
            pending.put_nowait(state)
        except Full:
            pass


def install(root):
    target = root / "srt/managers/scheduler.py"
    source = target.read_text()
    marker = "        self.process_pending_chunked_abort()\n"
    if source.count(marker) != 1 or source.count("import logging\n") != 1:
        raise ValueError("SGLang adapter observation hook changed")
    source = source.replace(
        "import logging\n",
        "import logging\nfrom sglang.srt import spindle_adapter_observer\n",
        1,
    )
    source = source.replace(
        marker,
        "        spindle_adapter_observer.snapshot(self, get_parallel())\n" + marker,
    )
    target.write_text(source)
    (root / "srt/spindle_adapter_observer.py").write_text(Path(__file__).read_text())


if __name__ == "__main__":
    install(Path("/sgl-workspace/sglang/python/sglang"))
