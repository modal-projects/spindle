import ast
from queue import Queue
from types import SimpleNamespace as NS

import pytest

from spindle.inference import sglang_adapter_observer as observer


def test_snapshot_is_cpu_catalog_only_and_replaces_pending_observation(monkeypatch):
    queue = Queue(maxsize=1)
    queue.put({"old": True})
    monkeypatch.setattr(observer, "pending", queue)
    monkeypatch.setattr(observer, "writer", object())
    monkeypatch.setattr(observer, "ENABLED", True)
    monkeypatch.setattr(observer, "last_snapshot", 0)
    manager = NS(loras={"uid": object()}, lora_refs={"uid": NS(lora_name="model/v1")})
    scheduler = NS(
        enable_lora=True, tp_worker=NS(model_runner=NS(lora_manager=manager))
    )
    parallel = NS(tp_rank=0, dp_size=1, pp_size=1)
    observer.snapshot(scheduler, parallel)
    state = queue.get_nowait()
    assert state["cpu_adapters"] == ["model/v1"]
    assert state["timestamp"] > 0
    observer.snapshot(scheduler, parallel)
    assert queue.empty()
    monkeypatch.setattr(observer, "last_snapshot", 0)
    parallel.tp_rank = 1
    observer.snapshot(scheduler, parallel)
    assert queue.empty()


def test_installer_inserts_hook_and_rejects_changed_anchor(tmp_path):
    root = tmp_path / "sglang"
    target = root / "srt/managers/scheduler.py"
    target.parent.mkdir(parents=True)
    source = "import logging\nclass Scheduler:\n    def step(self):\n        self.process_pending_chunked_abort()\n"
    target.write_text(source)
    observer.install(root)
    ast.parse(target.read_text())
    assert (
        "spindle_adapter_observer.snapshot(self, get_parallel())" in target.read_text()
    )
    assert (root / "srt/spindle_adapter_observer.py").exists()
    target.write_text(source.replace("process_pending_chunked_abort", "changed"))
    with pytest.raises(ValueError, match="hook changed"):
        observer.install(root)
