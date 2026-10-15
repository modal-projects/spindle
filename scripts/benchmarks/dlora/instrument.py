"""Fail-closed observation hooks for the pinned PR; no GPU synchronization."""

from pathlib import Path
from cache_instrumentation import instrument

root = Path("/opt/dlora-sglang/python/sglang")
sidecar = Path("/opt/dlora-spindle/spindle/inference/lora_sidecar.py")
instrument(root, sidecar)

# Experiment-only controls are reached through the authenticated Flash proxy.
source = sidecar.read_text()
marker = '    @app.get("/benchmark-state")\n'
assert source.count(marker) == 1
source = source.replace(
    marker,
    """    @app.get("/benchmark-config")
    async def benchmark_config():
        response = await app.state.benchmark_client.get("/server_info")
        response.raise_for_status()
        config = response.json()
        return {"replica": os.environ.get("MODAL_TASK_ID"), "engine": {
            k: config[k] for k in ("max_running_requests", "max_queued_requests", "enable_lora_merge", "lora_unmerge_mode")
        }}

    @app.post("/benchmark-mode")
    async def benchmark_mode(request: Request):
        scheduler = app.state.mode_scheduler
        if app.state.active or scheduler.active:
            return JSONResponse({"detail": "Drain before changing the policy"}, status_code=409)
        settings = await request.json()
        if scheduler.mode is not None:
            await scheduler._transition(None)
        scheduler.profile = MergeProfile(**settings)
        return {"ok": True}

    @app.post("/benchmark-transition/{action}")
    async def benchmark_transition(action: str, request: Request):
        if action not in {"merge_lora_weights", "unmerge_lora_weights", "flush_cache"}:
            return Response(status_code=404)
        if app.state.active:
            return Response(status_code=409)
        response = await app.state.client.post("/" + action, json=await request.json())
        if not response.is_error and action != "flush_cache":
            app.state.mode_scheduler.mode = response.json().get("merged_lora_name")
        app.state.prefixes.clear()
        app.state.routing_snapshot = {}
        return Response(content=response.content, status_code=response.status_code, media_type="application/json")

"""
    + marker,
)
sidecar.write_text(source)

# Observe actual executed decode batches, not merely outstanding requests.
metrics = root / "srt/lora/routing_benchmark.py"
with metrics.open("a") as stream:
    stream.write("""
def batch(reqs, merged):
    counts['merged_decode_steps' if merged else 'unmerged_decode_steps'] += 1
    adapters = Counter(req.lora_id for req in reqs)
    counts['decode_batch_sequences'] += len(reqs)
    counts['decode_batch_distinct_adapters'] += len(adapters)
    counts['decode_batch_dominant_sequences'] += max(adapters.values(), default=0)
    counts['decode_batch_size_' + str(len(reqs))] += 1
    flush()
""")
scheduler = root / "srt/managers/scheduler.py"
source = scheduler.read_text()
marker = '        """Run a batch."""\n'
assert source.count(marker) == 1
source = source.replace(
    marker,
    marker
    + "        if self.enable_lora and batch.forward_mode.is_decode():\n            routing_benchmark.batch(batch.reqs, self.tp_worker.model_runner.lora_manager.merged_lora_id is not None)\n",
)
scheduler.write_text(source)
