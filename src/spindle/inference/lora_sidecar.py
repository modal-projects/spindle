from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import modal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from stitch.types import VersionConstraint, VersionRef

from .bulletin import SnapshotBulletin, SnapshotNotFound
from .lora_routing import prefix_key


def create_app(
    bulletin: SnapshotBulletin,
    upstream_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=upstream_url.rstrip("/"),
            timeout=3000,
            trust_env=False,
            transport=transport,
        )
        app.state.adapter_lock = asyncio.Lock()
        app.state.registered_adapters = {}
        app.state.adapter_resolutions = {}
        app.state.active_adapters = Counter()
        app.state.recent_adapters = OrderedDict()
        app.state.prefixes = OrderedDict()
        app.state.active = 0
        app.state.adapter_loads = 0
        app.state.adapter_load_seconds = 0.0
        app.state.routing_requests = {}
        app.state.routing_snapshot = {}
        app.state.routing_read_at = 0.0
        try:
            yield
        finally:
            pending = list(app.state.adapter_resolutions.values())
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/routing-state")
    async def routing_state():
        now = time.monotonic()
        work = None
        if os.environ.get("SGLANG_SPINDLE_ROUTING_OBSERVATIONS") == "1":
            if now - app.state.routing_read_at > 0.5:
                app.state.routing_read_at = now
                path = Path("/tmp/spindle-routing-work.json")
                try:
                    app.state.routing_snapshot = json.loads(
                        await asyncio.to_thread(path.read_text)
                    )
                except (OSError, ValueError):
                    app.state.routing_snapshot = {}
            snapshot = app.state.routing_snapshot
            if snapshot:
                current = app.state.routing_requests
                rows = {
                    r["rid"]: r for r in snapshot["requests"] if r["rid"] in current
                }
                rows.update(
                    (rid, row) for rid, row in current.items() if rid not in rows
                )
                adapter_decode_maxima = {}
                for row in rows.values():
                    adapter = row["adapter"]
                    adapter_decode_maxima[adapter] = max(
                        adapter_decode_maxima.get(adapter, 0), row["decode_tokens"]
                    )
                work = dict(
                    age_seconds=time.time() - snapshot["timestamp"],
                    incarnation=snapshot["incarnation"],
                    sequence=snapshot["sequence"],
                    prefixes=snapshot["prefixes"],
                    cpu_adapters=snapshot["cpu_adapters"],
                    gpu_adapters=snapshot["gpu_adapters"],
                    request_ids=list(rows),
                    pending_adapters=list({r["adapter"] for r in rows.values()}),
                    prefill_tokens=sum(r["prefill_tokens"] for r in rows.values()),
                    decode_tokens=sum(r["decode_tokens"] for r in rows.values()),
                    adapter_decode_maxima=adapter_decode_maxima,
                    snapshot_seconds=snapshot["snapshot_seconds"],
                    cache_kind=snapshot["cache_kind"],
                )
        return {
            "replica": os.environ.get("MODAL_TASK_ID", "local"),
            "active": app.state.active,
            "adapters": dict(app.state.active_adapters),
            # Recent use is a residency estimate, not a GPU-cache assertion.
            "recent": list(app.state.recent_adapters),
            "prefixes": [k for k, t in app.state.prefixes.items() if now - t < 120],
            "adapter_loads": app.state.adapter_loads,
            "adapter_load_seconds": app.state.adapter_load_seconds,
            "work": work,
        }

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        payload = await request.json()
        rid = payload.setdefault("rid", f"spindle-{uuid.uuid4().hex}")
        constraint = VersionConstraint.from_payload(payload)
        run_id = payload.get("weight_run_id")
        adapter = (
            VersionRef(str(run_id), constraint.exact_version).identity
            if run_id is not None and constraint.exact_version is not None
            else payload.get("lora_path", "base")
        )
        if isinstance(rid, str):
            app.state.routing_requests[rid] = dict(
                rid=rid,
                adapter=adapter,
                prefill_tokens=len(payload.get("input_ids", ())),
                decode_tokens=payload.get("sampling_params", {}).get(
                    "max_new_tokens", 1
                ),
            )
        app.state.active += 1
        try:
            response = await generate_response(request, payload)
            response.headers["x-spindle-replica"] = os.environ.get(
                "MODAL_TASK_ID", "local"
            )
            return response
        finally:
            app.state.active -= 1
            if isinstance(rid, str):
                app.state.routing_requests.pop(rid, None)

    @app.get("/health")
    async def health() -> Response:
        response = await app.state.client.get("/health")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    async def generate_response(request: Request, payload: dict) -> Response:
        run_id = payload.pop("weight_run_id", None)
        constraint = VersionConstraint.from_payload(payload)
        payload.pop("weight_version", None)
        served_version = None
        if run_id is not None:
            try:
                ref, registration_error = await _prepare_adapter(
                    app, bulletin, str(run_id), constraint
                )
                if registration_error is not None:
                    return Response(
                        content=registration_error.content,
                        status_code=registration_error.status_code,
                        media_type=registration_error.headers.get("content-type"),
                    )
                payload["lora_path"] = ref.identity
                served_version = ref.version
            except (SnapshotNotFound, ValueError) as exc:
                return JSONResponse(
                    {"error": f"adapter version is not ready: {exc}"},
                    status_code=409,
                )

        adapter = payload.get("lora_path", "base")
        if isinstance(payload["rid"], str):
            app.state.routing_requests[payload["rid"]]["adapter"] = adapter
        affinity = payload.get("session_id")
        key = prefix_key(adapter, affinity) if isinstance(affinity, str) else None
        if key is not None:
            app.state.prefixes[key] = time.monotonic()
            app.state.prefixes.move_to_end(key)
            if len(app.state.prefixes) > 4096:
                app.state.prefixes.popitem(last=False)
        app.state.active_adapters[adapter] += 1
        try:
            response = await _post_generate(app.state.client, request, payload)
        finally:
            if key is not None and key in app.state.prefixes:
                app.state.prefixes[key] = time.monotonic()
                app.state.prefixes.move_to_end(key)
            app.state.active_adapters[adapter] -= 1
            if not app.state.active_adapters[adapter]:
                del app.state.active_adapters[adapter]
            app.state.recent_adapters[adapter] = time.monotonic()
            app.state.recent_adapters.move_to_end(adapter)
            while len(app.state.recent_adapters) > 32:
                app.state.recent_adapters.popitem(last=False)
        if response is None:
            return Response(status_code=499)
        try:
            body = response.json()
        except ValueError:
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )
        if served_version is not None and isinstance(body, dict):
            metadata = body.setdefault("meta_info", {})
            metadata["weight_version_start"] = served_version
            metadata["weight_version_end"] = served_version
        return JSONResponse(body, status_code=response.status_code)

    return app


async def _prepare_adapter(
    app: FastAPI,
    bulletin: SnapshotBulletin,
    run_id: str,
    constraint: VersionConstraint,
) -> tuple[VersionRef, httpx.Response | None]:
    ref = (
        VersionRef(run_id, constraint.exact_version)
        if constraint.exact_version is not None
        else None
    )
    if ref is not None and ref.identity in app.state.registered_adapters:
        return ref, None
    key = (run_id, constraint.exact_version, constraint.min_version)
    pending = app.state.adapter_resolutions
    task = pending.get(key)
    if task is None:

        async def prepare():
            # Reload must not hide paths while SGLang reads a new adapter.
            async with app.state.adapter_lock:
                if ref is not None and ref.identity in app.state.registered_adapters:
                    return ref, None
                if ref is not None:
                    resolved = ref
                    try:
                        path = await asyncio.to_thread(bulletin.resolve, resolved)
                    except SnapshotNotFound:
                        # Exact versions are immutable: reload only if this mount
                        # does not yet contain the complete requested snapshot.
                        await bulletin.refresh()
                        path = await asyncio.to_thread(bulletin.resolve, resolved)
                else:
                    # Latest/minimum requests must see newly published policies.
                    await bulletin.refresh()
                    resolved = bulletin.read_latest(run_id)
                    if resolved is None:
                        raise SnapshotNotFound(run_id)
                    if (
                        constraint.min_version is not None
                        and resolved.version < constraint.min_version
                    ):
                        raise SnapshotNotFound(
                            f"{run_id} latest={resolved.version} required={constraint.min_version}"
                        )
                    if resolved.identity in app.state.registered_adapters:
                        return resolved, None
                    path = await asyncio.to_thread(bulletin.resolve, resolved)
                started = time.monotonic()
                response = await app.state.client.post(
                    "/load_lora_adapter",
                    json={
                        "lora_name": resolved.identity,
                        "lora_path": str(path),
                        "pinned": False,
                    },
                )
                if response.is_error:
                    return resolved, response
                app.state.adapter_loads += 1
                app.state.adapter_load_seconds += time.monotonic() - started
                app.state.registered_adapters[resolved.identity] = str(path)
                return resolved, None

        task = asyncio.create_task(prepare())
        pending[key] = task

        def finished(completed):
            if pending.get(key) is completed:
                del pending[key]
            if not completed.cancelled():
                # A disconnected caller may leave no waiter for this failure.
                completed.exception()

        task.add_done_callback(finished)
    # Share only in-flight work. Later requests still resolve the latest version.
    # One disconnected request must not cancel registration for other callers.
    return await asyncio.shield(task)


async def _post_generate(
    client: httpx.AsyncClient,
    request: Request,
    payload: dict,
) -> httpx.Response | None:
    rid = payload.setdefault("rid", f"spindle-{uuid.uuid4().hex}")
    task = asyncio.create_task(client.post("/generate", json=payload))
    try:
        while not task.done():
            done, _ = await asyncio.wait((task,), timeout=0.25)
            if done:
                break
            if await request.is_disconnected():
                await _abort(client, rid)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return None
        return await task
    except asyncio.CancelledError:
        await _abort(client, rid)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise


async def _abort(client: httpx.AsyncClient, rid) -> None:
    rids = rid if isinstance(rid, list) else [rid]
    await asyncio.gather(
        *(client.post("/abort_request", json={"rid": str(item)}) for item in rids),
        return_exceptions=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--bulletin-root", required=True)
    parser.add_argument("--bulletin-volume", default="")
    args = parser.parse_args()

    refresh = None
    if args.bulletin_volume:
        refresh = modal.Volume.from_name(args.bulletin_volume, version=2).reload
    bulletin = SnapshotBulletin(Path(args.bulletin_root), refresh=refresh)
    uvicorn.run(
        create_app(bulletin, args.upstream_url),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
