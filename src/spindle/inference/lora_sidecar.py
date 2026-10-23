from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import modal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from stitch.types import VersionConstraint, VersionRef

from spindle.startup_timing import timeline

from .bulletin import SnapshotBulletin, SnapshotNotFound
from .request_replay import RequestReplay


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
            # SGLang owns admission. Long generations must not exhaust the
            # HTTP pool before its configured concurrency or block control calls.
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
        )
        app.state.adapter_lock = asyncio.Lock()
        app.state.registered_adapters = {}
        app.state.adapter_resolutions = {}
        app.state.routing_requests = {}
        app.state.request_work = {}
        app.state.routing_catalog = {}
        app.state.routing_read_at = 0.0
        app.state.request_replay = RequestReplay()
        try:
            yield
        finally:
            await app.state.request_replay.aclose()
            pending = list(app.state.adapter_resolutions.values())
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/routing-state")
    async def routing_state():
        now = time.monotonic()
        if now - app.state.routing_read_at > 0.5:
            app.state.routing_read_at = now
            try:
                app.state.routing_catalog = json.loads(
                    await asyncio.to_thread(
                        Path("/tmp/spindle-routing-work.json").read_text
                    )
                )
            except (OSError, ValueError):
                app.state.routing_catalog = {}
        snapshot = app.state.routing_catalog
        work = None
        if snapshot:
            prefixes = snapshot["prefixes"]
            rows = {
                r["rid"]: r
                for r in snapshot["requests"]
                if r["rid"] in app.state.routing_requests
            }
            rows.update(
                (rid, row)
                for rid, row in app.state.request_work.items()
                if rid not in rows
            )
            work = dict(
                age_seconds=time.time() - snapshot["timestamp"],
                prefixes=prefixes,
                cpu_adapters=snapshot["cpu_adapters"],
                gpu_adapters=snapshot["gpu_adapters"],
                request_ids=list(rows),
                pending_adapters=sorted(set(app.state.routing_requests.values())),
                prefill_tokens=sum(r["prefill_tokens"] for r in rows.values()),
                decode_tokens=sum(r["decode_tokens"] for r in rows.values()),
                snapshot_seconds=snapshot["snapshot_seconds"],
                cache_kind=snapshot["cache_kind"],
                running_requests=snapshot["running_requests"],
                waiting_requests=snapshot["waiting_requests"],
                kv_used_tokens=snapshot.get("kv_used_tokens", 0),
                kv_capacity_tokens=snapshot.get("kv_capacity_tokens", 0),
            )
        return {
            "replica": os.environ.get("MODAL_TASK_ID", "local"),
            "active": len(app.state.routing_requests),
            "adapters": dict(Counter(app.state.routing_requests.values())),
            "request_ids": list(app.state.routing_requests),
            "pending_adapters": sorted(set(app.state.routing_requests.values())),
            "cpu_adapters": snapshot.get("cpu_adapters", []),
            "age_seconds": time.time() - snapshot["timestamp"] if snapshot else None,
            "work": work,
        }

    @app.get("/health")
    async def health() -> Response:
        response = await app.state.client.get("/health")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        payload = await request.json()
        rid = payload.setdefault("rid", f"spindle-{uuid.uuid4().hex}")
        # Deduplicate scalar attempts. Preserve the existing batch-ID path;
        # its separate engine request IDs do not identify one shared attempt.
        replay_id = rid if isinstance(rid, str) else object()
        response = await app.state.request_replay.run(
            replay_id,
            payload,
            lambda: generate_once(payload),
            request.is_disconnected,
        )
        if response is None:
            response = Response(status_code=499)
        response.headers["x-spindle-replica"] = os.environ.get("MODAL_TASK_ID", "local")
        return response

    async def generate_once(payload: dict) -> Response:
        rid = payload["rid"]
        constraint = VersionConstraint.from_payload(payload)
        run_id = payload.get("weight_run_id")
        if isinstance(rid, str):
            app.state.routing_requests[rid] = (
                VersionRef(str(run_id), constraint.exact_version).identity
                if run_id is not None and constraint.exact_version is not None
                else payload.get("lora_path", "base")
            )
            app.state.request_work[rid] = dict(
                rid=rid,
                adapter=app.state.routing_requests.get(rid, "base"),
                prefill_tokens=len(payload.get("input_ids", ())),
                decode_tokens=payload.get("sampling_params", {}).get(
                    "max_new_tokens", 1
                ),
            )
        try:
            return await generate_response(payload)
        finally:
            if isinstance(rid, str):
                app.state.routing_requests.pop(rid, None)
                app.state.request_work.pop(rid, None)

    async def generate_response(payload: dict) -> Response:
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

        if isinstance(payload["rid"], str):
            app.state.routing_requests[payload["rid"]] = payload.get(
                "lora_path", "base"
            )
        response = await _post_generate(app.state.client, payload)
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
                if not app.state.registered_adapters:
                    timeline("sampler-sidecar").mark(
                        "first_adapter_loaded", adapter=resolved.identity
                    )
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
    payload: dict,
) -> httpx.Response:
    rid = payload.setdefault("rid", f"spindle-{uuid.uuid4().hex}")
    try:
        return await client.post("/generate", json=payload)
    except asyncio.CancelledError:
        await _abort(client, rid)
        raise


async def _abort(client: httpx.AsyncClient, rid) -> None:
    rids = rid if isinstance(rid, list) else [rid]
    try:
        async with asyncio.timeout(2):
            await asyncio.gather(
                *(
                    client.post("/abort_request", json={"rid": str(item)})
                    for item in rids
                ),
                return_exceptions=True,
            )
    except TimeoutError:
        pass


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
