import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from stitch.types import VersionRef

from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app
from spindle.inference.lora_routing import prefix_key


@pytest.mark.parametrize("limit", [2, 9])
def test_mode_controller_checks_native_adapter_capacity(tmp_path, monkeypatch, limit):
    monkeypatch.setenv(
        "SGLANG_SPINDLE_MERGE_PROFILE",
        json.dumps(
            dict(
                unmerged_token_seconds=0.1,
                merged_token_seconds=0.05,
                prefill_tokens_per_second=1000,
                merge_seconds=1,
                unmerge_seconds=1,
                capacity=4,
                max_batch_adapters=limit,
            )
        ),
    )

    async def engine(request):
        assert request.url.path == "/server_info"
        return httpx.Response(200, json={"max_loras_per_batch": 8})

    app = create_app(
        SnapshotBulletin(tmp_path),
        "http://engine",
        transport=httpx.MockTransport(engine),
    )
    if limit > 8:
        with pytest.raises(ValueError, match="exceeds the native batch limit"):
            with TestClient(app):
                pass
    else:
        with TestClient(app) as client:
            assert (
                client.get("/routing-state").json()["mode"]["profile"][
                    "max_batch_adapters"
                ]
                == limit
            )


def test_mode_controller_pins_latest_before_merging(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SGLANG_SPINDLE_MERGE_PROFILE",
        json.dumps(
            dict(
                unmerged_token_seconds=0.1,
                merged_token_seconds=0.01,
                prefill_tokens_per_second=10000,
                merge_seconds=0.1,
                unmerge_seconds=0.1,
                capacity=1,
            )
        ),
    )
    bulletin = _publish(tmp_path, 3)
    mode = None
    served = []

    async def engine(request):
        nonlocal mode
        body = json.loads(request.content)
        if request.url.path == "/merge_lora_weights":
            mode = body["lora_name"]
        elif request.url.path == "/unmerge_lora_weights":
            mode = None
        elif request.url.path == "/generate":
            assert mode is None or mode == body["lora_path"]
            served.append((body["lora_path"], mode))
        return httpx.Response(200, json={"success": True, "meta_info": {}})

    app = create_app(bulletin, "http://engine", transport=httpx.MockTransport(engine))
    with TestClient(app) as client:
        payload = dict(
            input_ids=[1],
            weight_run_id="model-a",
            weight_version={"min_version": 3},
            sampling_params={"max_new_tokens": 100},
        )
        first = client.post("/generate", json=payload)
        assert first.status_code == 200
        assert first.json()["meta_info"]["weight_version_start"] == 3
        _publish(tmp_path, 4)
        payload["sampling_params"]["max_new_tokens"] = 1
        second = client.post("/generate", json=payload)
        assert second.status_code == 200
        assert second.json()["meta_info"]["weight_version_start"] == 4
        assert served == [
            ("model-a/weight_v000003", "model-a/weight_v000003"),
            ("model-a/weight_v000004", None),
        ]


def test_mode_admission_bounds_latest_resolution_and_releases_on_cancel(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "SGLANG_SPINDLE_MERGE_PROFILE",
        json.dumps(
            dict(
                unmerged_token_seconds=0.1,
                merged_token_seconds=0.05,
                prefill_tokens_per_second=1000,
                merge_seconds=1,
                unmerge_seconds=1,
                capacity=1,
                queue_buffer=1,
                enabled=False,
            )
        ),
    )
    bulletin = _publish(tmp_path, 12)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def refresh():
            entered.set()
            await release.wait()

        bulletin._refresh = refresh

        def engine(request):
            calls.append(request.url.path)
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://engine", transport=httpx.MockTransport(engine)
        )
        payload = dict(
            input_ids=[1],
            weight_run_id="model-a",
            weight_version={"min_version": 12},
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://sidecar"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.sleep(0.01)
            assert app.state.active == 2
            assert app.state.mode_scheduler.active == 0
            rejected = await client.post("/generate", json=payload)
            assert rejected.status_code == 503
            assert rejected.headers["x-spindle-capacity-rejected"] == "1"
            assert app.state.mode_scheduler.rejections == 1
            assert calls == []
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert app.state.active == 1
            replacement = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.sleep(0.01)
            assert app.state.active == 2
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(second, replacement), 2)
            assert all(r.status_code == 200 for r in responses)
            assert all(
                r.json()["meta_info"]["weight_version_start"] == 12 for r in responses
            )
            assert calls.count("/load_lora_adapter") == 1
            assert calls.count("/generate") == 2
            assert app.state.active == 0
            assert not app.state.routing_requests

    asyncio.run(run())


def _publish(tmp_path, version):
    source = tmp_path / f"source-{version}"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(f"v{version}".encode())
    (source / "adapter_config.json").write_text(
        json.dumps({"r": 8}),
        encoding="utf-8",
    )
    bulletin = SnapshotBulletin(tmp_path / "bulletin")
    bulletin.publish(VersionRef("model-a", version), source)
    return bulletin


def test_admission_rejects_before_registration_and_releases_after_completion(tmp_path):
    async def run():
        entered, finish = asyncio.Event(), asyncio.Event()
        calls = []

        async def engine(request):
            calls.append(request.url.path)
            if request.url.path == "/generate":
                entered.set()
                await finish.wait()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            _publish(tmp_path, 3),
            "http://sglang",
            transport=httpx.MockTransport(engine),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://sidecar"
            ) as client:
                first = asyncio.create_task(
                    client.post(
                        "/generate",
                        json={"rid": "accepted", "input_ids": [1]},
                        headers={"X-Spindle-Routing-Limit": "1"},
                    )
                )
                await entered.wait()
                rejected = await client.post(
                    "/generate",
                    json={
                        "rid": "rejected",
                        "input_ids": [1],
                        "weight_run_id": "model-a",
                        "weight_version": {"exact_version": 3},
                    },
                    headers={"X-Spindle-Routing-Limit": "1"},
                )
                assert rejected.status_code == 429
                assert rejected.headers["x-spindle-routing-rejected"] == "1"
                assert rejected.json()["request_ids"] == ["accepted"]
                assert calls == ["/generate"]
                assert app.state.active == 1
                assert not app.state.registered_adapters
                finish.set()
                accepted = await first
                assert accepted.headers["x-spindle-routing-admitted"] == "1"
                assert app.state.active == 0 and not app.state.routing_requests
                invalid = await client.post(
                    "/generate",
                    json={"input_ids": [1]},
                    headers={"X-Spindle-Routing-Limit": "0"},
                )
                assert invalid.status_code == 400
                assert calls == ["/generate"]

    asyncio.run(run())


def test_decode_maxima_merge_engine_progress_with_unobserved_ingress(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SGLANG_SPINDLE_ROUTING_OBSERVATIONS", "1")
    app = create_app(_publish(tmp_path, 3), "http://sglang")

    def row(rid, adapter, decode):
        return dict(rid=rid, adapter=adapter, prefill_tokens=0, decode_tokens=decode)

    with TestClient(app) as client:
        app.state.routing_read_at = time.monotonic()
        app.state.routing_requests = {
            r["rid"]: r
            for r in [
                row("running", "m/v1", 100),
                row("new", "m/v1", 80),
                row("version2", "m/v2", 90),
            ]
        }
        app.state.routing_snapshot = dict(
            timestamp=time.time(),
            incarnation=1,
            sequence=1,
            prefixes=[],
            cpu_adapters=["m/v1", "m/v2"],
            gpu_adapters=["m/v1"],
            requests=[row("running", "m/v1", 20), row("completed", "m/v3", 1000)],
            snapshot_seconds=0.001,
            cache_kind="RadixCache",
        )
        work = client.get("/routing-state").json()["work"]
        assert work["decode_tokens"] == 190
        assert work["adapter_decode_maxima"] == {"m/v1": 80, "m/v2": 90}
        assert set(work["request_ids"]) == {"running", "new", "version2"}


def test_prefix_hints_are_scoped_to_the_resolved_adapter_version(tmp_path):
    bulletin = _publish(tmp_path, 3)
    _publish(tmp_path, 4)

    def upstream(request):
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(bulletin, "http://sglang", transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        for version in (3, 4):
            response = client.post(
                "/generate",
                json={
                    "input_ids": [1, 2],
                    "session_id": "same-prefix",
                    "weight_run_id": "model-a",
                    "weight_version": {"exact_version": version},
                },
            )
            assert response.status_code == 200
        state = client.get("/routing-state").json()
        assert set(state["prefixes"]) == {
            prefix_key(VersionRef("model-a", version).identity, "same-prefix")
            for version in (3, 4)
        }
        assert state["active"] == 0 and state["adapters"] == {}


def test_sidecar_resolves_exact_adapter_and_stamps_response(tmp_path) -> None:
    bulletin = _publish(tmp_path, 3)
    adapter = VersionRef("model-a", 3)

    def upstream(request):
        payload = json.loads(request.content)
        if request.url.path == "/load_lora_adapter":
            assert payload == {
                "lora_name": adapter.identity,
                "lora_path": str(bulletin.resolve(adapter)),
                "pinned": False,
            }
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/generate"
        assert payload["lora_path"] == adapter.identity
        assert payload["rid"].startswith("spindle-")
        assert "weight_run_id" not in payload
        assert "weight_version" not in payload
        return httpx.Response(200, json={"meta_info": {}, "text": "ok"})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": 3, "min_version": None},
            },
        )

    assert response.status_code == 200
    assert response.json()["meta_info"] == {
        "weight_version_start": 3,
        "weight_version_end": 3,
    }


def test_sidecar_uses_latest_adapter_satisfying_minimum(tmp_path) -> None:
    bulletin = _publish(tmp_path, 4)
    adapter = VersionRef("model-a", 4)

    def upstream(request):
        payload = json.loads(request.content)
        if request.url.path == "/load_lora_adapter":
            assert payload["lora_name"] == adapter.identity
            assert payload["lora_path"] == str(bulletin.resolve(adapter))
            return httpx.Response(200, json={"success": True})
        assert payload["lora_path"] == adapter.identity
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": None, "min_version": 3},
            },
        )

    assert response.status_code == 200
    assert response.json()["meta_info"]["weight_version_start"] == 4


def test_sidecar_returns_retryable_conflict_when_adapter_is_missing(tmp_path) -> None:
    called = False

    def upstream(_request):
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    app = create_app(
        SnapshotBulletin(tmp_path / "bulletin"),
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": 2, "min_version": None},
            },
        )

    assert response.status_code == 409
    assert not called


def test_sidecar_passes_base_model_requests_without_lora(tmp_path) -> None:
    def upstream(request):
        payload = json.loads(request.content)
        assert "lora_path" not in payload
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(
        SnapshotBulletin(tmp_path / "bulletin"),
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post("/generate", json={"input_ids": [1]})

    assert response.status_code == 200
    assert response.json()["meta_info"] == {}


def test_sidecar_registers_each_adapter_once(tmp_path) -> None:
    bulletin = _publish(tmp_path, 5)
    loads = 0

    def upstream(request):
        nonlocal loads
        if request.url.path == "/load_lora_adapter":
            loads += 1
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    payload = {
        "input_ids": [1],
        "weight_run_id": "model-a",
        "weight_version": {"exact_version": 5, "min_version": None},
    }
    with TestClient(app) as client:
        assert client.post("/generate", json=payload).status_code == 200
        assert client.post("/generate", json=payload).status_code == 200

    assert loads == 1


def test_sidecar_propagates_adapter_registration_failure(tmp_path) -> None:
    bulletin = _publish(tmp_path, 6)

    def upstream(request):
        assert request.url.path == "/load_lora_adapter"
        return httpx.Response(400, json={"success": False, "error": "invalid adapter"})

    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "input_ids": [1],
                "weight_run_id": "model-a",
                "weight_version": {"exact_version": 6, "min_version": None},
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid adapter"


def test_volume_refresh_cannot_overlap_adapter_loading(tmp_path) -> None:
    bulletin = _publish(tmp_path, 7)

    async def scenario():
        loading = asyncio.Event()
        release = asyncio.Event()
        refreshes = []

        async def refresh():
            assert not loading.is_set(), "volume refreshed during SGLang file reads"
            refreshes.append(True)

        bulletin._refresh = refresh

        async def upstream(request):
            if request.url.path == "/load_lora_adapter":
                loading.set()
                await release.wait()
                loading.clear()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"exact_version": 7},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(loading.wait(), 2)
            second = asyncio.create_task(
                client.post(
                    "/generate",
                    json={
                        **payload,
                        "weight_version": {"min_version": 7},
                    },
                )
            )
            await asyncio.sleep(0.05)
            assert len(refreshes) == 0
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 2)
            assert all(result.status_code == 200 for result in results)
            assert len(refreshes) == 1

    asyncio.run(scenario())


def test_loaded_exact_version_bypasses_refresh_and_registration_lock(tmp_path) -> None:
    bulletin = _publish(tmp_path, 8)

    async def scenario():
        refreshes = []
        loads = []

        async def refresh():
            refreshes.append(True)

        bulletin._refresh = refresh

        def upstream(request):
            if request.url.path == "/load_lora_adapter":
                loads.append(True)
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"exact_version": 8},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            assert (await client.post("/generate", json=payload)).status_code == 200
            async with app.state.adapter_lock:
                response = await asyncio.wait_for(
                    client.post("/generate", json=payload), 2
                )
            assert response.status_code == 200
            assert response.json()["meta_info"]["weight_version_start"] == 8
            assert len(refreshes) == 0
            assert len(loads) == 1
            payload["weight_version"] = {"min_version": 8}
            assert (await client.post("/generate", json=payload)).status_code == 200
            assert len(refreshes) == 1
            assert len(loads) == 1

    asyncio.run(scenario())


def test_latest_validates_each_version_once_and_observes_publications(
    tmp_path, monkeypatch
):
    bulletin = _publish(tmp_path, 8)
    resolved = []
    loaded = []
    original = bulletin.resolve

    def resolve(ref):
        resolved.append(ref.version)
        return original(ref)

    monkeypatch.setattr(bulletin, "resolve", resolve)

    def upstream(request):
        if request.url.path == "/load_lora_adapter":
            loaded.append(json.loads(request.content)["lora_name"])
        return httpx.Response(200, json={"meta_info": {}})

    app = create_app(bulletin, "http://sglang", transport=httpx.MockTransport(upstream))
    payload = {
        "input_ids": [1],
        "weight_run_id": "model-a",
        "weight_version": {"min_version": 8},
    }
    with TestClient(app) as client:
        for _ in range(3):
            response = client.post("/generate", json=payload)
            assert response.status_code == 200
            assert response.json()["meta_info"]["weight_version_start"] == 8
        _publish(tmp_path, 9)
        for _ in range(2):
            response = client.post("/generate", json=payload)
            assert response.json()["meta_info"]["weight_version_start"] == 9
        payload["weight_version"] = {"min_version": 10}
        assert client.post("/generate", json=payload).status_code == 409
    assert resolved == [8, 9]
    assert loaded == [VersionRef("model-a", v).identity for v in (8, 9)]


def test_concurrent_latest_requests_share_one_refresh(tmp_path):
    bulletin = _publish(tmp_path, 12)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        refreshes = 0
        loads = 0

        async def refresh():
            nonlocal refreshes
            refreshes += 1
            entered.set()
            await release.wait()

        bulletin._refresh = refresh

        def upstream(request):
            nonlocal loads
            if request.url.path == "/load_lora_adapter":
                loads += 1
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"min_version": 12},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            requests = [
                asyncio.create_task(client.post("/generate", json=payload))
                for _ in range(8)
            ]
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0.01)
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(*requests), 2)
            assert all(r.status_code == 200 for r in responses)
            assert refreshes == 1
            assert loads == 1
            # A later request still checks for a newer publication.
            _publish(tmp_path, 13)
            response = await client.post("/generate", json=payload)
            assert response.json()["meta_info"]["weight_version_start"] == 13
            assert refreshes == 2

    asyncio.run(scenario())


def test_cancelled_waiter_does_not_cancel_shared_adapter_registration(tmp_path):
    bulletin = _publish(tmp_path, 15)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        loads = 0

        async def upstream(request):
            nonlocal loads
            if request.url.path == "/load_lora_adapter":
                loads += 1
                entered.set()
                await release.wait()
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"min_version": 15},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.sleep(0.01)
            first.cancel()
            try:
                await first
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("expected cancelled request")
            assert not second.done()
            release.set()
            assert (await asyncio.wait_for(second, 1)).status_code == 200
            assert loads == 1
            assert not app.state.adapter_resolutions

    asyncio.run(scenario())


def test_shared_resolution_failure_is_retryable_and_minimums_are_distinct(tmp_path):
    bulletin = _publish(tmp_path, 16)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
                raise RuntimeError("refresh failed")

        bulletin._refresh = refresh
        app = create_app(
            bulletin,
            "http://sglang",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"meta_info": {}})
            ),
        )
        payload = {
            "input_ids": [1],
            "weight_run_id": "model-a",
            "weight_version": {"min_version": 16},
        }
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.sleep(0.01)
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            assert all(isinstance(r, RuntimeError) for r in results)
            assert calls == 1
            assert not app.state.adapter_resolutions
            current, future = await asyncio.gather(
                client.post("/generate", json=payload),
                client.post(
                    "/generate",
                    json={**payload, "weight_version": {"min_version": 17}},
                ),
            )
            assert current.status_code == 200
            assert future.status_code == 409
            assert calls == 3

    asyncio.run(scenario())


def test_exact_version_refreshes_only_when_snapshot_is_missing(tmp_path):
    bulletin = SnapshotBulletin(tmp_path / "bulletin")
    refreshes = []

    def refresh():
        refreshes.append(True)
        _publish(tmp_path, 21)

    bulletin._refresh = refresh
    app = create_app(
        bulletin,
        "http://sglang",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"meta_info": {}})
        ),
    )
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/generate",
                json={
                    "input_ids": [1],
                    "weight_run_id": "model-a",
                    "weight_version": {"exact_version": 21},
                },
            )
            assert response.status_code == 200
            assert response.json()["meta_info"]["weight_version_start"] == 21
    assert refreshes == [True]


def test_concurrent_models_with_same_version_keep_distinct_adapters_and_load_counts(
    tmp_path,
):
    bulletin = SnapshotBulletin(tmp_path / "bulletin")
    refs = [VersionRef("model-a", 1), VersionRef("model-b", 1)]
    for ref in refs:
        source = tmp_path / ref.run_id
        source.mkdir()
        (source / "adapter_model.safetensors").write_bytes(ref.run_id.encode())
        (source / "adapter_config.json").write_text('{"r":8}')
        bulletin.publish(ref, source)

    async def run():
        loaded = []
        release, started = asyncio.Event(), asyncio.Event()
        requests = 0

        async def upstream(request):
            nonlocal requests
            payload = json.loads(request.content)
            if request.url.path == "/load_lora_adapter":
                loaded.append(payload["lora_name"])
                return httpx.Response(200, json={"success": True})
            requests += 1
            if requests == 2:
                started.set()
            await release.wait()
            return httpx.Response(
                200, json={"adapter": payload["lora_path"], "meta_info": {}}
            )

        app = create_app(
            bulletin, "http://sglang", transport=httpx.MockTransport(upstream)
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://sidecar"
            ) as client,
        ):
            pending = [
                asyncio.create_task(
                    client.post(
                        "/generate",
                        json={
                            "input_ids": [1],
                            "weight_run_id": ref.run_id,
                            "weight_version": {"exact_version": 1},
                        },
                    )
                )
                for ref in refs
            ]
            await asyncio.wait_for(started.wait(), 2)
            state = (await client.get("/routing-state")).json()
            assert state["active"] == 2
            assert state["adapters"] == {ref.identity: 1 for ref in refs}
            release.set()
            results = await asyncio.gather(*pending)
            assert [r.json()["adapter"] for r in results] == [
                ref.identity for ref in refs
            ]
            state = (await client.get("/routing-state")).json()
            assert state["active"] == 0 and state["adapters"] == {}
            assert state["adapter_loads"] == 2
            assert len(loaded) == len(set(loaded)) == 2
            assert all(r.headers["x-spindle-replica"] == "local" for r in results)

    asyncio.run(run())
