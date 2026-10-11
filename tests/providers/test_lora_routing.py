import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from spindle.providers.modal import lora_routing


def test_flash_discovery_preserves_port_auth_and_checks_actual_replica(monkeypatch):
    async def run():
        nodes = [
            SimpleNamespace(task_id="a", host="host-a", port=443),
            SimpleNamespace(task_id="b", host="host-b", port=443),
        ]
        discover = AsyncMock(return_value=nodes)
        monkeypatch.setattr(lora_routing, "list_flash_containers_async", discover)
        monkeypatch.setattr(
            lora_routing,
            "proxy_auth_headers",
            lambda: {"Modal-Key": "key", "Modal-Secret": "secret"},
        )
        calls = []
        incorrect_proxy = False

        def handle(request):
            assert request.url == "https://gateway/routing-state"
            assert request.headers["Modal-Key"] == "key"
            assert "Modal-Session-ID" not in request.headers
            assert "Modal-Routing-Affinity-Key" not in request.headers
            upstream = request.headers["Modal-Flash-Upstream"]
            assert upstream.endswith(":443")
            actual = upstream.removeprefix("host-").removesuffix(":443")
            calls.append(actual)
            return httpx.Response(
                200,
                json={
                    "replica": "wrong" if incorrect_proxy else actual,
                    "active": 3,
                    "adapters": {},
                    "recent": ["m/v1"],
                },
            )

        original = httpx.AsyncClient
        monkeypatch.setattr(
            lora_routing.httpx,
            "AsyncClient",
            lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
        )
        router = lora_routing.create_router(
            "app", "https://gateway", target=16, slots=8
        )
        await router.refresh()
        assert set(router.replicas) == {"a", "b"}
        assert router.replicas["a"].upstream == "host-a:443"
        await router._refresh()
        assert len(calls) == 4
        discover.assert_awaited_once()
        incorrect_proxy = True
        await router._refresh()
        assert router.acquire("model", "adapter", "affinity") is None

    asyncio.run(run())


def test_zero_replica_discovery_does_not_prevent_flash_cold_start(monkeypatch):
    async def run():
        monkeypatch.setattr(
            lora_routing, "list_flash_containers_async", AsyncMock(return_value=[])
        )
        monkeypatch.setattr(lora_routing, "proxy_auth_headers", lambda: {})
        pool = lora_routing.create_router("app", "https://gateway", target=16, slots=8)
        await pool.refresh()
        assert pool.acquire("m", "v1", "r") is None

    asyncio.run(run())
