import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from spindle.providers.modal import model_routing


def test_discovery_reuses_membership_and_rejects_flash_misroutes(monkeypatch):
    async def scenario():
        containers = [
            SimpleNamespace(host=h, port=8000, task_id=h) for h in ("a", "b", "c")
        ]
        discover = AsyncMock(return_value=containers)
        monkeypatch.setattr(model_routing, "list_flash_containers_async", discover)
        monkeypatch.setattr(
            model_routing, "proxy_auth_headers", lambda: {"auth": "test"}
        )

        def handle(request):
            host = request.headers["Modal-Flash-Upstream"].split(":")[0]
            assert request.headers["auth"] == "test"
            return httpx.Response(
                200,
                json={
                    "replica": "wrong" if host == "b" else host,
                    "active": 2,
                    "request_ids": ["one", "two"],
                    "cpu_adapters": ["model/v1"],
                    "pending_adapters": ["model/v2"],
                    "age_seconds": None if host == "c" else 0.1,
                },
            )

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            model_routing.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
        )
        router = model_routing.create_model_router(
            "app", "https://gateway/", capacity=8
        )
        try:
            replicas = await router.discover()
            await router.discover()
            assert discover.await_count == 1
            assert [r.replica_id for r in replicas] == ["a", "c"]
            assert replicas[0].cpu_adapters == ("model/v1",)
            assert replicas[0].request_ids == {"one", "two"}
            assert replicas[1].age_seconds == float("inf")
            assert replicas[1].cpu_adapters == ()
        finally:
            await router.aclose()

    asyncio.run(scenario())
