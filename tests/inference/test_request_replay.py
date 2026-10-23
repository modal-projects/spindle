"""Exercise duplicate HTTP delivery against the real sidecar entrypoint."""

import asyncio
import json
import os

import httpx
from fastapi.responses import Response

from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app
from spindle.inference.request_replay import RequestReplay


async def wait_for_waiters(app, rid, count):
    async with asyncio.timeout(1):
        while app.state.request_replay.pending[rid].waiters < count:
            await asyncio.sleep(0)


def test_replayed_generation_shares_work_and_preserves_active_count(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def engine(request):
            nonlocal calls
            if request.url.path == "/generate":
                calls += 1
                if calls > 1:
                    return httpx.Response(400, json={"error": "Duplicate request ID"})
                entered.set()
                await release.wait()
                return httpx.Response(
                    200,
                    json={
                        "output_ids": [42],
                        "meta_info": {"completion_tokens": 1, "cached_tokens": 0},
                    },
                )
            return httpx.Response(200, json={})

        app = create_app(
            SnapshotBulletin(tmp_path),
            "http://engine",
            transport=httpx.MockTransport(engine),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            payload = {"rid": "same-attempt:0", "input_ids": [1, 2, 3]}
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(client.post("/generate", json=payload))
            await wait_for_waiters(app, payload["rid"], 2)
            await asyncio.sleep(0.02)
            state = {"active": len(app.state.request_replay.pending)}
            release.set()
            replies = await asyncio.wait_for(asyncio.gather(first, second), 1)
            assert state["active"] == 1
            assert [r.status_code for r in replies] == [200, 200]
            assert calls == 1
            # A reply lost after completion must not generate a second sample.
            replay = await client.post("/generate", json=payload)
            assert replay.json() == replies[0].json()
            assert calls == 1

    asyncio.run(scenario())


def test_disconnect_reconnect_preserves_generation_then_aborts_unclaimed_work():
    async def scenario():
        replay = RequestReplay(disconnect_grace=0.1)
        disconnected = asyncio.Event()
        entered = asyncio.Event()
        release = asyncio.Event()
        aborted = asyncio.Event()
        calls = 0

        async def generate():
            nonlocal calls
            calls += 1
            entered.set()
            try:
                await release.wait()
                return Response(b"result")
            except asyncio.CancelledError:
                aborted.set()
                raise

        async def connected():
            return False

        async def connection_lost():
            return disconnected.is_set()

        try:
            first = asyncio.create_task(
                replay.run("recover", {"input_ids": [1]}, generate, connection_lost)
            )
            await entered.wait()
            disconnected.set()
            assert await asyncio.wait_for(first, 1) is None
            second = asyncio.create_task(
                replay.run("recover", {"input_ids": [1]}, generate, connected)
            )
            await asyncio.sleep(0.15)
            assert not aborted.is_set()
            release.set()
            assert (await second).body == b"result"
            assert calls == 1

            # A disconnected request with no replay must not keep generating.
            release.clear()
            assert (
                await replay.run(
                    "abandoned", {"input_ids": [2]}, generate, connection_lost
                )
                is None
            )
            await asyncio.wait_for(aborted.wait(), 1)
            await asyncio.sleep(0)
            result = await replay.run(
                "abandoned", {"input_ids": [2]}, generate, connected
            )
            assert result.status_code == 409
            assert result.headers["x-spindle-replay-state"] == "cancelled"
            assert calls == 2
        finally:
            await replay.aclose()

    asyncio.run(scenario())


def test_replay_rejects_changed_payload_and_bounds_completed_results():
    async def scenario():
        replay = RequestReplay(max_entries=1, max_bytes=10)
        calls = 0

        async def generate():
            nonlocal calls
            calls += 1
            return Response(b"result")

        async def connected():
            return False

        try:
            await replay.run("a", {"input_ids": [1]}, generate, connected)
            conflict = await replay.run("a", {"input_ids": [2]}, generate, connected)
            assert conflict.status_code == 409
            assert "x-spindle-replay-state" not in conflict.headers
            assert calls == 1
            await replay.run("b", {"input_ids": [1]}, generate, connected)
            await replay.run("a", {"input_ids": [1]}, generate, connected)
            assert calls == 3
            replay.ttl = 0
            await replay.run("c", {"input_ids": [1]}, generate, connected)
            await replay.run("c", {"input_ids": [1]}, generate, connected)
            assert calls == 5
        finally:
            await replay.aclose()

    asyncio.run(scenario())


def test_cancelled_original_does_not_cancel_connected_replay(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def engine(request):
            calls.append((request.url.path, json.loads(request.content)))
            if request.url.path == "/generate":
                entered.set()
                await release.wait()
            return httpx.Response(200, json={"meta_info": {"completion_tokens": 1}})

        app = create_app(
            SnapshotBulletin(tmp_path),
            "http://engine",
            transport=httpx.MockTransport(engine),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            payload = {"rid": "original", "input_ids": [1]}
            first = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(client.post("/generate", json=payload))
            await wait_for_waiters(app, payload["rid"], 2)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            state = {"active": len(app.state.request_replay.pending)}
            release.set()
            result = await asyncio.wait_for(second, 1)
            assert result.status_code == 200
            assert state["active"] == 1
            assert [p for p, _ in calls] == ["/generate"]

    asyncio.run(scenario())


def test_last_cancellation_is_bounded_when_engine_abort_does_not_reply(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        abort_entered = asyncio.Event()

        async def engine(request):
            if request.url.path == "/generate":
                entered.set()
            elif request.url.path == "/abort_request":
                abort_entered.set()
            await asyncio.Future()

        app = create_app(
            SnapshotBulletin(tmp_path),
            "http://engine",
            transport=httpx.MockTransport(engine),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            task = asyncio.create_task(
                client.post("/generate", json={"rid": "cancel", "input_ids": [1]})
            )
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.wait_for(abort_entered.wait(), 1)
            cancelling = await client.post(
                "/generate", json={"rid": "cancel", "input_ids": [1]}
            )
            assert cancelling.status_code == 409
            assert cancelling.headers["x-spindle-replay-state"] == "cancelling"
            replica = os.environ.get("MODAL_TASK_ID", "local")
            assert cancelling.headers["x-spindle-replica"] == replica
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
            assert len(app.state.request_replay.pending) == 0
            cancelled = await client.post(
                "/generate", json={"rid": "cancel", "input_ids": [1]}
            )
            assert cancelled.status_code == 409
            assert cancelled.headers["x-spindle-replay-state"] == "cancelled"
            assert cancelled.headers["x-spindle-replica"] == replica
            conflict = await client.post(
                "/generate", json={"rid": "cancel", "input_ids": [2]}
            )
            assert conflict.status_code == 409
            assert "x-spindle-replay-state" not in conflict.headers

    asyncio.run(scenario())


def test_batch_request_ids_preserve_pass_through_without_retaining_results(tmp_path):
    async def scenario():
        calls = []

        async def engine(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=[{"text": "a"}, {"text": "b"}])

        app = create_app(
            SnapshotBulletin(tmp_path),
            "http://engine",
            transport=httpx.MockTransport(engine),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            payload = {"rid": ["a", "b"], "input_ids": [[1], [2]]}
            for _ in range(2):
                reply = await client.post("/generate", json=payload)
                assert reply.status_code == 200
                assert reply.json() == [{"text": "a"}, {"text": "b"}]
            assert [call["rid"] for call in calls] == [["a", "b"], ["a", "b"]]
            assert app.state.request_replay.completed_bytes == 0

    asyncio.run(scenario())


def test_batch_cancellation_aborts_each_engine_request(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        aborted = []

        async def engine(request):
            if request.url.path == "/generate":
                entered.set()
                await asyncio.Future()
            if request.url.path == "/abort_request":
                aborted.append(json.loads(request.content)["rid"])
            return httpx.Response(200, json={})

        app = create_app(
            SnapshotBulletin(tmp_path),
            "http://engine",
            transport=httpx.MockTransport(engine),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            task = asyncio.create_task(
                client.post(
                    "/generate",
                    json={"rid": ["a", "b"], "input_ids": [[1], [2]]},
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
            assert sorted(aborted) == ["a", "b"]
            assert not app.state.request_replay.pending

    asyncio.run(scenario())
