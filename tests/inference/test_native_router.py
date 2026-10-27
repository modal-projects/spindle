"""Real HTTP integration tests against Rust's runtime, without Modal/GPU access."""

import asyncio
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from unittest.mock import AsyncMock

import pytest

from spindle.inference.native_router import NativeRouter


@pytest.fixture
def worker():
    state = dict(
        active=0,
        requests=[],
        statuses=[],
        delay=0,
        arrived=threading.Event(),
        second_arrived=threading.Event(),
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200, *, replica="worker", replay_state=None):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            if replica is not None:
                self.send_header("x-spindle-replica", replica)
            if replay_state is not None:
                self.send_header("x-spindle-replay-state", replay_state)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            state.setdefault("observation_paths", []).append(self.path)
            replica = self.headers.get("Modal-Flash-Upstream", "worker:8000").split(
                ":"
            )[0]
            observation = state.get("observations", {}).get(replica, {})
            time.sleep(observation.get("delay", 0))
            self.reply(
                dict(
                    replica=replica,
                    active=state["active"],
                    work=dict(
                        age_seconds=observation.get("age", 0),
                        prefixes=observation.get("prefixes", []),
                        prefix_nodes=[],
                        cpu_adapters=["adapter/v1"],
                        gpu_adapters=["adapter/v1"],
                        pending_adapters=[],
                        request_ids=[],
                    ),
                ),
                replica=replica,
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["requests"].append((dict(self.headers), body))
            replica = self.headers.get("Modal-Flash-Upstream", "worker:8000").split(
                ":"
            )[0]
            if state.get("publish_prefix"):
                state["observations"][replica]["prefixes"] = [state["publish_prefix"]]
            state.setdefault("arrival_times", []).append(time.monotonic())
            state["arrived"].set()
            if len(state["requests"]) == 2:
                state["second_arrived"].set()
            time.sleep(
                state["delays"].pop(0) if state.get("delays") else state["delay"]
            )
            # A sidecar deduplicates work by RID; a lost HTTP response must not
            # turn a retry into a second logical generation.
            state.setdefault("generations", set()).add(body["rid"])
            fault = state.get("faults", []).pop(0) if state.get("faults") else None
            if fault in ("headers", "body"):
                if fault == "body":
                    self.send_response(200)
                    self.send_header("x-spindle-replica", "worker")
                    self.send_header("content-length", "1000")
                    self.end_headers()
                    self.wfile.write(b'{"text":')
                    self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            if fault == "proxy":
                self.reply({"error": "upstream reset"}, 502, replica=None)
                return
            if fault == "mismatch":
                self.reply({"text": "wrong worker"}, replica="different-worker")
                return
            if fault in ("cancelled", "cancelling", "conflict"):
                self.reply(
                    {"error": "cannot replay"},
                    409,
                    replay_state=None if fault == "conflict" else fault,
                )
                return
            status = state["statuses"].pop(0) if state["statuses"] else 200
            self.reply(
                {"meta_info": {"completion_tokens": 1}, "text": "ok"},
                status,
                replica=replica,
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["gateway"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def make_router(worker, policy="cache_wait"):
    return NativeRouter(
        AsyncMock(
            return_value=[
                dict(replica=r, upstream=f"{r}:8000")
                for r in worker.get("observations", {"worker": {}})
            ]
        ),
        gateway=worker["gateway"],
        headers={},
        policy=policy,
        capacity=24,
        prefill_tokens_per_second=1000.0,
        balance_absolute=worker.get("balance_absolute", 4),
        balance_relative=worker.get("balance_relative", 1.25),
    )


async def dispatch(router, worker, id="test", timeout=3):
    return await router.dispatch(
        router.prepare_request("adapter/v1", "group"),
        request_id=id,
        gateway=worker["gateway"],
        body=dict(input_ids=[1, 2], sampling_params=dict(max_new_tokens=4)),
        headers={"Modal-Session-ID": "must-be-removed"},
        deadline=asyncio.get_running_loop().time() + timeout,
    )


@pytest.mark.parametrize("policy", ["cache_wait"])
def test_native_dispatch_owns_transport_retries_and_releases(worker, policy):
    worker["statuses"] = [503, 200]

    async def run():
        router = make_router(worker, policy)
        try:
            result = await dispatch(router, worker)
            assert result.response.status_code == 200 and result.replica == "worker"
            assert len(worker["requests"]) == 2
            assert worker["observation_paths"]
            assert all(path == "/routing-state" for path in worker["observation_paths"])
            for headers, body in worker["requests"]:
                headers = {k.lower(): v for k, v in headers.items()}
                assert headers["modal-flash-upstream"] == "worker:8000"
                assert "modal-session-id" not in headers
                assert body["input_ids"] == [1, 2]
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["headers", "body", "proxy"])
def test_native_lost_response_reuses_worker_and_generation(worker, fault):
    worker["faults"] = [fault]

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker)
            assert result.response.status_code == 200
            assert len(worker["requests"]) == 2
            assert len(worker["generations"]) == 1
            assert worker["requests"][0] == worker["requests"][1]
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_native_replay_is_bounded_before_starting_new_attempt(worker):
    worker["faults"] = ["headers", "headers"]

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker)
            assert result.response.status_code == 200
            ids = [body["rid"] for _, body in worker["requests"]]
            assert len(ids) == 3
            assert ids[0] == ids[1] and ids[1] != ids[2]
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["headers", "proxy", "mismatch"])
def test_native_transport_budget_and_worker_identity_remain_enforced(worker, fault):
    worker["faults"] = [fault] * 30

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker, timeout=30)
            assert result.response.status_code == 502
            ids = [body["rid"] for _, body in worker["requests"]]
            if fault == "mismatch":
                assert len(ids) == 1
                assert "identity mismatch" in result.response.text
            else:
                assert len(ids) == 16
                assert len(set(ids)) == 8
                assert all(ids[i] == ids[i + 1] for i in range(0, 16, 2))
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_native_replay_obeys_cancellation_and_original_deadline(worker, cancel):
    worker["faults"] = ["headers"]
    worker["delays"] = [0, 2]

    async def run():
        router = make_router(worker)
        started = time.monotonic()
        try:
            task = asyncio.create_task(dispatch(router, worker, timeout=0.5))
            assert await asyncio.to_thread(worker["second_arrived"].wait, 1)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task).response.status_code == 504
            assert time.monotonic() - started < 1.2
            for _ in range(50):
                state = router.runtime.health()
                if state == dict(pending=0, requests=0):
                    break
                await asyncio.sleep(0.01)
            assert state == dict(pending=0, requests=0)
            assert len(worker["requests"]) == 2
            assert worker["requests"][0][1] == worker["requests"][1][1]
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("state", ["cancelled", "cancelling", "conflict"])
def test_native_only_retries_explicitly_cancelled_replay(worker, state):
    worker["faults"] = ["headers", state]

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker)
            assert result.response.status_code == (409 if state == "conflict" else 200)
            ids = [body["rid"] for _, body in worker["requests"]]
            assert ids[0] == ids[1]
            assert len(ids) == (2 if state == "conflict" else 3)
            if state != "conflict":
                assert ids[2] != ids[1]
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_native_capacity_pressure_does_not_exhaust_failure_budget(worker):
    worker["statuses"] = [503] * 9 + [200]

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker, timeout=15)
            assert result.response.status_code == 200
            assert len(worker["requests"]) == 10
            assert len({body["rid"] for _, body in worker["requests"]}) == 10
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("policy", ["cache_wait"])
def test_native_busy_retries_obey_original_deadline(worker, policy):
    worker["statuses"] = [503] * 100

    async def run():
        router = make_router(worker, policy)
        started = time.monotonic()
        try:
            result = await dispatch(router, worker, timeout=1.5)
            assert result.response.status_code == 504
            assert result.replica is None
            assert 1 <= len(worker["requests"]) <= 5
            assert time.monotonic() - started < 2.5
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_native_permanent_worker_failure_still_has_a_retry_limit(worker):
    worker["statuses"] = [500] * 20

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker, timeout=30)
            assert result.response.status_code == 502
            assert result.routing["reason"] == "router_error"
            assert len(worker["requests"]) == 8
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("active,delay", [(24, 0), (0, 2)])
@pytest.mark.parametrize("policy", ["cache_wait"])
def test_native_cancellation_releases_admission_and_inflight(
    worker, active, delay, policy
):
    worker["active"] = active
    worker["delay"] = delay

    async def run():
        router = make_router(worker, policy)
        try:
            task = asyncio.create_task(dispatch(router, worker))
            if delay:
                assert await asyncio.to_thread(worker["arrived"].wait, 2)
            else:
                await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(50):
                state = router.runtime.health()
                if state == dict(pending=0, requests=0):
                    break
                await asyncio.sleep(0.01)
            assert state == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("policy", ["cache_wait"])
def test_native_deadline_bounds_capacity_wait(worker, policy):
    worker["active"] = 24

    async def run():
        router = make_router(worker, policy)
        try:
            result = await dispatch(router, worker, timeout=0.2)
            assert result.response.status_code == 504
            assert not worker["requests"]
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_native_admission_sees_every_concurrent_request(worker):
    worker["active"] = 24

    async def run():
        router = make_router(worker)
        try:
            tasks = [
                asyncio.create_task(dispatch(router, worker, id=f"queued-{i}"))
                for i in range(300)
            ]
            for _ in range(100):
                if router.runtime.health()["requests"] == 300:
                    break
                await asyncio.sleep(0.01)
            assert router.runtime.health() == dict(pending=0, requests=300)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for _ in range(50):
                if router.runtime.health() == dict(pending=0, requests=0):
                    break
                await asyncio.sleep(0.01)
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_native_invalid_payload_does_not_reserve(worker):
    async def run():
        router = make_router(worker)
        try:
            result = await router.dispatch(
                router.prepare_request("", ""),
                request_id="bad",
                gateway=worker["gateway"],
                body=dict(input_ids=[1], sampling_params=dict(max_new_tokens=1)),
                headers={},
                deadline=asyncio.get_running_loop().time() + 1,
            )
            assert result.response.status_code == 400
            assert result.replica is None
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_native_logical_conflict_is_returned_without_retry(worker):
    worker["statuses"] = [409]

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker)
            assert result.response.status_code == 409
            assert len(worker["requests"]) == 1
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_native_dispatch_does_not_serialize_worker_requests(worker):
    worker["delay"] = 0.3

    async def run():
        router = make_router(worker)
        try:
            results = await asyncio.gather(
                *(dispatch(router, worker, id=f"parallel-{i}") for i in range(4))
            )
            assert all(r.response.status_code == 200 for r in results)
            arrivals = worker["arrival_times"]
            # All requests must arrive before the first response finishes.
            assert max(arrivals) - min(arrivals) < worker["delay"]
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


def test_slow_worker_does_not_refresh_other_workers_observations(worker):
    worker["observations"] = {"fast": {"age": 2.4}, "slow": {"age": 100, "delay": 1.7}}

    async def run():
        router = make_router(worker)
        try:
            result = await dispatch(router, worker, timeout=2.8)
            assert result.response.status_code == 504
            assert worker["observation_paths"]
            assert not worker["requests"]
        finally:
            await router.aclose()

    asyncio.run(run())


def test_siblings_wait_for_shared_prefix_in_native_runtime(worker):
    from array import array
    import xxhash

    worker["observations"] = {"a": {}, "b": {}}
    worker["balance_absolute"] = 64
    worker["balance_relative"] = 2.0
    prompt = [42] * 129
    worker["publish_prefix"] = xxhash.xxh3_128(
        b"adapter/v1\0" + array("q", prompt[:128]).tobytes()
    ).hexdigest()
    worker["delay"] = 1.2

    async def run():
        router = make_router(worker)
        prepared = router.prepare_request("adapter/v1", "group")
        try:

            async def sample(index):
                return await router.dispatch(
                    prepared,
                    request_id=f"sibling-{index}",
                    gateway=worker["gateway"],
                    body=dict(input_ids=prompt, sampling_params=dict(max_new_tokens=4)),
                    headers={},
                    deadline=asyncio.get_running_loop().time() + 5,
                )

            results = await asyncio.gather(*(sample(i) for i in range(4)))
            assert all(r.response.status_code == 200 for r in results)
            assert len({r.replica for r in results}) == 1
            assert sorted(r.routing["cached_tokens"] for r in results) == [
                0,
                128,
                128,
                128,
            ]
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())
