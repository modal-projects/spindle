"""Real HTTP integration tests against Rust's runtime, without Modal/GPU access."""

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest

from spindle.inference.native_router import NativeRouter


@pytest.fixture
def worker():
    state = dict(
        active=0, requests=[], header_items=[], statuses=[], faults=[], delay=0
    )
    state["arrived"] = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def replica(self):
            return self.headers.get("Modal-Flash-Upstream", "worker:8000").split(":")[0]

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
            replica = self.replica()
            observation = state.get("observations", {}).get(replica, {})
            time.sleep(observation.get("delay", 0))
            self.reply(
                dict(
                    replica=replica,
                    active=state["active"],
                    work=dict(
                        age_seconds=observation.get("age", 0),
                        prefixes=[],
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
            state["header_items"].append(self.headers.items())
            state["arrived"].set()
            time.sleep(state["delay"])
            fault = state["faults"].pop(0) if state["faults"] else None
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
            if fault in ("cancelled", "conflict"):
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
                replica=self.replica(),
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["gateway"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def with_router(worker, test, *, headers=None):
    """Run `test(router)`, then require every reservation to be released."""

    async def run():
        router = NativeRouter(
            AsyncMock(
                return_value=[
                    dict(replica=r, upstream=f"{r}:8000")
                    for r in worker.get("observations", {"worker": {}})
                ]
            ),
            gateway=worker["gateway"],
            headers=headers or {},
            policy="cache_wait",
            capacity=24,
            prefill_tokens_per_second=1000.0,
        )
        try:
            await test(router)
            # Cancellation releases from a Rust destructor shortly after the await.
            for _ in range(50):
                if router.runtime.health() == dict(pending=0, requests=0):
                    break
                await asyncio.sleep(0.01)
            assert router.runtime.health() == dict(pending=0, requests=0)
        finally:
            await router.aclose()

    asyncio.run(run())


async def dispatch(router, worker, timeout=3, request_id="test", headers=None):
    return await router.dispatch(
        router.prepare_request("adapter/v1", "session"),
        request_id=request_id,
        gateway=worker["gateway"],
        body=dict(input_ids=[1, 2], sampling_params=dict(max_new_tokens=4)),
        headers={"Modal-Session-ID": "session", **(headers or {})},
        deadline=asyncio.get_running_loop().time() + timeout,
    )


def rids(worker):
    return [body["rid"] for _, body in worker["requests"]]


def test_a_worker_error_is_returned_without_cooling_the_replica(worker):
    worker["statuses"] = [500, 200]

    async def test(router):
        failed = await dispatch(router, worker, request_id="first")
        assert failed.response.status_code == 500
        assert failed.replica == "worker"
        started = time.monotonic()
        result = await dispatch(router, worker, request_id="second", timeout=1)
        assert result.response.status_code == 200
        assert time.monotonic() - started < 0.5
        assert len(worker["requests"]) == 2
        for headers, _ in worker["requests"]:
            headers = {k.lower(): v for k, v in headers.items()}
            assert headers["modal-flash-upstream"] == "worker:8000"
            assert "modal-session-id" not in headers

    with_router(worker, test)


def test_caller_headers_replace_runtime_headers(worker):
    async def test(router):
        await dispatch(router, worker, headers={"Modal-Key": "caller"})
        [items] = worker["header_items"]
        assert [v for k, v in items if k.lower() == "modal-key"] == ["caller"]

    with_router(worker, test, headers={"Modal-Key": "runtime"})


@pytest.mark.parametrize("fault", ["headers", "body", "proxy"])
def test_lost_response_is_redelivered_with_the_same_rid(worker, fault):
    worker["faults"] = [fault]

    async def test(router):
        assert (await dispatch(router, worker)).response.status_code == 200
        assert worker["requests"][0] == worker["requests"][1]

    with_router(worker, test)


@pytest.mark.parametrize("fault", ["headers", "proxy"])
def test_transport_failures_have_a_retry_limit(worker, fault):
    worker["faults"] = [fault] * 30

    async def test(router):
        result = await dispatch(router, worker, timeout=30)
        assert result.response.status_code == 502
        if fault == "proxy":
            # The last upstream response, not a router-made error.
            assert "upstream reset" in result.response.text
        else:
            assert result.routing["reason"] == "router_error"
        # Eight attempts, each delivered twice with one RID.
        ids = rids(worker)
        assert len(ids) == 16 and len(set(ids)) == 8
        assert all(ids[i] == ids[i + 1] for i in range(0, 16, 2))

    with_router(worker, test)


def test_wrong_worker_identity_is_not_retried(worker):
    worker["faults"] = ["mismatch"]

    async def test(router):
        result = await dispatch(router, worker)
        assert result.response.status_code == 502
        assert "identity mismatch" in result.response.text
        assert len(worker["requests"]) == 1

    with_router(worker, test)


@pytest.mark.parametrize(
    "state,status,requests", [("cancelled", 200, 3), ("conflict", 409, 2)]
)
def test_only_a_cancelled_replay_starts_a_new_attempt(worker, state, status, requests):
    worker["faults"] = ["headers", state]

    async def test(router):
        assert (await dispatch(router, worker)).response.status_code == status
        ids = rids(worker)
        assert len(ids) == requests
        assert ids[0] == ids[1]
        assert len(set(ids)) == requests - 1

    with_router(worker, test)


def test_busy_responses_do_not_spend_the_failure_budget(worker):
    worker["statuses"] = [503] * 9 + [200]

    async def test(router):
        assert (await dispatch(router, worker, timeout=15)).response.status_code == 200
        assert len(set(rids(worker))) == 10

    with_router(worker, test)


@pytest.mark.parametrize("active,statuses", [(24, []), (0, [503] * 100)])
def test_deadline_bounds_capacity_waits_and_busy_retries(worker, active, statuses):
    worker["active"] = active
    worker["statuses"] = statuses

    async def test(router):
        started = time.monotonic()
        result = await dispatch(router, worker, timeout=1)
        assert result.response.status_code == 504
        assert time.monotonic() - started < 2
        assert bool(worker["requests"]) == bool(statuses)

    with_router(worker, test)


@pytest.mark.parametrize("active,delay", [(24, 0), (0, 2)])
def test_cancellation_releases_queued_and_in_flight_requests(worker, active, delay):
    worker["active"] = active
    worker["delay"] = delay

    async def test(router):
        task = asyncio.create_task(dispatch(router, worker))
        if delay:
            assert await asyncio.to_thread(worker["arrived"].wait, 2)
        else:
            await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with_router(worker, test)


def test_a_slow_peer_does_not_stale_a_fast_replica(worker):
    # Fast's row stays routable only if installed before the slow poll times out.
    worker["observations"] = {"fast": {"age": 1.5}, "slow": {"delay": 2.5}}

    async def test(router):
        result = await dispatch(router, worker, timeout=2)
        assert result.response.status_code == 200
        assert result.replica == "fast"

    with_router(worker, test)


@pytest.mark.parametrize("observations", [{}, {"stale": {"age": 100}}])
def test_unroutable_pools_fall_back_to_the_gateway(worker, observations):
    # No workers (a pool at zero replicas) or no fresh observations (a pool
    # without an observer): Modal's gateway picks or cold-starts a replica.
    worker["observations"] = observations

    async def test(router):
        started = time.monotonic()
        result = await dispatch(router, worker, timeout=10)
        assert result.response.status_code == 200
        assert result.routing["reason"] == "gateway_fallback"
        assert time.monotonic() - started < (1 if not observations else 7)
        [(headers, _)] = worker["requests"]
        headers = {k.lower(): v for k, v in headers.items()}
        assert "modal-flash-upstream" not in headers
        assert headers["modal-session-id"] == "session"

    with_router(worker, test)
