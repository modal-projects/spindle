import asyncio
import json
import pytest
import socket
from unittest.mock import AsyncMock, patch

import httpx
from spindle.inference import sampling
from spindle.inference.sampling import sample_task
from spindle.inference.model_routing import ModelRouter, Replica


def response(
    version: int,
    token: int = 5,
    *,
    end_version: int | None = None,
    cached_tokens: int = 0,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "meta_info": {
                "finish_reason": "stop",
                "output_token_logprobs": [[-0.1, token]],
                "weight_version_start": version,
                "weight_version_end": (version if end_version is None else end_version),
                "cached_tokens": cached_tokens,
            }
        },
    )


def task(*, latest: bool = False) -> dict:
    return {
        "request_id": "request-a",
        "sampling_session_id": "sample-a",
        "model_id": "model-a",
        "publish_version": 7,
        "latest": latest,
        "payload": {
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
            "sampling_params": {"max_tokens": 1, "seed": 10},
        },
    }


def routing_pool():
    return ModelRouter(
        AsyncMock(
            return_value=[
                Replica(h, h + ":8000", 0, frozenset(), (), (), 0) for h in ("a", "b")
            ]
        ),
        capacity=8,
    )


@pytest.mark.parametrize("failure", [503, "version", "transport"])
def test_model_routing_retries_elsewhere_and_releases_reservations(
    monkeypatch, failure
):
    monkeypatch.setattr(sampling, "RETRY_INITIAL_DELAY_SECONDS", 0)

    async def scenario():
        router = routing_pool()
        attempts = []

        def handle(request):
            assert "Modal-Session-ID" not in request.headers
            assert "Modal-Routing-Affinity-Key" not in request.headers
            assert json.loads(request.content)["weight_version"]["exact_version"] == 7
            attempts.append(request.headers["Modal-Flash-Upstream"])
            if len(attempts) == 1:
                if failure == "transport":
                    raise httpx.ReadError("closed")
                if failure == "version":
                    return response(6)
                return httpx.Response(failure)
            return response(7)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
            headers={
                "Modal-Session-ID": "inherited",
                "Modal-Routing-Affinity-Key": "old",
            },
        ) as client:
            try:
                result = await sample_task(
                    task(), "http://rollout", client=client, model_router=router
                )
                assert result["sequences"][0]["tokens"] == [5]
                assert len(attempts) == 2 and attempts[0] != attempts[1]
                assert not router.reservations and not router.host_reservations
            finally:
                await router.aclose()

    asyncio.run(scenario())


def test_model_routing_releases_cancelled_generation():
    async def scenario():
        router = routing_pool()
        entered = asyncio.Event()

        async def handle(request):
            entered.set()
            await asyncio.Future()

        operation = asyncio.create_task(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(handle),
                model_router=router,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert len(router.reservations) == 1
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert not router.reservations and not router.host_reservations
        finally:
            await router.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("latest", [False, True])
def test_model_routing_falls_back_to_affinity_or_bypasses_latest(latest):
    async def scenario():
        router = routing_pool()
        router.discover.return_value = []

        def handle(request):
            assert request.headers["Modal-Session-ID"] == "request-a"
            assert "Modal-Flash-Upstream" not in request.headers
            return response(7)

        try:
            await sample_task(
                task(latest=latest),
                "http://rollout",
                transport=httpx.MockTransport(handle),
                model_router=router,
            )
            assert router.discover.await_count == (0 if latest else 1)
            assert not router.reservations
        finally:
            await router.aclose()

    asyncio.run(scenario())


def test_exact_sampling_pins_version() -> None:
    body = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return response(7)

    result = asyncio.run(
        sample_task(
            task(),
            "http://rollout",
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["weight_run_id"] == "model-a"
    assert body["weight_version"] == {"min_version": None, "exact_version": 7}
    assert result["sequences"][0]["tokens"] == [5]


@pytest.mark.parametrize("first", [None, [None, 1], {"logprob": None, "token_id": 1}])
def test_prompt_logprobs_preserve_undefined_first_token(first) -> None:
    request = task()
    request["payload"]["prompt_logprobs"] = True
    request["payload"]["prompt"]["chunks"][0]["tokens"] = [1, 2]
    body = response(7).json()
    body["meta_info"]["input_token_logprobs"] = [first, [-0.25, 2]]
    result = asyncio.run(
        sample_task(
            request,
            "http://rollout",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
        )
    )
    assert result["prompt_logprobs"] == [None, -0.25]


def test_grouped_sampling_shares_session_and_dp_rank() -> None:
    request = task()
    request["payload"]["num_samples"] = 4
    sessions = []
    bodies = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        sessions.append(http_request.headers["Modal-Session-ID"])
        bodies.append(json.loads(http_request.content))
        return response(7)

    result = asyncio.run(
        sample_task(
            request,
            "http://rollout",
            data_parallel_size=4,
            transport=httpx.MockTransport(handle),
        )
    )

    assert len(result["sequences"]) == 4
    assert sessions == ["request-a"] * 4
    assert len({body["routed_dp_rank"] for body in bodies}) == 1
    assert all("session_id" not in body for body in bodies)
    assert 0 <= bodies[0]["routed_dp_rank"] < 4
    assert {body["sampling_params"]["sampling_seed"] for body in bodies} == {
        10,
        11,
        12,
        13,
    }


def test_multiturn_affinity_tracks_prefix_cache_hit_rate() -> None:
    cached_sequences: dict[tuple[str, int], list[int]] = {}
    sessions = []
    bodies = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        session_id = http_request.headers["Modal-Session-ID"]
        body = json.loads(http_request.content)
        route = (session_id, body["routed_dp_rank"])
        prompt = body["input_ids"]
        previous = cached_sequences.get(route, [])
        cached_tokens = 0
        for actual, expected in zip(prompt, previous, strict=False):
            if actual != expected:
                break
            cached_tokens += 1
        output_token = 100 + len(bodies)
        cached_sequences[route] = [*prompt, output_token]
        sessions.append(session_id)
        bodies.append(body)
        return response(7, token=output_token, cached_tokens=cached_tokens)

    prompt = [11, 12, 13, 14]
    prompt_lengths = []
    cache_hits = []
    expected_reusable = []
    for turn in range(4):
        request = task()
        request["request_id"] = f"sample-a:{turn}"
        request["payload"]["cache_affinity_key"] = "trajectory-a"
        request["payload"]["prompt"]["chunks"][0]["tokens"] = list(prompt)
        result = asyncio.run(
            sample_task(
                request,
                "http://rollout",
                data_parallel_size=4,
                transport=httpx.MockTransport(handle),
            )
        )
        prompt_lengths.append(len(prompt))
        cache_hits.append(result["prompt_cache_hit_tokens"])
        if turn:
            expected_reusable.append(len(prompt) - 1)
        prompt.extend(result["sequences"][0]["tokens"])
        prompt.append(20 + turn)

    assert len(set(sessions)) == 1
    assert len({body["routed_dp_rank"] for body in bodies}) == 1
    assert {body["session_id"] for body in bodies} == set(sessions)
    assert cache_hits == [0, *expected_reusable]
    eligible_prefix_hit_rate = sum(cache_hits[1:]) / sum(expected_reusable)
    assert eligible_prefix_hit_rate == 1.0
    assert sum(cache_hits) / sum(prompt_lengths) > 0.7


# A chat-shaped transcript: messages end with END, assistant turns open with HEADER.
END, START, ASSISTANT, TOOL = 200007, 200006, 173781, 99
HEADER = [END, START, ASSISTANT]


def conversation(question: list[int], replies: list[list[int]]) -> list[list[int]]:
    """Prompts of each turn: system + question, then reply and tool result per turn."""
    prompt = [7] * 20 + question + HEADER
    prompts = [list(prompt)]
    for reply in replies:
        prompt += [*reply, END, START, TOOL, 42, *HEADER]
        prompts.append(list(prompt))
    return prompts


def route_sessions(prompts: list[list[int]], sampling_session_id: str = "sample-a"):
    sessions = []

    def handle(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["Modal-Session-ID"])
        return response(7)

    for turn, prompt in enumerate(prompts):
        request = task()
        request["request_id"] = f"turn-{turn}"
        request["sampling_session_id"] = sampling_session_id
        request["payload"]["prompt"]["chunks"][0]["tokens"] = prompt
        asyncio.run(
            sample_task(
                request, "http://rollout", transport=httpx.MockTransport(handle)
            )
        )
    return sessions


def test_conversation_turns_share_a_route_without_affinity_key() -> None:
    first = route_sessions(conversation([1, 2], [[30, 31], [32], [33, 34]]))
    # The same question with a different first reply is a different conversation.
    second = route_sessions(conversation([1, 2], [[40], [32], [33, 34]]))

    assert first[0] == second[0] == "turn-0"
    assert len(set(first[1:])) == 1 and first[1].startswith("conversation-")
    assert len(set(second[1:])) == 1 and second[1] != first[1]


def test_single_turn_prompts_keep_request_routing() -> None:
    prompts = [[7] * 20 + [n] + HEADER for n in range(3)] + [[1, 2, 3]]
    assert route_sessions(prompts) == ["turn-0", "turn-1", "turn-2", "turn-3"]


def test_explicit_affinity_key_overrides_conversation_route() -> None:
    prompt = conversation([1, 2], [[30]])[1]
    sessions = []

    def handle(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["Modal-Session-ID"])
        return response(7)

    request = task()
    request["payload"]["cache_affinity_key"] = "trajectory-a"
    request["payload"]["prompt"]["chunks"][0]["tokens"] = prompt
    asyncio.run(
        sample_task(request, "http://rollout", transport=httpx.MockTransport(handle))
    )
    assert sessions[0].startswith("affinity-")


def test_affinity_route_is_namespaced_by_sampling_session_and_key() -> None:
    sessions = []

    def handle(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["Modal-Session-ID"])
        return response(7)

    for sampling_session_id, affinity_key in (
        ("sample-a", "trajectory-a"),
        ("sample-b", "trajectory-a"),
        ("sample-a", "trajectory-b"),
    ):
        request = task()
        request["sampling_session_id"] = sampling_session_id
        request["payload"]["cache_affinity_key"] = affinity_key
        asyncio.run(
            sample_task(
                request,
                "http://rollout",
                transport=httpx.MockTransport(handle),
            )
        )

    assert len(set(sessions)) == 3
    assert all(session.startswith("affinity-") for session in sessions)


@pytest.mark.parametrize("key", ["", "   ", 42, "x" * 257])
def test_invalid_cache_affinity_key_is_rejected(key) -> None:
    request = task()
    request["payload"]["cache_affinity_key"] = key
    with pytest.raises(ValueError, match="cache_affinity_key"):
        asyncio.run(sample_task(request, "http://rollout"))


def test_missing_max_tokens_defaults_to_remaining_context() -> None:
    request = task()
    request["payload"]["prompt"]["chunks"][0]["tokens"] = [1, 2, 3]
    request["payload"]["sampling_params"] = {"temperature": 0.7}
    body = {}

    def handle(http_request: httpx.Request) -> httpx.Response:
        body.update(json.loads(http_request.content))
        return response(7)

    asyncio.run(
        sample_task(
            request,
            "http://rollout",
            context_length=10,
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["sampling_params"]["max_new_tokens"] == 6


def test_explicit_max_tokens_wins_over_context_default() -> None:
    body = {}

    def handle(http_request: httpx.Request) -> httpx.Response:
        body.update(json.loads(http_request.content))
        return response(7)

    asyncio.run(
        sample_task(
            task(),
            "http://rollout",
            context_length=10,
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["sampling_params"]["max_new_tokens"] == 1


def test_prompt_filling_context_is_rejected() -> None:
    request = task()
    request["payload"]["prompt"]["chunks"][0]["tokens"] = [1, 2, 3]
    request["payload"]["sampling_params"] = {}

    with pytest.raises(ValueError, match="no room"):
        asyncio.run(
            sample_task(
                request,
                "http://rollout",
                context_length=3,
                transport=httpx.MockTransport(lambda _: response(7)),
            )
        )


def test_latest_sampling_sets_version_floor() -> None:
    body = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return response(8)

    asyncio.run(
        sample_task(
            task(latest=True),
            "http://rollout",
            transport=httpx.MockTransport(handle),
        )
    )

    assert body["weight_version"] == {"min_version": 7, "exact_version": None}


def test_latest_sampling_accepts_in_place_version_advance() -> None:
    result = asyncio.run(
        sample_task(
            task(latest=True),
            "http://rollout",
            transport=httpx.MockTransport(lambda _: response(7, end_version=8)),
        )
    )

    assert result["sequences"][0]["tokens"] == [5]


def test_exact_sampling_retries_in_place_version_advance() -> None:
    responses = iter((response(7, end_version=8), response(7)))
    sleep = AsyncMock()
    with patch("spindle.inference.sampling.asyncio.sleep", sleep):
        result = asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(lambda _: next(responses)),
            )
        )

    sleep.assert_awaited_once()
    assert result["sequences"][0]["tokens"] == [5]


def test_sampling_backs_off_version_conflicts() -> None:
    statuses = iter((409, 200))

    def handle(_: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return response(7) if status == 200 else httpx.Response(status)

    sleep = AsyncMock()
    with (
        patch("spindle.inference.sampling.asyncio.sleep", sleep),
        patch("spindle.inference.sampling.random.uniform", return_value=1),
    ):
        result = asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(handle),
            )
        )

    sleep.assert_awaited_once_with(1.0)
    assert result["sequences"][0]["tokens"] == [5]


def test_sampling_reroutes_immediate_overload_response() -> None:
    sessions = []

    def handle(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["Modal-Session-ID"])
        assert request.url.path == "/generate"
        if len(sessions) == 1:
            return httpx.Response(503, text="queue full")
        return response(7)

    with (
        patch(
            "spindle.inference.sampling.asyncio.sleep",
            AsyncMock(),
        ),
        patch("spindle.inference.sampling.random.uniform", return_value=1),
    ):
        result = asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(handle),
            )
        )

    assert sessions == ["request-a", "request-a:retry-1"]
    assert result["sequences"][0]["tokens"] == [5]


def test_sampling_sends_extra_headers() -> None:
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return response(7)

    asyncio.run(
        sample_task(
            task(),
            "http://rollout",
            transport=httpx.MockTransport(handle),
            api_key="key",
            headers={"Modal-Key": "wk-a", "Modal-Secret": "ws-a"},
        )
    )

    assert seen["modal-key"] == "wk-a"
    assert seen["modal-secret"] == "ws-a"
    assert seen["x-api-key"] == "key"


def test_sampling_reports_saturation_and_touches_while_waiting() -> None:
    on_wait = AsyncMock()
    with (
        patch("spindle.inference.sampling.asyncio.sleep", AsyncMock()),
        pytest.raises(RuntimeError, match=r"rollout saturated after 0s: HTTP 503"),
    ):
        asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                retry_timeout=0,
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(503, text="queue full")
                ),
                on_wait=on_wait,
            )
        )

    on_wait.assert_not_awaited()
    responses = iter((httpx.Response(503, text="queue full"), response(7)))
    with patch("spindle.inference.sampling.asyncio.sleep", AsyncMock()):
        asyncio.run(
            sample_task(
                task(),
                "http://rollout",
                transport=httpx.MockTransport(lambda _: next(responses)),
                on_wait=on_wait,
            )
        )

    on_wait.assert_awaited_once()


def test_exact_sampling_rejects_wrong_response_version() -> None:
    hosts = []

    async def gateways() -> tuple[str, ...]:
        return ("http://replica-a", "http://replica-b")

    def handle(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return response(6 if request.url.host == "replica-a" else 7)

    with patch(
        "spindle.inference.sampling.asyncio.sleep",
        AsyncMock(),
    ):
        result = asyncio.run(
            sample_task(
                task(),
                gateways,
                transport=httpx.MockTransport(handle),
            )
        )

    assert hosts == ["replica-a", "replica-b"]
    assert result["sequences"][0]["tokens"] == [5]


def test_default_transport_enables_tcp_keepalive() -> None:
    transport = sampling.keepalive_transport()
    assert transport._pool._socket_options == list(sampling.TCP_KEEPALIVE_OPTIONS)
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in sampling.TCP_KEEPALIVE_OPTIONS

    with patch.object(
        sampling,
        "keepalive_transport",
        return_value=httpx.MockTransport(lambda _: response(7)),
    ) as default_transport:
        asyncio.run(sample_task(task(), "http://rollout"))
    default_transport.assert_called_once_with()


def test_dynamic_route_wait_refreshes_demand_and_retries_deleted_gateway():
    async def check():
        lookups, requests, events = [], [], []

        async def resolve():
            lookups.append(True)  # The scoped resolver refreshes demand here.
            if len(lookups) == 1:
                return ()
            return ("http://old",) if len(lookups) == 2 else ("http://new",)

        def handle(request):
            requests.append(request)
            return httpx.Response(404) if request.url.host == "old" else response(7)

        request = task()
        request["payload"]["cache_affinity_key"] = "trajectory-a"
        with patch("spindle.inference.sampling.asyncio.sleep", AsyncMock()):
            result = await sample_task(
                request,
                resolve,
                headers={"Modal-Key": "proxy-id", "Modal-Secret": "proxy-secret"},
                data_parallel_size=4,
                on_event=events.append,
                transport=httpx.MockTransport(handle),
            )
        assert [r.url.host for r in requests] == ["old", "new"]
        assert len(lookups) == 3
        assert result["sequences"][0]["tokens"] == [5]
        assert all(r.headers["Modal-Key"] == "proxy-id" for r in requests)
        assert all(r.headers["Modal-Secret"] == "proxy-secret" for r in requests)
        bodies = [json.loads(r.content) for r in requests]
        assert len({b["session_id"] for b in bodies}) == 1
        assert len({b["routed_dp_rank"] for b in bodies}) == 1
        assert all(
            r.headers["Modal-Session-ID"] == b["session_id"]
            for r, b in zip(requests, bodies, strict=True)
        )
        attempts = [e["attrs"] for e in events if e["name"] == "sample_attempt"]
        assert [a["http_status"] for a in attempts] == [404, 200]
        assert [a["ok"] for a in attempts] == [False, True]

    asyncio.run(check())


def test_transport_error_on_nonfirst_sample_is_observable(capsys):
    request = task()
    request["payload"]["num_samples"] = 2
    attempts = {}

    def handle(http_request):
        seed = json.loads(http_request.content)["sampling_params"]["sampling_seed"]
        attempts[seed] = attempts.get(seed, 0) + 1
        if seed == 11 and attempts[seed] == 1:
            raise httpx.ReadError("connection reset", request=http_request)
        return response(7)

    with patch("spindle.inference.sampling.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            sample_task(
                request, "http://rollout", transport=httpx.MockTransport(handle)
            )
        )
    assert len(result["sequences"]) == 2
    logs = capsys.readouterr().out
    assert "sample=1 attempt=1 elapsed_seconds=" in logs
    assert "reason=ReadError: connection reset" in logs
    assert "sample=1 attempts=1 transport_retries=1" in logs


def test_aborted_generation_is_not_returned_as_successful_rollout():
    def handle(request):
        body = response(7).json()
        body["meta_info"]["finish_reason"] = {"type": "abort"}
        return httpx.Response(200, json=body)

    with pytest.raises(RuntimeError, match="aborted generation"):
        asyncio.run(
            sample_task(task(), "http://rollout", transport=httpx.MockTransport(handle))
        )


def test_empty_stop_list_disables_eos():
    seen = []
    request = task()
    request["payload"]["sampling_params"]["stop"] = []

    def handle(http_request):
        seen.append(json.loads(http_request.content)["sampling_params"])
        return response(7)

    asyncio.run(
        sample_task(request, "http://rollout", transport=httpx.MockTransport(handle))
    )
    assert seen[0]["ignore_eos"] is True
    assert "stop" not in seen[0]


def test_tokenizerless_sampler_rejects_stop_strings():
    request = task()
    request["payload"]["sampling_params"]["stop"] = ["</answer>"]

    def handle(http_request):
        raise AssertionError("stop strings must not reach a tokenizer-less engine")

    with pytest.raises(ValueError, match="token IDs"):
        asyncio.run(
            sample_task(
                request,
                "http://rollout",
                stop_token_ids_only=True,
                transport=httpx.MockTransport(handle),
            )
        )


def test_tokenizerless_sampler_accepts_stop_token_ids():
    seen = []
    request = task()
    request["payload"]["sampling_params"]["stop"] = [7, 9]

    def handle(http_request):
        seen.append(json.loads(http_request.content)["sampling_params"])
        return response(7)

    asyncio.run(
        sample_task(
            request,
            "http://rollout",
            stop_token_ids_only=True,
            transport=httpx.MockTransport(handle),
        )
    )
    assert seen[0]["stop_token_ids"] == [7, 9]


def test_large_http1_group_dispatches_before_any_generation_finishes():
    async def run():
        arrived, release = asyncio.Event(), asyncio.Event()
        count = 0
        body = response(7).content

        async def engine(reader, writer):
            nonlocal count
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            count += 1
            if count == 128:
                arrived.set()
            await release.wait()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + body
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(engine, "127.0.0.1", 0, backlog=256)
        async with server:
            port = server.sockets[0].getsockname()[1]
            request = task()
            request["payload"]["num_samples"] = 128
            samples = asyncio.create_task(
                sample_task(request, f"http://127.0.0.1:{port}", timeout=10)
            )
            try:
                # The old transport admits only 100 requests and deadlocks on
                # this barrier until the test releases their responses.
                await asyncio.wait_for(arrived.wait(), 5)
            finally:
                release.set()
                result = await asyncio.wait_for(samples, 10)
            assert len(result["sequences"]) == count == 128

    asyncio.run(run())
