import asyncio

import httpx
import pytest

from spindle.inference.dlora_scheduler import (
    MergeProfile,
    ReplicaBusy,
    ReplicaScheduler,
)


def profile(**overrides):
    return MergeProfile(
        **(
            dict(
                unmerged_token_seconds=0.02,
                merged_token_seconds=0.01,
                prefill_tokens_per_second=10000,
                merge_seconds=0.01,
                unmerge_seconds=0.01,
                capacity=4,
                queue_buffer=1,
            )
            | overrides
        )
    )


def test_merge_drain_version_switch_and_exact_adapter_requests():
    async def run():
        mode = None
        events = []
        release = asyncio.Event()
        entered = asyncio.Event()

        async def engine(request):
            nonlocal mode
            events.append(request.url.path)
            if request.url.path == "/merge_lora_weights":
                mode = "a/v1"
            elif request.url.path == "/unmerge_lora_weights":
                assert release.is_set()
                mode = None
            return httpx.Response(200, json={"success": True})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(engine), base_url="http://engine"
        ) as client:
            scheduler = ReplicaScheduler(
                client, profile(), invalidate=lambda: events.append("flush")
            )

            async def prepare():
                events.append("prepare")

            async def first():
                assert mode == "a/v1"
                entered.set()
                await release.wait()
                return "a/v1"

            async def second():
                assert mode is None
                return "a/v2"

            tasks = [
                asyncio.create_task(scheduler.submit("a/v1", 8, 100, prepare, first))
                for _ in range(4)
            ]
            await asyncio.wait_for(entered.wait(), 1)
            next_version = asyncio.create_task(
                scheduler.submit("a/v2", 8192, 1, prepare, second)
            )
            await asyncio.sleep(0.01)
            with pytest.raises(ReplicaBusy):
                await scheduler.submit("b/v1", 8, 1, prepare, second)
            assert "/unmerge_lora_weights" not in events
            release.set()
            assert await asyncio.gather(*tasks, next_version) == ["a/v1"] * 4 + ["a/v2"]
            assert events.count("flush") == 2
            await scheduler.aclose()

    asyncio.run(run())


def test_warm_long_prefix_rebuild_can_make_merge_unprofitable():
    async def run():
        events = []

        async def engine(request):
            events.append(request.url.path)
            return httpx.Response(200, json={"success": True})

        async def prepare():
            pass

        async def execute():
            return 1

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(engine), base_url="http://engine"
        ) as client:
            scheduler = ReplicaScheduler(client, profile(), invalidate=lambda: None)
            assert (
                await asyncio.gather(
                    *[
                        scheduler.submit("a", 8192, 32, prepare, execute)
                        for _ in range(4)
                    ]
                )
                == [1] * 4
            )
            assert events == []
            await scheduler.aclose()

    asyncio.run(run())


def test_first_merge_pays_cold_cost_then_uses_warm_cost():
    async def run():
        actions = []

        async def engine(request):
            actions.append(request.url.path)
            return httpx.Response(200, json={"success": True})

        async def prepare():
            pass

        async def execute():
            return 1

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(engine), base_url="http://engine"
        ) as client:
            scheduler = ReplicaScheduler(
                client, profile(initial_merge_seconds=2), invalidate=lambda: None
            )

            async def cohort(adapter, output):
                return await asyncio.gather(
                    *(
                        scheduler.submit(adapter, 8, output, prepare, execute)
                        for _ in range(4)
                    )
                )

            await cohort("a", 100)
            assert actions == []  # Warm cost would have justified this merge.
            await cohort("a", 400)
            assert actions == ["/merge_lora_weights"]
            await cohort("b", 1)
            assert scheduler.mode is None
            await cohort("a", 100)
            assert actions == [
                "/merge_lora_weights",
                "/unmerge_lora_weights",
                "/merge_lora_weights",
            ]
            await scheduler.aclose()

    asyncio.run(run())


def test_uncertain_merge_fails_closed_and_cancellation_releases_capacity():
    async def run():
        async def engine(request):
            return httpx.Response(500, json={"error": "failed transition"})

        async def prepare():
            pass

        async def execute():
            raise AssertionError("Generation must not run after a failed transition")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(engine), base_url="http://engine"
        ) as client:
            scheduler = ReplicaScheduler(
                client, profile(capacity=1), invalidate=lambda: None
            )
            with pytest.raises(RuntimeError, match="controller failed"):
                await scheduler.submit("a", 1, 100, prepare, execute)
            assert scheduler.failure
            await scheduler.aclose()

    asyncio.run(run())


def test_cancelled_pending_and_running_requests_release_local_admission():
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def prepare():
            pass

        async def execute():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async with httpx.AsyncClient(base_url="http://engine") as client:
            scheduler = ReplicaScheduler(
                client, profile(enabled=False, capacity=1), invalidate=lambda: None
            )
            first = asyncio.create_task(scheduler.submit("a", 1, 1, prepare, execute))
            await asyncio.wait_for(entered.wait(), 1)
            pending = asyncio.create_task(scheduler.submit("b", 1, 1, prepare, execute))
            await asyncio.sleep(0)
            assert scheduler.active == 2
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            assert scheduler.active == 1
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await asyncio.wait_for(cancelled.wait(), 1)
            assert scheduler.active == 0
            await scheduler.aclose()

    asyncio.run(run())
