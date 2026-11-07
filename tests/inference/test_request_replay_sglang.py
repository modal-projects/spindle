"""Live admission test: SPINDLE_TEST_SGLANG_URL must have running/queued limits of 1."""

import asyncio
import os
import time

import httpx
import pytest

from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app


SGLANG_URL = os.environ.get("SPINDLE_TEST_SGLANG_URL")
pytestmark = pytest.mark.skipif(
    not SGLANG_URL, reason="requires a dedicated live SGLang"
)


def test_real_queue_full_retries_same_id_before_replay_ttl(tmp_path):
    async def scenario():
        app = create_app(SnapshotBulletin(tmp_path), SGLANG_URL)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
            httpx.AsyncClient(base_url=SGLANG_URL, timeout=60) as engine,
        ):
            blockers = [
                asyncio.create_task(
                    engine.post(
                        "/generate",
                        json={
                            "rid": f"blocker-{i}",
                            "input_ids": [1, 2, 3, 4],
                            "sampling_params": {
                                "temperature": 0,
                                "ignore_eos": True,
                                "max_new_tokens": 8192,
                            },
                        },
                    )
                )
                for i in range(4)
            ]
            payload = None
            try:
                async with asyncio.timeout(10):
                    for attempt in range(100):
                        candidate = {
                            "rid": f"queue-retry-{attempt}",
                            "input_ids": [1, 2, 3, 4],
                            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                        }
                        response = await client.post("/generate", json=candidate)
                        if response.status_code == 503:
                            assert "queue is full" in response.text, response.text
                            payload = candidate
                            rejected_at = time.monotonic()
                            break
                        assert response.status_code == 200, response.text
                        await asyncio.sleep(0.02)
                assert payload is not None, "SGLang did not reject a full queue"
            finally:
                await asyncio.gather(
                    *(
                        engine.post("/abort_request", json={"rid": f"blocker-{i}"})
                        for i in range(4)
                    )
                )
                await asyncio.wait_for(asyncio.gather(*blockers), 10)

            async with asyncio.timeout(10):
                while True:
                    control = await engine.post(
                        "/generate", json={**payload, "rid": "capacity-control"}
                    )
                    if control.status_code == 200:
                        break
                    assert control.status_code == 503, control.text
                    await asyncio.sleep(0.02)
            recovered = await client.post("/generate", json=payload)
            assert recovered.status_code == 200, recovered.text
            assert len(recovered.json()["output_ids"]) > 0
            assert recovered.json()["output_ids"] == control.json()["output_ids"]
            elapsed = time.monotonic() - rejected_at
            assert elapsed < 30, f"Recovery took {elapsed}s; replay TTL is 30s"
            replay = await client.post("/generate", json=payload)
            assert replay.json() == recovered.json()
            print(
                f"Real queue rejection recovered with identical RID in {elapsed:.3f}s"
            )

    asyncio.run(scenario())
