"""Live SGLang regression; set SPINDLE_TEST_SGLANG_URL and SPINDLE_TEST_LORA_PATH."""

import asyncio
import os
from pathlib import Path

import httpx
import pytest
from stitch.types import VersionRef

from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app


SGLANG_URL = os.environ.get("SPINDLE_TEST_SGLANG_URL")
LORA_PATH = os.environ.get("SPINDLE_TEST_LORA_PATH")
pytestmark = pytest.mark.skipif(
    not SGLANG_URL or not LORA_PATH, reason="requires a live LoRA-enabled SGLang server"
)


class LoseLoadAcknowledgement(httpx.AsyncHTTPTransport):
    def __init__(self):
        super().__init__()
        self.loads = []

    async def handle_async_request(self, request):
        response = await super().handle_async_request(request)
        if request.url.path == "/load_lora_adapter":
            await response.aread()
            self.loads.append((response.status_code, response.json()))
            if len(self.loads) == 1:
                assert response.status_code == 200, response.text
                await response.aclose()
                raise httpx.ReadError("load acknowledgement lost", request=request)
        return response


def test_lost_acknowledgement_recovers_and_generates_with_real_adapter(tmp_path):
    async def scenario():
        bulletin = SnapshotBulletin(tmp_path / "bulletin")
        ref = VersionRef(f"lost-ack-{tmp_path.name}", 1)
        bulletin.publish(ref, Path(LORA_PATH))
        transport = LoseLoadAcknowledgement()
        app = create_app(bulletin, SGLANG_URL, transport=transport)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
            httpx.AsyncClient(base_url=SGLANG_URL, timeout=60) as engine,
        ):
            payload = {
                "rid": "lost-ack",
                "input_ids": [1, 2, 3, 4],
                "weight_run_id": ref.run_id,
                "weight_version": {"exact_version": 1},
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            }
            try:
                with pytest.raises(httpx.ReadError):
                    await client.post("/generate", json=payload)
                models = (await engine.get("/v1/models")).json()["data"]
                assert any(
                    m["id"] == ref.identity and m["root"] == str(bulletin.resolve(ref))
                    for m in models
                )
                recovered = await client.post("/generate", json=payload)
                assert recovered.status_code == 200, recovered.text
                assert len(transport.loads) == 2
                assert transport.loads[1][0] == 400
                assert "already loaded" in transport.loads[1][1]["error_message"]
                result = recovered.json()
                assert len(result["output_ids"]) > 0
                assert result["meta_info"]["weight_version_start"] == 1
                direct = await engine.post(
                    "/generate",
                    json={
                        "rid": "direct-control",
                        "input_ids": payload["input_ids"],
                        "lora_path": ref.identity,
                        "sampling_params": payload["sampling_params"],
                    },
                )
                assert direct.status_code == 200, direct.text
                assert result["output_ids"] == direct.json()["output_ids"]
                replay = await client.post("/generate", json=payload)
                assert replay.json() == result
                assert len(transport.loads) == 2
                print(
                    "Recovered real duplicate load and matched direct greedy adapter output"
                )
            finally:
                await engine.post(
                    "/unload_lora_adapter", json={"lora_name": ref.identity}
                )

    asyncio.run(scenario())
