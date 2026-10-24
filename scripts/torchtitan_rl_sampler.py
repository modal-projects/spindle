"""SGLang half of the bounded TorchTitan RL integration experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
import httpx
import modal
import uvicorn

from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app


def main():
    checkpoint = snapshot_download(
        os.environ["RL_MODEL"],
        revision=os.environ.get("RL_MODEL_REVISION"),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    options = {
        "dtype": "bfloat16",
        "language_only": True,
        "context_length": 2048,
        "mem_fraction_static": 0.75,
        "max_running_requests": 8,
        "enable_lora": True,
        "max_lora_rank": 16,
        "max_loras_per_batch": 4,
        "max_loaded_loras": 32,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "disable_cuda_graph": True,
    }
    print("SGLANG_OPTIONS=" + json.dumps(options), flush=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spindle.inference.sglang",
            checkpoint,
            json.dumps(options),
        ]
    )
    try:
        deadline = time.monotonic() + 900
        with httpx.Client(timeout=5) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"SGLang exited with {process.returncode}")
                try:
                    if client.get("http://127.0.0.1:8001/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("SGLang startup exceeded 900 seconds")
        volume = modal.Volume.from_name(os.environ["SPINDLE_BULLETIN_VOLUME"])
        bulletin = SnapshotBulletin("/bulletin", refresh=volume.reload)
        app = create_app(bulletin, "http://127.0.0.1:8001")

        @app.middleware("http")
        async def authorize(request, call_next):
            if request.headers.get("x-api-key") != os.environ["RL_API_KEY"]:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

        @app.get("/validation/manifests")
        async def manifests():
            await bulletin.refresh()
            return [
                json.loads(p.read_text())
                for p in Path("/bulletin").rglob("snapshot.json")
            ]

        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
