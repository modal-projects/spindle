"""Load the validation adapter into the normal Spindle SGLang entrypoint."""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import modal
import numpy as np


def main():
    volume = modal.Volume.from_name("spindle-glm53-pr26-validation")
    volume.reload()
    report = json.loads(Path("/validation/latest.json").read_text())
    settings = json.loads(sys.argv[1])
    settings.update(
        tp_size=1,
        ep_size=1,
        quantization=None,
        dtype="bfloat16",
        context_length=256,
        max_running_requests=4,
        max_lora_rank=8,
        mem_fraction_static=0.7,
        disable_cuda_graph=True,
    )
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spindle.inference.sglang",
            report["model_path"],
            json.dumps(settings),
        ]
    )
    try:
        with httpx.Client(base_url="http://127.0.0.1:8001", timeout=120) as client:
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"SGLang exited with code {server.returncode}")
                try:
                    if client.get("/health", timeout=2).is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("SGLang did not become ready")
            response = client.post(
                "/load_lora_adapter",
                json={
                    "lora_name": "validation",
                    "lora_path": report["adapter_path"],
                },
            )
            response.raise_for_status()
            tokens = report["tokens"] + [report["tokens"][-1] + 1]
            request = {
                "input_ids": tokens,
                "return_logprob": True,
                "logprob_start_len": 0,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            }
            base = client.post("/generate", json=request)
            base.raise_for_status()
            adapted = client.post(
                "/generate", json={**request, "lora_path": "validation"}
            )
            adapted.raise_for_status()
            base_lp = np.array(
                [x[0] for x in base.json()["meta_info"]["input_token_logprobs"][1:]]
            )
            adapted_lp = np.array(
                [x[0] for x in adapted.json()["meta_info"]["input_token_logprobs"][1:]]
            )
            train_lp = np.array(report["logprobs"])
            assert np.isfinite(adapted_lp).all()
            assert train_lp.shape == adapted_lp.shape
            effect = float(np.abs(adapted_lp - base_lp).max())
            difference = float(np.abs(adapted_lp - train_lp).mean())
            assert (
                effect > 1e-5
            ), "The exported adapter did not change inference logprobs"
            assert (
                difference < 0.15
            ), f"Mean trainer/sampler logprob difference: {difference}"
            result = {
                "adapter_effect_max": effect,
                "train_sample_logprob_diff_mean": difference,
                "completion_tokens": adapted.json()["meta_info"]["completion_tokens"],
            }
            print(json.dumps(result), flush=True)
            Path("/validation/sampling.json").write_text(json.dumps(result))
            volume.commit()
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
