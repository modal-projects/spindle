"""Load the validation adapter into the normal Spindle SGLang entrypoint."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import modal
import numpy as np


def main():
    settings = json.loads(sys.argv[1])
    full_model = settings.pop("validation_full_model", False)
    measure_loading = settings.pop("validation_measure_loading", False)
    parallel = settings.pop("validation_parallel_size", 1)
    volume = modal.Volume.from_name(
        "spindle-glm53-pr26-full-validation"
        if full_model
        else "spindle-glm53-pr26-validation"
    )
    volume.reload()
    report = json.loads(Path("/validation/latest.json").read_text())
    if not full_model:
        settings.update(
            tokenizer_path="zai-org/GLM-5.3-Flash",
            tp_size=parallel,
            ep_size=parallel,
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
            load_start = time.monotonic()
            response = client.post(
                "/load_lora_adapter",
                json={
                    "lora_name": "validation",
                    "lora_path": report["adapter_path"],
                },
            )
            response.raise_for_status()
            registration_s = time.monotonic() - load_start
            tokens = report["tokens"] + [report["tokens"][-1] + 1]
            request = {
                "input_ids": tokens,
                "return_logprob": True,
                "logprob_start_len": 0,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            }
            base = client.post("/generate", json=request)
            base.raise_for_status()
            adapted_start = time.monotonic()
            adapted = client.post(
                "/generate", json={**request, "lora_path": "validation"}
            )
            adapted.raise_for_status()
            first_sample_s = time.monotonic() - adapted_start
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
                "registration_s": registration_s,
                "first_adapter_sample_s": first_sample_s,
                "adapter_effect_max": effect,
                "train_sample_logprob_diff_mean": difference,
                "completion_tokens": adapted.json()["meta_info"]["completion_tokens"],
            }
            if measure_loading:
                first_token = threading.Event()
                token_times = []

                def stream_generation():
                    with httpx.Client(
                        base_url="http://127.0.0.1:8001", timeout=300
                    ) as stream_client:
                        with stream_client.stream(
                            "POST",
                            "/generate",
                            json={
                                "input_ids": tokens,
                                "lora_path": "validation",
                                "stream": True,
                                "sampling_params": {
                                    "max_new_tokens": 128,
                                    "temperature": 0,
                                    "ignore_eos": True,
                                },
                            },
                        ) as stream:
                            stream.raise_for_status()
                            for line in stream.iter_lines():
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    chunk = json.loads(line[6:])
                                    if chunk.get("output_ids"):
                                        token_times.append(
                                            (time.monotonic(), len(chunk["output_ids"]))
                                        )
                                        first_token.set()
                    return len(token_times)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    stream = executor.submit(stream_generation)
                    assert first_token.wait(120), "Streaming generation did not start"
                    begin = time.monotonic()
                    loaded = client.post(
                        "/load_lora_adapter",
                        json={
                            "lora_name": "validation-next",
                            "lora_path": report["adapter_path"],
                        },
                    )
                    loaded.raise_for_status()
                    end = time.monotonic()
                    stream.result(timeout=300)
                result["registration_during_decode_s"] = end - begin
                result["decode_chunks_during_registration"] = sum(
                    begin < t < end for t, _ in token_times
                )
                result["decode_max_chunk_gap_s"] = max(
                    (b[0] - a[0] for a, b in zip(token_times, token_times[1:])),
                    default=0,
                )
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
