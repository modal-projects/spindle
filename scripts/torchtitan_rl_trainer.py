"""Real SDK RL loops and numerical checks against the SGLang sampler."""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

from huggingface_hub import snapshot_download
import httpx
import tinker
from tinker import types
from transformers import AutoTokenizer
import uvicorn

from spindle.backends.torchtitan_lora import build_executor
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.sampling import sample_task
from spindle.providers.local import (
    InMemoryKeyValueStore,
    InMemorySessionKeyValueStores,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)


@contextmanager
def serve(app, port):
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("local HTTP server failed to start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def rl_datum(prompt, sequence, reward):
    tokens = prompt + list(sequence.tokens)
    prefix = len(prompt) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "logprobs": [0.0] * prefix + list(sequence.logprobs),
            "advantages": [0.0] * prefix
            + [reward / len(sequence.tokens)] * len(sequence.tokens),
        },
    )


def difference(left, right):
    assert len(left) == len(right)
    errors = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return {"max_absolute": max(errors), "mean_absolute": sum(errors) / len(errors)}


async def run_clients(base_url, result):
    model = os.environ["RL_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=os.environ.get("RL_MODEL_REVISION")
    )
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Pick one letter, A or B. Reply with only that letter.",
            }
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    probes = [
        prompt + tokenizer.encode(letter, add_special_tokens=False)
        for letter in ("A", "B")
    ]
    result["prompt_tokens"] = prompt
    result["probe_tokens"] = probes
    result["clients"] = []
    service = tinker.ServiceClient(base_url=base_url, api_key=os.environ["RL_API_KEY"])
    headers = {"x-api-key": os.environ["RL_API_KEY"]}
    async with httpx.AsyncClient(
        base_url=os.environ["RL_SAMPLER_URL"], headers=headers, timeout=180
    ) as http:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                response = await http.get("/health")
                if response.is_success:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            raise TimeoutError("sampler readiness deadline exceeded")

        async def sglang_probe(model_id, version):
            values = []
            versions = []
            for tokens in probes:
                response = await http.post(
                    "/generate",
                    json={
                        "input_ids": tokens,
                        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
                        "return_logprob": True,
                        "logprob_start_len": len(prompt) - 1,
                        "weight_run_id": model_id,
                        "weight_version": {"exact_version": version},
                    },
                )
                response.raise_for_status()
                metadata = response.json()["meta_info"]
                values.append(metadata["input_token_logprobs"][-1][0])
                versions.append(
                    [metadata["weight_version_start"], metadata["weight_version_end"]]
                )
            assert versions == [[version, version]] * len(probes), versions
            assert all(math.isfinite(value) for value in values), values
            return values

        async def titan_probe(training):
            data = [
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tokens[1:],
                        "weights": [1.0] * (len(tokens) - 1),
                    },
                )
                for tokens in probes
            ]
            future = await training.forward_async(data, "cross_entropy")
            output = await future.result_async()
            return [row["logprobs"].data[-1] for row in output.loss_fn_outputs]

        async def client_loop(index):
            await asyncio.sleep(
                index * float(os.environ.get("RL_STAGGER_SECONDS", "5"))
            )
            client = {
                "index": index,
                "preferred_letter": ("A", "B")[index],
                "steps": [],
            }
            result["clients"].append(client)
            training = await service.create_lora_training_client_async(
                base_model=model,
                rank=16,
                seed=100 + index,
                train_unembed=False,
            )
            model_id = (await training.get_info_async()).model_id
            client["model_id"] = model_id
            sampler = await training.save_weights_and_get_sampling_client_async()
            bulletin = SnapshotBulletin("/bulletin")
            version = bulletin.read_latest(model_id).version
            initial_version = version
            client["initial_version"] = initial_version
            baseline = await sglang_probe(model_id, version)
            titan_baseline = await titan_probe(training)
            client.update(
                initial_sglang=baseline,
                initial_torchtitan=titan_baseline,
                initial_difference=difference(baseline, titan_baseline),
            )
            print("RL_BASELINE=" + json.dumps(client), flush=True)
            assert client["initial_difference"]["max_absolute"] < 0.25, client
            assert client["initial_difference"]["mean_absolute"] < 0.05, client
            for step in range(int(os.environ.get("RL_STEPS", "5"))):
                record = {
                    "step": step + 1,
                    "sample_version": version,
                    "rollout_start": time.time(),
                }
                client["steps"].append(record)
                response = await sampler.sample_async(
                    types.ModelInput.from_ints(prompt),
                    num_samples=4,
                    sampling_params=types.SamplingParams(
                        max_tokens=4, temperature=1.0, seed=1000 + index * 100 + step
                    ),
                )
                record["rollout_end"] = time.time()
                texts = [
                    tokenizer.decode(sequence.tokens) for sequence in response.sequences
                ]
                rewards = [
                    1.0 if text.strip().startswith(client["preferred_letter"]) else -1.0
                    for text in texts
                ]
                record.update(
                    completions=texts,
                    rewards=rewards,
                    behavior_logprobs=[list(s.logprobs) for s in response.sequences],
                )
                # A deliberate first-step pause makes independent progress observable.
                if index == 1 and step == 0:
                    await asyncio.sleep(20)
                data = [
                    rl_datum(prompt, sequence, reward)
                    for sequence, reward in zip(
                        response.sequences, rewards, strict=True
                    )
                ]
                record["training_start"] = time.time()
                backward = await training.forward_backward_async(
                    data, "importance_sampling"
                )
                optimizer = await training.optim_step_async(
                    types.AdamParams(learning_rate=1e-4, grad_clip_norm=1.0)
                )
                backward_output = await backward.result_async()
                record["loss_metrics"] = backward_output.metrics
                record["training_logprobs"] = [
                    list(row["logprobs"].data[len(prompt) - 1 :])
                    for row in backward_output.loss_fn_outputs
                ]
                record["behavior_difference"] = difference(
                    [value for row in record["training_logprobs"] for value in row],
                    [value for row in record["behavior_logprobs"] for value in row],
                )
                record["optimizer_metrics"] = (await optimizer.result_async()).metrics
                assert record["optimizer_metrics"]["update_successful:mean"] == 1
                assert record["optimizer_metrics"]["grad_norm:mean"] > 0
                record["training_end"] = time.time()
                record["publication_start"] = time.time()
                sampler = await training.save_weights_and_get_sampling_client_async()
                published = bulletin.read_latest(model_id).version
                assert published > version
                version = published
                record["publication_end"] = time.time()
                record["published_version"] = version
                current = await sglang_probe(model_id, version)
                titan = await titan_probe(training)
                old = await sglang_probe(model_id, initial_version)
                record.update(
                    sglang=current,
                    torchtitan=titan,
                    difference=difference(current, titan),
                    old_version_drift=difference(old, baseline),
                    policy_change=difference(current, baseline),
                    delta_difference=difference(
                        [a - b for a, b in zip(current, baseline, strict=True)],
                        [a - b for a, b in zip(titan, titan_baseline, strict=True)],
                    ),
                )
                record["sampling_policy_ready"] = time.time()
                record["wall_seconds"] = (
                    record["sampling_policy_ready"] - record["rollout_start"]
                )
                print("RL_STEP=" + json.dumps({"client": index, **record}), flush=True)
                assert record["old_version_drift"]["max_absolute"] < 1e-4, record
                assert record["difference"]["max_absolute"] < 0.25, record
                assert record["difference"]["mean_absolute"] < 0.05, record
                assert record["policy_change"]["max_absolute"] > 1e-3, record
            client["final_version"] = version

        await asyncio.gather(client_loop(0), client_loop(1))
        clients = sorted(result["clients"], key=lambda c: c["index"])
        result["independent_progress"] = any(
            other["rollout_start"]
            < step["rollout_start"]
            < other["sampling_policy_ready"]
            for client in clients
            for step in client["steps"][1:]
            for peer in clients
            if peer is not client
            for other in peer["steps"]
            if other["step"] < step["step"]
        )
        assert result["independent_progress"], (
            "no observed overlapping independent progress"
        )
        response = await http.get("/validation/manifests")
        response.raise_for_status()
        result["manifests"] = response.json()


def main():
    result = {
        "model": os.environ["RL_MODEL"],
        "started_at": time.time(),
        "passed": False,
    }
    sampling_events = []
    try:
        checkpoint = snapshot_download(
            os.environ["RL_MODEL"],
            revision=os.environ.get("RL_MODEL_REVISION"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        )
        result["model_revision"] = Path(checkpoint).name
        os.environ["SPINDLE_BACKEND_CONFIG"] = json.dumps(
            {
                "torchtitan": {
                    "hf_checkpoint": checkpoint,
                    "model_family": "qwen3_8",
                    "model_flavor": "27B",
                    "max_context_length": 2048,
                    "max_lora_slots": 2,
                    "max_lora_rank": 16,
                    "max_batch_rows": 4,
                    "max_batch_tokens": 4096,
                }
            }
        )
        executor = build_executor()

        async def runner(task):
            event = {
                "model_id": task.model_id,
                "requested_version": task.publish_version,
                "request_id": task.request_id,
                "start": time.time(),
                "stats": {},
            }
            sampling_events.append(event)
            try:
                return await sample_task(
                    asdict(task),
                    os.environ["RL_SAMPLER_URL"],
                    api_key=os.environ["RL_API_KEY"],
                    stats=event["stats"],
                    timeout=180,
                    retry_timeout=30,
                    context_length=2048,
                )
            finally:
                event["end"] = time.time()

        with serve(create_backend_app(executor), 8002) as backend_url:
            platform = LocalEnginePlatform(
                "titan_rl", lambda: HttpBackendClient(backend_url), max_models=2
            )
            plane = ControlPlane(
                InMemoryKeyValueStore(),
                platform,
                sampling_tasks=LocalSamplingTaskPlatform(runner),
                sampling_task_stores=InMemorySessionKeyValueStores(),
            )
            definitions = [
                SimpleNamespace(
                    definition_id="titan_rl",
                    name="titan_rl",
                    model=os.environ["RL_MODEL"],
                    parameterization="lora",
                    max_context_length=2048,
                )
            ]
            app = create_control_plane_app(
                plane, definitions, api_key=os.environ["RL_API_KEY"], retrieve_window=5
            )
            with serve(app, 8000) as base_url:
                asyncio.run(run_clients(base_url, result))
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        result.update(finished_at=time.time(), sampling_events=sampling_events)
        result["local_manifests"] = [
            json.loads(p.read_text()) for p in Path("/bulletin").rglob("snapshot.json")
        ]
        print("RL_RESULT=" + json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
