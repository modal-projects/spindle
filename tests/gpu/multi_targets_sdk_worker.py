"""Full SDK/HTTP/Miles training regression using Qwen3-0.6B on one GPU."""

import contextlib
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import tinker
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from gpu_test_support import SingleEnginePlatform, serve
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine import Engine
from spindle.engine.backend_http import HttpBackendClient
from spindle.engine.http import HttpEngineClient, create_engine_app
from spindle.providers.local import InMemoryKeyValueStore

MODEL = "Qwen/Qwen3-0.6B"
ROOT = Path("/tmp/multi-target-sdk")


def dense_datum(tokens, targets, weights):
    return tinker.Datum(
        tinker.ModelInput.from_ints(tokens),
        {
            "target_tokens": tinker.TensorData(
                data=targets.flatten().tolist(),
                dtype="int64",
                shape=list(targets.shape),
            ),
            "weights": tinker.TensorData(
                data=weights.flatten().tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
        },
    )


def scores(output, shape):
    value = output.loss_fn_outputs[0]["logprobs"].to_torch()
    assert list(value.shape) == shape
    assert torch.isfinite(value).all()
    return value


async def metadata(path):
    return json.loads((Path(path) / "metadata.json").read_text())


def checkpoint_tensors(uri):
    model, _, name = uri.removeprefix("tinker://").split("/")
    source = ROOT / "checkpoints" / name / model / "miles"
    destination = ROOT / f"{name}-tensors.pt"
    dcp_to_torch_save(str(source), str(destination))
    state = torch.load(destination, map_location="cpu", weights_only=False)

    def flatten(value, prefix=""):
        if isinstance(value, torch.Tensor):
            return {prefix: value}
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                result.update(flatten(child, f"{prefix}/{key}"))
            return result
        if isinstance(value, (tuple, list)):
            return flatten(dict(enumerate(value)), prefix)
        return {}

    return flatten(state)


def exercise(url, tokens, targets, weights, hf_scores, request_types):
    service = tinker.ServiceClient(base_url=url, api_key="tml-gpu-test")
    training = service.create_lora_training_client(
        base_model=MODEL, rank=8, train_unembed=False
    )
    sparse = tinker.Datum(
        tinker.ModelInput.from_ints(tokens),
        {"target_tokens": targets, "weights": weights},
    )
    assert all(
        sparse.loss_fn_inputs[key].sparse_crow_indices is not None
        for key in ("target_tokens", "weights")
    )
    dense = dense_datum(tokens, targets, weights)
    baseline = training.forward([sparse], "cross_entropy").result(timeout=180)
    before = scores(baseline, [1631, 5])
    dense_forward = training.forward([dense], "cross_entropy").result(timeout=180)
    torch.testing.assert_close(before, scores(dense_forward, [1631, 5]), atol=0, rtol=0)
    single_scores = []
    for column in range(5):
        datum = tinker.Datum(
            tinker.ModelInput.from_ints(tokens),
            {
                "target_tokens": targets[:, column],
                "weights": weights[:, column],
            },
        )
        single_scores.append(
            scores(
                training.forward([datum], "cross_entropy").result(timeout=180), [1631]
            )
        )
    single_scores = torch.stack(single_scores, dim=-1)
    torch.testing.assert_close(before, single_scores, atol=2e-5, rtol=2e-5)
    # HF/Megatron model-conversion parity is diagnostic here. The supported
    # one-dimensional Miles path is the reference for multi-target behavior.
    print(
        "SDK_FORWARD_RESULT "
        + json.dumps(
            {
                "max_multi_single_error": (before - single_scores).abs().max().item(),
                "max_multi_hf_error": (before - hf_scores).abs().max().item(),
                "max_single_hf_error": (single_scores - hf_scores).abs().max().item(),
            }
        ),
        flush=True,
    )
    expected_loss = -(before * weights).sum().item()
    assert abs(baseline.metrics["loss:sum"] - expected_loss) < 0.05
    initial = training.save_state("initial").result(timeout=180).path
    print("SDK_INITIAL_CHECKPOINT " + initial, flush=True)
    initial_tensors = checkpoint_tensors(initial)
    outcomes = {}
    for mode in ("dense_control", "sparse", "dense", "single"):
        print("SDK_TRAINING_START " + mode, flush=True)
        training.load_state_with_optimizer(initial).result(timeout=180)
        restored_path = training.save_state("restored-" + mode).result(timeout=180).path
        restored_tensors = checkpoint_tensors(restored_path)
        added = restored_tensors.keys() - initial_tensors.keys()
        removed = initial_tensors.keys() - restored_tensors.keys()
        print(
            "SDK_CHECKPOINT_KEYS "
            + json.dumps({"added": sorted(added), "removed": sorted(removed)}),
            flush=True,
        )
        # An unused Adam optimizer may materialize zero moments when loaded.
        # Newly allocated zero optimizer state is equivalent to absent state.
        for key in added:
            assert (
                "optimizer" in key.lower()
                and torch.count_nonzero(restored_tensors[key]) == 0
            ), f"Unexpected added checkpoint tensor: {key}"
        for key in removed:
            assert (
                "optimizer" in key.lower()
                and torch.count_nonzero(initial_tensors[key]) == 0
            ), f"Missing checkpoint tensor: {key}"
        for key in initial_tensors.keys() & restored_tensors.keys():
            torch.testing.assert_close(
                initial_tensors[key],
                restored_tensors[key],
                atol=0,
                rtol=0,
                msg=lambda message: f"Checkpoint restore mismatch {key}: {message}",
            )
        print("SDK_RESTORE_IDENTICAL " + mode, flush=True)
        restored = scores(
            training.forward([dense], "cross_entropy").result(timeout=180), [1631, 5]
        )
        torch.testing.assert_close(before, restored, atol=0, rtol=0)
        if mode == "single":
            loss = 0.0
            for column in range(5):
                datum = tinker.Datum(
                    tinker.ModelInput.from_ints(tokens),
                    {
                        "target_tokens": targets[:, column],
                        "weights": weights[:, column],
                    },
                )
                result = training.forward_backward([datum], "cross_entropy").result(
                    timeout=180
                )
                torch.testing.assert_close(
                    scores(result, [1631]), before[:, column], atol=2e-5, rtol=2e-5
                )
                loss += result.metrics["loss:sum"]
        else:
            result = training.forward_backward(
                [sparse if mode == "sparse" else dense], "cross_entropy"
            ).result(timeout=180)
            torch.testing.assert_close(
                scores(result, [1631, 5]), before, atol=2e-5, rtol=2e-5
            )
            loss = result.metrics["loss:sum"]
        assert abs(loss - expected_loss) < 0.05
        update = training.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(
            timeout=180
        )
        after = scores(
            training.forward([dense], "cross_entropy").result(timeout=180), [1631, 5]
        )
        change = (after - before).abs().max().item()
        assert change > 1e-4, "optimizer update did not affect model predictions"
        outcomes[mode] = after
        print(
            "SDK_TRAINING_RESULT "
            + json.dumps(
                {
                    "mode": mode,
                    "loss": loss,
                    "prediction_change": change,
                    "optim": update.metrics,
                }
            ),
            flush=True,
        )
    print(
        "SDK_UPDATE_DIAGNOSTICS "
        + json.dumps(
            {
                "dense_repeat_error": (outcomes["dense_control"] - outcomes["dense"])
                .abs()
                .max()
                .item(),
                "sparse_dense_error": (outcomes["sparse"] - outcomes["dense"])
                .abs()
                .max()
                .item(),
                "single_dense_error": (outcomes["single"] - outcomes["dense"])
                .abs()
                .max()
                .item(),
            }
        ),
        flush=True,
    )
    torch.testing.assert_close(
        outcomes["dense_control"], outcomes["dense"], atol=0, rtol=0
    )
    torch.testing.assert_close(outcomes["sparse"], outcomes["dense"], atol=0, rtol=0)
    # A combined backward and several accumulated BF16 backwards need not
    # produce identical parameter gradients. Measure that effect using the
    # established 1-D API alone, without invoking the multi-target hook.
    one_dimensional = {}
    for count in (1, 5):
        training.load_state_with_optimizer(initial).result(timeout=180)
        restored = scores(
            training.forward([dense], "cross_entropy").result(timeout=180), [1631, 5]
        )
        torch.testing.assert_close(restored, before, atol=0, rtol=0)
        control = tinker.Datum(
            tinker.ModelInput.from_ints(tokens),
            {
                "target_tokens": targets[:, 0],
                "weights": weights[:, 0] * (5 / count),
            },
        )
        total = 0.0
        for _ in range(count):
            result = training.forward_backward([control], "cross_entropy").result(
                timeout=180
            )
            torch.testing.assert_close(
                scores(result, [1631]), before[:, 0], atol=2e-5, rtol=2e-5
            )
            total += result.metrics["loss:sum"]
        expected = -(before[:, 0] * weights[:, 0] * 5).sum().item()
        assert abs(total - expected) < 0.05
        update = training.optim_step(tinker.AdamParams(learning_rate=1e-4)).result(
            timeout=180
        )
        one_dimensional[count] = scores(
            training.forward([dense], "cross_entropy").result(timeout=180), [1631, 5]
        )
        print(
            "SDK_1D_CONTROL "
            + json.dumps(
                {
                    "backward_calls": count,
                    "loss": total,
                    "grad_norm": update.metrics["grad_norm:mean"],
                }
            ),
            flush=True,
        )
    split_difference = (one_dimensional[1] - one_dimensional[5]).abs().max().item()
    print("SDK_1D_SPLIT_DIFFERENCE " + str(split_difference), flush=True)
    print(
        "SDK_E2E_RESULT "
        + json.dumps(
            {
                "model": MODEL,
                "shape": [1631, 5],
                "sparse_inputs": ["weights", "target_tokens"],
                "max_hf_logprob_error": (before - hf_scores).abs().max().item(),
                "max_sparse_dense_update_error": (
                    outcomes["sparse"] - outcomes["dense"]
                )
                .abs()
                .max()
                .item(),
                "max_1d_split_update_difference": split_difference,
                "max_multi_single_update_difference": (
                    outcomes["sparse"] - outcomes["single"]
                )
                .abs()
                .max()
                .item(),
                "http_request_types": sorted(request_types),
            }
        ),
        flush=True,
    )


def main():
    ROOT.mkdir(exist_ok=True)
    checkpoint = snapshot_download(
        MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pattern = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    tokens = (pattern * (1631 // len(pattern) + 1))[:1631]
    targets = torch.zeros(1631, 5, dtype=torch.long)
    targets[1200:] = torch.tensor([10, 20, 30, 40, 50])
    weights = torch.zeros(1631, 5)
    weights[1200:] = 0.2
    hf_model = (
        AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    with torch.inference_mode():
        logits = hf_model(torch.tensor([tokens], device="cuda")).logits[0]
        hf_scores = logits.float().log_softmax(-1).gather(-1, targets.cuda()).cpu()
    del logits, hf_model
    gc.collect()
    torch.cuda.empty_cache()
    config = {
        "miles": {
            "hf_checkpoint": checkpoint,
            "model_type": "qwen3-0.6B",
            "actor_num_gpus_per_node": 1,
            "max_lora_slots": 1,
            "max_lora_rank": 8,
            "default_lora_alpha": 16,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "max_tokens_per_gpu": 2048,
            "cli_options": {"attention_backend": "unfused", "deterministic_mode": True},
        },
        "checkpoint_dir": str(ROOT / "checkpoints"),
        "capture_dir": str(ROOT / "captures"),
    }
    env = {
        **os.environ,
        "SPINDLE_BACKEND_CONFIG": json.dumps(config),
        "SPINDLE_BASE_MODEL": MODEL,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "spindle.engine.backend_http",
            "spindle.backends.miles_lora:build_executor",
            "8911",
        ],
        env=env,
    )
    try:
        deadline = time.monotonic() + 900
        while True:
            if backend.poll() is not None:
                raise RuntimeError(f"backend exited: {backend.returncode}")
            try:
                if httpx.get("http://127.0.0.1:8911/healthz", timeout=2).is_success:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("backend startup")
            time.sleep(2)
        engine = Engine(
            HttpBackendClient("http://127.0.0.1:8911", read_timeout=180), max_models=1
        )
        with serve(create_engine_app(engine, token="test-engine")) as engine_url:
            client = HttpEngineClient(engine_url, token="test-engine")
            plane = ControlPlane(
                InMemoryKeyValueStore(),
                SingleEnginePlatform("gpu-test", client),
                checkpoint_root=str(ROOT / "checkpoints"),
                read_checkpoint_metadata=metadata,
            )
            definition = SimpleNamespace(
                definition_id="gpu-test",
                name="gpu-test",
                model=MODEL,
                parameterization="lora",
                max_context_length=2048,
            )
            app = create_control_plane_app(
                plane, (definition,), api_key="tml-gpu-test", retrieve_window=5.0
            )
            request_types = set()

            @app.middleware("http")
            async def record_type(request, call_next):
                if "forward" in request.url.path:
                    request_types.add(request.headers.get("content-type", ""))
                return await call_next(request)

            with serve(app) as url:
                exercise(url, tokens, targets, weights, hf_scores, request_types)
    finally:
        backend.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            backend.wait(timeout=30)
        if backend.poll() is None:
            backend.kill()
            backend.wait()


if __name__ == "__main__":
    main()
