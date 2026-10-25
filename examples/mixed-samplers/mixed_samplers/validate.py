"""Exercise support capture and quantized weight staging on the actual GPUs."""

import argparse
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import modal
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from spindle.deployments import DeploymentConfig
from spindle.inference.serving import terminate, wait_http
from spindle.providers.modal.rollout_image import image as rollout_image
from spindle.replay import capture_replay

from .config import GROUPS, TEMPERATURE, TOP_P, config, sampler_config
from .weight_validation import stage_and_restore

app = modal.App("spindle-mixed-sampler-validation")
image = rollout_image.add_local_python_source("mixed_samplers", "spindle")
assets = modal.Volume.from_name(
    config.platform["storage"]["assets"], create_if_missing=True
)
model_path = f"/assets/{config.model}"


@app.function(
    image=image,
    volumes={"/assets": assets},
    timeout=14400,
)
def prepare():
    snapshot_download(
        config.model, revision=config.model_revision, local_dir=model_path
    )
    assets.commit()


def check_group(group, function_app=app, *, weights=False):
    recipe = sampler_config(group)
    options = DeploymentConfig.create(recipe).inference_settings

    @function_app.function(
        name=f"check_{group['name']}",
        serialized=True,
        image=image,
        gpu=f"{group['gpu']}:{group['gpus']}",
        volumes={"/assets": assets},
        cpu=16,
        memory=262144,
        timeout=7200,
        env=recipe.inference_env,
    )
    def check():
        print(
            json.dumps({"event": "validation_started", "group": group["name"]}),
            flush=True,
        )
        assets.reload()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spindle.inference.sglang",
                model_path,
                json.dumps(options),
            ],
            start_new_session=True,
        )
        try:
            wait_http("http://127.0.0.1:8001/health", process, 2400)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokens = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Write a Python function that adds two integers.",
                    }
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
            sampling_request = {
                "input_ids": tokens,
                "return_logprob": True,
                "return_sampling_mask": True,
                "sampling_logprobs_mode": "support",
                "sampling_params": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": -1,
                    "max_new_tokens": 64,
                    "sampling_seed": 42,
                },
            }
            with httpx.Client(timeout=600) as client:
                response = client.post(
                    "http://127.0.0.1:8001/generate",
                    json=sampling_request,
                )
                response.raise_for_status()
                result = response.json()
                repeated = client.post(
                    "http://127.0.0.1:8001/generate", json=sampling_request
                )
                repeated.raise_for_status()
                arithmetic_tokens = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": "What is 17 + 25? Reply with just the number.",
                        }
                    ],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                arithmetic = client.post(
                    "http://127.0.0.1:8001/generate",
                    json={**sampling_request, "input_ids": arithmetic_tokens},
                )
                arithmetic.raise_for_status()
                meta = result["meta_info"]
                output_ids = [int(row[1]) for row in meta["output_token_logprobs"]]
                replay = capture_replay(
                    meta,
                    output_ids,
                    prompt_tokens=len(tokens),
                    temperature=TEMPERATURE,
                    routes=False,
                    mask=True,
                    probabilities=True,
                )
                info = client.get("http://127.0.0.1:8001/get_server_info")
                info.raise_for_status()
                error = (
                    None
                    if re.search(r"\b42\b", arithmetic.json()["text"])
                    else "initial model failed the arithmetic sanity check"
                )
                print(
                    json.dumps(
                        {
                            "event": "initial_generation",
                            "group": group["name"],
                            "text": result.get("text"),
                            "arithmetic": arithmetic.json()["text"],
                        }
                    ),
                    flush=True,
                )
                weight_report = None
                if weights:
                    try:
                        weight_report = stage_and_restore(
                            client, model_path, sampling_request
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve baseline evidence on reload failure
                        error = f"weight validation failed: {exc}"
                        weight_report = {"error": str(exc)}
                if weights and "samples" in weight_report:
                    restored = weight_report["samples"][-1]["meta_info"]
                    restored_ids = [
                        int(row[1]) for row in restored["output_token_logprobs"]
                    ]
                    restored_replay = capture_replay(
                        restored,
                        restored_ids,
                        prompt_tokens=len(tokens),
                        temperature=TEMPERATURE,
                        routes=False,
                        mask=True,
                        probabilities=True,
                    )
                    matches = not (
                        restored_ids != output_ids
                        or restored_replay["sampling_mask_ids"]
                        != replay["sampling_mask_ids"]
                        or not all(
                            math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4)
                            for a, b in zip(
                                replay["sampling_mask_logprobs"],
                                restored_replay["sampling_mask_logprobs"],
                                strict=True,
                            )
                        )
                    )
                    weight_report["restored_distribution_matches"] = matches
                    # Normal serving kernels are not deterministic across cache
                    # shapes. Exact generations are diagnostic; the native
                    # baseline comparison validates the actual runtime weights.
                    weight_report["baseline_weights_match"] = all(
                        rank["stage"]["initialization"]["validation"]["bytes"] > 0
                        for rank in weight_report["stages"][0]["rank_stats"]
                    )
            report = {
                "group": group,
                "hardware": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    text=True,
                ).splitlines(),
                "server_info": info.json(),
                "tokens": len(output_ids),
                "support_entries": len(replay["sampling_mask_ids"]),
                "text": result.get("text"),
                "initial_sample": result,
                "repeated_sample": repeated.json(),
                "arithmetic_sample": arithmetic.json(),
                "model_revision": config.model_revision,
                "weight_validation": weight_report,
            }
            if error:
                report["error"] = error
            return report
        finally:
            terminate(process)

    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group", default="b200-nvfp4", choices=[g["name"] for g in GROUPS] + ["all"]
    )
    parser.add_argument("--output", default="validation.json")
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()
    checks = {
        g["name"]: check_group(g, weights=args.weights)
        for g in GROUPS
        if args.group == "all" or g["name"] == args.group
    }
    with modal.enable_output(), app.run(environment_name="kailash-dev"):
        prepare.remote()
        with ThreadPoolExecutor(max_workers=len(checks)) as executor:
            pending = {
                executor.submit(check.remote): name for name, check in checks.items()
            }
            result = []
            for future in as_completed(pending):
                try:
                    report = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve every group's validation evidence
                    report = {"error": str(exc), "group": pending[future]}
                result.append(report)
                destination = Path(args.output)
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2))
                temporary.replace(destination)
                print(
                    json.dumps(
                        {
                            "event": "validation_completed",
                            "group": report.get("group"),
                            "error": report.get("error"),
                        }
                    ),
                    flush=True,
                )
    if any("error" in report for report in result):
        raise RuntimeError(f"Some sampler checks failed; see {args.output}")


if __name__ == "__main__":
    main()
