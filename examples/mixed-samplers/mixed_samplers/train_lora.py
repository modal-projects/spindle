"""Concurrent independent LoRA runs on one heterogeneous sampler fleet."""

import argparse
import asyncio
import atexit
import json
import os
import signal
from contextlib import ExitStack
from pathlib import Path

import httpx
import modal
import tinker

from spindle.deployments import DeploymentConfig
from spindle.providers.modal.deployment_apps import build_rollout_app
from spindle.providers.modal.lora_pool import LoraPoolSpec

from . import config as common
from . import config_lora, train
from .data import load_tasks
from .workers import app as worker_app
from .workers import measure_capacity


def unload_model(url, key, model_id):
    try:
        response = httpx.post(
            f"{url}/api/v1/unload_model",
            headers={"X-API-Key": key},
            json={"model_id": model_id},
            timeout=300,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(
            json.dumps(
                {
                    "event": "trainer_cleanup_failed",
                    "model_id": model_id,
                    "error": str(exc),
                }
            ),
            flush=True,
        )


def exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def client_args(args, index):
    output = args.output / f"client-{index}"
    checkpoint = output / "checkpoint.json"
    if args.fresh and output.exists() and any(output.iterdir()):
        raise ValueError(f"--fresh requires an empty client directory: {output}")
    saved = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists() and not args.fresh
        else {}
    )
    output.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
        **{
            **vars(args),
            "output": str(output),
            "checkpoint": saved.get("path"),
            "start_step": saved.get("step", 0),
        }
    )


def create_clients(service, base_model, args, url, key):
    outputs = [client_args(args, index) for index in range(args.lora_clients)]
    clients = []
    for index, options in enumerate(outputs):
        training = service.create_lora_training_client(
            base_model, rank=32, train_mlp=True, train_attn=True, train_unembed=False
        )
        atexit.register(unload_model, url, key, training.model_id)
        if options.checkpoint:
            training.load_state_with_optimizer(options.checkpoint).result(timeout=7200)
        publication = training.save_weights_for_sampler(
            f"initial-{options.start_step}"
        ).result(timeout=3600)
        seed = common.SEED + 1_000_003 * index
        manifest = {
            "model": base_model,
            "model_revision": config_lora.config.model_revision,
            "model_id": training.model_id,
            "seed": seed,
            "lr": config_lora.LEARNING_RATE,
            "batch": [common.PROMPTS_PER_UPDATE, common.TRAJECTORIES_PER_PROMPT],
            "groups": config_lora.GROUPS,
            "initial_publication": publication.path,
            "checkpoint": options.checkpoint,
            "start_step": options.start_step,
        }
        (Path(options.output) / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        clients.append((training, options, seed))
        print(json.dumps({"event": "trainer_created", **manifest}), flush=True)
    return clients


async def run_clients(clients, gateways, tasks):
    measured = await asyncio.gather(
        *(
            measure_capacity.remote.aio(url, clients[0][0].model_id, 1)
            for url in gateways.values()
        )
    )
    weights = [measurement["tokens_per_second"] for measurement in measured]
    async with asyncio.TaskGroup() as group:
        for training, options, seed in clients:
            (Path(options.output) / "startup-validation.json").write_text(
                json.dumps(dict(zip(gateways, measured, strict=True)), indent=2)
            )
            group.create_task(
                train.train(
                    training,
                    gateways,
                    weights,
                    options,
                    tasks,
                    seed=seed,
                    learning_rate=config_lora.LEARNING_RATE,
                )
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-clients", type=int, default=4)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch; requires empty client directories.",
    )
    parser.add_argument("--steps", type=int, default=common.TRAINING_STEPS)
    parser.add_argument("--inflight-prompts", type=int, default=192)
    args = parser.parse_args(argv)
    if not 1 <= args.lora_clients <= config_lora.config.miles_cfg["max_lora_slots"]:
        parser.error("--lora-clients must be between 1 and 6 (trainer slot capacity)")
    if args.steps < 1 or args.inflight_prompts < 1:
        parser.error("--steps and --inflight-prompts must be positive")
    signal.signal(signal.SIGINT, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    cfg = config_lora.config
    environment = cfg.platform["modal"]["environment"]
    url = modal.Function.from_name(
        cfg.platform["frontend"], "server", environment_name=environment
    ).get_web_url()
    key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ["TINKER_API_KEY"]
    )
    service = tinker.ServiceClient(base_url=url, api_key=key)
    tasks = load_tasks()
    clients = create_clients(service, cfg.model, args, url, key)
    gateways = {}
    with modal.enable_output(), ExitStack() as stack:
        stack.enter_context(worker_app.run(environment_name=environment))
        for group in config_lora.GROUPS:
            deployment = DeploymentConfig.create(config_lora.sampler_config(group))
            pool = LoraPoolSpec(deployment.definition_id)
            app, server = build_rollout_app(deployment, pool)
            stack.enter_context(app.run(environment_name=environment))
            gateways[group["name"]] = server.get_url()
        asyncio.run(run_clients(clients, gateways, tasks))


if __name__ == "__main__":
    main()
