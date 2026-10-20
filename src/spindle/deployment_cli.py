"""Operator CLI for Python deployments. Only `deploy` provisions Modal resources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import Future
from functools import partial
from pathlib import Path

import modal
from rich.console import Console

from spindle import deploy_wizard
from spindle.configuration import BaseConfig
from spindle.deployments import (
    DeploymentConfig,
    config_path,
    load,
    validate_frontend,
)
from spindle.providers.modal.deployment_apps import (
    build_inference_app,
    build_rollout_app,
    build_sampler_image,
    build_trainer_app,
)
from spindle.providers.modal.deployment_configs import (
    CONFIGS_FILE_ENV,
    PLATFORM_ENV,
    deployed_configs,
    deployed_platform,
    deployed_pools,
)
from spindle.providers.modal.fft_pool import FFTPoolSpec
from spindle.providers.modal.lora_pool import LoraPoolSpec


NO_CONFIGS = (
    "No configs given. Run `spindle deploy` in a terminal to pick models interactively, "
    "pass config files or bundled config names (e.g. `spindle deploy qwen35-9b-lora-16k`), "
    "or pass --all to deploy every bundled config."
)


def compile_configs(paths):
    recipes = [load(path) for path in paths]
    return [DeploymentConfig.create(recipe) for recipe in recipes]


def config_file(arg):
    """A config file path, or the name of a bundled config."""
    if arg.endswith(".py"):
        return Path(arg)
    path = config_path(arg)
    if not path.is_file():
        raise ValueError(
            f"unknown config {arg!r}; run `spindle deploy` to pick from the bundled models"
        )
    return path


def interactive():
    return sys.stdin.isatty() and sys.stdout.isatty()


def app_is_missing(app_name, environment):
    try:
        modal.App.lookup(app_name, environment_name=environment)
    except modal.exception.NotFoundError:
        return True
    return False


def apps_to_publish(configs, current_configs, environment, trainers, inference):
    """Add names whose apps are missing or whose trainer/inference settings changed."""
    current_by_name = {deployment.name: deployment for deployment in current_configs}
    for deployment in configs:
        current = current_by_name.get(deployment.name)
        if app_is_missing(deployment.trainer_app_name, environment) or (
            current is not None
            and deployment.trainer_identity() != current.trainer_identity()
        ):
            trainers.add(deployment.name)
        if app_is_missing(deployment.inference_app_name, environment) or (
            current is not None
            and deployment.inference_identity() != current.inference_identity()
        ):
            inference.add(deployment.name)
    return trainers, inference


def deploy(configs, *, refresh_trainers=(), refresh_inference=(), run_steps=None):
    """Deploy missing or changed trainer and inference apps, then the frontend."""
    if sys.version_info[:2] != (3, 12):
        raise ValueError(
            "Python deployment requires Python 3.12 to match the serialized GPU runtime images"
        )
    trainers = set(refresh_trainers)
    inference = set(refresh_inference)
    names = {deployment.name for deployment in configs}
    unknown = (trainers | inference) - names
    if unknown:
        raise ValueError(f"unknown configs to refresh: {sorted(unknown)}")
    validate_frontend([deployment.recipe for deployment in configs])
    platform = configs[0].recipe.platform
    environment = platform["modal"]["environment"]
    current_configs = [
        DeploymentConfig.model_validate(row)
        for row in deployed_configs(platform["frontend"], environment)
    ]
    current_platform = deployed_platform(platform["frontend"], environment)
    if current_platform is not None and current_platform != platform:
        trainers.update(names)
        inference.update(names)
    trainers, inference = apps_to_publish(
        configs, current_configs, environment, trainers, inference
    )
    pools = deployed_pools(platform["frontend"], environment) if inference else []

    def deploy_trainer(deployment):
        build_trainer_app(deployment, platform)[0].deploy(environment_name=environment)

    def deploy_inference(deployment):
        build_inference_app(deployment, platform)[0].deploy(
            environment_name=environment
        )
        for record in pools:
            if record["definition_id"] != deployment.definition_id:
                continue
            pool = (
                LoraPoolSpec.from_dict(record)
                if deployment.parameterization == "lora"
                else FFTPoolSpec.from_dict(record)
            )
            if not app_is_missing(pool.app_name, environment):
                build_rollout_app(deployment, pool, platform)[0].deploy(
                    environment_name=environment
                )

    def deploy_frontend():
        command = [sys.executable, "-m", "modal", "deploy"]
        if environment:
            command += ["--env", environment]
        captured = {"capture_output": True, "text": True} if run_steps else {}
        with tempfile.TemporaryDirectory() as directory:
            configs_file = Path(directory, "deployment_configs.json")
            configs_file.write_text(
                json.dumps(
                    [deployment.model_dump(mode="json") for deployment in configs]
                )
            )
            subprocess.run(
                [*command, "-m", "spindle.providers.modal.app"],
                check=True,
                env={
                    **os.environ,
                    CONFIGS_FILE_ENV: str(configs_file),
                    PLATFORM_ENV: json.dumps(platform),
                    "SPINDLE_APP_NAME": platform["frontend"],
                },
                **captured,
            )

    steps = []
    for deployment in configs:
        if deployment.name in trainers:
            steps.append(
                (f"Trainer    {deployment.name}", partial(deploy_trainer, deployment))
            )
        if deployment.name in inference:
            steps.append(
                (f"Inference  {deployment.name}", partial(deploy_inference, deployment))
            )
    steps.append(("API server", deploy_frontend))
    build_image = partial(build_sampler_image, platform["frontend"], environment)
    if run_steps is not None:
        run_steps(steps, [("Sampler image", in_background(build_image))])
        return
    with modal.enable_output():
        building = in_background(build_image)
        for _, step in steps:
            step()
        try:
            building.result()
        except Exception as exc:
            print(
                f"Sampler image build failed: {exc}\n"
                "The first sampling request builds it instead.",
                file=sys.stderr,
            )


def in_background(step):
    """Run `step` on a daemon thread so a failed or interrupted deploy exits promptly."""
    future = Future()

    def run():
        try:
            future.set_result(step())
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return future


def parser():
    result = argparse.ArgumentParser(prog="spindle")
    commands = result.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config").add_subparsers(dest="action", required=True)
    init = config.add_parser("init")
    init.add_argument("--preset", required=True)
    validate = config.add_parser("validate")
    validate.add_argument("files", nargs="+")
    apply = commands.add_parser(
        "deploy",
        help="Deploy the complete active Python config set behind one frontend",
    )
    apply.add_argument(
        "files",
        nargs="*",
        help="Python config files or bundled config names (none: pick interactively)",
    )
    apply.add_argument("--all", action="store_true", help="Deploy every bundled config")
    apply.add_argument("--app")
    apply.add_argument("--env")
    apply.add_argument("--region")
    apply.add_argument(
        "--refresh-trainer", action="append", default=[], metavar="CONFIG_NAME"
    )
    apply.add_argument(
        "--refresh-inference", action="append", default=[], metavar="CONFIG_NAME"
    )
    management = commands.add_parser("deployment").add_subparsers(
        dest="action", required=True
    )
    retry = management.add_parser("retry")
    retry.add_argument("--frontend", required=True)
    retry.add_argument("--env")
    retry.add_argument("definition_id")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "config":
            if args.action == "init":
                module = config_path(args.preset).stem
                if not config_path(args.preset).is_file():
                    raise ValueError(f"unknown example config: {args.preset}")
                print(
                    f'from spindle.configs.{module} import Config as Parent\n\n\nclass Config(Parent):\n    name = "my-model"\n\n\nconfig = Config()'
                )
            else:
                recipes = [load(path) for path in args.files]
                validate_frontend(recipes)
                print(
                    f"Validated {len(recipes)} deployment(s). Backend integration settings are checked when preparing trainers and pools; native options are checked at engine startup."
                )
        elif args.command == "deploy":
            console = Console()
            frontend = args.app or BaseConfig.platform["frontend"]
            environment = args.env or os.environ.get("MODAL_ENVIRONMENT")
            chosen = []
            if args.files:
                paths = [config_file(arg) for arg in args.files]
            elif args.all:
                paths = deploy_wizard.bundled_config_paths()
            elif interactive():
                chosen = deploy_wizard.choose(console, frontend, environment)
                paths = [preset.path for preset in chosen]
            else:
                raise ValueError(NO_CONFIGS)
            configs = compile_configs(paths)
            for deployment in configs:
                platform = deployment.recipe.platform
                if args.app is not None:
                    platform["frontend"] = args.app
                if args.env is not None:
                    platform["modal"]["environment"] = args.env
                if args.region is not None:
                    platform["modal"]["region"] = args.region
            deploy(
                configs,
                refresh_trainers=args.refresh_trainer,
                refresh_inference=args.refresh_inference,
                run_steps=partial(
                    deploy_wizard.deploy_with_progress,
                    console,
                    retry_command=deploy_wizard.deploy_command(chosen),
                )
                if chosen
                else None,
            )
            if chosen:
                deploy_wizard.show_next_steps(console, chosen, frontend, environment)
        else:
            modal.Function.from_name(
                args.frontend, "clear_deployment_failure", environment_name=args.env
            ).remote(args.definition_id)
    except deploy_wizard.Cancelled:
        cli.exit(1, "Deploy cancelled.\n")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
