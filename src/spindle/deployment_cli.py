"""Operator CLI for Python deployments. Only `deploy` provisions Modal resources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

from spindle.deployments import (
    DeploymentConfig,
    config_path,
    load,
    validate_frontend,
)
from spindle.providers.modal.deployment_apps import (
    build_inference_app,
    build_rollout_app,
    build_trainer_app,
)
from spindle.providers.modal.deployment_configs import (
    CONFIGS_ENV,
    PLATFORM_ENV,
    deployed_configs,
    deployed_platform,
    deployed_pools,
)
from spindle.providers.modal.fft_pool import FFTPoolSpec
from spindle.providers.modal.lora_pool import LoraPoolSpec


def compile_configs(paths):
    recipes = [load(path) for path in paths]
    return [DeploymentConfig.create(recipe) for recipe in recipes]


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


def deploy(configs, *, refresh_trainers=(), refresh_inference=()):
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

    with modal.enable_output():
        for deployment in configs:
            if deployment.name in trainers:
                build_trainer_app(deployment, platform)[0].deploy(
                    environment_name=environment
                )
            if deployment.name in inference:
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
    command = [sys.executable, "-m", "modal", "deploy"]
    if environment:
        command += ["--env", environment]
    subprocess.run(
        [*command, "-m", "spindle.providers.modal.app"],
        check=True,
        env={
            **os.environ,
            CONFIGS_ENV: json.dumps(
                [deployment.model_dump(mode="json") for deployment in configs]
            ),
            PLATFORM_ENV: json.dumps(platform),
            "SPINDLE_APP_NAME": platform["frontend"],
        },
    )


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
        "files", nargs="*", help="Python configs (default: all bundled configs)"
    )
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
            paths = args.files or sorted(
                Path(__file__).with_name("configs").glob("[!_]*.py")
            )
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
            )
        else:
            modal.Function.from_name(
                args.frontend, "clear_deployment_failure", environment_name=args.env
            ).remote(args.definition_id)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
