"""Operator CLI for Python deployments. Only `deploy` provisions Modal resources."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from functools import partial
from importlib import resources
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
from spindle.providers.modal.deployment_apps import build_sampler_image
from spindle.providers.modal.deployment_configs import (
    FRONTEND_SETTINGS_ENV,
    deployed_configs,
    deployed_platform,
    seed_registry_rows,
)
from spindle.control_plane.deployments import DEPLOY_TIMEOUT_S, FrontendSettings
from spindle.version import __version__, deployment_tag, source_commit


NO_CONFIGS = (
    "No configs given. Run `spindle deploy` in a terminal to pick models interactively, "
    "pass config files or bundled config names (e.g. `spindle deploy qwen35-9b-lora-16k`), "
    "or pass --all to deploy every bundled config."
)


DEPLOY_POLL_INTERVAL_S = 10


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


def deploy(configs, *, refresh_trainers=(), refresh_inference=(), run_steps=None):
    """Deploy recipes through the frontend registry."""
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
    current_platform = deployed_platform(platform["frontend"], environment)
    if current_platform is not None and current_platform != platform:
        trainers.update(names)
        inference.update(names)

    def seed_registry():
        """Migrate a pre-registry frontend's recipes across the upgrade.

        A frontend that still ships a configs file answers `deployed_configs`
        with bare configs instead of registry rows. Every recipe it serves is
        written to the registry as ready before the frontend swaps over, so
        nothing drops out and its apps stay put, except a listed recipe whose
        config changed or is being refreshed: that one deploys as usual.
        """
        served = deployed_configs(platform["frontend"], environment)
        legacy = [config for config in served if "state" not in config]
        if not legacy:
            return
        listed = {
            deployment.name: deployment.model_dump(mode="json")
            for deployment in configs
        }
        keep = [
            config
            for config in legacy
            if config["recipe"]["name"] not in listed
            or (
                listed[config["recipe"]["name"]] == config
                and config["recipe"]["name"] not in trainers | inference
            )
        ]
        if keep:
            seed_registry_rows(platform["frontend"], environment, keep)

    def deploy_frontend():
        seed_registry()
        command = [sys.executable, "-m", "modal", "deploy", "--tag", deployment_tag()]
        if environment:
            command += ["--env", environment]
        captured = {"capture_output": True, "text": True} if run_steps else {}
        settings = FrontendSettings.from_recipe(configs[0].recipe)
        subprocess.run(
            [*command, "-m", "spindle.providers.modal.app"],
            check=True,
            env={
                **os.environ,
                FRONTEND_SETTINGS_ENV: json.dumps(settings.as_dict()),
                "SPINDLE_APP_NAME": platform["frontend"],
            },
            **captured,
        )
        deploy_recipes()

    def deploy_recipes():
        function = modal.Function.from_name(
            platform["frontend"], "deploy_recipes", environment_name=environment
        )
        function.remote(
            json.dumps([d.model_dump(mode="json") for d in configs]),
            sorted(trainers),
            sorted(inference),
        )

    def wait_for(deployment):
        deadline = time.monotonic() + DEPLOY_TIMEOUT_S
        while time.monotonic() < deadline:
            rows = deployed_configs(platform["frontend"], environment)
            row = next(
                (r for r in rows if r["config"]["recipe"]["name"] == deployment.name),
                None,
            )
            if row and row["state"] == "ready":
                return
            if row and row["state"] == "failed":
                raise ValueError(row.get("error") or "deployment failed")
            time.sleep(DEPLOY_POLL_INTERVAL_S)
        raise ValueError(f"timed out waiting for deployment {deployment.name}")

    steps = []
    steps.append(("API server", deploy_frontend))
    for deployment in configs:
        steps.append((f"Recipe     {deployment.name}", partial(wait_for, deployment)))
    build_image = partial(build_sampler_image, platform["frontend"], environment)
    if run_steps is not None:
        run_steps(steps, [("Sampler image", in_background(build_image))])
        return
    with modal.enable_output():
        building = in_background(build_image)
        for label, step in steps:
            note = step()
            if note:
                print(f"{' '.join(label.split())}: {note}")
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


class PrintVersion(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            help="show the Spindle version and source commit, then exit",
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"spindle {__version__} ({source_commit() or 'unknown commit'})")
        parser.exit()


def parser():
    result = argparse.ArgumentParser(prog="spindle")
    result.add_argument("--version", action=PrintVersion)
    commands = result.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config").add_subparsers(dest="action", required=True)
    init = config.add_parser("init")
    init.add_argument("--preset", required=True)
    validate = config.add_parser("validate")
    validate.add_argument("files", nargs="+")
    apply = commands.add_parser(
        "deploy",
        help="Add or update Python configs behind one frontend",
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
    skills = commands.add_parser(
        "skills", help="Manage the bundled agent skills"
    ).add_subparsers(dest="action", required=True)
    install = skills.add_parser(
        "install", help="Install the bundled skills into a project"
    )
    install.add_argument(
        "--project-dir",
        type=Path,
        help="Project root (default: nearest Git repository)",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Replace existing skills and Claude links",
    )
    return result


def install_skills(project_dir: Path | None, force: bool) -> None:
    if project_dir is None:
        cwd = Path.cwd().resolve()
        project_dir = next(
            (p for p in (cwd, *cwd.parents) if (p / ".git").exists()), None
        )
        if project_dir is None:
            raise ValueError("Run inside a Git repository or pass --project-dir.")
    skills_dir = Path(resources.files("spindle") / "skills")
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        source = skill_md.parent
        destination = project_dir / ".agents" / "skills" / source.name
        if destination.exists() and not force:
            print(
                f"Skipped {source.name}: {destination} exists; "
                "use --force to replace it.",
                file=sys.stderr,
            )
            continue
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)
        link = project_dir / ".claude" / "skills" / source.name
        if link.is_symlink() or not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(os.path.relpath(destination, link.parent))
        print(f"Installed {source.name} at {destination}")


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
        elif args.command == "skills":
            install_skills(args.project_dir, args.force)
        else:
            modal.Function.from_name(
                args.frontend, "clear_deployment_failure", environment_name=args.env
            ).remote(args.definition_id)
    except deploy_wizard.Cancelled:
        cli.exit(1, "Deploy cancelled.\n")
    except deploy_wizard.SetupIncomplete:
        cli.exit(1, "\nRun `spindle deploy` again once the secrets exist.\n")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
