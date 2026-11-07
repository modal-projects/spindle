"""Operator CLI for Python deployments. Only `deploy` provisions Modal resources; `stop` releases them."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

import httpx
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
from spindle.control_plane.deployments import FRONTEND_FIELDS, FrontendSettings
from spindle.providers.modal.deployment_configs import (
    FRONTEND_SETTINGS_ENV,
    delete_registry,
    deployed_configs,
    deployed_platform,
    deployed_pools,
    frontend_app_id,
    start_weights_download,
    write_registry,
)
from spindle.providers.modal.fft_pool import FFTPoolSpec
from spindle.providers.modal.lora_pool import LoraPoolSpec
from spindle.version import __version__, deployment_tag, source_commit


RECIPE_FIELDS = frozenset(key for key in vars(BaseConfig) if not key.startswith("_"))
ADD_POLL_S = 10
ADD_TIMEOUT_S = 60 * 60
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
    """Deploy missing or changed trainer and inference apps, then the frontend.

    The listed configs are added to or updated in the frontend's registry;
    recipes it already serves and that are not listed keep serving.
    """
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
    served = deployed_configs(platform["frontend"], environment)
    current_configs = [DeploymentConfig.from_registry(row) for row in served]
    unlisted = [row for row in served if row["recipe"]["name"] not in names]
    current_platform = deployed_platform(platform["frontend"], environment)
    if current_platform is not None and current_platform != platform:
        if unlisted:
            raise ValueError(
                "This deploy changes the frontend's platform settings, so "
                "recipes it does not list would keep apps built for the old "
                f"settings: {sorted(row['recipe']['name'] for row in unlisted)}. "
                "List them in this deploy too."
            )
        trainers.update(names)
        inference.update(names)
    # The frontend's idle timeouts and sweep interval come from the listed recipes,
    # so they must match the recipes this deploy leaves serving.
    changed = {
        field: sorted(
            current.name
            for current in current_configs
            if current.name not in names
            and getattr(current.recipe, field) != getattr(configs[0].recipe, field)
        )
        for field in FRONTEND_FIELDS
    }
    if any(changed.values()):
        raise ValueError(
            "This deploy changes frontend-wide settings that recipes it does not "
            "list rely on: "
            + "; ".join(
                f"{field} for {recipes}"
                for field, recipes in changed.items()
                if recipes
            )
            + ". Match their settings or list them in this deploy too."
        )
    trainers, inference = apps_to_publish(
        configs, current_configs, environment, trainers, inference
    )
    pools = deployed_pools(platform["frontend"], environment) if inference else []

    def deploy_trainer(deployment):
        build_trainer_app(deployment, platform)[0].deploy(
            environment_name=environment, tag=deployment_tag()
        )

    def deploy_inference(deployment):
        build_inference_app(deployment, platform)[0].deploy(
            environment_name=environment, tag=deployment_tag()
        )
        for record in pools:
            if record["definition_id"] != deployment.definition_id:
                continue
            pool = (
                LoraPoolSpec.from_dict(record)
                if deployment.parameterization == "lora"
                else FFTPoolSpec.from_dict(record)
            )
            # Pools named for another frontend (from a release before --app
            # named them) are never routed to again; leave them to be stopped.
            if pool.frontend == platform["frontend"] and not app_is_missing(
                pool.app_name, environment
            ):
                build_rollout_app(deployment, pool, platform)[0].deploy(
                    environment_name=environment, tag=deployment_tag()
                )

    def register():
        # An older frontend serves recipes without a registry; carry them over.
        return write_registry(
            platform["frontend"],
            environment,
            [deployment.model_dump(mode="json") for deployment in configs],
            keep_existing=unlisted,
        )

    def deploy_frontend():
        # Fill the registry before the swap when the frontend already exists,
        # so nothing it serves drops out; a new frontend gets it afterwards.
        registered = register()
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
                FRONTEND_SETTINGS_ENV: json.dumps(asdict(settings)),
                "SPINDLE_APP_NAME": platform["frontend"],
            },
            **captured,
        )
        if not registered:
            register()

    def start_weights(deployment):
        try:
            downloading = start_weights_download(
                platform["frontend"], environment, deployment.definition_id
            )
        except Exception as exc:
            return f"not started ({exc}); the first request starts the download"
        if downloading is None:
            return "not started; the first request starts the download"
        return "downloading in the background" if downloading else "ready"

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
    for deployment in configs:
        steps.append(
            (f"Weights    {deployment.name}", partial(start_weights, deployment))
        )
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


# Stop order: the frontend and inference apps create pools, so they go before
# the pools; trainers only use them.
APP_ROLES = ("inference", "fft", "lora", "trainer")


def deployment_apps(frontend, environment):
    """Running apps of one deployment, in the order to stop them."""
    command = [sys.executable, "-m", "modal", "app", "list", "--json"]
    if environment:
        command += ["--env", environment]
    rows = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
    )
    prefixes = [f"{frontend}-{role}-" for role in APP_ROLES]

    def rank(name):
        if name == frontend:
            return 0
        return next((i for i, p in enumerate(prefixes, 1) if name.startswith(p)), None)

    names = {
        row["description"] or ""
        for row in rows
        if not str(row["state"]).startswith("stop")
    }
    return sorted(
        (name for name in names if rank(name) is not None),
        key=lambda name: (rank(name), name),
    )


def _already_stopped(output):
    return "App is already stopped." in output or "No App with name" in output


def stop(frontend, environment):
    """Stop every app of the deployment, ending its runs."""
    names = deployment_apps(frontend, environment)
    # Looked up even when the frontend is already stopping, so its folder is deleted.
    app_id = frontend_app_id(frontend, environment)
    if not names:
        print(f"No running apps for {frontend}.")
    for name in names:
        command = [sys.executable, "-m", "modal", "app", "stop", "-y", name]
        if environment:
            command += ["--env", environment]
        result = subprocess.run(command, capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()
        if result.returncode and not _already_stopped(output):
            raise ValueError(f"stopping {name} failed: {output}")
        print(f"Stopped {name}")
    if app_id is not None:
        delete_registry(app_id, environment)
        print(f"Deleted the registry for {frontend} ({app_id})")


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


def _frontend_client():
    base_url = os.environ.get("TINKER_BASE_URL")
    api_key = os.environ.get("TINKER_API_KEY")
    if not base_url or not api_key:
        raise ValueError(
            "Set TINKER_BASE_URL and TINKER_API_KEY to the Spindle frontend's URL and API key."
        )
    headers = {"X-API-Key": api_key}
    if os.environ.get("MODAL_PROXY_TOKEN_ID"):
        headers["Modal-Key"] = os.environ["MODAL_PROXY_TOKEN_ID"]
        headers["Modal-Secret"] = os.environ.get("MODAL_PROXY_TOKEN_SECRET", "")
    return httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=60)


def _response_body(response, action):
    if response.status_code == 405:
        raise ValueError(
            "This frontend predates the deployments API; redeploy it once with spindle deploy."
        )
    if response.is_error:
        try:
            message = response.json()["message"]
        except (ValueError, KeyError, TypeError):
            message = response.text
        raise ValueError(f"{action} failed ({response.status_code}): {message}")
    return response.json()


def add_deployment(
    name, *, base, overrides, poll_s=ADD_POLL_S, timeout_s=ADD_TIMEOUT_S
):
    """Register one config through the frontend's API and wait until its apps are ready."""
    path = "/api/v1/spindle/deployments"
    deadline = time.monotonic() + timeout_s
    with _frontend_client() as client:
        body = {"name": name, "base": base, "overrides": overrides}
        _response_body(client.post(path, json=body), f"Adding {name}")
        print(f"Registered {name}; the frontend is deploying its apps.")
        while True:
            listed = _response_body(client.get(path), "Listing deployments")
            row = next((d for d in listed["deployments"] if d["name"] == name), None)
            if row is None:
                raise ValueError(f"{name} is no longer registered")
            if "apps" not in row:
                raise ValueError(
                    "This frontend does not deploy apps; redeploy it once with spindle deploy."
                )
            if row["apps"] == "ready":
                break
            if row["apps"] == "missing":
                # No deploy is running (one never started); posting again starts it.
                _response_body(client.post(path, json=body), f"Adding {name}")
            if row["apps"] == "failed":
                raise ValueError(
                    f"Deploying {name}'s apps failed: {row.get('apps_error')}. "
                    "Run the same spindle add again to retry."
                )
            if time.monotonic() > deadline:
                raise ValueError(f"{name}'s apps were not ready after {timeout_s}s.")
            time.sleep(poll_s)
    print(f"Added {name}. Train on it by passing base_model={name!r}.")


def remove_deployment(name, *, force=False, confirm=None):
    """Remove one config from a running frontend, warning first if models train on it."""
    path = f"/api/v1/spindle/deployments/{name}"
    with _frontend_client() as client:
        response = client.delete(path)
        if response.status_code == 409:
            print(f"Warning: removing {name} ends training for these models:")
            for model in response.json().get("models", []):
                print(f"  {model['model_id']} (session {model['session_id']})")
            if not force and (confirm is None or not confirm(f"Remove {name} anyway?")):
                raise ValueError(
                    f"Kept {name}. Rerun with --force to remove it anyway."
                )
            response = client.delete(path, params={"force": "true"})
        _response_body(response, f"Removing {name}")
    print(f"Removed {name} and stopped its apps.")


def parse_fields(tokens):
    """`--FIELD=VALUE` / `--FIELD VALUE` tokens into recipe overrides."""
    overrides = {}
    tokens = iter(tokens)
    for token in tokens:
        field, separator, value = token.removeprefix("--").partition("=")
        if not token.startswith("--") or not field:
            raise ValueError(f"expected a --FIELD=VALUE recipe field, got {token!r}")
        if field.split(".", 1)[0] not in RECIPE_FIELDS:
            raise ValueError(f"unknown recipe field: {field}")
        if field == "name":
            raise ValueError("pass the config name as the first argument, not --name")
        if field in overrides:
            raise ValueError(f"duplicate field: {field}")
        if not separator:
            value = next(tokens, None)
            if value is None or value.startswith("--"):
                raise ValueError(f"missing value for --{field}")
        try:
            overrides[field] = json.loads(value)
        except ValueError:
            overrides[field] = value
    return overrides


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
    add = commands.add_parser(
        "add",
        help="Add a config to a running frontend: spindle add NAME --base BASE --FIELD=VALUE ...",
        allow_abbrev=False,
    )
    add.add_argument("name")
    add.add_argument("--base", required=True, help="Deployed config to start from")
    remove = commands.add_parser(
        "remove", help="Remove a config from a running frontend"
    )
    remove.add_argument("name")
    remove.add_argument(
        "--force",
        action="store_true",
        help="Remove it even if models still train on it, without asking",
    )
    halt = commands.add_parser(
        "stop", help="Stop a deployment's frontend, trainers, and sampler pools"
    )
    halt.add_argument("--app")
    halt.add_argument("--env")
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
    args, fields = cli.parse_known_args(argv)
    if fields and args.command != "add":
        cli.error(f"unrecognized arguments: {' '.join(fields)}")
    try:
        if args.command == "add":
            add_deployment(args.name, base=args.base, overrides=parse_fields(fields))
        elif args.command == "remove":
            remove_deployment(
                args.name,
                force=args.force,
                confirm=(
                    (
                        lambda question: (
                            input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
                        )
                    )
                    if interactive()
                    else None
                ),
            )
        elif args.command == "config":
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
        elif args.command == "stop":
            stop(
                args.app or BaseConfig.platform["frontend"],
                args.env or os.environ.get("MODAL_ENVIRONMENT"),
            )
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
