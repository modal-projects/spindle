"""Interactive model picker behind a bare `spindle deploy`."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import modal
import questionary
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig, load

CONFIG_DIR = Path(__file__).with_name("configs")
SERVED_TIMEOUT_S = 20
CONTEXT_CHOICES = (2_048, 16_384, 32_768, 65_536, 131_072, 262_144)
DEFAULT_CONTEXT = 32_768
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
PROVIDER_ORDER = ("Qwen", "openai", "nvidia", "deepseek-ai", "zai-org", "moonshotai")
MODE_LABELS = {"lora": "LoRA", "full": "Full fine-tune"}


class Cancelled(Exception):
    pass


class SetupIncomplete(Exception):
    pass


class RequiredSecret(NamedTuple):
    platform_key: str
    keys: tuple[str, ...]
    purpose: str
    create: str


REQUIRED_SECRETS = (
    RequiredSecret(
        "api",
        ("TINKER_API_KEY",),
        "API key your Tinker clients send to Spindle",
        'modal secret create {name} TINKER_API_KEY="tml-$(openssl rand -hex 32)"',
    ),
    RequiredSecret(
        "sampler_proxy",
        ("MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"),
        "Modal proxy token Spindle uses to reach its sampler pools",
        "modal workspace proxy-tokens create, allow it for your environment, then\n"
        "    modal secret create {name} MODAL_PROXY_TOKEN_ID=wk-... "
        "MODAL_PROXY_TOKEN_SECRET=ws-...",
    ),
    RequiredSecret(
        "huggingface",
        ("HF_TOKEN",),
        "Hugging Face read token for downloading model weights",
        "modal secret create {name} HF_TOKEN=hf_...  "
        "(token from huggingface.co/settings/tokens)",
    ),
)


@dataclass(frozen=True)
class Preset:
    path: Path
    config: BaseConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def mode(self) -> str:
        return self.config.parameterization

    @property
    def context(self) -> int:
        return self.config.max_context_length

    @property
    def trainer_gpu_count(self) -> int:
        return self.config.trainer_nodes * self.config.trainer_gpus_per_node

    @property
    def trainer(self) -> str:
        gpu = self.config.trainer_gpu
        gpu = "/".join(gpu) if isinstance(gpu, list) else gpu
        return f"{self.trainer_gpu_count}× {gpu}"


def bundled_config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.glob("[!_]*.py"))


def bundled_presets() -> list[Preset]:
    """One preset per (model, mode, context); hardware variants stay reachable by name."""
    presets = {}
    for path in bundled_config_paths():
        preset = Preset(path, load(path))
        presets.setdefault((preset.model, preset.mode, preset.context), preset)
    return list(presets.values())


def provider(model: str) -> str:
    return model.split("/")[0]


def models_by_provider(presets: list[Preset]) -> list[str]:
    smallest = {}
    for preset in presets:
        smallest[preset.model] = min(
            smallest.get(preset.model, preset.trainer_gpu_count),
            preset.trainer_gpu_count,
        )

    def rank(model):
        name = provider(model)
        order = (
            PROVIDER_ORDER.index(name)
            if name in PROVIDER_ORDER
            else len(PROVIDER_ORDER)
        )
        return order, name.lower(), smallest[model], model.lower()

    return sorted(smallest, key=rank)


def modes_for(presets: list[Preset], model: str) -> list[str]:
    return [
        mode
        for mode in MODE_LABELS
        if any(p.model == model and p.mode == mode for p in presets)
    ]


def pick(presets: list[Preset], model: str, mode: str, context: int) -> Preset:
    """The shortest context that fits, or the longest available when none does."""
    candidates = sorted(
        (p for p in presets if p.model == model and p.mode == mode),
        key=lambda p: p.context,
    )
    return next((p for p in candidates if p.context >= context), candidates[-1])


def format_context(tokens: int) -> str:
    return f"{tokens // 1024}K"


def model_title(presets: list[Preset], model: str, width: int) -> str:
    own = [p for p in presets if p.model == model]
    modes = ", ".join(MODE_LABELS[mode] for mode in modes_for(presets, model))
    longest = format_context(max(p.context for p in own))
    smallest = min(own, key=lambda p: p.trainer_gpu_count)
    return (
        f"{short_name(model):<{width}}  {modes} · up to {longest} · "
        f"from {smallest.trainer}"
    )


def short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def secret_problem(
    name: str, keys: tuple[str, ...], environment: str | None
) -> str | None:
    try:
        modal.Secret.from_name(
            name, environment_name=environment, required_keys=list(keys)
        ).hydrate()
    except modal.exception.NotFoundError:
        return "not found"
    except modal.exception.ConflictError as exc:
        return str(exc)
    return None


def check_setup(console: Console, environment: str | None) -> None:
    names = BaseConfig.platform["secrets"]
    table = Table(
        title="Setup check", title_justify="left", show_header=False, box=None
    )
    table.add_row(
        "[green]✓[/green]",
        "MODAL_ENVIRONMENT",
        environment or "unset: your profile's default environment",
    )
    fixes = []
    with console.status("Checking your Modal secrets..."):
        for secret in REQUIRED_SECRETS:
            name = names[secret.platform_key]
            label = f"{name} · {', '.join(secret.keys)}"
            problem = secret_problem(name, secret.keys, environment)
            if problem is None:
                table.add_row("[green]✓[/green]", label, secret.purpose)
            else:
                table.add_row(
                    "[red]✗[/red]", label, f"{secret.purpose} [red]({problem})[/red]"
                )
                fixes.append(secret.create.format(name=name))
    console.print(table)
    if fixes:
        env = f" (with MODAL_ENVIRONMENT={environment})" if environment else ""
        console.print(f"\nCreate the missing secrets{env}:")
        for fix in fixes:
            console.print(f"  [bold]{fix}[/bold]")
        raise SetupIncomplete
    console.print()


def ask(question):
    answer = question.ask()
    if answer is None:
        raise Cancelled
    return answer


def check_served(frontend: str, environment: str | None):
    try:
        return modal.Function.from_name(
            frontend, "deployed_configs", environment_name=environment
        ).spawn()
    except modal.exception.NotFoundError:
        return None


def served_configs(console: Console, call) -> list[str] | None:
    """Names the current frontend serves, or None if it doesn't answer in time."""
    if call is None:
        return []
    with console.status("Checking what's already deployed..."):
        try:
            rows = call.get(timeout=SERVED_TIMEOUT_S)
        except (TimeoutError, modal.exception.TimeoutError):
            call.cancel()
            return None
    return [DeploymentConfig.model_validate(row).name for row in rows]


def choose(console: Console, frontend: str, environment: str | None) -> list[Preset]:
    served = check_served(frontend, environment)
    presets = bundled_presets()
    console.print(
        Panel.fit(
            "[bold]Welcome to Spindle[/bold], Modal's Training API.",
            border_style="green",
        )
    )
    check_setup(console, environment)

    models = models_by_provider(presets)
    width = max(len(short_name(model)) for model in models)
    choices = []
    for index, model in enumerate(models):
        if index == 0 or provider(model) != provider(models[index - 1]):
            choices.append(questionary.Separator(f"── {provider(model)} ──"))
        choices.append(
            questionary.Choice(
                model_title(presets, model, width),
                value=model,
                checked=model == DEFAULT_MODEL,
            )
        )
    selected = ask(
        questionary.checkbox(
            "Which models would you like to post-train?",
            choices=choices,
            instruction="(space to select, enter to confirm)",
            validate=lambda picked: bool(picked) or "Pick at least one model.",
        )
    )

    modes = {}
    for model in selected:
        available = modes_for(presets, model)
        if len(available) == 1:
            modes[model] = available
            continue
        modes[model] = ask(
            questionary.checkbox(
                f"How do you want to train {model}?",
                choices=[
                    questionary.Choice(
                        "LoRA: adapters share one base model; works with the stock Tinker SDK",
                        value="lora",
                        checked=True,
                    ),
                    questionary.Choice(
                        "Full fine-tune: dedicated trainers; uses spindle.create_full_training_client",
                        value="full",
                    ),
                ],
                validate=lambda picked: bool(picked) or "Pick at least one.",
            )
        )

    context = ask(
        questionary.select(
            "What's the longest sequence you'll train on (prompt + completion)?",
            choices=[
                questionary.Choice(f"{format_context(tokens)} tokens", value=tokens)
                for tokens in CONTEXT_CHOICES
            ],
            default=DEFAULT_CONTEXT,
            instruction="(not sure? 32K covers most RL and SFT runs)",
        )
    )

    chosen = [
        pick(presets, model, mode, context)
        for model in selected
        for mode in modes[model]
    ]
    show_plan(
        console,
        chosen,
        context,
        served_configs(console, served),
        stop_command(frontend, environment),
    )
    if not ask(
        questionary.confirm(
            f"Deploy {len(chosen)} config{'s' if len(chosen) != 1 else ''} "
            f"to {environment or 'your default environment'}?",
            default=True,
        )
    ):
        raise Cancelled
    return chosen


def deploy_command(chosen: list[Preset]) -> str:
    return "spindle deploy " + " ".join(preset.name for preset in chosen)


def stop_command(frontend: str, environment: str | None) -> str:
    return f"modal app stop {frontend}" + (f" -e {environment}" if environment else "")


def show_plan(
    console: Console,
    chosen: list[Preset],
    context: int,
    served: list[str] | None,
    stop: str,
) -> None:
    console.print()
    table = Table(title="Deployment plan", title_justify="left")
    for column in ("Model", "Training", "Context", "Trainer", "Config"):
        table.add_column(column)
    for preset in chosen:
        fits = preset.context >= context
        table.add_row(
            preset.model,
            MODE_LABELS[preset.mode],
            format_context(preset.context)
            + ("" if fits else " [yellow](max)[/yellow]"),
            preset.trainer,
            preset.name,
        )
    console.print(table)
    if served is None:
        console.print(
            "[yellow]The Spindle frontend already deployed here didn't respond. "
            f"If it's broken, stop it with `{stop}`, then run spindle deploy again.[/yellow]"
        )
        served = []
    dropped = sorted(set(served) - {preset.name for preset in chosen})
    if dropped:
        console.print(
            f"[yellow]This replaces the current deployment: {len(dropped)} config(s) "
            "not picked here will stop being served.[/yellow]"
        )
    console.print(
        f"Skip these prompts next time with:\n  [bold]{deploy_command(chosen)}[/bold]\n"
    )


def deploy_with_progress(
    console: Console, steps, background=(), *, retry_command: str
) -> None:
    """One row per step, in order; `background` (label, future) rows finish last and only warn."""
    console.print(
        "[dim]The first deploy in a workspace builds container images, which can take several minutes.[/dim]"
    )
    progress = Progress(
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )

    def run(task, label, call):
        try:
            call()
        except Exception as exc:
            progress.update(task, description=f"[red]{label} failed[/red]")
            output = getattr(exc, "stderr", None) or getattr(exc, "stdout", None)
            if output:
                console.print(output)
            console.print(
                f"Rerun with full Modal logs:\n  [bold]{retry_command}[/bold]"
            )
            raise
        progress.update(task, completed=1)

    def mark_done(task, future):
        if future.exception() is None:
            progress.update(task, completed=1)

    def finish(task, label, future):
        try:
            future.result()
        except Exception as exc:
            progress.update(task, description=f"[yellow]{label} skipped[/yellow]")
            console.print(
                f"[yellow]{label} failed:[/yellow] {escape(str(exc))}\n"
                "The first request that needs it builds it instead."
            )

    started = time.monotonic()
    with progress:
        waiting = []
        for label, future in background:
            task = progress.add_task(label, total=1)
            future.add_done_callback(partial(mark_done, task))
            waiting.append((task, label, future))
        rows = [
            (progress.add_task(label, total=1, start=False), label, step)
            for label, step in steps
        ]
        for task, label, step in rows:
            progress.start_task(task)
            run(task, label, step)
        for task, label, future in waiting:
            finish(task, label, future)
    console.print(
        f"[green]✓[/green] Deployed in {format_duration(time.monotonic() - started)}"
    )


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def show_next_steps(
    console: Console, chosen: list[Preset], frontend: str, environment: str | None
) -> None:
    url = modal.Function.from_name(
        frontend, "server", environment_name=environment
    ).get_web_url()
    preset = chosen[0]
    if preset.mode == "lora":
        client = (
            f'service.create_lora_training_client(base_model="{preset.model}", rank=16)'
        )
    else:
        client = f'spindle.create_full_training_client(service, "{preset.model}")'
    console.print(
        Panel.fit(
            "[bold green]Spindle is live.[/bold green] Point any Tinker script at it:\n\n"
            f"  export TINKER_BASE_URL={url}\n"
            "  export TINKER_API_KEY=<the key in your spindle-api secret>\n\n"
            f"  {client}",
            border_style="green",
        )
    )
