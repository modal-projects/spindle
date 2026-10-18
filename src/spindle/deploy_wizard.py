"""Interactive model picker behind a bare `spindle deploy`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import modal
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig, load

CONFIG_DIR = Path(__file__).with_name("configs")
SERVED_TIMEOUT_S = 20
CONTEXT_CHOICES = (2_048, 16_384, 32_768, 65_536, 131_072, 262_144)
DEFAULT_CONTEXT = 32_768
MODE_LABELS = {"lora": "LoRA", "full": "Full fine-tune"}


class Cancelled(Exception):
    pass


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


def models_by_size(presets: list[Preset]) -> list[str]:
    smallest = {}
    for preset in presets:
        smallest[preset.model] = min(
            smallest.get(preset.model, preset.trainer_gpu_count),
            preset.trainer_gpu_count,
        )
    return sorted(smallest, key=lambda model: (smallest[model], model.lower()))


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
    return f"{model:<{width}}  {modes} · up to {longest} · from {smallest.trainer}"


def ask(question):
    answer = question.ask()
    if answer is None:
        raise Cancelled
    return answer


def check_served(frontend: str, environment: str | None):
    """Ask the current frontend what it serves without waiting; None if there is none."""
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
            "[bold]Welcome to Spindle[/bold], the API for training on Modal.\n\n"
            "Pick the models you want to post-train. Spindle deploys only those,\n"
            "and GPUs start when your first training request arrives.",
            subtitle=f"environment: {environment or 'default'}",
            border_style="green",
        )
    )

    models = models_by_size(presets)
    width = max(len(model) for model in models)
    selected = ask(
        questionary.checkbox(
            "Which models would you like to post-train?",
            choices=[
                questionary.Choice(model_title(presets, model, width), value=model)
                for model in models
            ],
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
    show_plan(console, chosen, context, served_configs(console, served))
    if not ask(
        questionary.confirm(
            f"Deploy {len(chosen)} config{'s' if len(chosen) != 1 else ''} "
            f"to {environment or 'your default environment'}?",
            default=True,
        )
    ):
        raise Cancelled
    return chosen


def show_plan(
    console: Console, chosen: list[Preset], context: int, served: list[str]
) -> None:
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
            "[yellow]The current Spindle frontend in this environment didn't respond. "
            "If it's broken, stop it with `modal app stop` before deploying.[/yellow]"
        )
        served = []
    dropped = sorted(set(served) - {preset.name for preset in chosen})
    if dropped:
        console.print(
            f"[yellow]This replaces the current deployment: {len(dropped)} config(s) "
            "not picked here will stop being served.[/yellow]"
        )
    console.print(
        "Skip these prompts next time with:\n"
        f"  [bold]spindle deploy {' '.join(preset.name for preset in chosen)}[/bold]\n"
    )


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
            f"  {client}\n\n"
            "[dim]The first request for each model starts GPUs and loads weights,\n"
            "which can take 10+ minutes.[/dim]",
            border_style="green",
        )
    )
