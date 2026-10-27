import io
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

LOOKBACK_S = 90 * 24 * 3600


@dataclass(frozen=True)
class RunPoint:
    ts: float
    timings: dict[str, float]
    app_name: str
    total_duration_s: float
    status: str = "success"
    url: str | None = None


_BG = "#181818"
_PANEL = "#232323"
_GRID = "#2f2f2f"
_SPINE = "#464646"
_TEXT = "#a3a3a3"
_TITLE = "#d1d1d1"
_BAR = "#85aef8"


def _style_axes(ax) -> None:
    ax.set_facecolor(_PANEL)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.tick_params(colors=_TEXT, length=3, width=0.8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TITLE)
    for spine in ax.spines.values():
        spine.set_color(_SPINE)
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_timing_history_chart(history: list[RunPoint], *, probe_name: str) -> bytes:
    from matplotlib import pyplot

    cutoff = time.time() - LOOKBACK_S
    points = [p for p in history if p.status == "success" and p.ts >= cutoff]
    phases: list[str] = []
    for point in points:
        for phase in point.timings:
            if phase not in phases:
                phases.append(phase)
    if not phases:
        return b""

    series = {phase: [p.timings.get(phase) for p in points] for phase in phases}
    tick_labels = [
        datetime.fromtimestamp(p.ts, tz=timezone.utc).strftime("%m-%d\n%H:%M")
        for p in points
    ]
    xs = list(range(len(points)))
    n_cols = max(1, math.ceil(math.sqrt(len(phases))))
    n_rows = math.ceil(len(phases) / n_cols)
    fig, axes = pyplot.subplots(
        n_rows,
        n_cols,
        figsize=(2 + 4 * n_cols, 2 + 4.2 * n_rows),
        squeeze=False,
        sharey=False,
        layout="constrained",
    )
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        f"{probe_name} (n={len(points)} runs)",
        color=_TITLE,
        fontsize=13,
        fontweight="medium",
    )
    for i, phase in enumerate(phases):
        ax = axes[i // n_cols][i % n_cols]
        for x, y in zip(xs, series[phase], strict=True):
            if y is not None:
                ax.bar(x, y, color=_BAR, width=0.72, zorder=2)
        ax.set_title(phase.replace("_", " "), fontsize=10, pad=10)
        ax.set_ylabel("seconds")
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        _style_axes(ax)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time (UTC)")
    for j in range(len(phases), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)
    fig.get_layout_engine().set(h_pad=0.18, hspace=0.18)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_BG)
    pyplot.close(fig)
    return buf.getvalue()
