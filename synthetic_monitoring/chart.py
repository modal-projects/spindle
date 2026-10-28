import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone

LOOKBACK_S = 90 * 24 * 3600

STAGES = {
    "trainer ready": "#6ac345",
    "first step": "#85aef8",
    "sample": "#ffa556",
    "forward_backward": "#40c6ed",
    "optim_step": "#c29bfc",
    "save weights": "#fd76f8",
}

_GRAY_00 = "#181818"
_GRAY_10 = "#232323"
_GRAY_20 = "#2f2f2f"
_GRAY_30 = "#464646"
_GRAY_50 = "#a3a3a3"
_GRAY_70 = "#d1d1d1"


@dataclass(frozen=True)
class RunPoint:
    ts: float
    timings: dict[str, float]
    total_duration_s: float
    status: str = "success"
    modal_app_url: str | None = None


def render_timing_history_chart(
    history: list[RunPoint],
    *,
    config_name: str,
) -> bytes | None:
    cutoff = time.time() - LOOKBACK_S
    points = [
        p
        for p in history
        if p.status == "success"
        and p.ts >= cutoff
        and STAGES.keys() <= p.timings.keys()
    ]
    if not points:
        return None

    from matplotlib import pyplot

    tick_labels = [
        datetime.fromtimestamp(p.ts, tz=timezone.utc).strftime("%m-%d\n%H:%M")
        for p in points
    ]
    xs = list(range(len(points)))
    fig, axes = pyplot.subplots(2, 3, figsize=(14, 10.4), layout="constrained")
    fig.patch.set_facecolor(_GRAY_00)
    fig.suptitle(
        f"{config_name} (n={len(points)} runs)",
        color=_GRAY_70,
        fontsize=13,
        fontweight="medium",
    )

    for ax, (stage, color) in zip(axes.flat, STAGES.items()):
        for x, point in enumerate(points):
            ax.bar(x, point.timings[stage], color=color, width=0.72, zorder=2)
        ax.set_title(stage, fontsize=10, pad=10)
        ax.set_ylabel("seconds")
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_facecolor(_GRAY_10)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=_GRAY_20, linewidth=0.8)
        ax.xaxis.grid(False)
        ax.tick_params(colors=_GRAY_50, length=3, width=0.8)
        ax.xaxis.label.set_color(_GRAY_50)
        ax.yaxis.label.set_color(_GRAY_50)
        ax.title.set_color(_GRAY_70)
        for spine in ax.spines.values():
            spine.set_color(_GRAY_30)
            spine.set_linewidth(0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("time (UTC)")

    fig.get_layout_engine().set(h_pad=0.18, hspace=0.18)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_GRAY_00)
    pyplot.close(fig)
    return buf.getvalue()
