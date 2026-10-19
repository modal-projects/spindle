"""Modal app tags that identify Spindle apps by role, model, and release."""

from __future__ import annotations

import re

from spindle.version import version_tags

SOURCE = "spindle"


def app_tags(
    job_type: str, *, framework: str | None = None, model: str | None = None
) -> dict[str, str]:
    tags = {"_modal_source": SOURCE, "_modal_job_type": job_type}
    if framework:
        tags["_modal_framework"] = framework
    if model:
        tags["_modal_model_name"] = tag_value(model)
    return {**tags, **version_tags()}


def tag_value(value: str) -> str:
    name = value.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9_.-]+", "-", name).strip("-_.")
