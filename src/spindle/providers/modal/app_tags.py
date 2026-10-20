"""Attribution for every Modal app created by Spindle."""

from spindle.resource_scope import SessionAttribution, MODAL_MAX_APP_TAGS

JOB_TYPES = {
    "frontend": "control-plane",
    "trainer": "training_spindle",
    "inference-provisioner": "serving",
    "inference": "serving",
    "scoped": "training_spindle",
}


def app_tags(
    role: str,
    *,
    model: str = "",
    framework: str = "",
    attribution: SessionAttribution | None = None,
) -> dict[str, str]:
    tags = {
        "_modal_source": "spindle",
        "spindle_role": role,
        # Dojo's report also matches job_type=training without checking source.
        "_modal_job_type": JOB_TYPES[role],
    }
    if framework:
        tags["_modal_framework"] = framework
    if model:
        tags["_modal_model_name"] = model.rsplit("/", 1)[-1].lower()
    if attribution is not None:
        tags.update(attribution.app_tags())
        if len(tags) > MODAL_MAX_APP_TAGS:
            tags.pop("_modal_framework", None)
    if len(tags) > MODAL_MAX_APP_TAGS:
        raise ValueError("Modal permits at most eight app tags")
    return tags
