"""Attribution for every Modal app created by Spindle."""

from spindle.version import version_tags

JOB_TYPES = {
    "frontend": "control-plane",
    "trainer": "training_spindle",
    "inference-provisioner": "serving",
    "inference": "serving",
    "scoped": "training_spindle",
}


def app_tags(role: str, *, model: str = "", framework: str = "") -> dict[str, str]:
    tags = {
        "_modal_source": "spindle",
        "spindle_role": role,
        # Dojo's report also matches job_type=training without checking source.
        "_modal_job_type": JOB_TYPES[role],
        **version_tags(),
    }
    if framework:
        tags["_modal_framework"] = framework
    if model:
        tags["_modal_model_name"] = model.rsplit("/", 1)[-1].lower()
    return tags
