"""Attribution for every Modal app created by Spindle."""

JOB_TYPES = {
    "frontend": "control-plane",
    "trainer": "training_spindle",
    "inference-provisioner": "serving",
    "inference": "serving",
    "scoped": "training_spindle",
}


RESERVED_PREFIXES = ("_modal_", "spindle_")


def app_tags(
    role: str,
    *,
    model: str = "",
    framework: str = "",
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    extra = extra or {}
    reserved = sorted(key for key in extra if key.startswith(RESERVED_PREFIXES))
    if reserved:
        raise ValueError(f"app tag keys {reserved} are reserved for Spindle")
    tags = {
        **extra,
        "_modal_source": "spindle",
        "spindle_role": role,
        # Dojo's report also matches job_type=training without checking source.
        "_modal_job_type": JOB_TYPES[role],
    }
    if framework:
        tags["_modal_framework"] = framework
    if model:
        tags["_modal_model_name"] = model.rsplit("/", 1)[-1].lower()
    return tags
