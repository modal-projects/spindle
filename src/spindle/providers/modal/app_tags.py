"""Attribution for every Modal app created by Spindle."""

import re

from spindle.version import version_tags

JOB_TYPES = {
    "frontend": "control-plane",
    "trainer": "training_spindle",
    "inference-provisioner": "serving",
    "inference": "serving",
    "scoped": "training_spindle",
}


RESERVED_PREFIXES = ("_modal_", "spindle_")
# Modal's tag rule, checked up front so a bad tag fails before any app deploys.
TAG_PATTERN = re.compile(r"[a-zA-Z0-9._-]{1,63}")


def validate_user_tags(tags: dict[str, str]) -> None:
    for key, value in tags.items():
        for text in (key, value):
            if not isinstance(text, str) or not TAG_PATTERN.fullmatch(text):
                raise ValueError(
                    f"invalid app tag {key!r}: {value!r}; keys and values must be "
                    "1-63 letters, digits, '.', '_' or '-'"
                )
        if key.startswith(RESERVED_PREFIXES):
            raise ValueError(f"app tag key {key!r} is reserved for Spindle")


def metadata_tags(metadata: dict[str, object]) -> dict[str, str]:
    """Tags from Tinker user_metadata, rewriting what Modal would reject."""
    tags = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        key, value = (
            re.sub(r"[^a-zA-Z0-9._-]", "-", text)[:63] for text in (key, value)
        )
        if key and value and not key.startswith(RESERVED_PREFIXES):
            tags[key] = value
    return tags


def app_tags(
    role: str,
    *,
    model: str = "",
    framework: str = "",
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    extra = extra or {}
    validate_user_tags(extra)
    tags = {
        **extra,
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
