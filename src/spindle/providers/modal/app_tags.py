"""Fixed attribution for every Modal app created by Spindle."""


def app_tags(role: str) -> dict[str, str]:
    return {"_modal_source": "spindle", "spindle_role": role}
