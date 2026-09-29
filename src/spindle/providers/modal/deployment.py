import os

APP_NAME_ENV = "SPINDLE_APP_NAME"


FORWARDED_DEPLOYMENT_ENVS = (
    APP_NAME_ENV,
    "SPINDLE_TORCH_PROFILE_STEP",
    "SPINDLE_TORCH_PROFILE_DIR",
    "SPINDLE_TORCH_PROFILE_RANKS",
    "SPINDLE_REQUEST_TIMING",
)


def trainer_deployment_env() -> dict[str, str]:
    return {
        name: value
        for name in FORWARDED_DEPLOYMENT_ENVS
        if (value := os.environ.get(name)) is not None
    }
