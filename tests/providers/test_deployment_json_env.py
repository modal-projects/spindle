import json

from spindle.providers.modal.deployment_configs import image_env_json, parse_env_json


def test_nested_json_survives_image_environment_and_reimport():
    config = {
        "inference_env": {
            "SGLANG_SPINDLE_MERGE_PROFILE": json.dumps({"enabled": True})
        },
        "path": "a\\b",
        "text": 'quoted "value"',
    }
    raw = json.dumps(config)
    encoded = image_env_json(raw)
    assert '"' not in encoded and "\\" not in encoded
    assert parse_env_json(encoded) == parse_env_json(raw) == config
    assert image_env_json(encoded) == encoded
