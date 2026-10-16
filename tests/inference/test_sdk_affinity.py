import importlib.util
import json
from pathlib import Path

import httpx


spec = importlib.util.spec_from_file_location(
    "sdk_affinity",
    Path(__file__).parents[2] / "scripts/benchmarks/lora_routing/sdk_affinity.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def request(prompt, **extra):
    return httpx.Request(
        "POST",
        "https://spindle.test/api/v1/asample",
        json=dict(
            prompt={"chunks": [{"type": "encoded_text", "tokens": prompt}]},
            num_samples=8,
            seq_id=4,
            sampling_session_id="session",
            **extra,
        ),
        headers={"x-api-key": "test-key"},
        extensions={"timeout": {"read": 300}},
    )


def test_affinity_extension_preserves_sdk_payload_auth_and_timeout():
    original = request([1, 2, 3])
    changed = module.with_prompt_affinity(original)
    body = json.loads(changed.content)
    key = body.pop("cache_affinity_key")
    assert body == json.loads(original.content)
    assert len(key) == 64
    assert changed.headers["x-api-key"] == "test-key"
    assert int(changed.headers["content-length"]) == len(changed.content)
    assert changed.extensions == original.extensions
    assert (
        key
        == json.loads(module.with_prompt_affinity(request([1, 2, 3])).content)[
            "cache_affinity_key"
        ]
    )
    assert (
        key
        != json.loads(module.with_prompt_affinity(request([1, 2, 4])).content)[
            "cache_affinity_key"
        ]
    )


def test_other_commands_and_explicit_keys_are_preserved():
    original = httpx.Request("POST", "https://spindle.test/api/v1/optim_step", json={})
    assert module.with_prompt_affinity(original) is original
    explicit = module.with_prompt_affinity(request([1], cache_affinity_key="given"))
    assert json.loads(explicit.content)["cache_affinity_key"] == "given"
