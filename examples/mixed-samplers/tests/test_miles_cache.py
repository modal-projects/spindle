import json
import random

import pytest
from mixed_samplers import train_miles


def make_cache(path, tasks):
    manifest = {
        "backend": "miles",
        "parameterization": "lora",
        "model": train_miles.config.model,
        "model_revision": train_miles.config.model_revision,
        "rank": 32,
        "alpha": 32,
        "dataset": train_miles.DATASET,
        "dataset_revision": train_miles.DATASET_REVISION,
        "groups": train_miles.GROUPS,
        "top_p": train_miles.TOP_P,
        "temperature": train_miles.TEMPERATURE,
        "batch_per_client": [128, 8],
        "clients": 2,
        "client_seeds": [42, 1000042],
        "initial_versions": [1, 1],
        "capacity": {
            group["name"]: {"tokens_per_second": index + 1.0}
            for index, group in enumerate(train_miles.GROUPS)
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    for client, seed in enumerate(manifest["client_seeds"]):
        directory = path / f"client-{client}"
        directory.mkdir()
        order = list(range(len(tasks)))
        random.Random(seed).shuffle(order)
        rows = [
            {
                "event": "prompt_started",
                "prompt": 0,
                "instance_id": tasks[order[0]]["instance_id"],
                "groups": ["b200-nvfp4"] * 8,
            },
            {"event": "trajectory_completed", "prompt": 0, "index": client},
        ]
        (directory / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return manifest


def test_initial_cache_keeps_clients_seeds_and_original_group_capacity(tmp_path):
    tasks = [{"instance_id": str(index)} for index in range(20)]
    manifest = make_cache(tmp_path, tasks)
    cache = train_miles.initial_client_rollouts(tmp_path, tasks, 2)
    assert cache["capacity"] == manifest["capacity"]
    assert cache["clients"][0]["completed"] == {(0, 0)}
    assert cache["clients"][1]["completed"] == {(0, 1)}
    with pytest.raises(ValueError, match="different task order"):
        train_miles.initial_client_rollouts(tmp_path, list(reversed(tasks)), 2)
    with (tmp_path / "client-1/events.jsonl").open("a") as out:
        out.write('{"event":"training_completed"}\n')
    with pytest.raises(ValueError, match="precede any completed update"):
        train_miles.initial_client_rollouts(tmp_path, tasks, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "megatron"),
        ("rank", 16),
        ("initial_versions", [1, 2]),
        ("client_seeds", [42, 42]),
        ("dataset_revision", "different"),
        ("top_p", 1.0),
        ("capacity", {}),
    ],
)
def test_initial_cache_rejects_incompatible_policies(tmp_path, field, value):
    tasks = [{"instance_id": str(index)} for index in range(20)]
    manifest = make_cache(tmp_path, tasks)
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        train_miles.initial_client_rollouts(tmp_path, tasks, 2)
