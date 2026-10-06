"""CPU checks; run with scripts and the pinned Miles checkout on PYTHONPATH."""

import ast
import base64
import struct
import json
from contextlib import suppress
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import glm53_dapo_data as data_helpers

import numpy as np
import pytest

pytest.importorskip("miles.rollout.rm_hub.math_dapo_utils")

from glm53_dapo_data import (
    collect_training_groups,
    grade_sample,
    group_advantages,
    sample_problems,
    save_rollouts,
    write_json,
    training_data,
)


def response(answer="34", *, truncated=False):
    return {
        "text": f"Calculation.\nAnswer: \\boxed{{{answer}}}",
        "output_ids": [7, 8],
        "meta_info": {
            "finish_reason": {"type": "length" if truncated else "stop"},
            "output_token_logprobs": [[-0.2, 7, None], [-0.3, 8, None]],
            "weight_version_start": 3,
        },
    }


def test_dataset_reward_and_token_alignment():
    assert grade_sample(response(), {"answer": "34"})["reward"] == 1
    assert grade_sample(response("35"), {"answer": "34"})["reward"] == -1
    bad = response()
    bad["meta_info"]["output_token_logprobs"][0][1] = 99
    with pytest.raises(AssertionError, match="do not match"):
        grade_sample(bad, {"answer": "34"})


def test_incomplete_responses_do_not_create_a_training_signal():
    samples = [
        {"reward": 1.0, "truncated": False},
        {"reward": 1.0, "truncated": False},
        {"reward": -1.0, "truncated": True},
    ]
    np.testing.assert_array_equal(group_advantages(samples), [0, 0, 0])
    samples[1]["reward"] = -1.0
    advantages = group_advantages(samples)
    assert advantages[2] == 0
    assert advantages[0] == pytest.approx(-advantages[1])
    assert np.std(advantages[:2], ddof=1) == pytest.approx(1.0, abs=1e-5)


def test_prompt_mask_and_global_token_normalization():
    good = grade_sample(response(), {"answer": "34"})
    bad = grade_sample(response("35"), {"answer": "34"})
    bad.update(tokens=[7, 8, 9, 10], logprobs=[-0.2, -0.3, -0.4, -0.5])
    truncated = grade_sample(response(truncated=True), {"answer": "34"})
    groups = [{"problem": {"tokens": [1, 2, 3]}, "samples": [good, bad, truncated]}]
    data, replay, total = training_data(groups)
    assert total == 6  # Excludes the two incomplete-response tokens.
    assert data[0].model_input.to_ints() == [1, 2, 3, 7]
    assert data[0].loss_fn_inputs["target_tokens"].data == [2, 3, 7, 8]
    advantages = group_advantages(groups[0]["samples"])
    for datum, sample, advantage in zip(
        data, groups[0]["samples"], advantages, strict=True
    ):
        weights = datum.loss_fn_inputs["advantages"].data
        assert weights[:2] == [0, 0]
        np.testing.assert_allclose(weights[2:], advantage / total)
        np.testing.assert_allclose(
            datum.loss_fn_inputs["logprobs"].data[2:], sample["logprobs"], rtol=1e-7
        )
    assert replay[1] == (2, bad["logprobs"])


def test_mixed_rollouts_pin_each_clients_adapter():
    received = []

    def generate(operation, payload):
        assert operation == "generate"
        received.append(payload)
        return response()

    config = {
        "run_id": "experiment",
        "group_size": 2,
        "seed": 42,
        "max_tokens": 8192,
        "concurrency": 4,
    }
    sampler = SimpleNamespace(request=SimpleNamespace(remote=generate))
    problem = {"id": 5, "tokens": [1, 2], "answer": "34"}
    groups = sample_problems(sampler, [(0, problem), (1, problem)], config, 3)
    assert [g["client"] for g in groups] == [0, 1]
    assert {p["weight_run_id"] for p in received} == {
        "experiment-client0",
        "experiment-client1",
    }
    assert all(p["weight_version"] == {"exact_version": 3} for p in received)
    assert len({p["sampling_params"]["sampling_seed"] for p in received}) == 4
    with pytest.raises(AssertionError, match="wrong policy"):
        sample_problems(sampler, [(0, problem)], config, 4)


def test_fixed_batches_reuse_saved_groups_and_keep_constant_rewards(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(data_helpers, "write_status", MagicMock())
    problems = [{"id": i, "tokens": [i + 10], "answer": "34"} for i in range(3)]
    dataset = {"train": problems, "orders": [[0, 1, 2], [0, 1, 2]]}
    config = {
        "run_id": "test",
        "clients": 2,
        "groups": 3,
        "group_size": 2,
        "concurrency": 4,
        "seed": 42,
        "max_tokens": 8192,
        "continue_run": "test",
    }
    cached_response = response()
    cached_response["meta_info"]["weight_version_start"] = 0
    cached_sample = grade_sample(cached_response, problems[0])
    for client in range(2):
        save_rollouts(
            tmp_path / f"client{client}-train-0001.jsonl.gz",
            [
                {
                    "client": client,
                    "problem": problems[0],
                    "samples": [cached_sample, cached_sample],
                }
            ],
        )
    received = []

    def generate(operation, payload):
        received.append(payload)
        return cached_response

    cursors = [0, 0]
    sampler = SimpleNamespace(request=SimpleNamespace(remote=generate))
    groups = collect_training_groups(sampler, dataset, config, 1, cursors, tmp_path)
    assert [len(batch) for batch in groups] == [3, 3]
    assert cursors == [3, 3]
    assert len(received) == 8
    assert all(p["input_ids"] != [10] for p in received)
    assert all(
        not np.any(group_advantages(g["samples"])) for batch in groups for g in batch
    )
    assert not list(tmp_path.glob("*.tmp"))
    # A fresh controller can reuse all completed groups without any generation.
    received.clear()
    collect_training_groups(sampler, dataset, config, 1, [0, 0], tmp_path)
    assert received == []


def test_sampling_refills_before_a_slow_group_finishes():
    later_started = Event()
    config = {
        "run_id": "test",
        "group_size": 2,
        "seed": 42,
        "max_tokens": 8192,
        "concurrency": 4,
    }
    jobs = [(0, {"id": i, "tokens": [i], "answer": "34"}) for i in range(3)]

    def generate(operation, payload):
        if payload["input_ids"] == [0]:
            assert later_started.wait(5), "Waited for a whole wave before refilling"
        if payload["input_ids"] == [2]:
            later_started.set()
        return response()

    sampler = SimpleNamespace(request=SimpleNamespace(remote=generate))
    groups = sample_problems(sampler, jobs, config, 3)
    assert len(groups) == 3
    assert all(len(g["samples"]) == 2 for g in groups)


def controller_fixture(tmp_path):
    # Execute the controller body with fake Modal objects; importing the launcher
    # would build CUDA images and bind deployment resources in this CPU test.
    tree = ast.parse(Path("scripts/run_glm53_dapo.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    function.decorator_list = []
    call = MagicMock(object_id="fc-existing")
    sampler = MagicMock()
    sampler.request.remote.return_value = {"dtype": "bfloat16", "quantization": None}
    sampler.request.spawn.return_value.get.return_value = {
        "dtype": "bfloat16",
        "quantization": None,
        "cuda_graph_config": {"decode": {"backend": "full"}},
    }
    helpers = {
        "prepare": MagicMock(),
        "baseline": MagicMock(),
        "write_json": write_json,
        "write_status": MagicMock(),
    }
    env = {
        "Path": lambda *parts: tmp_path.joinpath(str(parts[0]).lstrip("/"), *parts[1:]),
        "sys": SimpleNamespace(path=[]),
        "runpy": SimpleNamespace(run_path=lambda _: helpers),
        "json": json,
        "suppress": suppress,
        "assets": MagicMock(),
        "results": MagicMock(),
        "Sampler": MagicMock(return_value=sampler),
        "train": MagicMock(),
        "modal": MagicMock(),
    }
    env["train"].spawn.return_value = call
    env["modal"].FunctionCall.from_id.return_value = call
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "controller", "exec"), env
    )
    config = {
        "run_id": "test",
        "app_id": "ap-test",
        "resume": "",
        "prepare_only": False,
        "rollout_replicas": 1,
    }
    write_json(tmp_path / "checkpoints/test/baseline.json", {})
    return env, config, call, sampler


def test_controller_reattaches_after_cpu_preemption(tmp_path):
    env, config, call, sampler = controller_fixture(tmp_path)
    call.get.side_effect = [KeyboardInterrupt(), {"ok": True}]
    with pytest.raises(KeyboardInterrupt):
        env["run"]({}, {}, "{}", config)
    call.cancel.assert_not_called()
    sampler.update_autoscaler.assert_not_called()
    sampler.request.spawn.assert_called_with("ready")
    type(call).object_id = PropertyMock(side_effect=AttributeError("unhydrated"))
    assert env["run"]({}, {}, "{}", config) == {"ok": True}
    env["train"].spawn.assert_called_once()
    env["modal"].FunctionCall.from_id.assert_called_once_with("fc-existing")
    sampler.request.remote.assert_called_with("stop")
    sampler.update_autoscaler.assert_called_once_with(
        min_containers=0, max_containers=0
    )


def test_controller_cancels_trainer_on_failure(tmp_path):
    env, config, call, sampler = controller_fixture(tmp_path)
    call.get.side_effect = RuntimeError("trainer failed")
    with pytest.raises(RuntimeError, match="trainer failed"):
        env["run"]({}, {}, "{}", config)
    call.cancel.assert_called_once_with(terminate_containers=True)
    sampler.request.remote.assert_called_with("stop")
    sampler.update_autoscaler.assert_called_once_with(
        min_containers=0, max_containers=0
    )


def test_sampler_stop_does_not_call_decorated_exit_method():
    tree = ast.parse(Path("scripts/run_glm53_dapo.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Sampler"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "request"
    )
    method.decorator_list = []
    env = {"terminate": MagicMock()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "sampler", "exec"), env)
    sampler = SimpleNamespace(sidecar="sidecar", server="server")
    env["request"](sampler, "stop")
    assert [call.args for call in env["terminate"].call_args_list] == [
        ("sidecar",),
        ("server",),
    ]


@pytest.mark.parametrize("extra_final_token", [False, True])
def test_routing_replay_capture_reaches_training(extra_final_token):
    routes = ([-1] * (3 * 8) + list(range(8)) * 42) * (3 + extra_final_token)
    encoded = base64.b64encode(struct.pack(f"<{len(routes)}i", *routes)).decode()
    result = response()
    result["meta_info"]["routed_experts"] = encoded
    requests = []

    def generate(operation, payload):
        requests.append(payload)
        return result

    sampler = SimpleNamespace(request=SimpleNamespace(remote=generate))
    config = dict(
        run_id="test",
        group_size=2,
        seed=42,
        max_tokens=8,
        concurrency=2,
        routing_replay=True,
    )
    groups = sample_problems(
        sampler, [(0, {"id": 0, "tokens": [1, 2], "answer": "34"})], config, 3
    )
    assert all(
        p["return_routed_experts"] and p["routed_experts_start_len"] == 0
        for p in requests
    )
    datums, _, _ = training_data(groups, routing_replay=True)
    for datum in datums:
        tensor = datum.loss_fn_inputs["routed_experts"]
        assert tensor.shape == [3, 45, 8]
        assert tensor.data == routes[: 3 * 45 * 8]
    groups[0]["samples"][0]["routed_experts"] = None
    with pytest.raises(ValueError, match="missing expert routes"):
        training_data(groups, routing_replay=True)
    requests.clear()
    sample_problems(
        sampler,
        [(0, {"id": 0, "tokens": [1, 2], "answer": "34"})],
        config,
        3,
        evaluation=True,
    )
    assert all("return_routed_experts" not in p for p in requests)


def test_recipe_change_cannot_reuse_old_rollouts(tmp_path, monkeypatch):
    write_json(tmp_path / "config.json", {"recipe": "old"})
    monkeypatch.setattr(
        data_helpers,
        "Path",
        lambda value: tmp_path if value == "/checkpoints" else Path(value),
    )
    with pytest.raises(ValueError, match="fresh run"):
        data_helpers.prepare({"run_id": ".", "recipe": "new"})


def test_clients_route_to_separate_replicas():
    received = [[], []]

    def replica(index):
        def generate(operation, payload):
            received[index].append(payload)
            return response()

        return SimpleNamespace(request=SimpleNamespace(remote=generate))

    config = dict(run_id="test", group_size=2, seed=42, max_tokens=8, concurrency=2)
    problem = {"id": 0, "tokens": [1, 2], "answer": "34"}
    sample_problems([replica(0), replica(1)], [(0, problem), (1, problem)], config, 3)
    assert all(p["weight_run_id"] == "test-client0" for p in received[0])
    assert all(p["weight_run_id"] == "test-client1" for p in received[1])
    assert list(map(len, received)) == [2, 2]


def test_controller_rejects_eager_rollouts_before_launching_trainer(tmp_path):
    env, config, call, sampler = controller_fixture(tmp_path)
    sampler.request.spawn.return_value.get.return_value["cuda_graph_config"]["decode"][
        "backend"
    ] = "disabled"
    with pytest.raises(AssertionError, match="require decode CUDA graphs"):
        env["run"]({}, {}, "{}", config)
    env["train"].spawn.assert_not_called()
    sampler.update_autoscaler.assert_called_once_with(
        min_containers=0, max_containers=0
    )
