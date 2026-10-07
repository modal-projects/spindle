"""Exercise independent client progress through the real Engine scheduler on CPU."""

import asyncio
import base64
import struct
import time

import pytest
from types import SimpleNamespace

import glm53_dapo as experiment
from spindle.engine import Engine
from tests.support import EchoExecutor


@pytest.mark.parametrize("clients", [2, 8])
def test_clients_advance_independently_and_publish_their_own_version(
    tmp_path, monkeypatch, clients
):
    forwarded = []

    class Executor(EchoExecutor):
        async def execute_forward_backward_batch(self, commands):
            results = []
            for command in commands:
                forwarded.append(command.model_id)
                results.append(
                    {
                        "loss_fn_outputs": [
                            {
                                "logprobs": {
                                    "data": datum.loss_fn_inputs["logprobs"].data
                                }
                            }
                            for datum in command.payload.data
                        ]
                    }
                )
                assert all(
                    "routed_experts" in d.loss_fn_inputs for d in command.payload.data
                )
            return tuple(results)

        async def execute(self, model_id, kind, payload):
            return {"metrics": {"update_successful:mean": 1, "grad_norm:mean": 1.0}}

        async def capture_snapshot(self, model_id, kind, payload):
            return payload

        async def persist_snapshot(self, model_id, kind, payload, capture):
            await asyncio.sleep(0.01)
            return {}

    def sample(samplers, jobs, config, version):
        client, problem = jobs[0]
        if client == 1 and version == 0:
            time.sleep(0.2)
        routes = ([-1] * 24 + list(range(8)) * 42) * 3
        encoded = base64.b64encode(struct.pack(f"<{len(routes)}i", *routes)).decode()
        return [
            {
                "client": client,
                "problem": problem,
                "samples": [
                    {
                        "tokens": [3, 4],
                        "logprobs": [-0.1, -0.2],
                        "reward": 1.0,
                        "correct": True,
                        "truncated": False,
                        "routed_experts": encoded,
                    }
                    for _ in range(2)
                ],
            }
        ]

    monkeypatch.setattr(experiment, "sample_problems", sample)
    probes = []

    def probe(operation, payload):
        probes.append(
            (payload["weight_run_id"], payload["weight_version"]["exact_version"])
        )
        return {
            "meta_info": {
                "weight_version_start": payload["weight_version"]["exact_version"]
            }
        }

    config = dict(
        run_id="test",
        clients=clients,
        steps=3,
        groups=1,
        group_size=2,
        lora_rank=16,
        concurrency=clients * 2,
        stagger_s=0,
        functional_validation=True,
        clip_low=0.8,
        clip_high=1.28,
        learning_rate=1e-5,
        beta1=0.9,
        beta2=0.98,
        weight_decay=0.1,
    )
    dataset = {
        "orders": [list(range(3)) for _ in range(clients)],
        "train": [{"id": i, "tokens": [1, 2]} for i in range(3)],
        "eval": [{"tokens": [1, 2]}],
    }
    samplers = [SimpleNamespace(request=SimpleNamespace(remote=probe))] * 2

    async def run():
        engine = Engine(Executor(), max_models=clients)
        try:
            return await asyncio.wait_for(
                experiment.run_clients(
                    engine,
                    samplers,
                    config,
                    dataset,
                    tmp_path,
                    SimpleNamespace(commit=lambda: None),
                ),
                timeout=15,
            )
        finally:
            await engine.close()

    report = asyncio.run(run())
    assert len(report["client_steps"]) == clients * 3
    rows = {(r["client"], r["step"]): r for r in report["client_steps"]}
    assert rows[0, 2]["start_time"] < rows[1, 1]["end_time"]
    assert sorted(probes) == [
        (f"test-client{c}", s) for c in range(clients) for s in (1, 2, 3)
    ]
    assert len(forwarded) == clients * 3
    for client in range(clients):
        assert [
            rows[client, step]["behavior_policy_version"] for step in (1, 2, 3)
        ] == [0, 0, 1]
        for step in (2, 3):
            assert rows[client, step]["start_time"] < rows[client, step - 1]["end_time"]
            assert rows[client, step]["policy_lag"] == 1
