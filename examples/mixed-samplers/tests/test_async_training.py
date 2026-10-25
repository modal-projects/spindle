import asyncio
import gzip
import importlib
import json
import pickle
import time
from types import SimpleNamespace

training_module = importlib.import_module("mixed_samplers.train")


def test_producers_continue_during_native_training_and_use_old_rollouts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training_module, "PROMPTS_PER_UPDATE", 2)
    monkeypatch.setattr(training_module, "TRAJECTORIES_PER_PROMPT", 2)

    async def sample(task, gateway, group, model_id, version, seed):
        await asyncio.sleep(0.01)
        row = {
            "tokens": [1, 2],
            "active": [0, 1],
            "logq": [0.0, 0.0],
            "offsets": [0, 0, 1],
            "support": [2],
            "support_logq": [0.0],
            "reward": float(seed % 3 == 0),
            "format_error": False,
            "generated_tokens": 1,
            "versions": [{"start": 1, "end": 1}],
            "group": group,
            "started_at": time.time(),
            "finished_at": time.time(),
            "stop_reason": "submit",
            "turns": 1,
        }
        return gzip.compress(pickle.dumps(row))

    monkeypatch.setattr(
        training_module,
        "sample_trajectory",
        SimpleNamespace(remote=SimpleNamespace(aio=sample)),
    )

    class Future:
        def __init__(self, result):
            self.result = result

        async def result_async(self):
            await asyncio.sleep(0.03)
            return self.result

    class Training:
        model_id = "test"

        def forward_backward(self, data, loss, config):
            assert loss == "score_centered_mis"
            assert len(data) == 2
            return Future(SimpleNamespace(metrics={"entropy:mean": 0.0}))

        def optim_step(self, params):
            return Future(SimpleNamespace(metrics={"update_successful:mean": 1.0}))

        def save_weights_for_sampler(self, name):
            return Future(SimpleNamespace(path=name))

        def save_state(self, name):
            return Future(SimpleNamespace(path=name))

    args = SimpleNamespace(
        output=str(tmp_path), inflight_prompts=3, start_step=0, steps=2
    )
    asyncio.run(
        training_module.train(
            Training(), {"b200-nvfp4": "unused"}, [1.0], args, [{"instance_id": "one"}]
        )
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    starts = [e for e in events if e["event"] == "training_started"]
    finishes = [e for e in events if e["event"] == "training_completed"]
    assert len(finishes) == 2
    assert any(
        starts[0]["at"] < e["at"] < finishes[0]["at"]
        for e in events
        if e["event"] == "prompt_started"
    )
    assert finishes[1]["version"] == 3
    # The second update consumed the still-valid version-1 samples.
    assert all(
        e["versions"][0]["start"] == 1
        for e in events
        if e["event"] == "trajectory_completed"
    )
