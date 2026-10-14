from pathlib import Path
import runpy

import pytest


BENCHMARK = Path(__file__).resolve().parents[2] / "scripts/benchmarks/lora_routing"


def client(monkeypatch, rewards):
    monkeypatch.syspath_prepend(str(BENCHMARK))
    cls = runpy.run_path(str(BENCHMARK / "client.py"))["Client"]
    instance = cls.__new__(cls)
    instance.cfg = dict(
        groups_per_update=2,
        minimum_refill_groups=2,
        max_candidate_groups=4,
        max_tokens=8,
    )
    instance.next_group = 0
    instance.event = lambda *args, **kwargs: None
    instance.group = lambda index, snapshot: dict(
        prompt=[1, 2],
        sequences=[
            dict(tokens=[3, 4], logprobs=[-0.2, -0.3], reward=r) for r in rewards[index]
        ],
    )
    return instance


def test_refills_keep_exact_training_batch_and_account_for_unused_mixed_groups(
    monkeypatch,
):
    runner = client(monkeypatch, [[0, 1], [0, 0], [0, 1], [0, 1]])
    data, stats = runner.rollout((None, 0), 1)
    assert len(data) == 4
    assert stats["groups_sampled"] == 4
    assert stats["groups_filtered"] == stats["groups_unused"] == 1
    assert stats["groups_retained"] == 2
    assert stats["output_tokens"] == 16
    assert stats["used_output_tokens"] == 8


def test_refills_stop_at_candidate_budget_without_training_constant_reward_groups(
    monkeypatch,
):
    runner = client(monkeypatch, [[0, 0]] * 4)
    with pytest.raises(RuntimeError, match="Too few mixed-reward"):
        runner.rollout((None, 0), 1)
    assert runner.next_group == 4
