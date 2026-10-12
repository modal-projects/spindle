import asyncio
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / "client.py"
spec = importlib.util.spec_from_file_location("dapo_client", PATH)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def test_numeric_grading_uses_final_box_and_rejects_nonfinite():
    assert client.answer("thinking 12; \\boxed{1,234}") == 1234
    assert client.answer("\\boxed{1} then \\boxed{2}") == 2
    assert client.answer("NaN") is None
    assert client.answer("Infinity") is None
    assert client.answer("some 42 prose") is None


def test_shift_and_token_normalization():
    row = client.datum([1, 2, 3], [4, 5], [-0.2, -0.3], 2.0, 8)
    assert row.model_input.to_ints() == [1, 2, 3, 4]
    assert row.loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5]
    assert row.loss_fn_inputs["advantages"].data == [0, 0, 0.25, 0.25]
    assert row.loss_fn_inputs["logprobs"].data == pytest.approx([0, 0, -0.2, -0.3])
    with pytest.raises(ValueError):
        client.datum([1], [2], [], 1.0, 1)


def test_dynamic_filter_counts_all_generation(tmp_path):
    c = client.Client.__new__(client.Client)
    c.cfg = dict(groups_per_update=1, max_candidate_groups=4, max_tokens=4)
    c.next_group = 0
    logged = []
    c.event = lambda event, **kwargs: logged.append((event, kwargs))

    def group(i, snapshot):
        return dict(
            prompt=[1, 2],
            sequences=[
                dict(tokens=[3, 4], logprobs=[-0.1, -0.2], reward=0.0),
                dict(tokens=[5], logprobs=[-0.3], reward=float(i > 0)),
            ],
        )

    c.group = group
    data, stats = c.rollout((None, 0), 1)
    assert len(data) == 2
    assert stats["groups_filtered"] == 1
    assert stats["output_tokens"] == 6
    assert stats["used_output_tokens"] == 3
    assert stats["sampling_input_tokens"] == 8
    assert stats["training_input_tokens"] == 5
    assert c.next_group == 2


def test_async_loop_has_exactly_30_updates_and_at_most_one_policy_lag():
    from types import SimpleNamespace

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self, timeout):
            return self.value

    class Sampler:
        def sample(self, *args, **kwargs):
            return Future(SimpleNamespace(sequences=[SimpleNamespace(tokens=[1])]))

    trained = []
    published = []
    sampled = []
    logged = []
    queued = []

    class Trainer:
        def forward_backward(self, data, loss_fn, loss_fn_config):
            trained.append(data)
            assert loss_fn == "ppo"
            return Future(SimpleNamespace(metrics={"loss:sum": 0.1}))

        def optim_step(self, params):
            return Future(SimpleNamespace(metrics={"update_successful:mean": 1.0}))

        def save_weights_for_sampler(self, name):
            queued.append(name)
            return Future(SimpleNamespace(path=name))

    c = client.Client.__new__(client.Client)
    c.cfg = client.read_config(PATH.with_name("config.json"))
    c.trainer = Trainer()
    c.prompts = [[1, 2]]
    c.stops = [3]
    c.event = lambda event, **kwargs: logged.append(dict(event=event, **kwargs))

    def publish(update, future=None):
        assert (future is not None) == (update > 0)
        published.append(update)
        return Sampler(), update

    def rollout(snapshot, update):
        sampled.append(update)
        return [update], dict(policy_version=snapshot[1])

    c.publish = publish
    c.rollout = rollout
    c.run()
    assert len(trained) == 30
    assert sampled == list(range(1, 31))
    assert published == list(range(30))
    assert queued == [f"policy-{i:03d}" for i in range(1, 30)]
    assert all(r["policy_lag"] in (0, 1) for r in logged if r["event"] == "update_done")
    assert logged[-1]["event"] == "completed"


def test_shared_model_creation_omits_unsupported_seed():
    calls = []

    class Service:
        def create_lora_training_client(self, model, **kwargs):
            calls.append((model, kwargs))
            assert "seed" not in kwargs
            return "trainer"

    assert client.create_trainer(Service(), "Qwen/Qwen3.8-27B", 32) == "trainer"
    assert calls == [
        (
            "Qwen/Qwen3.8-27B",
            dict(rank=32, train_attn=True, train_mlp=True, train_unembed=False),
        )
    ]


def test_fixed_groups_keep_uniform_rewards_without_replacements():
    c = client.Client.__new__(client.Client)
    c.cfg = dict(
        groups_per_update=8,
        max_candidate_groups=256,
        max_tokens=4,
        dynamic_group_filtering=False,
    )
    c.next_group = 0
    c.event = lambda *args, **kwargs: None

    def group(i, snapshot):
        return dict(
            prompt=[1, 2],
            sequences=[
                dict(tokens=[3, 4], logprobs=[-0.1, -0.2], reward=float(i % 2))
                for _ in range(8)
            ],
        )

    c.group = group
    data, stats = c.rollout((None, 0), 1)
    assert c.next_group == 8
    assert len(data) == 64
    assert stats["groups_sampled"] == 8
    assert stats["groups_filtered"] == 0
    assert stats["output_tokens"] == stats["used_output_tokens"] == 128
    assert all(not any(d.loss_fn_inputs["advantages"].data) for d in data)


def test_resource_accounting_excludes_previous_experiment(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "dapo_supervise", PATH.with_name("run_spindle.py")
    )
    supervise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervise)
    infos = {
        name: SimpleNamespace(
            started_at=start,
            finished_at=end,
            gpu_type="H200",
            gpu_config=SimpleNamespace(count=2),
        )
        for name, start, end in [
            ("old", 90.0, 99.0),
            ("new", 101.0, 0.0),
            ("pending", 0.0, 0.0),
        ]
    }

    class Stub:
        async def TaskList(self, request):
            return SimpleNamespace(
                tasks=[
                    SimpleNamespace(task_id=name, app_description="owned")
                    for name in infos
                ]
                + [SimpleNamespace(task_id="unrelated", app_description="other")]
            )

        async def TaskGetInfo(self, request):
            return SimpleNamespace(info=infos[request.task_id])

    async def from_env():
        return SimpleNamespace(stub=Stub())

    monkeypatch.setattr(supervise, "_Client", SimpleNamespace(from_env=from_env))
    known = {"old": {"task_id": "old"}}
    result = asyncio.run(supervise.resources("env", ["owned"], known, since=100.0))
    assert set(result) == {"new", "pending"}
    assert result["new"]["started_at"] == 101.0
