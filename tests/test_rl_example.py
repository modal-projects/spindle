"""CPU checks of workload semantics at the SDK boundary; no model downloads."""

import json
import math
import sys
from types import SimpleNamespace

import pytest

from scripts import rl_example as rl


class Future:
    def __init__(self, value, on_result=lambda: None):
        self.value, self.on_result = value, on_result

    def result(self, timeout=None):
        self.on_result()
        return self.value


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, **kwargs):
        return [11, 12]

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"].endswith("Give the final answer in \\boxed{}.")
        assert kwargs == dict(
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=False,
        )
        return [11, 12]

    def convert_tokens_to_ids(self, text):
        return 98

    def decode(self, tokens):
        return r"\boxed{4}" if tokens[0] == 4 else r"\boxed{5}"


class Trainer:
    model_id = "adapter"

    def __init__(self):
        self.calls, self.forwards, self.optimizers, self.samples = [], [], [], []

    def save_weights_for_sampler(self, name):
        self.calls.append(("publish", name))
        return Future(
            SimpleNamespace(path=f"tinker://adapter:train:0/sampler_weights/{name}"),
            lambda: self.calls.append(("publish_result", name)),
        )

    def forward_backward(self, data, loss, **kwargs):
        self.calls.append(("forward", loss))
        self.forwards.append(
            (
                [
                    dict(
                        model_input=d.model_input.to_ints(),
                        loss_fn_inputs={
                            k: {"data": v.to_numpy().tolist()}
                            for k, v in d.loss_fn_inputs.items()
                        },
                    )
                    for d in data
                ],
                loss,
                kwargs,
            )
        )
        return Future(
            SimpleNamespace(metrics={"loss:mean": 0.2}),
            lambda: self.calls.append(("forward_result",)),
        )

    def optim_step(self, params):
        self.calls.append(("optim",))
        self.optimizers.append(params.model_dump())
        return Future(
            SimpleNamespace(
                metrics={"update_successful:mean": 1, "grad_norm:mean": 0.5}
            ),
            lambda: self.calls.append(("optim_result",)),
        )


class Sampler:
    def __init__(self, trainer, version):
        self.trainer, self.version = trainer, version

    def sample(self, prompt, count, params):
        self.trainer.samples.append((self.version, count, params.model_dump()))
        sequences = [
            SimpleNamespace(
                tokens=[4 if i % 2 == 0 else 5, 99],
                logprobs=[-0.1, -0.2],
                stop_reason="stop",
            )
            for i in range(count)
        ]
        return Future(SimpleNamespace(sequences=sequences))

    def compute_logprobs(self, prompt):
        return Future(
            [None] + [-0.1 - self.version * 0.01] * (len(prompt.to_ints()) - 1)
        )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    trainer = Trainer()
    creates = []

    class Service:
        def __init__(self, **kwargs):
            pass

        def create_lora_training_client(self, model, **kwargs):
            creates.append(("lora", model, kwargs))
            return trainer

        def create_sampling_client(self, model_path):
            return Sampler(trainer, int(model_path.rsplit("-", 1)[-1]))

    def full(service, model):
        creates.append(("full", model))
        return trainer

    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps({"prompt": "Compute 2+2.", "answer": "4"}) + "\n")
    monkeypatch.setattr(rl, "DAPO_DATA", dataset)
    monkeypatch.setattr(rl.tinker, "ServiceClient", Service)
    monkeypatch.setattr(rl, "create_full_training_client", full)
    monkeypatch.setattr(
        rl.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer()
    )
    return trainer, creates


def trace(root):
    return [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]


def test_dapo_preserves_payload_sampling_and_async_order(backend, tmp_path):
    trainer, creates = backend
    cfg = rl.read_config(rl.DAPO_CONFIG)
    cfg["updates"] = 3
    root = tmp_path / "dapo"
    rl.Client(cfg, root, None, validate=True).run()
    assert creates == [
        (
            "lora",
            cfg["model"],
            dict(rank=32, train_attn=True, train_mlp=True, train_unembed=False),
        )
    ]
    assert [r["policy_lag"] for r in trace(root) if r["event"] == "train_start"] == [
        0,
        1,
        1,
    ]
    assert len(trainer.forwards) == 3
    data, loss, options = trainer.forwards[0]
    assert loss == "ppo"
    assert options == {
        "loss_fn_config": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
    }
    assert len(data) == 64
    assert data[0]["loss_fn_inputs"]["target_tokens"]["data"] == [12, 4, 99]
    # Four successes/four failures per group; sample std and global token normalization.
    expected = 0.5 / (math.sqrt(2 / 7) + 1e-6) / 128
    assert data[0]["loss_fn_inputs"]["advantages"]["data"] == pytest.approx(
        [0, expected, expected]
    )
    assert data[1]["loss_fn_inputs"]["advantages"]["data"] == pytest.approx(
        [0, -expected, -expected]
    )
    assert (
        trainer.optimizers[0]
        == rl.types.AdamParams(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.95,
            eps=1e-12,
            weight_decay=0.0,
            grad_clip_norm=1.0,
        ).model_dump()
    )
    samples = [
        params for _, _, params in trainer.samples if params["max_tokens"] == 16384
    ]
    assert len(samples) == 24
    assert sorted(p["seed"] for p in samples) == list(range(4242, 4242 + 24))
    assert all(
        p["temperature"] == p["top_p"] == 1
        and p["top_k"] == -1
        and p["stop"] == [98, 99]
        for p in samples
    )
    assert trainer.calls[:7] == [
        ("publish", "policy-000"),
        ("publish_result", "policy-000"),
        ("forward", "ppo"),
        ("optim",),
        ("publish", "policy-001"),
        ("forward_result",),
        ("optim_result",),
    ]
    rows = trace(root)
    assert [r["update"] for r in rows if r["event"] == "publish"] == [0, 1, 2, 3]
    assert (
        next(r for r in rows if r["event"] == "policy_probe")["max_logprob_change"] > 0
    )
    assert all(
        r["groups_filtered"] == 0
        and r["output_tokens"] == r["used_output_tokens"] == 128
        for r in rows
        if r["event"] == "rollout_done"
    )


@pytest.mark.parametrize(
    "parameterization,expected", [("full", [0, 1, 1]), ("lora", [0, 0.5, 0.5])]
)
def test_toy_preserves_full_reward_and_lora_centering(
    backend, tmp_path, parameterization, expected
):
    trainer, creates = backend
    cfg = rl.toy_config(parameterization)
    root = tmp_path / "toy"
    rl.Client(cfg, root, None, task="toy").run()
    assert creates[0][0] == parameterization
    assert len(trainer.forwards) == cfg["updates"]
    data, loss, options = trainer.forwards[0]
    assert loss == "importance_sampling" and options == {}
    assert data[0]["loss_fn_inputs"]["advantages"]["data"] == expected
    assert all(r["policy_lag"] == 0 for r in trace(root) if r["event"] == "train_start")
    assert not any(r["event"] == "policy_probe" for r in trace(root))
    assert all(p["max_tokens"] == 16 for _, _, p in trainer.samples)


def test_wandb_only_observes_training_and_finishes_on_failure(
    backend, monkeypatch, tmp_path
):
    trainer, _ = backend
    logs = []
    logger = SimpleNamespace(log=lambda values, step: logs.append((step, values)))
    cfg = rl.toy_config("lora")
    cfg["updates"] = 2
    rl.Client(cfg, tmp_path / "plain", None, task="toy").run()
    before = trainer.forwards.copy()
    rl.Client(cfg, tmp_path / "logged", None, task="toy", wandb_run=logger).run()
    assert trainer.forwards[2:] == before
    assert [step for step, _ in logs] == [0, 1]
    assert "train/loss:mean" in logs[0][1]
    finished = []
    monkeypatch.setattr(
        rl,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: SimpleNamespace(finish=lambda: finished.append(True))
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rl_example.py",
            "--wandb-project",
            "test",
            "--output",
            str(tmp_path / "failed"),
        ],
    )

    def fail(self):
        raise RuntimeError("training failed")

    monkeypatch.setattr(rl.Client, "run", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        rl.main()
    assert finished == [True]
