import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from stitch.types import VersionRef
from tinker import AdamParams, Datum, LoraConfig, ModelInput, TensorData

from spindle.backends import miles_full
from spindle.backends.contract import ForwardBatch, ForwardItem, ModelSpec
from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_full import MilesFullBackend
from spindle.errors import BackendFailed
from spindle.inference.fft_bulletin import FFTSnapshotBulletin


class FakeRuntime:
    revision = "miles-test"

    def __init__(self) -> None:
        self.calls = []
        self.optimizer_outcome = {"grad_norm": 0.5}
        self.aligned = False

    def forward_backward(self, slot_rows, *, loss_fn, loss_fn_config, forward_only):
        self.calls.append(("forward_backward", len(slot_rows), loss_fn, forward_only))
        return [
            {"loss": 1.0, "logprobs": [-0.1] * row["target_len"]}
            for _, row in slot_rows
        ]

    def full_optim_step(self, adam_params):
        self.calls.append(("optim_step", adam_params["learning_rate"]))
        return self.optimizer_outcome

    def save_full_checkpoint(self, path, *, include_optimizer, metadata=None):
        self.calls.append(("save", path, include_optimizer))
        Path(path).mkdir(parents=True)
        (Path(path) / "shard.distcp").write_bytes(b"x")

    def load_full_checkpoint(self, path, *, restore_optimizer):
        self.calls.append(("load", path, restore_optimizer))

    def full_delta_aligned_with(self, identity):
        return self.aligned

    def capture_full_delta(self, **kwargs):
        self.calls.append(("capture", kwargs["publish_version"]))
        ref = VersionRef(kwargs["model_id"], kwargs["publish_version"])
        return {
            "ref": ref.identity,
            "parent": VersionRef(
                kwargs["model_id"], kwargs["publish_version"] - 1
            ).identity,
            "metadata": {"optimizer_step": kwargs["optimizer_step"]},
            "weight_map": {"model.weight": "pp00-delta-00001.safetensors"},
            "metrics": {},
        }

    def reload_base_weights(self):
        self.calls.append(("reload_base",))

    def close(self):
        self.calls.append(("close",))


def _config(**overrides) -> MilesBackendConfig:
    settings = {
        "hf_checkpoint": "/assets/model",
        "model_type": "qwen3.5-4B",
        "actor_num_gpus_per_node": 4,
        "parameterization": "full",
        "tensor_model_parallel_size": 2,
    }
    settings.update(overrides)
    return MilesBackendConfig(**settings)


def _backend(tmp_path, runtime=None) -> tuple[MilesFullBackend, FakeRuntime]:
    runtime = runtime or FakeRuntime()
    backend = MilesFullBackend(
        _config(),
        checkpoint_dir=tmp_path / "checkpoints",
        base_model="Qwen/Test",
        runtime=runtime,
    )
    backend.accept_model(
        "run-a", ModelSpec(base_model="Qwen/Test", parameterization="full")
    )
    return backend, runtime


def _datum(tokens) -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]
            ),
            "weights": TensorData(
                data=[1.0] * (len(tokens) - 1), dtype="float32", shape=[len(tokens) - 1]
            ),
        },
    )


def _batch(model_id="run-a", forward_only=False) -> ForwardBatch:
    return ForwardBatch(
        items=(ForwardItem(model_id=model_id, data=(_datum([1, 2, 3, 4]),)),),
        loss_fn="cross_entropy",
        forward_only=forward_only,
    )


def test_full_config_emits_no_lora_arguments() -> None:
    arguments = _config(pipeline_model_parallel_size=2).miles_arguments()
    assert "--multi-lora-n-adapters" not in arguments
    assert "--lora-rank" not in arguments
    assert "--target-modules" not in arguments
    assert arguments[arguments.index("--pipeline-model-parallel-size") + 1] == "2"
    assert _config(pipeline_model_parallel_size=2).data_parallel_size == 1


def test_lora_config_rejects_pipeline_parallelism() -> None:
    with pytest.raises(ValueError, match="pipeline parallelism"):
        _config(parameterization="lora", pipeline_model_parallel_size=2).validate()


def test_full_config_rejects_context_parallelism() -> None:
    with pytest.raises(ValueError, match="context parallelism"):
        _config(context_parallel_size=2).validate()


def test_export_mode_is_selectable() -> None:
    arguments = _config(megatron_to_hf_mode="raw").miles_arguments()
    assert arguments[arguments.index("--megatron-to-hf-mode") + 1] == "raw"
    _config(parameterization="lora", megatron_to_hf_mode="raw").validate()
    with pytest.raises(ValueError, match="bridge or raw"):
        _config(megatron_to_hf_mode="direct").validate()


def test_backend_hosts_one_full_model(tmp_path) -> None:
    backend, runtime = _backend(tmp_path)
    with pytest.raises(ValueError, match="full parameterization"):
        backend.accept_model(
            "run-b",
            ModelSpec(
                base_model="Qwen/Test",
                parameterization="lora",
                lora_config=LoraConfig(rank=8),
            ),
        )
    with pytest.raises(ValueError, match="already hosts"):
        backend.accept_model(
            "run-b", ModelSpec(base_model="Qwen/Test", parameterization="full")
        )
    backend.unload_model("run-a")
    backend.accept_model(
        "run-b", ModelSpec(base_model="Qwen/Test", parameterization="full")
    )
    assert ("reload_base",) in runtime.calls


def test_forward_backward_accumulates_until_optim_step(tmp_path) -> None:
    backend, runtime = _backend(tmp_path)
    with pytest.raises(ValueError, match="no accumulated gradients"):
        backend.optim_step(("run-a",), AdamParams(learning_rate=1e-5))
    outputs = backend.forward_backward(_batch())
    assert outputs[0].metrics["loss:sum"] == 1.0
    assert backend.accumulating
    (response,) = backend.optim_step(("run-a",), AdamParams(learning_rate=1e-5))
    assert response.metrics["grad_norm:mean"] == 0.5
    assert backend.optimizer_step == 1
    assert not backend.accumulating
    assert runtime.calls[-2:] == [
        ("forward_backward", 2, "cross_entropy", False),
        ("optim_step", 1e-5),
    ]


def test_nonfinite_step_does_not_advance(tmp_path) -> None:
    backend, runtime = _backend(tmp_path)
    runtime.optimizer_outcome = {"skipped_nonfinite": 1.0}
    backend.forward_backward(_batch())
    (response,) = backend.optim_step(("run-a",), AdamParams(learning_rate=1e-5))
    assert response.metrics["update_successful:mean"] == 0.0
    assert backend.optimizer_step == 0
    runtime.optimizer_outcome = {"error": "cell lost"}
    backend.forward_backward(_batch())
    with pytest.raises(BackendFailed, match="cell lost"):
        backend.optim_step(("run-a",), AdamParams(learning_rate=1e-5))


def test_checkpoint_roundtrip_validates_metadata(tmp_path, monkeypatch) -> None:
    backend, runtime = _backend(tmp_path)
    monkeypatch.setenv("SPINDLE_DEFINITION_ID", "qwen-fft")
    backend.forward_backward(_batch())
    backend.optim_step(("run-a",), AdamParams(learning_rate=1e-5))
    backend.capture_checkpoint(
        "run-a", "snap", destination="ckpt-1", include_optimizer=True
    )
    uri = backend.persist_checkpoint("snap", "ckpt-1")
    metadata = json.loads((Path(uri) / "metadata.json").read_text())
    assert metadata["parameterization"] == {"type": "full"}
    assert metadata["lora_config"] is None
    assert metadata["optimizer_step"] == 1
    assert metadata["topology"]["pipeline_model_parallel_size"] == 1
    with pytest.raises(FileExistsError):
        backend.capture_checkpoint(
            "run-a", "again", destination="ckpt-1", include_optimizer=True
        )

    backend.optimizer_step = 7
    backend.load_checkpoint("run-a", uri, restore_optimizer=True)
    assert backend.optimizer_step == 1
    assert runtime.calls[-1] == ("load", str(Path(uri) / "miles"), True)
    backend.load_checkpoint("run-a", uri, restore_optimizer=False)
    assert backend.optimizer_step == 0

    metadata["has_optimizer"] = False
    (Path(uri) / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="optimizer state"):
        backend.load_checkpoint("run-a", uri, restore_optimizer=True)
    metadata["topology"]["tensor_model_parallel_size"] = 4
    (Path(uri) / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="topology"):
        backend.load_checkpoint("run-a", uri, restore_optimizer=False)


def test_sampler_publication_is_contiguous_and_staged_by_trainer(
    tmp_path, monkeypatch
) -> None:
    backend, runtime = _backend(tmp_path)
    root = tmp_path / "bulletin"
    monkeypatch.setenv("SPINDLE_BULLETIN_ROOT", str(root))
    monkeypatch.setenv("SPINDLE_BULLETIN_VOLUME", "bulletin")
    board = FFTSnapshotBulletin(root)
    board.claim("run-a")
    published = []
    monkeypatch.setattr(miles_full, "FFTSnapshotBulletin", lambda root: board)
    monkeypatch.setattr(miles_full, "bulletin_board", lambda root, volume: board)
    reloaded = []
    monkeypatch.setattr(miles_full, "reload_bulletin_volume", reloaded.append)
    monkeypatch.setattr(
        miles_full,
        "publish_staged",
        lambda board, *, ref, metadata: published.append((ref, metadata)),
    )

    publication = backend.capture_sampler_snapshot(
        "run-a", "capture-1", requested_version=9
    )
    assert publication.publish_version == 1
    assert runtime.calls[-1] == ("capture", 1)
    backend.publish_sampler_snapshot("capture-1")
    assert published == [(VersionRef("run-a", 1), {"optimizer_step": 0})]
    assert reloaded == ["bulletin"]

    board.advance(VersionRef("run-a", 1))
    monkeypatch.setattr(board, "metadata", lambda ref: {"optimizer_step": 0})
    runtime.aligned = True
    repeat = backend.capture_sampler_snapshot(
        "run-a", "capture-2", requested_version=10
    )
    assert repeat.publish_version == 1
    backend.publish_sampler_snapshot("capture-2")
    assert len(published) == 1

    backend.accumulating = True
    with pytest.raises(ValueError, match="accumulated gradients"):
        backend.capture_sampler_snapshot("run-a", "capture-3", requested_version=11)


def test_failed_publication_discards_staged_delta(tmp_path, monkeypatch) -> None:
    backend, _ = _backend(tmp_path)
    monkeypatch.setenv("SPINDLE_BULLETIN_ROOT", str(tmp_path / "bulletin"))
    monkeypatch.setenv("SPINDLE_BULLETIN_VOLUME", "bulletin")
    discarded = []
    monkeypatch.setattr(miles_full, "bulletin_board", lambda root, volume: "board")
    monkeypatch.setattr(miles_full, "reload_bulletin_volume", lambda volume: None)
    monkeypatch.setattr(miles_full, "write_staged_index", lambda board, **kwargs: None)

    def fail(board, **kwargs):
        raise RuntimeError("pool wake failed")

    monkeypatch.setattr(miles_full, "publish_staged", fail)
    monkeypatch.setattr(
        miles_full,
        "discard_staged",
        lambda board, *, ref, parent: discarded.append((ref, parent)),
    )
    backend.capture_sampler_snapshot("run-a", "capture-1", requested_version=1)
    with pytest.raises(RuntimeError, match="pool wake failed"):
        backend.publish_sampler_snapshot("capture-1")
    assert discarded == [(VersionRef("run-a", 1), VersionRef("run-a", 0))]
    assert "capture-1" not in backend._sampler_captures


def test_build_executor_runs_single_process(monkeypatch, tmp_path) -> None:
    constructed = []
    monkeypatch.setenv(
        "SPINDLE_BACKEND_CONFIG",
        json.dumps({"miles": asdict(_config()), "checkpoint_dir": str(tmp_path)}),
    )
    monkeypatch.setenv("SPINDLE_BASE_MODEL", "Qwen/Test")
    monkeypatch.setattr(
        miles_full,
        "MilesFullBackend",
        lambda config, **kwargs: constructed.append(kwargs) or SimpleNamespace(),
    )
    executor = miles_full.build_executor()
    assert executor.command_group is None
    assert constructed[0]["checkpoint_dir"] == tmp_path
