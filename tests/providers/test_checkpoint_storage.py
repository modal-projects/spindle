import asyncio
import importlib
import runpy
from pathlib import Path
from unittest.mock import patch, sentinel

import modal

from spindle.backends.deployment import backend_config
from spindle.providers.modal.app import DEFINITIONS, PLATFORM
from spindle.providers.modal.checkpoint_storage import (
    CHECKPOINT_ROOT,
    ModalCheckpointStorage,
    _scan_checkpoints,
)
from spindle.providers.modal.deployment_apps import volumes_for

FULL_DEFINITIONS = tuple(
    definition for definition in DEFINITIONS if definition.parameterization == "full"
)


def test_checkpoint_storage_creates_one_v2_volume_without_live_lookup() -> None:
    storage_path = Path(__file__).parents[2] / (
        "src/spindle/providers/modal/checkpoint_storage.py"
    )
    with patch.object(
        modal.Volume,
        "from_name",
        return_value=sentinel.checkpoint_volume,
    ) as from_name:
        storage = runpy.run_path(str(storage_path))

    from_name.assert_called_once_with(
        "spindle-checkpoints",
        create_if_missing=True,
        version=2,
    )
    assert storage["CHECKPOINT_ROOT"] == "/checkpoints"


def test_deployments_use_configured_checkpoint_storage():
    for definition in DEFINITIONS:
        with patch.object(
            modal.Volume, "from_name", side_effect=lambda name, **kwargs: (name, kwargs)
        ):
            volumes = volumes_for(PLATFORM)
        assert volumes[CHECKPOINT_ROOT] == (
            PLATFORM["storage"]["checkpoints"],
            {"create_if_missing": True, "version": 2},
        )
        assert volumes["/bulletin"][0] == PLATFORM["storage"]["bulletin"]
        assert backend_config(definition.recipe)["checkpoint_dir"] == CHECKPOINT_ROOT


def test_same_checkpoint_name_isolated_by_model(tmp_path, monkeypatch) -> None:
    app = importlib.import_module("spindle.providers.modal.app")

    def scan(model_id):
        return _scan_checkpoints(str(tmp_path), model_id)

    monkeypatch.setattr(app, "CHECKPOINT_ROOT", str(tmp_path))
    for relative in ("final/run-a", "final/run-b"):
        checkpoint = tmp_path / relative
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text("{}")
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("notes")
    entries = scan(None)
    assert {(e["model_id"], e["name"]) for e in entries} == {
        ("run-a", "final"),
        ("run-b", "final"),
    }
    assert len(entries) == 2
    assert scan("run-a")[0]["path"] == str(tmp_path / "final/run-a")
    assert scan("missing") == []
    with (
        patch.object(app.checkpoint_volume, "reload"),
        patch.object(app.checkpoint_volume, "commit"),
    ):
        storage = ModalCheckpointStorage(app.checkpoint_volume, str(tmp_path))
        asyncio.run(storage.delete(str(tmp_path / "final/run-a")))
    assert scan("run-a") == []
    assert len(scan("run-b")) == 1
