import subprocess
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

from spindle.providers.modal import app_tags


def test_revision_uses_spindle_checkout_not_launch_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(app_tags.REVISION_ENV, raising=False)
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.chdir(tmp_path)
    assert app_tags.app_tags("trainer") == {
        "_modal_source": "spindle",
        "spindle_rev": expected,
        "spindle_role": "trainer",
    }


def test_remote_provisioner_keeps_deployed_revision(monkeypatch):
    monkeypatch.setenv(app_tags.REVISION_ENV, "a" * 40)
    assert app_tags.app_tags("inference")["spindle_rev"] == "a" * 40
    assert app_tags.revision_env() == {"SPINDLE_REVISION": "a" * 40}


def test_git_install_revision_without_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv(app_tags.REVISION_ENV, raising=False)
    monkeypatch.setattr(
        app_tags, "__file__", str(tmp_path / "src/spindle/providers/modal/app_tags.py")
    )
    monkeypatch.setattr(
        app_tags,
        "distribution",
        lambda name: SimpleNamespace(
            read_text=lambda name: '{"vcs_info": {"commit_id": "installed-commit"}}'
        ),
    )
    assert app_tags.spindle_revision() == "installed-commit"


def test_missing_revision_still_tags_app(monkeypatch, tmp_path):
    monkeypatch.delenv(app_tags.REVISION_ENV, raising=False)
    monkeypatch.setattr(
        app_tags, "__file__", str(tmp_path / "src/spindle/providers/modal/app_tags.py")
    )

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(app_tags, "distribution", missing)
    assert app_tags.app_tags("frontend") == {
        "_modal_source": "spindle",
        "spindle_rev": "unknown",
        "spindle_role": "frontend",
    }
