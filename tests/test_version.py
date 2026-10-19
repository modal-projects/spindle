import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest

import spindle
from spindle import deployment_cli as cli
from spindle import version as spindle_version


@pytest.fixture(autouse=True)
def fresh_commit(monkeypatch):
    monkeypatch.delenv(spindle_version.COMMIT_ENV, raising=False)
    spindle_version._local_commit.cache_clear()
    yield
    spindle_version._local_commit.cache_clear()


class Distribution:
    def __init__(self, direct_url):
        self.direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return self.direct_url


def test_package_metadata_matches_module_version():
    assert version("modal-spindle") == spindle.__version__


def test_deployed_containers_use_the_commit_recorded_at_deploy(monkeypatch):
    monkeypatch.setenv(spindle_version.COMMIT_ENV, "abc123")
    assert spindle_version.source_commit() == "abc123"
    assert spindle_version.version_tags() == {
        "spindle_version": spindle.__version__,
        "spindle_commit": "abc123",
    }


def test_git_installs_report_the_locked_commit(monkeypatch):
    direct_url = {
        "url": "https://github.com/modal-projects/spindle.git",
        "vcs_info": {"vcs": "git", "commit_id": "370f5f2"},
    }
    monkeypatch.setattr(
        spindle_version,
        "distribution",
        lambda name: Distribution(json.dumps(direct_url)),
    )
    assert spindle_version.source_commit() == "370f5f2"


def test_checkouts_fall_back_to_git(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(spindle_version, "distribution", missing)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="def456\n"),
    )
    assert spindle_version.source_commit() == "def456"


def test_unknown_commit_without_git(monkeypatch):
    monkeypatch.setattr(
        spindle_version, "distribution", lambda name: Distribution(None)
    )

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert spindle_version.source_commit() is None
    assert spindle_version.version_tags()["spindle_commit"] == "unknown"


def test_cli_prints_version_and_commit(monkeypatch, capsys):
    monkeypatch.setenv(spindle_version.COMMIT_ENV, "abc123")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"spindle {spindle.__version__} (abc123)\n"
