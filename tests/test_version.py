import json
import subprocess
from importlib.metadata import version

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


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def project_repo(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    (tmp_path / ".gitignore").write_text(".venv/\n")
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-qm", "project")
    monkeypatch.setattr(
        spindle_version, "distribution", lambda name: Distribution(None)
    )
    return tmp_path


def test_installs_inside_another_repo_do_not_report_its_commit(
    project_repo, monkeypatch
):
    installed = project_repo / ".venv" / "site-packages" / "spindle" / "version.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("")
    monkeypatch.setattr(spindle_version, "__file__", str(installed))
    assert spindle_version.source_commit() is None


def test_spindle_checkouts_report_head(project_repo, monkeypatch):
    source = project_repo / "src" / "spindle" / "version.py"
    source.parent.mkdir(parents=True)
    source.write_text("")
    git(project_repo, "add", "src")
    git(project_repo, "commit", "-qm", "spindle")
    monkeypatch.setattr(spindle_version, "__file__", str(source))
    head = git(project_repo, "rev-parse", "HEAD")
    assert spindle_version.source_commit() == head

    source.write_text("edited")
    spindle_version._local_commit.cache_clear()
    assert spindle_version.source_commit() == f"{head}-dirty"
    assert spindle_version.version_tags()["spindle_commit"] == f"{head[:7]}-dirty"


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
