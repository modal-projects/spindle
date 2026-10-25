import hashlib
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from mixed_samplers import environment as environments
from mixed_samplers.environment import execute_command


class ShellEnvironment:
    def __init__(self, startup_warning="", output_limit=None):
        self.paths = []
        self.commands = []
        self.startup_warning = startup_warning
        self.output_limit = output_limit
        self.env = SimpleNamespace(
            sandbox=SimpleNamespace(
                filesystem=SimpleNamespace(
                    read_text=lambda path: Path(path).read_text()
                )
            )
        )

    def copy_text_to(self, text, destination):
        path = Path(destination)
        self.paths.append(path)
        path.write_text(text)

    def execute(self, command, *, cwd, timeout=10):
        assert len(command.encode("utf-8")) < 65536
        self.commands.append(command)
        result = subprocess.run(
            ["bash", "-lc", f"cd {shlex.quote(cwd)} && {command}"],
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr + self.startup_warning
        if self.output_limit is not None:
            output = output[-self.output_limit :]
        return {
            "returncode": result.returncode,
            "output": output,
        }

    def setup_environment(self):
        pass

    def cleanup(self):
        pass


def test_large_unicode_heredoc_preserves_bytes_cwd_and_exit_status(tmp_path):
    environment = ShellEnvironment()
    payload = "🙂" * 20_000 + " quotes ' \" $HOME `false`"
    command = f"cat <<'SPINDLE_EOF'\n{payload}\nSPINDLE_EOF\npwd\nexit 7"
    result = execute_command(environment, command, cwd=str(tmp_path), timeout=10)
    assert result["returncode"] == 7
    expected = payload + "\n" + str(tmp_path) + "\n"
    assert (
        hashlib.sha256(result["output"].encode()).digest()
        == hashlib.sha256(expected.encode()).digest()
    )
    assert len(environment.paths) == 1
    assert not environment.paths[0].exists()


def test_short_commands_keep_the_existing_execution_path(tmp_path):
    environment = ShellEnvironment()
    command = "printf '%s' unchanged"
    assert execute_command(environment, command, cwd=str(tmp_path), timeout=10) == {
        "returncode": 0,
        "output": "unchanged",
    }
    assert environment.commands == [command]
    assert environment.paths == []


@pytest.mark.parametrize(
    "operation", ["metadata_removed", "worktree_replaced", "worktree_deleted"]
)
@pytest.mark.parametrize("startup_warning", ["", "/root/.bashrc: missing activate\n"])
def test_patch_survives_agent_commits_and_repository_deletion(
    tmp_path, monkeypatch, operation, startup_warning
):
    worktree = tmp_path / "agent repository"
    worktree.mkdir()
    (worktree / "file.bin").write_bytes(b"\x00original")
    (worktree / "deleted.txt").write_text("original\n")
    environment = ShellEnvironment(startup_warning)
    monkeypatch.setattr(environments, "create_environment", lambda task: environment)
    task = {"cwd": str(worktree)}
    _, baseline = environments.setup(task)
    try:
        verifier = tmp_path / "verifier"
        shutil.copytree(worktree, verifier)
        if operation == "metadata_removed":
            (worktree / "file.bin").write_bytes(b"\x00changed")
            (worktree / "deleted.txt").unlink()
            (worktree / "added.txt").write_text("submitted\n")
            committed = environment.execute(
                "git add -A && git -c user.name=Agent "
                "-c user.email=agent@example.invalid commit -qm submitted",
                cwd=str(worktree),
                timeout=10,
            )
            assert committed["returncode"] == 0
            shutil.rmtree(worktree / ".git")
        else:
            shutil.rmtree(worktree)
            if operation == "worktree_replaced":
                worktree.mkdir()
                (worktree / "file.bin").write_bytes(b"\x00changed")
                (worktree / "added.txt").write_text("submitted\n")
        patch = environments.capture_patch(environment, task, baseline)
        patch_path = tmp_path / "submitted.patch"
        patch_path.write_text(patch)
        applied = environment.execute(
            f"git apply --binary {shlex.quote(str(patch_path))}",
            cwd=str(verifier),
            timeout=10,
        )
        assert applied["returncode"] == 0, applied["output"]
        actual = {
            path.name: path.read_bytes()
            for path in verifier.iterdir()
            if path.is_file()
        }
        expected = (
            {}
            if operation == "worktree_deleted"
            else {"file.bin": b"\x00changed", "added.txt": b"submitted\n"}
        )
        assert actual == expected
    finally:
        shutil.rmtree(baseline["git_dir"])


def test_shell_warning_does_not_become_an_empty_patch(tmp_path, monkeypatch):
    environment = ShellEnvironment("/root/.bashrc: missing activate\n")
    monkeypatch.setattr(environments, "create_environment", lambda task: environment)
    task = {"cwd": str(tmp_path)}
    _, baseline = environments.setup(task)
    try:
        assert environments.capture_patch(environment, task, baseline) == ""
        environment.repo_path = str(tmp_path)
        environment.instance_id = "noisy-image"
        assert environments.capture_base_ref(environment) == baseline["commit"]
        broken = environment.execute(
            "git symbolic-ref HEAD refs/heads/missing", cwd=str(tmp_path), timeout=10
        )
        assert broken["returncode"] == 0
        with pytest.raises(RuntimeError, match="could not resolve HEAD"):
            environments.capture_base_ref(environment)
    finally:
        shutil.rmtree(baseline["git_dir"])


def test_patch_transfer_exceeds_command_output_limit(tmp_path, monkeypatch):
    worktree = tmp_path / "repo"
    worktree.mkdir()
    environment = ShellEnvironment("shell startup warning\n", output_limit=1024)
    monkeypatch.setattr(environments, "create_environment", lambda task: environment)
    task = {"cwd": str(worktree)}
    _, baseline = environments.setup(task)
    try:
        verifier = tmp_path / "verifier"
        shutil.copytree(worktree, verifier)
        payload = "new repository content\n" * 100_000
        (worktree / "large.txt").write_text(payload)
        patch = environments.capture_patch(environment, task, baseline)
        assert len(patch) > environment.output_limit
        patch_path = tmp_path / "patch"
        patch_path.write_text(patch)
        applied = environment.execute(
            f"git apply --binary {shlex.quote(str(patch_path))}",
            cwd=str(verifier),
            timeout=30,
        )
        assert applied["returncode"] == 0, applied["output"]
        assert (verifier / "large.txt").read_text() == payload
    finally:
        shutil.rmtree(baseline["git_dir"])
