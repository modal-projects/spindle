import hashlib
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from mixed_samplers import environment as environments
from mixed_samplers.environment import execute_command


class ShellEnvironment:
    def __init__(self):
        self.paths = []
        self.commands = []

    def copy_text_to(self, text, destination):
        path = Path(destination)
        self.paths.append(path)
        path.write_text(text)

    def execute(self, command, *, cwd, timeout):
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
        return {
            "returncode": result.returncode,
            "output": result.stdout + result.stderr,
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
def test_patch_survives_agent_commits_and_repository_deletion(
    tmp_path, monkeypatch, operation
):
    worktree = tmp_path / "agent repository"
    worktree.mkdir()
    (worktree / "file.bin").write_bytes(b"\x00original")
    (worktree / "deleted.txt").write_text("original\n")
    environment = ShellEnvironment()
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
