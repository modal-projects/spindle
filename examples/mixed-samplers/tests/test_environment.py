import hashlib
import shlex
import subprocess
from pathlib import Path

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
        return {"returncode": result.returncode, "output": result.stdout}


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
