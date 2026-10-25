"""Run the public task setup; grade the submitted diff in a fresh sandbox."""

import os
import shlex
import uuid

from mimoagent.environments.datasets.base import parse_git_head_output
from mimoagent.environments.datasets.opensource_code import OpenSourceCodeEnvironment
from mimoagent.environments.utils import make_dataset_env


def capture_base_ref(environment):
    # The pinned dataset wrapper takes the last whitespace-delimited word,
    # but some images append shell startup diagnostics after successful stdout.
    result = environment.execute("git rev-parse HEAD", cwd=environment.repo_path)
    commit = parse_git_head_output(result.get("output", ""))
    if result.get("returncode", 1) != 0 or commit is None:
        raise RuntimeError(
            f"{environment.instance_id}: could not resolve HEAD at "
            f"{environment.repo_path}: {result.get('output', '')[-2000:]}"
        )
    return commit


OpenSourceCodeEnvironment._capture_base_ref = capture_base_ref


def create_environment(task):
    return make_dataset_env(
        task,
        environment_class="modal",
        environment_name="kailash-dev",
        app_name=os.environ.get(
            "SPINDLE_MIMO_SANDBOX_APP", "spindle-mimo-mixed-sandboxes"
        ),
        anti_hack_cleanup=True,
        git_leak_prevention="strip",
        reward_mode="programmatic",
        cpu=2,
        memory=8192,
        sandbox_timeout=14400,
        timeout=120,
        raise_on_transport_error=True,
    )


def setup(task):
    environment = create_environment(task)
    try:
        environment.setup_environment()
        git_dir = f"/tmp/spindle-baseline-{uuid.uuid4().hex}.git"
        # Start from one root commit. This removes all pre-task history, refs,
        # reflogs, unreachable objects, hooks, remotes and repository config.
        result = environment.execute(
            "find . -name .git -prune -exec rm -rf -- {} + && git init -q && git add -A && "
            "git -c user.name=Spindle -c user.email=spindle@example.invalid "
            "commit -qm baseline --allow-empty && git rev-parse HEAD",
            cwd=task["cwd"],
            timeout=300,
        )
        if result["returncode"] != 0:
            raise RuntimeError(
                f"repository sanitization failed: {result['output'][-2000:]}"
            )
        baseline = parse_git_head_output(result["output"])
        if baseline is None:
            raise RuntimeError("sanitized repository did not return a commit hash")
        copied = environment.execute(
            f"cp -a -- .git {shlex.quote(git_dir)}",
            cwd=task["cwd"],
            timeout=300,
        )
        if copied["returncode"] != 0:
            raise RuntimeError(
                f"could not preserve repository baseline: {copied['output'][-2000:]}"
            )
        return environment, {"commit": baseline, "git_dir": git_dir}
    except BaseException:
        environment.cleanup()
        raise


def capture_patch(environment, task, baseline):
    # Agent commands can commit changes or replace the entire working directory.
    # Keep our original tree and index independent of its editable .git metadata.
    worktree = shlex.quote(task["cwd"])
    git = (
        f"git --git-dir={shlex.quote(baseline['git_dir'])} --work-tree={worktree} "
        "-c core.hooksPath=/dev/null"
    )
    marker = f"spindle-patch-{uuid.uuid4().hex}"
    result = environment.execute(
        f"mkdir -p -- {worktree} && cd -- {worktree} && {git} add -A && "
        f"printf '%s\\n' {marker}-begin && "
        f"{git} diff --binary {shlex.quote(baseline['commit'])} -- . && "
        f"printf '%s\\n' {marker}-end",
        cwd="/",
        timeout=120,
    )
    if result["returncode"] != 0:
        raise RuntimeError(
            f"could not capture the submitted patch (exit {result['returncode']}): "
            f"{result['output'][-2000:]}"
        )
    # Shell startup warnings are outside these markers, including when the
    # transport appends stderr after stdout. Preserve patch bytes and an empty
    # diff rather than treating a warning as a submitted patch.
    return (
        result["output"].split(f"{marker}-begin\n", 1)[1].split(f"{marker}-end\n", 1)[0]
    )


def execute_command(environment, command, *, cwd, timeout):
    # Modal limits the complete exec argument vector to 64 KiB. Large generated
    # heredocs are valid shell programs; transfer their bytes outside argv.
    if len(command.encode("utf-8")) < 60_000:
        return environment.execute(command, cwd=cwd, timeout=timeout)
    path = f"/tmp/spindle-shell-{uuid.uuid4().hex}.sh"
    environment.copy_text_to(command, path)
    quoted = shlex.quote(path)
    return environment.execute(
        f"bash {quoted}; spindle_command_status=$?; "
        f"rm -f -- {quoted}; exit $spindle_command_status",
        cwd=cwd,
        timeout=timeout,
    )


def grade_patch(task, patch):
    # Verifier setup has its own baseline tracking. No agent shell state or
    # modified test harness is carried over from the rollout sandbox.
    environment = create_environment(task)
    try:
        environment.setup_environment()
        if patch:
            environment.copy_text_to(patch, "/tmp/submitted.patch")
            applied = environment.execute(
                "git -c core.hooksPath=/dev/null apply --binary /tmp/submitted.patch",
                cwd=task["cwd"],
                timeout=120,
            )
            if applied["returncode"] != 0:
                return 0.0, {"patch_apply_failed": True}
        reward, output, details = environment.calculate_reward()
        if (
            details.get("transport_error")
            or details.get("error_category")
            or "verifier_returncode" not in details
        ):
            raise RuntimeError(
                f"verifier infrastructure failure: {details}; {output[-2000:]}"
            )
        return reward, {k: v for k, v in details.items() if k != "model_patch"}
    finally:
        environment.cleanup()
