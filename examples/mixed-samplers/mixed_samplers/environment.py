"""Run the public task setup; grade the submitted diff in a fresh sandbox."""

import shlex

from mimoagent.environments.utils import make_dataset_env


def create_environment(task):
    return make_dataset_env(
        task,
        environment_class="modal",
        environment_name="kailash-dev",
        app_name="spindle-mimo-mixed-sandboxes",
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
        baseline = result["output"].strip().splitlines()[-1]
        return environment, baseline
    except BaseException:
        environment.cleanup()
        raise


def capture_patch(environment, task, baseline):
    result = environment.execute(
        f"git add -A && git diff --binary {shlex.quote(baseline)} -- .",
        cwd=task["cwd"],
        timeout=120,
    )
    if result["returncode"] != 0:
        raise RuntimeError("could not capture the submitted patch")
    return result["output"]


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
