"""SWE-Gym execution in Modal; hidden tests run in a separate fresh sandbox."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import threading
from pathlib import Path

import modal
from swegym.harness.constants import MAP_REPO_VERSION_TO_SPECS, APPLY_PATCH_PASS
from swegym.harness.grading import get_eval_report, get_logs_eval
from swegym.harness.test_spec import TestSpec, make_eval_script_list

SANDBOX_APP = "swe-gym-9b-256k-sandboxes"


class InfrastructureError(RuntimeError):
    pass


def test_spec(instance):
    inst = dict(instance)
    inst["repo"] = inst["repo"].lower()
    inst["instance_id"] = inst["instance_id"].lower()
    specs = MAP_REPO_VERSION_TO_SPECS[inst["repo"]][inst["version"]]
    commands = make_eval_script_list(
        inst, specs, "testbed", "/testbed", inst["base_commit"], inst["test_patch"]
    )
    return TestSpec(
        inst["instance_id"],
        inst["repo"],
        inst["version"],
        [],
        commands,
        [],
        "x86_64",
        inst["FAIL_TO_PASS"],
        inst["PASS_TO_PASS"],
    )


def command(sb, code, timeout=120):
    p = sb.exec("/bin/bash", "-lc", code, timeout=timeout)
    # Drain both streams concurrently: large test output must not deadlock.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as pool:
        out, err = pool.submit(p.stdout.read), pool.submit(p.stderr.read)
        stdout, stderr = out.result(), err.result()
    p.wait()
    return p.returncode, stdout + stderr


def put(sb, path, text):
    sb.filesystem.write_text(text, path)


class Environments:
    def __init__(self, root):
        self.root = Path(root)
        self.app = modal.App.lookup(
            os.environ.get("SWE_SANDBOX_APP", SANDBOX_APP),
            environment_name=os.environ.get("MODAL_ENVIRONMENT") or None,
            create_if_missing=True,
        )
        self.images = {}
        self.lock = threading.Lock()

    def image(self, inst):
        key = inst["instance_id"]
        with self.lock:
            if key not in self.images:
                suffix = key.replace("__", "_s_").lower()
                # All 293 SkyRL tasks use this registry. Capture Modal's immutable
                # image ID in each episode; Docker tags alone are not a pin.
                self.images[key] = modal.Image.from_registry(
                    f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"
                ).entrypoint([])
            return self.images[key]

    @contextlib.contextmanager
    def sandbox(self, inst):
        sb = modal.Sandbox.create(
            "sleep",
            "7200",
            app=self.app,
            image=self.image(inst),
            cpu=2,
            memory=16384,
            timeout=7200,
            block_network=True,
        )
        try:
            rc, output = command(
                sb,
                "cd /testbed && git reset --hard "
                + shlex.quote(inst["base_commit"])
                + " && git clean -fd",
                120,
            )
            if rc:
                raise InfrastructureError("Cannot reset repository: " + output[-2000:])
            yield sb
        finally:
            sb.terminate()

    def prepare_agent(self, sb):
        # Discard image build scripts, remote refs and unreachable future commits.
        # The original grader sandbox remains untouched.
        rc, out = command(
            sb,
            """set -e
rm -f /eval.sh /patch.diff /test.patch /root/patch.diff
cd /testbed
git checkout --detach HEAD
git remote | xargs -r -n1 git remote remove
git for-each-ref --format='%(refname)' | while read r; do git update-ref -d "$r"; done
git reflog expire --expire=now --all
git gc --prune=now
""",
            180,
        )
        if rc:
            raise InfrastructureError("Agent preparation failed: " + out[-2000:])

    def patch(self, sb):
        # Git lists embedded repositories as directories. They are scratch
        # environments, not files to add as broken submodules to the patch.
        rc, out = command(
            sb,
            """cd /testbed && python - <<'PY'
import os, subprocess
subprocess.run(['git', 'add', '-u'], check=True)
paths = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z']).split(b'\\0')
files = [p for p in paths if p and (os.path.isfile(p) or os.path.islink(p))]
if files:
    subprocess.run(['git', 'add', '--pathspec-from-file=-', '--pathspec-file-nul'], input=b'\\0'.join(files)+b'\\0', check=True)
subprocess.run(['git', 'diff', '--cached', '--binary', 'HEAD', '--', '.', ':(exclude)*.pyc', ':(exclude)__pycache__', ':(exclude).pytest_cache'], check=True)
PY""",
            120,
        )
        if rc or len(out) > 2_000_000:
            raise InfrastructureError(
                f"Cannot capture patch: exit={rc}, bytes={len(out)}; {out[-2000:]}"
            )
        return out

    def grade(self, inst, patch, directory):
        directory = Path(directory) / inst["instance_id"].lower()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prediction.patch").write_text(patch)
        spec = test_spec(inst)
        with self.sandbox(inst) as sb:
            if patch:
                put(sb, "/prediction.patch", patch)
                rc, out = command(
                    sb, "cd /testbed && git apply --whitespace=nowarn /prediction.patch"
                )
                if rc:
                    (directory / "apply.log").write_text(out)
                    return {"reward": 0.0, "invalid_patch": True}
            put(sb, "/hidden_eval.sh", spec.eval_script)
            rc, out = command(sb, "bash /hidden_eval.sh", 1200)
            path = directory / "test_output.txt"
            path.write_text(f"{APPLY_PATCH_PASS} (pred)\n" + out)
            statuses, parsed = get_logs_eval(str(path))
            if not parsed or not statuses:
                # A candidate can break imports before pytest collects a test.
                # The empty/gold controls validate the baseline separately.
                # Keep unrelated unparseable output as an infrastructure error.
                if patch and (
                    "ImportError while loading conftest" in out
                    or re.search(r"_{3,} ERROR collecting .*? _{3,}", out)
                ):
                    report = {
                        "reward": 0.0,
                        "resolved": False,
                        "test_collection_failed": True,
                    }
                    (directory / "report.json").write_text(json.dumps(report, indent=2))
                    return report
                raise InfrastructureError(
                    f"No trustworthy test results: {path}; exit={rc}"
                )
            report = get_eval_report(
                spec,
                {"instance_id": spec.instance_id, "model_patch": patch},
                str(path),
                True,
            )[spec.instance_id]
            report["reward"] = float(report["resolved"])
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            return report

    def calibrate(self, inst):
        path = self.root / "calibration" / inst["instance_id"]
        done = path / "validated.json"
        if done.exists():
            return json.loads(done.read_text())
        empty = self.grade(inst, "", path / "empty")
        gold = self.grade(inst, inst["patch"], path / "gold")
        if empty["reward"] != 0 or gold["reward"] != 1:
            raise InfrastructureError(
                f"Grader control failed: {inst['instance_id']}: empty={empty['reward']}, gold={gold['reward']}"
            )
        result = {
            "instance_id": inst["instance_id"],
            "empty": 0,
            "gold": 1,
            "modal_image_id": self.image(inst).object_id,
        }
        done.write_text(json.dumps(result, indent=2))
        return result
