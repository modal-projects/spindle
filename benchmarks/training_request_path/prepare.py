"""Make isolated benchmark source copies. No production app is redeployed."""

import argparse
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("arm", choices=["before", "after"])
p.add_argument("destination", type=Path)
a = p.parse_args()
a.destination.mkdir(parents=True, exist_ok=False)
if a.arm == "before":
    archive = subprocess.run(
        ["git", "archive", "7279288e095d7d1b42cd6abb902cbf87ddc6039d"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(a.destination)], input=archive, check=True)
else:
    shutil.copytree(root / "src", a.destination / "src")
    shutil.copy2(root / "pyproject.toml", a.destination / "pyproject.toml")
# Both arms receive exactly the same CPU allocation and audit instrumentation.
app = a.destination / "src/spindle/providers/modal/app.py"
s = app.read_text().replace(
    '    name="server",', '    name="server",\n    cpu=4,\n    memory=8192,'
)
app.write_text(s)
apps = a.destination / "src/spindle/providers/modal/deployment_apps.py"
apps.write_text(
    apps.read_text().replace(
        '"spindle.backends.miles_lora:build_executor"',
        '"spindle.request_path_benchmark:build_executor"',
    )
)
# Match the archived experiment's broad US trainer placement and disable
# compilation in Ray workers in both arms, as that experiment did.
s = apps.read_text().replace('region=platform["modal"]["region"],', 'region="us",', 1)
apps.write_text(s)
runtime = a.destination / "src/spindle/backends/miles_runtime/runtime.py"
s = runtime.read_text().replace(
    "_WORKER_ENV_VARS = (", '_WORKER_ENV_VARS = (\n    "TORCH_COMPILE_DISABLE",', 1
)
runtime.write_text(s)
shutil.copy2(
    Path(__file__).with_name("audit.py"),
    a.destination / "src/spindle/request_path_benchmark.py",
)
config = (
    Path(__file__)
    .with_name("deployment.py")
    .read_text()
    .replace("request-path-before-", f"request-path-{a.arm}-")
)
(a.destination / "deployment.py").write_text(config)
print(a.destination)
