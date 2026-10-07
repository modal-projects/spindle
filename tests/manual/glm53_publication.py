"""Compare copy and rename publication using the full-model exported adapter.

PYTHONPATH=src modal run --detach tests/manual/glm53_publication.py
Staging is timed separately; publication includes hashing and volume commit.
"""

import json
import shutil
import time
import uuid
from pathlib import Path

import modal
from stitch.types import VersionRef

from spindle.inference.bulletin import SnapshotBulletin
from spindle.providers.modal.glm53_image import trainer_image

app = modal.App("spindle-glm53-publication-validation")
artifacts = modal.Volume.from_name("spindle-glm53-pr26-full-validation")


@app.function(
    image=trainer_image,
    cpu=8,
    memory=65536,
    timeout=1800,
    volumes={"/validation": artifacts},
)
def measure():
    report = json.loads(Path("/validation/latest.json").read_text())
    source = Path(report["adapter_path"])
    root = Path("/validation") / f"publication-{uuid.uuid4().hex}"
    root.mkdir()
    results = []
    try:
        for index, consume in enumerate((False, True, True, False)):
            stage = root / f"stage-{index}"
            start = time.monotonic()
            shutil.copytree(source, stage)
            stage_s = time.monotonic() - start
            # Commit staging before both variants so this measures publication.
            artifacts.commit()
            commits = []

            def commit():
                start = time.monotonic()
                artifacts.commit()
                commits.append(time.monotonic() - start)

            bulletin = SnapshotBulletin(root / "bulletin", commit=commit)
            ref = VersionRef(f"trial-{index}", 1)
            inode = (stage / "adapter_model.safetensors").stat().st_ino
            start = time.monotonic()
            bulletin.publish(ref, stage, consume=consume)
            publish_s = time.monotonic() - start
            target = bulletin.snapshot_dir(ref)
            if consume:
                assert not stage.exists()
                assert (target / "adapter_model.safetensors").stat().st_ino == inode
            start = time.monotonic()
            assert bulletin.resolve(ref) == target
            result = {
                "consume": consume,
                "stage_copy_s": stage_s,
                "publish_s": publish_s,
                "commit_s": commits[0],
                "resolve_s": time.monotonic() - start,
                "adapter_bytes": report["adapter_bytes"],
            }
            results.append(result)
            print("PUBLICATION", json.dumps(result), flush=True)
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(target)
        Path("/validation/publication.json").write_text(json.dumps(results, indent=2))
        return results
    finally:
        shutil.rmtree(root)
        artifacts.commit()


@app.local_entrypoint()
def main():
    print(measure.remote())
