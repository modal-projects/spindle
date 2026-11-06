"""Restore a warmed Spindle 9B Miles trainer, including its Ray process tree.

Snapshots require deployment. See docs/trainer-memory-snapshots.md for commands.
No production trainer, sampler, frontend, or registry is used.
"""

import contextlib
import json
import os
import time
import uuid
from pathlib import Path

import modal

APP_NAME = os.environ.get("SPINDLE_SNAPSHOT_APP", "spindle-qwen9b-memory-snapshot-1105")
MODEL = "Qwen/Qwen3.5-9B-Base"
ASSET_VOLUME = "spindle-qwen9b-runtime-snapshot-assets-1105"
parity_mode = os.environ.get("SPINDLE_SNAPSHOT_PARITY", "0") == "1"
te_deterministic = os.environ.get("SPINDLE_SNAPSHOT_TE_DETERMINISTIC", "0") == "1"
assets = modal.Volume.from_name(ASSET_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)
if modal.is_local():
    from spindle.providers.modal.miles_image import image as trainer_image

    image = trainer_image.add_local_file(
        Path(__file__).with_name("qwen9b_snapshot_probe.py"),
        "/root/qwen9b_snapshot_probe.py",
    )
    if parity_mode:
        image = image.add_local_file(
            Path(__file__).with_name("qwen9b_snapshot_parity.py"),
            "/root/qwen9b_snapshot_parity.py",
        )
else:
    image = modal.Image.debian_slim()
transport = os.environ.get("SPINDLE_SNAPSHOT_TRANSPORT", "default")
ray_ip = os.environ.get("SPINDLE_SNAPSHOT_RAY_IP", "auto")
comms_lifecycle = os.environ.get("SPINDLE_SNAPSHOT_COMMS", "none")
if comms_lifecycle not in {"none", "suspend"}:
    raise ValueError(f"unknown communicator lifecycle {comms_lifecycle}")
trainer_env = {
    "SPINDLE_SNAPSHOT_APP": APP_NAME,
    "SPINDLE_SNAPSHOT_TRANSPORT": transport,
    "SPINDLE_SNAPSHOT_RAY_IP": ray_ip,
    "SPINDLE_SNAPSHOT_COMMS": comms_lifecycle,
    "SPINDLE_SNAPSHOT_PARITY": str(int(parity_mode)),
    "SPINDLE_SNAPSHOT_TE_DETERMINISTIC": str(int(te_deterministic)),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": "/root:/root/miles:/root/Megatron-LM",
    "NCCL_DEBUG": "INFO",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
}
if te_deterministic:
    trainer_env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
if transport == "socket":
    trainer_env.update(
        NCCL_P2P_DISABLE="1",
        NCCL_SHM_DISABLE="1",
        NCCL_IB_DISABLE="1",
        NCCL_NVLS_ENABLE="0",
        NCCL_CUMEM_ENABLE="0",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_SOCKET_IFNAME="lo",
        GLOO_SOCKET_IFNAME="lo",
    )
elif transport == "no-nvls":
    trainer_env["NCCL_NVLS_ENABLE"] = "0"
elif transport != "default":
    raise ValueError(f"unknown transport {transport}")
if comms_lifecycle == "suspend":
    # This probe is single-container. Keep bootstrap/control sockets local;
    # this does not disable the GPU P2P transport.
    # Miles defaults cuMem off. Explicitly use the allocation path exercised
    # by the successful standalone NCCL suspend/resume snapshot controls.
    trainer_env.update(
        NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo", NCCL_CUMEM_ENABLE="1"
    )


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub"),
    volumes={"/assets": assets},
    timeout=1800,
)
def prepare():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, local_dir=f"/assets/{MODEL}")
    assets.commit()


def make_batch(forward_only=False):
    from spindle.backends.contract import ForwardBatch, ForwardItem
    from tinker import Datum, ModelInput, TensorData

    tokens = [100 + i % 100 for i in range(129)]
    datum = Datum(
        ModelInput.from_ints(tokens[:-1]),
        {
            "target_tokens": TensorData(data=tokens[1:], dtype="int64", shape=[128]),
            "weights": TensorData(data=[1.0] * 128, dtype="float32", shape=[128]),
        },
    )
    return ForwardBatch(
        items=(ForwardItem("snapshot-probe", (datum,)),),
        loss_fn="cross_entropy",
        loss_fn_config={},
        forward_only=forward_only,
    )


class Trainer:
    def initialize_runtime(self):
        import subprocess

        from spindle.backends import miles_runtime
        from spindle.backends.contract import ModelSpec
        from spindle.backends.miles_config import MilesBackendConfig
        from spindle.backends.miles_lora import MilesCommandBackend
        from spindle.configs.qwen35_9b_lora_16k import Config
        from spindle.deployments import DeploymentConfig
        from tinker import AdamParams, LoraConfig

        self.initialization_id = uuid.uuid4().hex
        self.snapshot_creator_task_id = os.environ["MODAL_TASK_ID"]
        self.hardware = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()
        started = time.monotonic()
        print("SNAPSHOT_PHASE initialize", self.initialization_id, flush=True)
        print("SNAPSHOT_PHASE trainer_env", json.dumps(trainer_env), flush=True)
        assets.reload()
        assert Path(f"/assets/{MODEL}/model.safetensors.index.json").exists()
        runtime_module = miles_runtime.runtime
        if ray_ip != "auto":
            if ray_ip == "127.0.0.1":
                runtime_module.ray._private.services.resolve_ip_for_localhost = (
                    lambda host: host
                )
            original_ray_init = runtime_module.ray.init

            def local_ray_init(*args, **kwargs):
                kwargs.setdefault("_node_ip_address", ray_ip)
                return original_ray_init(*args, **kwargs)

            runtime_module.ray.init = local_ray_init
        runtime_module._configure_actor_spec(runtime_module.train_specs)
        original = runtime_module.train_specs._compute_spec_trainer

        def actor_spec(*args, **kwargs):
            spec = original(*args, **kwargs)
            if spec.worker_class.endswith(".SpindleMilesTrainRayActor"):
                spec = spec.model_copy(
                    update={
                        "worker_class": (
                            "qwen9b_snapshot_parity.SnapshotParityActor"
                            if parity_mode
                            else "qwen9b_snapshot_probe.SnapshotProbeActor"
                        )
                    }
                )
            return spec

        runtime_module.train_specs._compute_spec_trainer = actor_spec
        settings = DeploymentConfig.create(Config()).trainer_settings["miles"]
        self.backend = MilesCommandBackend(
            MilesBackendConfig(**settings),
            checkpoint_dir=Path("/tmp/checkpoints"),
            capture_dir=Path("/tmp/captures"),
            base_model=MODEL,
        )
        self.ray_nodes = [
            node["NodeManagerAddress"]
            for node in runtime_module.ray.nodes()
            if node["Alive"]
        ]
        if ray_ip != "auto":
            assert self.ray_nodes == [ray_ip], self.ray_nodes
        self.backend.accept_model(
            "snapshot-probe",
            ModelSpec(
                base_model=MODEL,
                parameterization="lora",
                lora_config=LoraConfig(rank=32),
            ),
        )
        self.backend.forward_backward(make_batch())
        self.backend.optim_step(("snapshot-probe",), AdamParams(learning_rate=1e-4))
        self.reference = self.forward()
        self.before = self.probe()
        self.initialization_s = time.monotonic() - started
        self.suspended_comms = []
        self.suspend_control = None
        if comms_lifecycle == "suspend":
            suspended = self.actor_command("snapshot_suspend_comms")
            resumed = self.actor_command("snapshot_resume_comms")
            assert self.probe() == self.before
            control_forward = self.forward()
            control_error = max(
                abs(a - b) for a, b in zip(control_forward, self.reference, strict=True)
            )
            assert control_error <= 1e-5, control_error
            self.before = self.probe()
            self.suspend_control = {
                "suspended": suspended,
                "resumed": resumed,
                "max_logprob_error": control_error,
            }
            print(
                "SNAPSHOT_PHASE suspend_control_passed",
                json.dumps(self.suspend_control),
                flush=True,
            )
            if parity_mode:
                self.prepare_parity()
            self.suspended_comms = self.actor_command("snapshot_suspend_comms")
            print(
                "SNAPSHOT_PHASE comms_suspended",
                json.dumps(self.suspended_comms),
                flush=True,
            )
        elif parity_mode:
            self.prepare_parity()
        print(
            "SNAPSHOT_PHASE ready_to_capture",
            json.dumps(
                {
                    "initialization_id": self.initialization_id,
                    "initialization_s": self.initialization_s,
                    "ranks": self.before,
                }
            ),
            flush=True,
        )

    def forward(self):
        (output,) = self.backend.forward_backward(make_batch(forward_only=True))
        return list(output.loss_fn_outputs[0]["logprobs"].data)

    def probe(self):
        return self.actor_command("snapshot_probe")

    def actor_command(self, name, **kwargs):
        runtime = self.backend.runtime
        return runtime._run(runtime._trainer._execute_slots(name, **kwargs))

    def prepare_parity(self):
        self.parity_step = self.backend.jobs["snapshot-probe"].optimizer_step
        self.actor_command("parity_save")
        self.parity_controls = [
            self.parity_trial(f"control_{i}", reset=True) for i in range(3)
        ]
        if te_deterministic:
            self.parity_controls.extend(
                [
                    self.parity_trial(
                        f"fa3_{i}", reset=True, attention_deterministic=False
                    )
                    for i in range(2)
                ]
            )
        self.actor_command("parity_reset")
        self.forward()
        self.before = self.probe()
        self.parity_inventory = self.actor_command("parity_save")
        self.parity_audit_before = self.actor_command("parity_audit")
        print(
            "PARITY_CONTROLS",
            json.dumps(
                [
                    {
                        "label": row["label"],
                        "pre_step_logprob_error": row["pre_step_logprob_error"],
                        "grad_norm": row["optimizer"]["grad_norm:mean"],
                        "gradients": [
                            {
                                key: rank.get(key)
                                for key in (
                                    "rank",
                                    "sha256",
                                    "different_elements",
                                    "max_abs",
                                    "attention_backend",
                                )
                            }
                            for rank in row["gradients"]
                        ],
                    }
                    for row in self.parity_controls
                ]
            ),
            flush=True,
        )

    def parity_trial(self, label, *, reset, attention_deterministic=None):
        from tinker import AdamParams

        if reset:
            self.actor_command("parity_reset")
        if attention_deterministic is not None:
            self.actor_command(
                "parity_set_te_determinism", enabled=attention_deterministic
            )
        job = self.backend.jobs["snapshot-probe"]
        job.optimizer_step = self.parity_step
        job.accumulating = False
        before = self.forward()
        error = max(abs(a - b) for a, b in zip(before, self.reference, strict=True))
        self.backend.forward_backward(make_batch())
        gradients = self.actor_command("parity_observe", stage="gradients", label=label)
        (optimizer,) = self.backend.optim_step(
            ("snapshot-probe",), AdamParams(learning_rate=1e-4)
        )
        assert optimizer.metrics["update_successful:mean"] == 1
        updated = self.actor_command("parity_observe", stage="updated", label=label)
        return {
            "label": label,
            "pre_step_logprob_error": error,
            "gradients": gradients,
            "updated": updated,
            "optimizer": optimizer.metrics,
            "post_update_logprobs": self.forward(),
        }

    @modal.enter(snap=False)
    def restored(self):
        self.recover_runtime()

    def recover_runtime(self):
        self.restore_hook_at = time.monotonic()
        self.boot_id = uuid.uuid4().hex
        print("SNAPSHOT_PHASE post_restore", self.boot_id, flush=True)
        self.resumed_comms = []
        if comms_lifecycle == "suspend":
            self.resumed_comms = self.actor_command("snapshot_resume_comms")
            print(
                "SNAPSHOT_PHASE comms_resumed",
                json.dumps(self.resumed_comms),
                flush=True,
            )
        if parity_mode:
            self.parity_audit_after = self.actor_command("parity_audit")

    @modal.method()
    def verify(self):
        import math
        from tinker import AdamParams

        if parity_mode:
            after = self.probe()
            assert after == self.before, (self.before, after)
            trials = [
                self.parity_trial("after_restore_0", reset=False),
                self.parity_trial("after_restore_1", reset=True),
            ]
            return {
                "initialization_id": self.initialization_id,
                "task_id": os.environ["MODAL_TASK_ID"],
                "snapshot_creator_task_id": self.snapshot_creator_task_id,
                "restored_from_different_container": os.environ["MODAL_TASK_ID"]
                != self.snapshot_creator_task_id,
                "hardware": self.hardware,
                "trainer_env": trainer_env,
                "parity_inventory": self.parity_inventory,
                "parity_audit_before": self.parity_audit_before,
                "parity_audit_after": self.parity_audit_after,
                "controls": self.parity_controls,
                "trials": trials,
            }

        started = time.monotonic()
        print("SNAPSHOT_PHASE verifying", flush=True)
        after = self.probe()
        assert after == self.before, (self.before, after)
        actual = self.forward()
        assert all(math.isfinite(x) for x in actual)
        error = max(abs(a - b) for a, b in zip(actual, self.reference, strict=True))
        assert error <= 1e-5, error
        self.backend.forward_backward(make_batch())
        (optimizer,) = self.backend.optim_step(
            ("snapshot-probe",), AdamParams(learning_rate=1e-4)
        )
        assert optimizer.metrics["update_successful:mean"] == 1, optimizer.metrics
        trained = self.forward()
        assert all(math.isfinite(x) for x in trained)
        updated = self.probe()
        assert any(
            a["model_sha256"] != b["model_sha256"]
            for a, b in zip(after, updated, strict=True)
        ), "weights did not change"
        assert any(
            a["optimizer_sha256"] != b["optimizer_sha256"]
            for a, b in zip(after, updated, strict=True)
        ), "optimizer did not change"
        return {
            "initialization_id": self.initialization_id,
            "boot_id": self.boot_id,
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "snapshot_creator_task_id": self.snapshot_creator_task_id,
            "restored_from_different_container": (
                os.environ["MODAL_TASK_ID"] != self.snapshot_creator_task_id
            ),
            "transport": transport,
            "trainer_env": trainer_env,
            "comms_lifecycle": comms_lifecycle,
            "suspend_control": self.suspend_control,
            "suspended_comms": self.suspended_comms,
            "resumed_comms": self.resumed_comms,
            "ray_ip": ray_ip,
            "ray_nodes": self.ray_nodes,
            "hardware": self.hardware,
            "initialization_s": self.initialization_s,
            "verification_s": time.monotonic() - started,
            "max_logprob_error": error,
            "reference_logprobs": self.reference,
            "post_update_logprobs": trained,
            "ranks": after,
            "updated_ranks": updated,
            "optimizer": optimizer.metrics,
            "optimizer_step": self.backend.jobs["snapshot-probe"].optimizer_step,
        }

    @modal.exit()
    def close(self):
        if hasattr(self, "backend"):
            self.backend.close()


options = dict(
    image=image,
    gpu=os.environ.get("SPINDLE_SNAPSHOT_GPU", "H100!:4"),
    cpu=16,
    memory=98304,
    volumes={"/assets": assets},
    env=trainer_env,
    timeout=600,
    startup_timeout=1800,
    max_containers=1,
    single_use_containers=True,
    retries=0,
)


@app.cls(**options)
class Baseline(Trainer):
    @modal.enter()
    def restored(self):
        self.initialize_runtime()
        self.recover_runtime()


@app.cls(
    **options,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class Snapshot(Trainer):
    @modal.enter(snap=True)
    def initialize(self):
        self.initialize_runtime()


@app.local_entrypoint()
def main(
    mode: str = "snapshot",
    repeats: int = 2,
    output: str = "/tmp/qwen9b-memory-snapshot.json",
    settle_seconds: float = 60.0,
    require_restore: bool = False,
):
    if repeats < 1 or settle_seconds < 0:
        raise ValueError("repeats must be positive and settle_seconds nonnegative")
    if require_restore and mode != "snapshot":
        raise ValueError("require_restore is only valid in snapshot mode")
    if mode == "prepare":
        prepare.remote()
        return
    if mode not in {"baseline", "snapshot"}:
        raise ValueError("mode must be prepare, baseline, or snapshot")
    cls = modal.Cls.from_name(APP_NAME, mode.capitalize())
    rows = []
    for _ in range(repeats):
        if rows and mode == "snapshot" and settle_seconds:
            # Publishing a large snapshot continues after the original resumes.
            print("SNAPSHOT_SETTLE", settle_seconds, flush=True)
            time.sleep(settle_seconds)
        started = time.monotonic()
        call = cls().verify.spawn()
        print("SNAPSHOT_CALL", call.object_id, flush=True)
        try:
            row = call.get(timeout=2400)
        except BaseException as exc:
            with contextlib.suppress(Exception):
                call.cancel(terminate_containers=True)
            Path(output).write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "rows": rows,
                        "failure": {
                            "call_id": call.object_id,
                            "error": str(exc),
                            "elapsed_s": time.monotonic() - started,
                        },
                    },
                    indent=2,
                )
            )
            raise
        row["call_to_result_s"] = time.monotonic() - started
        rows.append(row)
        Path(output).write_text(json.dumps({"mode": mode, "rows": rows}, indent=2))
        print("SNAPSHOT_RESULT", json.dumps(row), flush=True)
    if require_restore and not any(
        row.get("restored_from_different_container", False) for row in rows
    ):
        raise RuntimeError(
            "No restore into a different container was observed. Check snapshot "
            "publication and worker compatibility; initialization alone is not a pass."
        )
