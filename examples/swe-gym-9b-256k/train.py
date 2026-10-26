"""Eight independent SWE-Gym clients. Each resumes from its own checkpoint."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import tinker
from datasets import load_dataset
from transformers import AutoTokenizer

from swe_gym_env import Environments
from swe_gym_lora_benchmark import (
    SYSTEM,
    TOOLS,
    Job,
    Log,
    atomic_json,
    config,
    publish_policy,
    request_restart,
)

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run"
DATASET = "SWE-Gym/SWE-Gym"
os.environ.setdefault("SWE_SANDBOX_APP", "swe-gym-9b-256k-sandboxes")


def load_tasks():
    ids = json.loads((ROOT / "data" / "validated_workload_ids.json").read_text())
    wanted = set(ids)
    rows = {}
    for row in load_dataset(DATASET, split="train"):
        instance_id = row["instance_id"]
        if instance_id not in wanted:
            continue
        task = dict(row)
        task["FAIL_TO_PASS"] = list(task["FAIL_TO_PASS"])
        task["PASS_TO_PASS"] = list(task["PASS_TO_PASS"])
        rows[instance_id] = task
    missing = [instance_id for instance_id in ids if instance_id not in rows]
    if missing:
        raise SystemExit(
            f"{DATASET} is missing {len(missing)} validated tasks, including {missing[0]}"
        )
    return [rows[instance_id] for instance_id in ids]


def main():
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY")
    manifest = json.loads((RUN / "launch.json").read_text())
    if manifest.get("environment"):
        os.environ["MODAL_ENVIRONMENT"] = manifest["environment"]
    os.environ["SWE_SANDBOX_APP"] = manifest.get(
        "sandbox_app", "swe-gym-9b-256k-sandboxes"
    )
    tasks = load_tasks()
    cfg = config(
        SimpleNamespace(
            rollout_workers=64,
            inflight_groups=32,
            grading_workers=64,
            grading_queue_size=64,
            steps=manifest["steps_per_job"],
            hours=manifest["hours"],
            tasks=len(tasks),
        )
    )
    cfg["sandbox_memory_mib"] = 16384
    cfg["client_stagger_seconds"] = manifest["client_stagger_seconds"]
    cfg["validated_workload_tasks"] = len(tasks)
    atomic_json(RUN / "config.json", cfg)
    log = Log(RUN)
    env = Environments(RUN)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=manifest["model_revision"]
    )
    tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Check rollout template."},
        ],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    service = tinker.ServiceClient(base_url=manifest["base_url"], timeout=3600)
    reference = service.create_sampling_client(base_model=cfg["model"])
    cancelled = threading.Event()
    ready = [threading.Event() for _ in range(8)]
    starts = {}
    models = {}
    next_groups = {job: 0 for job in range(8)}
    eventfile = RUN / "events.jsonl"
    if eventfile.exists():
        for line in eventfile.open():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (
                row.get("phase") == "eight"
                and "group" in row
                and row.get("job") in next_groups
            ):
                next_groups[row["job"]] = max(next_groups[row["job"]], row["group"] + 1)

    def run(job):
        try:
            checkpoint = RUN / f"eight-job{job}-checkpoint.json"
            record = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
            step = record["step"] + 1 if record else 0
            if step >= cfg["steps"]:
                starts[job] = time.monotonic()
                ready[job].set()
                return
            # Load before release, then space actual rollout starts. A slow cold
            # start cannot compress later clients into one burst.
            if job:
                while not ready[job - 1].wait(1):
                    if cancelled.is_set():
                        raise RuntimeError("Another client failed")
                delay = (
                    starts[job - 1] + cfg["client_stagger_seconds"] - time.monotonic()
                )
                if delay > 0 and cancelled.wait(delay):
                    raise RuntimeError("Another client failed")
            if record:
                trainer = service.create_training_client_from_state_with_optimizer(
                    record["checkpoint"]
                )
            else:
                trainer = service.create_lora_training_client(
                    cfg["model"],
                    rank=32,
                    train_unembed=False,
                    user_metadata={"experiment": "swe256", "job": str(job)},
                )
            if not record:
                initial_path = RUN / "initial.json"
                if job == 0 and not initial_path.exists():
                    initial = trainer.save_state("initial").result(timeout=3600).path
                    atomic_json(initial_path, {"path": initial})
                else:
                    trainer.load_state_with_optimizer(
                        json.loads(initial_path.read_text())["path"]
                    ).result(timeout=3600)
            sampler = publish_policy(
                trainer, service, f"eight-job{job}-start-{time.time_ns()}"
            )
            client = Job(
                job, trainer, service, tokenizer, env, tasks, cfg, log, reference
            )
            client.sampler = sampler
            with log.lock:
                models[str(job)] = trainer.model_id
                atomic_json(
                    RUN / "trainers.json", {"models": dict(models), "time": time.time()}
                )
            starts[job] = time.monotonic()
            log.event(
                "client_ready",
                phase="eight",
                job=job,
                model_id=trainer.model_id,
                start_step=step,
            )
            ready[job].set()
            client.run("eight", manifest["absolute_deadline"], step, next_groups[job])
            log.event("client_done", phase="eight", job=job)
        except BaseException as exc:
            cancelled.set()
            request_restart(log, job, exc)
            raise

    log.event(
        "phase_start",
        phase="eight",
        active_jobs=8,
        continuation=any(
            (RUN / f"eight-job{job}-checkpoint.json").exists() for job in range(8)
        ),
    )
    with ThreadPoolExecutor(8) as pool:
        jobs = [pool.submit(run, job) for job in range(8)]
        for job in jobs:
            job.result()
    log.event("completed")


if __name__ == "__main__":
    main()
