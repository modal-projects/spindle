from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Coroutine
from contextlib import contextmanager, suppress
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Any

from spindle.backends.miles_arguments import defer_config_overrides
from spindle.backends.miles_config import MilesBackendConfig
from spindle.errors import BackendFailed

from .replay_data import install_bridge_replay, validate_routed_experts

# CPU-side data helpers can be imported without the optional GPU runtime.
_runtime_import_error = None
try:
    import ray
    from miles.ray.specs import train as train_specs
    from miles.ray.train.group import TrainerController
    from miles.ray.wiring import launch_worker_manager
    from miles.tinker import runtime as tinker_runtime
    from miles.utils import object_store
    from miles.utils.arguments import parse_args
    from miles.utils.audit_utils import process_identity
    from miles.utils.external_utils.model_args_utils import load_model_args
    from miles.utils.logging_utils import configure_logger
    from miles.utils.lora import arguments as lora_arguments
except ModuleNotFoundError as exc:
    if exc.name not in {"ray", "miles"}:
        raise
    _runtime_import_error = exc

try:
    # Miles v0.1.2 runs the trainer controller as a worker reached by handle.
    from miles.backends.megatron_utils.megatron_config import compute_trainer_args
    from miles.ray.placement_group import create_trainer_handles
    from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs
    from miles.ray.wiring import shutdown_worker_manager
except ImportError:
    create_trainer_handles = None


class MilesRuntime:
    """Synchronous owner of Miles's asynchronous Ray trainer controller."""

    def __init__(self, config: MilesBackendConfig) -> None:
        if _runtime_import_error is not None:
            raise ImportError(
                "MilesRuntime requires Miles and Ray in the trainer image"
            ) from _runtime_import_error
        self.config = config
        # Record the commit actually installed from the branch in this image.
        self.revision = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(tinker_runtime.__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="spindle-miles-runtime",
            daemon=True,
        )
        self._thread.start()
        self._closed = False
        self._failure: BaseException | None = None
        self._trainer = None
        self._worker_manager = None
        self._bridge = None
        self._owns_ray = False
        self._owns_object_store = False
        self._unit_ids = count(1)
        try:
            self._call(self._start())
        except BaseException:
            with suppress(BaseException):
                self._call(self._close())
            self._stop_loop()
            raise

    def load_slot(
        self,
        slot: int,
        rank: int,
        alpha: float,
        *,
        checkpoint: str | None = None,
        restore_optimizer: bool = True,
    ) -> None:
        self._run(
            self._bridge.load_slot(
                slot,
                rank,
                alpha,
                ckpt_path=checkpoint,
                load_optimizer=restore_optimizer,
            )
        )

    def unload_slot(self, slot: int) -> None:
        self._run(self._bridge.unload_slot(slot))

    def forward_backward(
        self,
        slot_rows: tuple[tuple[int, dict[str, Any]], ...],
        *,
        loss_fn: str,
        loss_fn_config: dict[str, float],
        forward_only: bool,
    ) -> list[dict[str, Any]]:
        if any("routed_experts" in row for _, row in slot_rows):
            validate_routed_experts(
                slot_rows,
                num_layers=self._args.num_layers,
                num_experts=self._args.num_experts,
                topk=self._args.moe_router_topk,
            )
        unit_id = next(self._unit_ids)
        method = (
            self._bridge.forward_only if forward_only else self._bridge.forward_backward
        )
        return self._run(
            method(
                unit_id,
                list(slot_rows),
                loss_fn,
                loss_fn_config,
            )
        )

    def optim_step(
        self, adam_params_by_slot: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        return self._run(self._bridge.optim_step(adam_params_by_slot))

    def save_slot(
        self, slot: int, path: str, *, include_optimizer: bool = True
    ) -> None:
        if include_optimizer:
            self._run(self._bridge.save_slot(slot, path))
        else:
            self._run(
                self._trainer._execute_slots("save_slot_weights", slot=slot, path=path)
            )
        _materialize_capture(path)

    def export_slot_peft(
        self,
        *,
        slot: int,
        path: str,
        rank: int,
        alpha: float,
        base_model: str,
        target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> None:
        self._run(self._bridge.export_slot(slot, rank, alpha, path))
        _materialize_capture(path)

    def full_optim_step(self, adam_params: dict[str, float]) -> dict[str, float]:
        return _first(
            self._run(
                self._trainer._execute_slots("optim_step", adam_params=adam_params)
            )
        )

    def save_full_checkpoint(
        self,
        path: str,
        *,
        include_optimizer: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._run(
            self._trainer._execute_slots(
                "save_checkpoint",
                path=path,
                include_optimizer=include_optimizer,
                metadata=metadata,
            )
        )

    def load_full_checkpoint(self, path: str, *, restore_optimizer: bool) -> None:
        self._run(
            self._trainer._execute_slots(
                "load_checkpoint", path=path, restore_optimizer=restore_optimizer
            )
        )

    def capture_full_delta(self, **kwargs: Any) -> dict[str, Any]:
        """Stage a sampler delta on the writer ranks and return the merged
        manifest."""
        results = self._run(
            self._trainer._execute_slots("capture_sampler_delta", **kwargs)
        )
        manifests = [result for result in results if result is not None]
        if not manifests:
            raise BackendFailed("no trainer rank staged a sampler delta")
        return merge_delta_manifests(manifests)

    def full_delta_aligned_with(self, identity: str) -> bool:
        return all(
            self._run(
                self._trainer._execute_slots("is_delta_aligned_with", identity=identity)
            )
        )

    def forget_full_delta_parent(self) -> None:
        self._run(self._trainer._execute_slots("forget_delta_parent"))

    def reload_base_weights(self) -> None:
        self._run(self._trainer._execute_slots("reload_base_weights"))

    def torch_profile_start(self) -> None:
        self._run_best_effort(self._trainer._execute_slots("torch_profile_start"))

    def torch_profile_stop(self, output_dir: str) -> list[dict | None]:
        return self._run_best_effort(
            self._trainer._execute_slots("torch_profile_stop", output_dir=output_dir)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._call(self._close())
        finally:
            self._stop_loop()

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        if self._closed:
            coroutine.close()
            raise BackendFailed("Miles runtime is closed")
        if self._failure is not None:
            coroutine.close()
            raise BackendFailed("Miles trainer is unavailable") from self._failure
        try:
            result = self._call(coroutine)
            _check_result(result)
            return result
        except Exception as exc:
            self._failure = exc
            raise BackendFailed(f"Miles trainer failed: {exc}") from exc

    def _run_best_effort(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Like ``_run`` but an exception does not mark the trainer as failed."""
        if self._closed or self._failure is not None:
            coroutine.close()
            raise BackendFailed("Miles trainer is unavailable")
        result = self._call(coroutine)
        _check_result(result)
        return result

    def _call(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _stop_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("Miles runtime event loop did not stop")
        self._loop.close()

    async def _start(self) -> None:
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.setdefault("no_proxy", "127.0.0.1")

        install_bridge_replay(tinker_runtime)
        _configure_actor_spec(train_specs, full=self.config.full)
        _allow_context_parallel_multi_lora(lora_arguments)
        architecture = (
            shlex.split(load_model_args(self.config.model_type))
            if self.config.model_type
            else []
        )
        with _temporary_argv([*architecture, *self.config.miles_arguments()]):
            if self.config.cli_options:

                def configure(parser):
                    return defer_config_overrides(
                        parser, self.config.cli_options, sys.argv
                    )

                args = parse_args(add_custom_arguments=configure, entry="serve")
            else:
                args = parse_args(entry="serve")
        self._args = args

        # Match the megatron LoRA backend and PEFT/Tinker: kaiming-uniform A, zero B.
        args.lora_A_init_method = "kaiming"
        args.use_dynamic_global_batch_size = True
        args.delay_split_train_data_by_dp = True
        if self.config.full:
            _neutralize_lr_schedule(args, self.config.data_parallel_size)
        configure_logger(args, source=_main_process_identity())

        if ray.is_initialized():
            raise RuntimeError("MilesRuntime requires exclusive ownership of Ray")
        address = os.environ.get("SPINDLE_RAY_ADDRESS")
        if address:
            # A multi-node training cluster already exists.
            ray.init(
                address=address,
                ignore_reinit_error=False,
                log_to_driver=True,
                runtime_env={"env_vars": _worker_env()},
            )
        else:
            ray.init(
                ignore_reinit_error=False,
                include_dashboard=False,
                log_to_driver=True,
                num_cpus=max(4, self.config.world_size * 2),
                num_gpus=self.config.world_size,
            )
        self._owns_ray = True
        _require_cluster_nodes(
            ray,
            nodes=self.config.actor_num_nodes,
            world_size=self.config.world_size,
        )

        self._worker_manager = launch_worker_manager(args)
        object_store.init_instance(args, contribute_segment=False)
        self._owns_object_store = True
        if create_trainer_handles is not None:
            configs = compute_trainer_configs(args)
            (actor_config,) = [c for c in configs if c.role == ACTOR_ROLE]
            handles = create_trainer_handles(args, trainer_configs=configs)
            self._trainer = handles[actor_config.trainer_id]
            await self._trainer.init(compute_trainer_args(args, actor_config))
        else:
            self._trainer = TrainerController(
                args=args,
                role="actor",
                with_ref=False,
                with_opd_teacher=False,
                inference_controller=None,
                rollout_executor=None,
            )
            await self._trainer.init()
        self._bridge = tinker_runtime.MilesBackend(
            self._trainer, router_url="", dp_size=1
        )

    async def _close(self) -> None:
        try:
            if self._trainer is not None and create_trainer_handles is not None:
                with suppress(BaseException):
                    await self._trainer.dispose()
            elif self._trainer is not None:
                cell_ids = list(self._trainer.cell_ids)
                try:
                    await self._trainer.dispose()
                finally:
                    if self._worker_manager is not None and cell_ids:
                        with suppress(BaseException):
                            await self._worker_manager.stop_cells.remote(cell_ids)
        finally:
            if self._worker_manager is not None and create_trainer_handles is not None:
                with suppress(BaseException):
                    await shutdown_worker_manager(self._worker_manager)
            elif self._worker_manager is not None:
                with suppress(BaseException):
                    ray.kill(self._worker_manager, no_restart=True)
            if self._owns_object_store:
                object_store._INSTANCE = None
                self._owns_object_store = False
            if self._owns_ray and ray.is_initialized():
                ray.shutdown()
            self._owns_ray = False
            self._bridge = None
            self._trainer = None
            self._worker_manager = None


_WORKER_ENV_VARS = (
    "SPINDLE_CHECKPOINT_VOLUME",
    "SPINDLE_BULLETIN_VOLUME",
    "SPINDLE_BULLETIN_ROOT",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "EXPERT_LORA_COMPAT",
)


def _worker_env() -> dict[str, str]:
    """Settings the trainer actors need that a pre-existing Ray cluster lacks."""
    return {name: os.environ[name] for name in _WORKER_ENV_VARS if name in os.environ}


def _require_cluster_nodes(
    ray, *, nodes: int, world_size: int, timeout: float = 120.0
) -> None:
    """Fail fast if the cluster never exposes every node's GPUs, instead of hanging in actor placement."""
    deadline = time.monotonic() + timeout
    while True:
        gpu_nodes = [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) > 0
        ]
        alive_gpu_nodes = len(gpu_nodes)
        total_gpus = sum(node["Resources"].get("GPU", 0) for node in gpu_nodes)
        if alive_gpu_nodes >= nodes and total_gpus >= world_size:
            return
        if time.monotonic() >= deadline:
            raise BackendFailed(
                f"Ray cluster exposes {total_gpus} GPUs on {alive_gpu_nodes} nodes, "
                f"trainer needs {world_size} GPUs on {nodes} nodes"
            )
        time.sleep(2.0)


@contextmanager
def _temporary_argv(arguments: list[str]):
    original = sys.argv
    sys.argv = ["spindle-miles-backend", *arguments]
    try:
        yield
    finally:
        sys.argv = original


def _check_result(result) -> None:
    if isinstance(result, (list, tuple)):
        for item in result:
            _check_result(item)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Miles operation failed: {result['error']}")


def _first(results):
    return results[0]


def _main_process_identity():
    # Miles v0.1.2 folded the per-process identity classes into one.
    if hasattr(process_identity, "MainProcessIdentity"):
        return process_identity.MainProcessIdentity()
    return process_identity.SimpleProcessIdentity(component="main")


_ACTOR_CLASSES = {
    "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor": (
        "spindle.backends.miles_runtime.actor.SpindleMilesTrainRayActor"
    ),
    "miles.backends.megatron_utils.actor.MegatronTrainRayActor": (
        "spindle.backends.miles_runtime.full_actor.SpindleMilesFullTrainRayActor"
    ),
}


def merge_delta_manifests(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine the manifests of the writer ranks, one per pipeline stage."""
    first = manifests[0]
    weight_map: dict[str, str] = {}
    metrics: dict[str, float] = {}
    for manifest in manifests:
        for key in ("ref", "parent", "metadata"):
            if manifest[key] != first[key]:
                raise BackendFailed(f"staged delta manifests disagree on {key}")
        overlap = weight_map.keys() & manifest["weight_map"].keys()
        if overlap:
            raise BackendFailed(
                f"staged delta manifests overlap: {sorted(overlap)[:5]}"
            )
        weight_map.update(manifest["weight_map"])
        for key, value in manifest["metrics"].items():
            metrics[key] = (
                max(metrics.get(key, 0.0), value)
                if key == "capture_seconds"
                else metrics.get(key, 0.0) + value
            )
    return {**first, "weight_map": weight_map, "metrics": metrics}


def _neutralize_lr_schedule(args, data_parallel_size: int) -> None:
    """Miles sizes its LR schedule from a rollout plan; Spindle sets optimizer
    hyperparameters on every ``optim_step`` instead."""
    args.num_rollout = 1
    args.rollout_batch_size = data_parallel_size
    args.n_samples_per_prompt = 1
    args.global_batch_size = data_parallel_size
    args.micro_batch_size = 1


def _configure_actor_spec(train_specs, *, full: bool = False) -> None:
    original = train_specs._compute_spec_trainer
    if getattr(original, "_spindle_actor_spec", None) == full:
        return
    if getattr(original, "_spindle_actor_spec", None) is not None:
        original = original.__wrapped__

    @wraps(original)
    def compute(*args, **kwargs):
        spec = original(*args, **kwargs)
        replacement = _ACTOR_CLASSES.get(spec.worker_class)
        if replacement is None or (
            not full and spec.worker_class.endswith("MegatronTrainRayActor")
        ):
            return spec
        return spec.model_copy(update={"worker_class": replacement})

    compute._spindle_actor_spec = full
    train_specs._compute_spec_trainer = compute


def _allow_context_parallel_multi_lora(lora_arguments) -> None:
    """Miles rejects multi-LoRA with CP>1 because its Tinker losses zip
    full-length per-datum vectors against CP-sharded log probs. The Spindle actor
    gathers those log probs back to full response length before the loss runs,
    so the guard does not apply to this path."""
    original = lora_arguments.validate_multi_lora_args
    if getattr(original, "__spindle_allows_cp__", False):
        return

    @wraps(original)
    def validate(args) -> None:
        context_parallel_size = args.context_parallel_size
        args.context_parallel_size = 1
        try:
            original(args)
        finally:
            args.context_parallel_size = context_parallel_size

    validate.__spindle_allows_cp__ = True
    lora_arguments.validate_multi_lora_args = validate


def _materialize_capture(path: str) -> None:
    """Detach Miles's completed version directory before Spindle copies/cleans it."""
    capture = Path(path)
    if not capture.is_symlink():
        return
    version = capture.resolve(strict=True)
    if version.parent != capture.parent.resolve() or not version.name.startswith(
        f"_version_{capture.name}_"
    ):
        raise RuntimeError(f"unexpected Miles capture target: {version}")
    capture.unlink()
    version.rename(capture)
