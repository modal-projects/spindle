from __future__ import annotations

import asyncio
import inspect
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
from spindle.startup_timing import BACKEND_COMPONENT, timeline

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

    def load_dimuon_slot(
        self,
        slot: int,
        rank: int,
        alpha: float,
        *,
        checkpoint: str | None = None,
        restore_optimizer: bool = True,
    ) -> None:
        """Loads a slot whose per-slot Miles optimizer is Dimuon.

        Raises BackendFailed when the deployed Miles revision has no per-slot optimizer families.
        """
        if "optimizer" not in inspect.signature(self._bridge.load_slot).parameters:
            raise BackendFailed(
                f"Miles revision {self.revision} does not support the dimuon optimizer"
            )
        self._run(
            self._bridge.load_slot(
                slot,
                rank,
                alpha,
                ckpt_path=checkpoint,
                load_optimizer=restore_optimizer,
                optimizer="dimuon",
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
        _configure_actor_spec(train_specs)
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
        configure_logger(args, source=_main_process_identity())
        startup = timeline(BACKEND_COMPONENT)
        startup.mark("miles_args_parsed")

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
        startup.mark("ray_started")

        self._worker_manager = launch_worker_manager(args)
        object_store.init_instance(args, contribute_segment=False)
        self._owns_object_store = True
        startup.mark("ray_workers_launched")
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
        startup.mark("trainer_actors_initialized")
        self._bridge = tinker_runtime.MilesBackend(
            self._trainer, router_url="", dp_size=1
        )
        startup.mark("miles_backend_ready")

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
    "TILELANG_CACHE_DIR",
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


def _main_process_identity():
    # Miles v0.1.2 folded the per-process identity classes into one.
    if hasattr(process_identity, "MainProcessIdentity"):
        return process_identity.MainProcessIdentity()
    return process_identity.SimpleProcessIdentity(component="main")


def _configure_actor_spec(train_specs) -> None:
    original = train_specs._compute_spec_trainer
    if getattr(original, "_spindle_actor_spec", False):
        return

    @wraps(original)
    def compute(*args, **kwargs):
        spec = original(*args, **kwargs)
        if (
            spec.worker_class
            == "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"
        ):
            return spec.model_copy(
                update={
                    "worker_class": "spindle.backends.miles_runtime.actor.SpindleMilesTrainRayActor"
                }
            )
        return spec

    compute._spindle_actor_spec = True
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
