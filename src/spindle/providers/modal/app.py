from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict

import modal
from huggingface_hub import snapshot_download
from stitch.pools.modal_flash import ModalFlashPool

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.control_plane.keys import model_key, placement_key, trainer_demand_key
from spindle.control_plane.records import ModelRecord
from spindle.deployments import validate_frontend
from spindle.inference.sampling import sample_task
from spindle.providers.contracts import (
    Parameterization,
    SamplingTask,
)
from spindle.startup_timing import StartupTimeline
from spindle.telemetry.otlp import sample_trace

from .app_tags import app_tags
from .checkpoint_storage import (
    CHECKPOINT_ROOT,
    ModalCheckpointStorage,
)
from .deployment import trainer_deployment_env
from .deployment_configs import (
    CONFIGS_FILE_ENV,
    CONFIGS_IMAGE_PATH,
    PLATFORM_ENV,
    configs_from_env,
    platform_from_env,
)
from .engines import EngineInstanceRecord, ModalEnginePlatform
from .fft_pool import (
    FFTPoolSpec,
    deploy_pool,
    pool_gateway,
    proxy_auth_headers,
    stop_pool,
)
from .image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .kv import (
    ModalSessionKeyValueStores,
    current_app_id,
    fft_pool_kv,
    shared_kv,
)
from .lora_pool import (
    LoraPoolSpec,
)
from .lora_pool import (
    deploy_pool as deploy_lora_pool,
)
from .lora_pool import (
    pool_gateway as lora_pool_gateway,
)
from .lora_pool import (
    stop_pool as stop_lora_pool,
)
from .sampling import ModalSamplingTaskPlatform
from .trainer_reconciler import (
    complete_reconcile,
    pending_reconciliations,
    reconcile_trainers,
    release_reconcile_call,
    request_reconcile,
)

DEFINITIONS = tuple(configs_from_env())
validate_frontend([definition.recipe for definition in DEFINITIONS])
SETTINGS = DEFINITIONS[0]
PLATFORM = platform_from_env()
APP_NAME = PLATFORM["frontend"]
ROUTING_REGION = PLATFORM["modal"]["region"]
MODEL_ASSET_ROOT = "/assets"
SESSION_IDLE_TIMEOUT = SETTINGS.recipe.session_idle_timeout_s
FFT_POOL_IDLE_TIMEOUT = LORA_POOL_IDLE_TIMEOUT = SETTINGS.recipe.pool_idle_timeout_s
FFT_POOL_TOUCH_INTERVAL = 60.0
WEIGHTS_POLL_INTERVAL = 10.0
LORA_POOL_CHECK_INTERVAL = 60.0
SWEEP_PERIOD = modal.Period(seconds=SETTINGS.recipe.sweep_interval_s)
CHECKPOINT_READ_LOCK = asyncio.Lock()
_pool_touches: dict[str, float] = {}
_lora_pool_gateways: dict[str, tuple[float, str]] = {}
_lora_pool_checks: dict[str, asyncio.Lock] = {}
_weights_ready: set[str] = set()
_weights_starts: dict[str, asyncio.Lock] = {}
CHECKPOINT_VOLUME_NAME = PLATFORM["storage"]["checkpoints"]
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
TRAINER_DEPLOYMENT_ENV = {
    **trainer_deployment_env(),
    CONFIGS_FILE_ENV: CONFIGS_IMAGE_PATH,
    PLATFORM_ENV: os.environ.get(PLATFORM_ENV, ""),
    "SPINDLE_APP_NAME": APP_NAME,
}
app = modal.App(APP_NAME, tags=app_tags("frontend"))


async def _read_checkpoint_metadata(uri: str) -> dict[str, object]:
    return await ModalCheckpointStorage(
        checkpoint_volume, CHECKPOINT_ROOT, lock=CHECKPOINT_READ_LOCK
    ).read_metadata(uri)


async def _list_checkpoints(model_id: str | None) -> list[dict[str, object]]:
    return await ModalCheckpointStorage(
        checkpoint_volume, CHECKPOINT_ROOT, lock=CHECKPOINT_READ_LOCK
    ).list(model_id)


async def _delete_checkpoint(uri: str) -> None:
    await ModalCheckpointStorage(
        checkpoint_volume, CHECKPOINT_ROOT, lock=CHECKPOINT_READ_LOCK
    ).delete(uri)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*CORE_PACKAGES, TINKER_PACKAGE)
    .pip_install(STITCH_PACKAGE, "huggingface-hub")
    .add_local_file(os.environ[CONFIGS_FILE_ENV], CONFIGS_IMAGE_PATH, copy=True)
    .env(TRAINER_DEPLOYMENT_ENV)
    .add_local_python_source("spindle", ignore=ignore_config_source)
)


@app.function(image=image)
def deployed_configs():
    return [row.model_dump(mode="json") for row in configs_from_env()]


@app.function(image=image)
def deployed_platform():
    return PLATFORM


@app.function(image=image)
async def deployed_pools():
    return [
        value
        for registry, prefix in (
            (shared_kv(), "lora_pool:"),
            (fft_pool_kv(), "fft_pool:"),
        )
        for _, value in await registry.list_items(prefix)
    ]


model_assets = modal.Volume.from_name(
    PLATFORM["storage"]["assets"],
    create_if_missing=True,
)
API_SECRET_NAME = PLATFORM["secrets"]["api"]
HF_SECRET_NAME = PLATFORM["secrets"]["huggingface"]
proxy_secret = modal.Secret.from_name(
    PLATFORM["secrets"]["sampler_proxy"],
    required_keys=["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"],
)


@app.function(
    image=image,
    volumes={MODEL_ASSET_ROOT: model_assets},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)] if HF_SECRET_NAME else [],
    timeout=4 * 60 * 60,
    max_containers=1,
    retries=2,
)
async def prepare_model_assets(definition_id: str) -> None:
    definition = module_for(definition_id)
    checkpoint = os.path.normpath(definition.asset_path)
    if (
        os.path.commonpath((MODEL_ASSET_ROOT, checkpoint)) != MODEL_ASSET_ROOT
        or checkpoint == MODEL_ASSET_ROOT
    ):
        raise ValueError(f"invalid model asset path: {checkpoint}")
    started = time.time()
    await asyncio.to_thread(
        snapshot_download,
        repo_id=definition.weights_repo,
        local_dir=checkpoint,
    )
    await model_assets.commit.aio()
    StartupTimeline("model-assets", origin=started, labels={"path": checkpoint}).mark(
        "assets_ready"
    )
    await shared_kv().put(_weights_ready_key(definition_id), time.time())


@app.function(image=image)
async def start_weights_download(definition_id: str) -> bool:
    """Start the base-weight download for `spindle deploy`; True while it runs."""
    module_for(definition_id)
    return await _start_weights_download(definition_id) is not None


async def _touch_fft_pool(spec: FFTPoolSpec) -> None:
    if spec.latest:
        return
    now = time.time()
    if now - _pool_touches.get(spec.app_name, 0.0) < FFT_POOL_TOUCH_INTERVAL:
        return
    _pool_touches[spec.app_name] = now
    await fft_pool_kv().put(_touch_key(spec), {"touched_at": now})


def _touch_key(spec: FFTPoolSpec) -> str:
    return f"fft_pool_touch:{spec.app_name}"


async def _last_touched(registry, spec: FFTPoolSpec, record: dict) -> float:
    touch = await registry.get(_touch_key(spec))
    return max(
        float(record["touched_at"]),
        float(touch["touched_at"]) if touch is not None else 0.0,
    )


@app.function(image=image, max_containers=1, timeout=20 * 60, retries=2)
async def ensure_fft_pool(spec: dict) -> str:
    pool = FFTPoolSpec.from_dict(spec)
    gateway = await asyncio.to_thread(deploy_pool, pool)
    await fft_pool_kv().put(
        f"fft_pool:{pool.app_name}",
        {**pool.as_dict(), "touched_at": time.time()},
    )
    return gateway


@app.function(image=image, max_containers=1, timeout=20 * 60, retries=2)
async def ensure_lora_pool(spec: dict) -> str:
    pool = LoraPoolSpec.from_dict(spec)
    gateway = await asyncio.to_thread(deploy_lora_pool, pool)
    await shared_kv().put(
        f"lora_pool:{pool.app_name}",
        {**pool.as_dict(), "touched_at": time.time()},
    )
    return gateway


class WeightsDownloadFailed(ValueError):
    pass


def _weights_ready_key(definition_id: str) -> str:
    return f"model_assets_ready:{definition_id}"


def _weights_call_key(definition_id: str) -> str:
    return f"model_assets_call:{definition_id}"


async def _weights_are_ready(definition_id: str) -> bool:
    if definition_id in _weights_ready:
        return True
    if await shared_kv().get(_weights_ready_key(definition_id)) is None:
        return False
    _weights_ready.add(definition_id)
    return True


async def _call_running(call_id: str) -> bool:
    """Whether a spawned call is still running; raises its error if it failed."""
    try:
        await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        return True
    except modal.exception.OutputExpiredError:
        return False
    return False


async def _weights_download(definition_id: str) -> float | None:
    """When the running base-weight download started, or None if none is running."""
    if await _weights_are_ready(definition_id):
        return None
    record = await shared_kv().get(_weights_call_key(definition_id))
    if record is None:
        return None
    try:
        running = await _call_running(record["call_id"])
    except Exception as exc:
        if await _weights_are_ready(definition_id):
            return None
        raise WeightsDownloadFailed(
            f"Downloading weights for {definition_id} failed: {exc}"
        ) from exc
    return float(record["started_at"]) if running else None


async def _start_weights_download(definition_id: str) -> float | None:
    """Join the running download, or start one unless the weights are ready."""
    lock = _weights_starts.setdefault(definition_id, asyncio.Lock())
    async with lock:
        try:
            started = await _weights_download(definition_id)
        except WeightsDownloadFailed:
            started = None
        if started is not None or await _weights_are_ready(definition_id):
            return started
        call = await prepare_model_assets.spawn.aio(definition_id)
        started = time.time()
        await shared_kv().put(
            _weights_call_key(definition_id),
            {"call_id": call.object_id, "started_at": started},
        )
        return started


async def _wait_for_weights(definition_id: str) -> bool:
    """Wait out a running base-weight download; True if there was one."""
    waited = False
    while await _weights_download(definition_id) is not None:
        waited = True
        await asyncio.sleep(WEIGHTS_POLL_INTERVAL)
    return waited


async def _trainer_booted(registry, definition_id: str) -> bool:
    return any(
        record.definition_id == definition_id
        and record.state in {"starting", "running"}
        for _, value in await registry.list_items("engine_instance:")
        for record in (EngineInstanceRecord.model_validate(value),)
    )


async def _ready_lora_pool(spec: LoraPoolSpec) -> str:
    """Refresh warm pools locally; serialize only missing-pool deployment."""
    key = spec.app_name
    cached = _lora_pool_gateways.get(key)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    lock = _lora_pool_checks.setdefault(key, asyncio.Lock())
    async with lock:
        cached = _lora_pool_gateways.get(key)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        try:
            gateway = await lora_pool_gateway(spec)
        except modal.exception.NotFoundError:
            # The remote function retains single concurrency, and deploy_pool
            # rechecks existence before deploying to avoid competing deploys.
            gateway = await ensure_lora_pool.remote.aio(spec.as_dict())
        else:
            await shared_kv().put(
                f"lora_pool:{key}",
                {**spec.as_dict(), "touched_at": time.time()},
            )
        _lora_pool_gateways[key] = (
            time.monotonic() + LORA_POOL_CHECK_INTERVAL,
            gateway,
        )
        return gateway


@app.function(
    image=image,
    min_containers=0,
    timeout=60 * 60,
    retries=2,
    nonpreemptible=True,
    secrets=[proxy_secret, modal.Secret.from_name(API_SECRET_NAME)],
)
@modal.concurrent(max_inputs=128, target_inputs=16)
async def execute_sample(task: dict) -> dict:
    stats: dict = {}
    with sample_trace(task, stats):
        return await _execute_sample(task, stats)


async def _execute_sample(task: dict, stats: dict) -> dict:
    definition_id = str(task["engine_definition_id"])
    parameterization = parameterization_for(definition_id)
    if parameterization not in {"full", "lora"}:
        raise ValueError(f"unsupported sampling definition: {definition_id}")
    definition = module_for(definition_id)
    rollout_world_size = definition.recipe.inference_gpus_per_node
    rollout_tensor_parallel_size = definition.rollout_tensor_parallel_size
    if rollout_world_size % rollout_tensor_parallel_size:
        raise ValueError("rollout GPU count must be divisible by tensor parallel size")
    rollout_data_parallel_size = rollout_world_size // rollout_tensor_parallel_size
    stop_token_ids_only = bool(definition.inference_settings.get("skip_tokenizer_init"))
    waited = task["model_id"] is None and await _wait_for_weights(definition_id)
    if parameterization == "lora":
        spec = LoraPoolSpec(definition_id)
        gateway = await _ready_lora_pool(spec)

        async def keep_pool_ready() -> None:
            # Retries can outlive the cached readiness check. The pool's URL
            # is stable across redeployment of the same definition/revision.
            await _ready_lora_pool(spec)

        return await sample_task(
            task,
            gateway,
            data_parallel_size=rollout_data_parallel_size,
            headers=proxy_auth_headers(),
            context_length=definition.max_context_length,
            stop_token_ids_only=stop_token_ids_only,
            on_wait=keep_pool_ready,
            stats=stats,
        )
    spec = (
        FFTPoolSpec.base(definition_id)
        if task["model_id"] is None
        else FFTPoolSpec(
            definition_id=definition_id,
            model_id=str(task["model_id"]),
            latest=bool(task.get("latest")),
            version=int(task["publish_version"]),
        )
    )
    if waited:
        await ensure_fft_pool.remote.aio(spec.as_dict())
    await _touch_fft_pool(spec)
    return await sample_task(
        task,
        await pool_gateway(spec),
        data_parallel_size=rollout_data_parallel_size,
        headers=proxy_auth_headers(),
        on_wait=lambda: _touch_fft_pool(spec),
        context_length=definition.max_context_length,
        stop_token_ids_only=stop_token_ids_only,
        stats=stats,
    )


def _latest_pool(model) -> FFTPoolSpec:
    return FFTPoolSpec(
        model.engine_definition_id,
        model.model_id,
        True,
        0,
        **(model.spec.get("rollout") or {}),
    )


async def _model_record(kv, model_id: str):
    return ModelRecord.model_validate(await kv.get(model_key(model_id)))


def module_for(definition_id: str):
    for definition in DEFINITIONS:
        if definition.definition_id == definition_id:
            return definition
    raise KeyError(definition_id)


def parameterization_for(definition_id: str) -> Parameterization | None:
    try:
        return module_for(definition_id).parameterization
    except KeyError:
        return None


def trainer_autoscaling(definition_id: str) -> bool:
    return parameterization_for(definition_id) is not None


@app.function(
    image=image,
    max_containers=1,
    timeout=20 * 60,
    retries=3,
)
async def trainer_reconciler(delay_seconds: float = 0.0) -> None:
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)
    call_id = modal.current_function_call_id()

    async def run(definition_id: str, token: str) -> None:
        parameterization = parameterization_for(definition_id)
        if (
            parameterization is None
            or await deployment_error(definition_id)
            or not await _weights_are_ready(definition_id)
        ):
            await complete_reconcile(definition_id, token)
            return
        module = module_for(definition_id)
        maximum_instances = module.recipe.trainer_max_instances
        try:
            await reconcile_trainers(
                shared_kv(),
                ModalEnginePlatform(shared_kv(), _spawn_engine),
                definition_id,
                revision=None,
                maximum_instances=maximum_instances,
                models_per_instance=module.recipe.trainer_max_clients_per_instance,
                scale_up=trainer_autoscaling(definition_id),
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "trainer reconcile %s",
                definition_id,
            )
            raise
        await complete_reconcile(definition_id, token)

    pending = await pending_reconciliations()
    try:
        await asyncio.gather(
            *(run(definition_id, token) for definition_id, token in pending.items())
        )
    finally:
        await release_reconcile_call(call_id)
    remaining = await pending_reconciliations()
    if remaining:
        await kick_trainer_reconciler(next(iter(remaining)))


async def kick_trainer_reconciler(definition_id: str) -> None:
    if parameterization_for(definition_id) is None:
        return

    async def spawn(delay_seconds: float) -> str:
        call = await trainer_reconciler.spawn.aio(delay_seconds)
        return call.object_id

    await request_reconcile(spawn, definition_id)


async def _spawn_engine(definition_id: str, instance_id: str) -> str:
    if error := await deployment_error(definition_id):
        raise ValueError(error)
    definition = module_for(definition_id)
    engine = modal.Function.from_name(
        definition.trainer_app_name,
        "trainer",
        environment_name=PLATFORM["modal"]["environment"],
    )
    call = await engine.spawn.aio(
        instance_id, definition.model_dump_json(), current_app_id()
    )
    return call.object_id


async def deployment_error(definition_id: str) -> str | None:
    registry = shared_kv()
    key = f"deployment_failure:{definition_id}"
    record = await registry.get(key)
    if record:
        if time.time() - record.get("failed_at", 0) >= 300:
            await registry.delete(key)
            return None
        return f"Trainer startup failed for {definition_id}: {record['error']}. See Modal call logs for instance {record['instance_id']}; automatic retries resume after five minutes, or run spindle deployment retry."
    return None


@app.function(image=image)
async def clear_deployment_failure(definition_id: str) -> None:
    module_for(definition_id)
    await shared_kv().delete(f"deployment_failure:{definition_id}")
    await kick_trainer_reconciler(definition_id)


def _plane():
    kv = shared_kv()
    task_stores = ModalSessionKeyValueStores()
    engines = ModalEnginePlatform(kv, _spawn_engine)

    async def spawn_sampling(task: SamplingTask) -> str:
        call = await execute_sample.spawn.aio(asdict(task))
        return call.object_id

    async def prepare_model(model) -> None:
        parameterization = parameterization_for(model.engine_definition_id)
        if parameterization is None:
            return
        if await _start_weights_download(model.engine_definition_id) is not None:
            return
        if parameterization == "full":
            await ensure_fft_pool.spawn.aio(_latest_pool(model).as_dict())
        elif await _trainer_booted(kv, model.engine_definition_id):
            # Otherwise the trainer deploys the pool once it boots.
            await ensure_lora_pool.spawn.aio(
                LoraPoolSpec(model.engine_definition_id).as_dict()
            )

    async def ensure_pool(session) -> None:
        definition_id = session.engine_definition_id
        parameterization = parameterization_for(definition_id)
        if parameterization is None:
            return
        if (
            session.model_id is None
            and await _start_weights_download(definition_id) is not None
        ):
            return
        if parameterization == "lora":
            await _ready_lora_pool(LoraPoolSpec(definition_id))
            return
        if session.model_id is None:
            pool = FFTPoolSpec.base(definition_id)
        elif session.publish_version is None:
            return
        else:
            pool = FFTPoolSpec(
                definition_id=definition_id,
                model_id=session.model_id,
                latest=session.latest,
                version=session.publish_version,
            )
        registry = fft_pool_kv()
        key = f"fft_pool:{pool.app_name}"
        record = await registry.get(key)
        if record is None:
            if pool.latest:
                pool = _latest_pool(await _model_record(kv, session.model_id))
            await ensure_fft_pool.remote.aio(pool.as_dict())
        else:
            await _touch_fft_pool(pool)

    async def kick_trainers(definition_id: str) -> bool:
        await kick_trainer_reconciler(definition_id)
        if not trainer_autoscaling(definition_id):
            return False
        maximum = module_for(definition_id).recipe.trainer_max_instances
        instances = [
            instance
            for instance in await engines.list_instances()
            if instance.definition_id == definition_id and not instance.terminal
        ]
        return any(
            instance.state in {"starting", "draining"} for instance in instances
        ) or len(instances) < int(maximum)

    return ControlPlane(
        kv,
        engines,
        sampling_tasks=ModalSamplingTaskPlatform(task_stores, spawn_sampling),
        session_idle_timeout=SESSION_IDLE_TIMEOUT,
        ensure_sampling_pool=ensure_pool,
        prepare_model=prepare_model,
        downloading_weights=_weights_download,
        creation_error=deployment_error,
        sampling_task_stores=task_stores,
        read_checkpoint_metadata=_read_checkpoint_metadata,
        list_checkpoints=_list_checkpoints,
        delete_checkpoint=_delete_checkpoint,
        checkpoint_root=CHECKPOINT_ROOT,
        reconcile_trainers=kick_trainers,
        trainer_autoscaling=trainer_autoscaling,
    )


@app.function(
    image=image,
    name="server",
    routing_region=ROUTING_REGION,
    timeout=20 * 60,
    volumes={CHECKPOINT_ROOT: checkpoint_volume},
    secrets=[modal.Secret.from_name(API_SECRET_NAME, required_keys=["TINKER_API_KEY"])],
)
@modal.concurrent(max_inputs=128, target_inputs=16)
@modal.asgi_app(requires_proxy_auth=False)
def server():
    return create_control_plane_app(
        _plane(),
        DEFINITIONS,
        api_key=os.environ["TINKER_API_KEY"],
        checkpoint_volume=CHECKPOINT_VOLUME_NAME,
    )


async def _lose_undefined_models() -> tuple[str, ...]:
    kv = shared_kv()
    lost = []
    for _, value in await kv.list_items("model:"):
        model = ModelRecord.model_validate(value)
        if parameterization_for(model.engine_definition_id) is not None:
            continue
        await kv.delete(placement_key(model.model_id))
        await kv.delete(trainer_demand_key(model.model_id))
        lost.append(model.model_id)
    return tuple(lost)


async def _cleanup_fft_pools() -> tuple[str, ...]:
    active_latest = {
        FFTPoolSpec(
            model.engine_definition_id,
            model.model_id,
            True,
            0,
        ).app_name
        for _, value in await shared_kv().list_items("model:")
        for model in (ModelRecord.model_validate(value),)
        if parameterization_for(model.engine_definition_id) == "full"
    }
    stopped = []
    registry = fft_pool_kv()
    for key, value in await registry.list_items("fft_pool:"):
        try:
            spec = FFTPoolSpec.from_dict(value)
            if spec.latest:
                if spec.app_name in active_latest:
                    continue
            else:
                if await _last_touched(registry, spec, value) > (
                    time.time() - FFT_POOL_IDLE_TIMEOUT
                ):
                    continue
                try:
                    replicas = await ModalFlashPool(
                        spec.app_name,
                        "Server",
                    ).discover_replicas_async()
                except modal.exception.NotFoundError:
                    await registry.delete(key)
                    await registry.delete(_touch_key(spec))
                    continue
                if replicas:
                    continue
                await registry.delete(_touch_key(spec))
            await asyncio.to_thread(stop_pool, spec)
            await registry.delete(key)
            stopped.append(spec.app_name)
            if not spec.latest and await registry.get(_touch_key(spec)) is not None:
                await ensure_fft_pool.spawn.aio(spec.as_dict())
        except Exception:
            # Retain failed entries for retry without starving unrelated pools.
            logging.getLogger(__name__).exception("Failed to clean FFT pool %s", key)
    return tuple(stopped)


async def _cleanup_lora_pools() -> tuple[str, ...]:
    registry = shared_kv()
    active = {
        LoraPoolSpec(model.engine_definition_id).app_name
        for _, value in await registry.list_items("model:")
        for model in (ModelRecord.model_validate(value),)
        if parameterization_for(model.engine_definition_id) == "lora"
    }
    stopped = []
    for key, value in await registry.list_items("lora_pool:"):
        try:
            spec = LoraPoolSpec.from_dict(value)
            if spec.app_name in active:
                continue
            if float(value.get("touched_at", 0.0)) > (
                time.time() - LORA_POOL_IDLE_TIMEOUT
            ):
                continue
            await asyncio.to_thread(stop_lora_pool, spec)
            await registry.delete(key)
            stopped.append(spec.app_name)
        except Exception:
            # One failed stop must not starve cleanup of other shared pools.
            logging.getLogger(__name__).exception("Failed to clean LoRA pool %s", key)
    return tuple(stopped)


@app.function(image=image, schedule=SWEEP_PERIOD)
def cleaner():
    async def run() -> None:
        plane = _plane()
        await plane.sweep_idle_sessions(SESSION_IDLE_TIMEOUT)
        await plane.sweep_idle_models(SESSION_IDLE_TIMEOUT)
        await plane.sweep_idle_engines()
        await _lose_undefined_models()
        await _cleanup_fft_pools()
        await _cleanup_lora_pools()

    asyncio.run(run())
