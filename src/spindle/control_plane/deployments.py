"""The registry of recipes a frontend serves, and how requests pick one."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig
from spindle.providers.contracts import KeyValueStore

logger = logging.getLogger(__name__)

REGISTRY_VERSION_KEY = "registry_version:deployments"
REGISTRY_CLAIM_PREFIX = "registry_claim:"
# A claim this old was left by a container that died mid-add.
CLAIM_TIMEOUT_S = 60.0
# How long a container serves its list between registry version checks.
VERSION_CHECK_S = 1.0
MAX_APP_NAME_LENGTH = 63
FRONTEND_FIELDS = ("session_idle_timeout_s", "pool_idle_timeout_s", "sweep_interval_s")


class DeploymentConflict(ValueError):
    """A recipe name is already registered with different settings."""


class RegistryUnavailable(RuntimeError):
    """The registry could not be read, so a lookup can't tell whether a recipe exists."""


class DeploymentNotFound(KeyError):
    """No recipe is registered under that name."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class DeploymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    base: str | None = None
    recipe: dict[str, Any] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> DeploymentRequest:
        if (self.base is None) == (self.recipe is None):
            raise ValueError("pass exactly one of base or recipe")
        if any(key.split(".")[0] == "platform" for key in self.overrides):
            raise ValueError("platform settings come from the frontend")
        return self


class Platform(TypedDict):
    frontend: str
    modal: dict[str, str | None]
    secrets: dict[str, str]
    storage: dict[str, str]


@dataclass(frozen=True)
class FrontendSettings:
    platform: Platform
    session_idle_timeout_s: int
    pool_idle_timeout_s: int
    sweep_interval_s: int

    @classmethod
    def from_recipe(cls, recipe: BaseConfig) -> FrontendSettings:
        return cls(
            platform=recipe.platform,
            **{field: getattr(recipe, field) for field in FRONTEND_FIELDS},
        )


class Routable(Protocol):
    definition_id: str
    name: str
    model: str
    weights_repo: str
    parameterization: str
    max_context_length: int


async def mark_registry_changed(versions: KeyValueStore) -> str:
    """Record that the registry changed, so every frontend container rereads it."""
    version = uuid.uuid4().hex
    await versions.put(REGISTRY_VERSION_KEY, version)
    return version


class DeploymentRoutes:
    def __init__(
        self,
        kv: Callable[[], KeyValueStore] | None,
        versions: Callable[[], KeyValueStore] | None = None,
    ):
        # Factories, since a Modal frontend learns its app ID only at runtime.
        self.kv = kv
        self.versions = versions
        self._version_store: KeyValueStore | None = None
        self._definitions: list[Routable] = []
        self._version: object = None
        self._loaded = False
        self._checked_at = float("-inf")
        self._skipped: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}

    @classmethod
    def from_definitions(cls, definitions: Iterable[Routable]) -> DeploymentRoutes:
        routes = cls(None)
        routes._definitions = list(definitions)
        return routes

    def _versions(self) -> KeyValueStore | None:
        # Built once: every new Modal Dict handle costs a lookup round trip.
        if self._version_store is None and self.versions is not None:
            self._version_store = self.versions()
        return self._version_store

    def _shared(self, name: str, call: Callable[[], Any]) -> asyncio.Task:
        # One in-flight call per name and container; tests run several event loops.
        loop = asyncio.get_running_loop()
        task = self._tasks.get(name)
        if task is None or task.done() or task.get_loop() is not loop:
            task = self._tasks[name] = loop.create_task(call())
        return task

    async def _registry_version(self) -> object:
        versions = self._versions()
        return None if versions is None else await versions.get(REGISTRY_VERSION_KEY)

    async def _load(self, *, check: bool = False) -> list[Routable]:
        """The recipes; the registry version is checked now with `check`, else at most every VERSION_CHECK_S."""
        if self.kv is None:
            return self._definitions
        if self._loaded and not check:
            if time.monotonic() - self._checked_at < VERSION_CHECK_S:
                return self._definitions
        try:
            if check:
                version = await self._registry_version()
            else:
                version = await asyncio.shield(
                    self._shared("version", self._registry_version)
                )
            self._checked_at = time.monotonic()
            if not self._loaded or version != self._version:
                await self._reload(version)
        except Exception as exc:
            self._checked_at = time.monotonic()
            if check or not self._loaded:
                # A miss must not answer "not found" from a list it couldn't refresh.
                raise RegistryUnavailable(f"registry read failed: {exc}") from exc
            logger.warning(
                "Registry read failed (%s); serving the last recipe list", exc
            )
        return self._definitions

    async def _reload(self, version: object = None) -> None:
        await asyncio.shield(self._shared("read", self._read_registry))
        if self._loaded and self._version != version:
            # The read we joined started before `version` was written.
            await asyncio.shield(self._shared("read", self._read_registry))

    async def _read_registry(self) -> None:
        # Version first, so a change that lands mid-read is caught by the next check.
        version = await self._registry_version()
        kv = self.kv()
        names = await kv.list_keys("")
        rows = await asyncio.gather(
            *(kv.get(name) for name in names), return_exceptions=True
        )
        definitions, skipped = [], set()
        for name, row in zip(names, rows):
            try:
                if isinstance(row, BaseException):
                    raise row
                definitions.append(DeploymentConfig.from_registry(row))
            except (ValueError, TypeError) as error:
                if name not in self._skipped:
                    logger.error("Skipping unreadable recipe %r: %s", name, error)
                skipped.add(name)
        self._definitions, self._skipped = definitions, skipped
        self._version, self._loaded = version, True

    @staticmethod
    def _exact(definitions: Iterable[Routable], definition_id: str) -> Routable | None:
        return next((d for d in definitions if d.definition_id == definition_id), None)

    def cached(self, definition_id: str) -> Routable | None:
        """The recipe under this exact id from the last registry read, without I/O."""
        return self._exact(self._definitions, definition_id)

    async def get(self, definition_id: str) -> Routable | None:
        found = self._exact(await self._load(), definition_id)
        if found is None and self.kv is not None:
            found = self._exact(await self._load(check=True), definition_id)
        return found

    async def select(self, model: str, mode: str | None = None) -> Routable | None:
        """The exact recipe named `model`, else the longest-context one for it."""
        found = self._select(await self._load(), model, mode)
        if found is None and self.kv is not None:
            found = self._select(await self._load(check=True), model, mode)
        return found

    @staticmethod
    def _select(
        definitions: list[Routable], model: str, mode: str | None
    ) -> Routable | None:
        exact = DeploymentRoutes._exact(definitions, model)
        if exact is not None:
            return exact if mode is None or exact.parameterization == mode else None
        matches = [
            d
            for d in definitions
            if d.model == model and (mode is None or d.parameterization == mode)
        ]
        return max(matches, key=lambda d: d.max_context_length, default=None)

    async def definitions(self) -> list[Routable]:
        return list(await self._load(check=True))

    async def build(self, request: DeploymentRequest) -> DeploymentConfig:
        """Compile a request into a config under this frontend's settings."""
        recipe = request.recipe
        if recipe is None:
            parent = await self.get(request.base)
            if not isinstance(parent, DeploymentConfig):
                raise ValueError("unknown base deployment")
            recipe = vars(parent.recipe)
        registered = [d for d in await self._load() if isinstance(d, DeploymentConfig)]
        if not registered:
            raise ValueError("this frontend has no registered configs")
        frontend = FrontendSettings.from_recipe(registered[0].recipe)
        settings = {**asdict(frontend), **recipe, "platform": frontend.platform}
        try:
            deployment = DeploymentConfig.create(
                BaseConfig(
                    **settings, overrides={**request.overrides, "name": request.name}
                )
            )
        except (ArithmeticError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid config: {exc}") from exc
        for app_name in (deployment.trainer_app_name, deployment.inference_app_name):
            if len(app_name) > MAX_APP_NAME_LENGTH:
                raise ValueError(
                    f"name too long: app {app_name!r} would exceed "
                    f"{MAX_APP_NAME_LENGTH} characters"
                )
        if FrontendSettings.from_recipe(deployment.recipe) != frontend:
            raise ValueError(
                "deployments on one frontend must share lifecycle settings"
            )
        return deployment

    async def add(self, deployment: DeploymentConfig) -> dict[str, Any]:
        """Register a config; the same config again is a no-op."""
        store, config = self._store(), deployment.model_dump(mode="json")
        current = await store.get(deployment.name)
        versions = self._versions()
        if current is None and versions is None:
            await store.put(deployment.name, config)
            current = config
        elif current is None:
            # The Volume has no atomic create, so a first add holds a short claim.
            claim_key = REGISTRY_CLAIM_PREFIX + deployment.name
            claim = await versions.put_if_absent(
                claim_key, {"recipe": config["recipe"], "at": time.time()}
            )
            if not claim.created and time.time() - claim.value["at"] < CLAIM_TIMEOUT_S:
                current = claim.value
            else:
                try:
                    current = await store.get(deployment.name)
                    if current is None:
                        await store.put(deployment.name, config)
                        current = config
                finally:
                    await versions.delete(claim_key)
        # Also on a retry, in case the version update of the first attempt failed.
        await self._changed()
        # The recipe alone, so a frontend upgrade's spindle_version is no conflict.
        if not isinstance(current, dict) or current.get("recipe") != config["recipe"]:
            raise DeploymentConflict(
                f"deployment {deployment.name!r} already exists with different settings"
            )
        return config

    async def remove(self, name: str) -> DeploymentConfig:
        """Unregister a config; its apps scale down on their own."""
        store = self._store()
        row = await store.get(name)
        if row is None:
            await self._changed()  # a retry: the first attempt's update may be lost
            raise DeploymentNotFound(f"no deployment named {name!r}")
        await store.delete(name)
        await self._changed()
        return DeploymentConfig.from_registry(row)

    async def _changed(self) -> None:
        versions = self._versions()
        version = None if versions is None else await mark_registry_changed(versions)
        try:
            await self._reload(version)
        except Exception:
            # The write and version update landed; other containers will see them.
            logger.exception("Registry reread after a write failed")
        self._checked_at = time.monotonic()

    def _store(self) -> KeyValueStore:
        if self.kv is None:
            raise ValueError("this frontend serves a fixed set of recipes")
        return self.kv()

    async def capabilities(self) -> list[dict[str, Any]]:
        definitions = await self.definitions()
        rows = []
        for model in dict.fromkeys(d.model for d in definitions):
            recipes = [d for d in definitions if d.model == model]
            context = min(
                max(
                    candidate.max_context_length
                    for candidate in recipes
                    if candidate.parameterization == d.parameterization
                )
                for d in recipes
            )
            rows.append({"model_name": model, "max_context_length": context})
            if len(recipes) > 1:
                rows += [
                    {
                        "model_name": d.definition_id,
                        "max_context_length": d.max_context_length,
                    }
                    for d in recipes
                ]
        return rows
