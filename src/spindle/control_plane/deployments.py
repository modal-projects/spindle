"""One registry of every recipe a frontend serves, however it was deployed."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from collections.abc import Callable
from typing import Any

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig, validate_frontend
from spindle.providers.contracts import KeyValueStore
from spindle.providers.local.kv import InMemoryKeyValueStore

REGISTRY_TTL_S = 5.0
DEPLOY_TIMEOUT_S = 60 * 60
FRONTEND_FIELDS = ("session_idle_timeout_s", "pool_idle_timeout_s", "sweep_interval_s")
DEFINITION_FIELDS = (
    "definition_id",
    "name",
    "model",
    "weights_repo",
    "parameterization",
    "max_context_length",
)


class DeploymentStore:
    """Rows keyed by recipe name. Deploys complete at once here; hosts with
    apps to deploy override `deploy` and `running`."""

    def __init__(self, kv: Callable[[], KeyValueStore]):
        # A factory, since a Modal frontend learns its app ID only at runtime.
        self.kv = kv

    async def list(self) -> list[dict[str, Any]]:
        return sorted(
            (row for _, row in await self.kv().list_items("")),
            key=lambda row: row.get("ready_at", row["added_at"]),
        )

    async def get(self, name: str) -> dict[str, Any] | None:
        return await self.kv().get(name)

    async def put(self, name: str, row: dict[str, Any]) -> None:
        await self.kv().put(name, row)

    async def deploy(
        self, row: dict[str, Any], *, refresh_trainer: bool, refresh_inference: bool
    ) -> None:
        pass

    async def running(self, row: dict[str, Any]) -> bool:
        return True


@dataclass(frozen=True)
class FrontendSettings:
    """What the frontend itself needs before any recipe is registered."""

    platform: dict[str, Any]
    session_idle_timeout_s: int
    pool_idle_timeout_s: int
    sweep_interval_s: int

    @classmethod
    def from_recipe(cls, recipe: BaseConfig) -> FrontendSettings:
        return cls(
            platform=recipe.platform,
            **{field: getattr(recipe, field) for field in FRONTEND_FIELDS},
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def recipe(self) -> BaseConfig:
        return BaseConfig(name="<frontend>", **asdict(self))


@dataclass(frozen=True)
class Definition:
    """The routing facts about a recipe; stands in for a DeploymentConfig."""

    definition_id: str
    name: str
    model: str
    weights_repo: str
    parameterization: str
    max_context_length: int


def decode_definition(config: dict[str, Any]):
    if "trainer_settings" in config:
        return DeploymentConfig.model_validate(config)
    recipe = config["recipe"]
    return Definition(**{field: recipe[field] for field in DEFINITION_FIELDS})


def _encode(definition) -> dict[str, Any]:
    if isinstance(definition, DeploymentConfig):
        return definition.model_dump(mode="json")
    recipe = {field: getattr(definition, field, None) for field in DEFINITION_FIELDS}
    recipe["name"] = recipe["name"] or definition.definition_id
    recipe["weights_repo"] = recipe["weights_repo"] or definition.model
    return {"recipe": recipe}


@dataclass
class _Entry:
    row: dict[str, Any]
    desired: Any
    ready: Any  # the definition requests route to, or None


class DeploymentRoutes:
    """Reads the registry through a short cache; writes go straight through."""

    def __init__(self, store: DeploymentStore, frontend: FrontendSettings):
        self.store = store
        self.frontend = frontend
        self._entries: list[_Entry] | None = None
        self._cached_at = 0.0

    @classmethod
    def from_definitions(cls, definitions, frontend=None) -> DeploymentRoutes:
        """Serve fixed definitions, as tests and local servers do."""
        definitions = tuple(definitions)
        if frontend is None:
            recipe = next(
                (d.recipe for d in definitions if isinstance(d, DeploymentConfig)),
                BaseConfig(),
            )
            frontend = FrontendSettings.from_recipe(recipe)
        rows = {}
        # Rows list by ready_at, so number them to keep the given order.
        for order, definition in enumerate(definitions):
            config = _encode(definition)
            if "trainer_settings" not in config:
                config["recipe"]["platform"] = frontend.platform
            rows[definition.definition_id] = {
                "config": config,
                "ready": config,
                "state": "ready",
                "attempt": "initial",
                "added_at": 0.0,
                "updated_at": 0.0,
                "ready_at": float(order),
                "origin": "deploy",
            }
        kv = InMemoryKeyValueStore(rows)
        return cls(DeploymentStore(lambda: kv), frontend)

    def _entry(self, row: dict[str, Any]) -> _Entry:
        return _Entry(
            row,
            decode_definition(row["config"]),
            decode_definition(row["ready"]) if row.get("ready") is not None else None,
        )

    def invalidate(self) -> None:
        self._entries = None

    async def _load(self, *, refresh: bool = False) -> list[_Entry]:
        if (
            refresh
            or self._entries is None
            or time.monotonic() - self._cached_at >= REGISTRY_TTL_S
        ):
            self._entries = [self._entry(row) for row in await self.store.list()]
            self._cached_at = time.monotonic()
        return self._entries

    async def rows(self) -> list[dict[str, Any]]:
        return [entry.row for entry in await self._load()]

    @staticmethod
    def _exact(entries: list[_Entry], definition_id: str):
        return next(
            (
                entry.ready
                for entry in entries
                if entry.ready is not None
                and entry.ready.definition_id == definition_id
            ),
            None,
        )

    def cached(self, definition_id: str):
        """The ready definition under this exact id, from the cache only."""
        return self._exact(self._entries or [], definition_id)

    async def get(self, definition_id: str):
        found = self._exact(await self._load(), definition_id)
        if found is None and await self.store.get(definition_id) is not None:
            found = self._exact(await self._load(refresh=True), definition_id)
        return found

    load = get

    async def select(self, model: str, mode: str | None = None):
        """The exact recipe named `model`, else the longest-context one for it."""
        exact = await self.get(model)
        if exact is not None:
            return exact if mode is None or exact.parameterization == mode else None
        matches = [
            entry.ready
            for entry in await self._load()
            if entry.ready is not None
            and entry.ready.model == model
            and (mode is None or entry.ready.parameterization == mode)
        ]
        return max(matches, key=lambda d: d.max_context_length, default=None)

    async def definitions(self):
        return [e.ready for e in await self._load() if e.ready is not None]

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

    async def put(
        self,
        deployment: DeploymentConfig,
        *,
        origin: str,
        refresh_trainer: bool = False,
        refresh_inference: bool = False,
    ) -> dict[str, Any]:
        """Register a recipe and start deploying it if anything changed."""
        name = deployment.name
        others = [
            entry.ready.recipe
            for entry in await self._load(refresh=True)
            if entry.ready is not None and entry.ready.name != name
        ]
        validate_frontend([self.frontend.recipe(), *others, deployment.recipe])
        desired = deployment.model_dump(mode="json")
        now = time.time()
        current = await self.store.get(name)
        if (
            current is not None
            and current["config"] == desired
            and not (refresh_trainer or refresh_inference)
        ):
            normalized = current
            if normalized["state"] == "ready" or (
                normalized["state"] == "deploying" and await self.store.running(current)
            ):
                return normalized
        row = {
            key: value
            for key, value in (current or {}).items()
            if key not in {"error", "call_id"}
        }
        row.update(
            config=desired,
            state="deploying",
            attempt=uuid.uuid4().hex,
            added_at=row.get("added_at", now),
            updated_at=now,
            origin=origin,
        )
        await self.store.put(name, row)
        await self._start(row, refresh_trainer, refresh_inference)
        return row

    async def _start(self, row, refresh_trainer, refresh_inference):
        self.invalidate()
        await self.store.deploy(
            row, refresh_trainer=refresh_trainer, refresh_inference=refresh_inference
        )
