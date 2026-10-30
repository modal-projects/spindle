"""Redirect unmodified Tinker SDK code to a Spindle server on Modal.

    import spindle

    @spindle.on_modal
    def main():
        service = tinker.ServiceClient()  # now talks to Spindle
        ...

`on_modal` also works as `with spindle.on_modal(...):`, and `patch_tinker(...)`
applies the same redirect for the rest of the process.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

TINKER_HOSTED_DOMAINS = ("thinkingmachines.dev", "thinkingmachines.ai")
SERVER_FUNCTION = "server"
_ENV_KEYS = ("TINKER_BASE_URL", "TINKER_API_KEY")
_lock = threading.RLock()


def is_tinker_hosted(url: object) -> bool:
    host = urlparse(str(url)).hostname or ""
    return any(
        host == domain or host.endswith("." + domain)
        for domain in TINKER_HOSTED_DOMAINS
    )


def lookup_server_url(app_name: str = "spindle", environment: str | None = None) -> str:
    """Return the web URL of a deployed Spindle frontend."""
    import modal

    function = modal.Function.from_name(
        app_name, SERVER_FUNCTION, environment_name=environment
    )
    url = function.get_web_url()
    if not url:
        raise RuntimeError(f"{app_name}.{SERVER_FUNCTION} has no web URL")
    return url


def lookup_api_key(
    secret_name: str = "spindle-api", environment: str | None = None
) -> str:
    """Read TINKER_API_KEY from the deployment's Modal secret.

    Runs a short CPU function in an ephemeral app because secret values are
    only readable from inside Modal containers.
    """
    import modal

    app = modal.App("spindle-api-key-lookup")

    @app.function(
        image=modal.Image.debian_slim(),
        secrets=[
            modal.Secret.from_name(
                secret_name,
                environment_name=environment,
                required_keys=["TINKER_API_KEY"],
            )
        ],
        serialized=True,
    )
    def read_key() -> str:
        import os

        return os.environ["TINKER_API_KEY"]

    with app.run(environment_name=environment):
        return read_key.remote()


@dataclass(frozen=True)
class Target:
    base_url: str
    api_key: str


def resolve_target(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    app_name: str = "spindle",
    environment: str | None = None,
    api_secret: str = "spindle-api",
) -> Target:
    """Resolve the Spindle URL and key from arguments, env, or the Modal deployment.

    TINKER_BASE_URL / TINKER_API_KEY are only reused when the URL is not hosted
    Tinker, so an existing Thinking Machines key is never sent to Spindle.
    """
    env_url = os.environ.get("TINKER_BASE_URL") or None
    env_points_at_spindle = env_url is not None and not is_tinker_hosted(env_url)
    base_url = (
        base_url
        or os.environ.get("SPINDLE_BASE_URL")
        or (env_url if env_points_at_spindle else None)
        or lookup_server_url(app_name, environment)
    )
    api_key = (
        api_key
        or os.environ.get("SPINDLE_API_KEY")
        or (os.environ.get("TINKER_API_KEY") if env_points_at_spindle else None)
        or lookup_api_key(api_secret, environment)
    )
    return Target(base_url=base_url.rstrip("/"), api_key=api_key)


@dataclass
class _Patch:
    target: Target
    model_map: Mapping[str, str]
    check_models: bool
    _supported: frozenset[str] | None = None
    _undo: list[Callable[[], None]] = field(default_factory=list)

    def supported_models(self) -> frozenset[str]:
        if self._supported is None:
            import httpx

            response = httpx.get(
                f"{self.target.base_url}/api/v1/get_server_capabilities",
                headers={"X-API-Key": self.target.api_key},
                timeout=30,
            )
            response.raise_for_status()
            self._supported = frozenset(
                row["model_name"] for row in response.json()["supported_models"]
            )
        return self._supported

    def map_model(self, base_model: str | None) -> str | None:
        if base_model is None:
            return None
        mapped = self.model_map.get(base_model, base_model)
        if self.check_models and mapped not in self.supported_models():
            hint = "" if mapped != base_model else " Pass model_map={...} to remap it."
            raise ValueError(
                f"Spindle at {self.target.base_url} does not serve {mapped!r}; "
                f"supported: {sorted(self.supported_models())}.{hint}"
            )
        return mapped


_active: list[_Patch] = []


def _current() -> _Patch | None:
    return _active[-1] if _active else None


def _install_wrappers() -> Callable[[], None]:
    from tinker import ServiceClient

    originals: dict[str, Any] = {}

    original_init = ServiceClient.__init__

    @functools.wraps(original_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        patch = _current()
        if patch is not None:
            url = kwargs.get("base_url")
            if url is None or is_tinker_hosted(url):
                kwargs["base_url"] = patch.target.base_url
                kwargs["api_key"] = patch.target.api_key
        original_init(self, *args, **kwargs)

    originals["__init__"] = original_init
    ServiceClient.__init__ = __init__

    for name, method in list(vars(ServiceClient).items()):
        if name.startswith("_") or not callable(method):
            continue
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            continue
        if "base_model" not in signature.parameters:
            continue
        originals[name] = method
        setattr(ServiceClient, name, _wrap_base_model(method, signature))

    def undo() -> None:
        for name, value in originals.items():
            setattr(ServiceClient, name, value)

    return undo


def _wrap_base_model(method: Callable, signature: inspect.Signature) -> Callable:
    def remap(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        patch = _current()
        if patch is None:
            return args, kwargs
        bound = signature.bind(*args, **kwargs)
        if "base_model" in bound.arguments:
            bound.arguments["base_model"] = patch.map_model(
                bound.arguments["base_model"]
            )
        return bound.args, bound.kwargs

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = remap(args, kwargs)
            return await method(*args, **kwargs)

        return async_wrapper

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        args, kwargs = remap(args, kwargs)
        return method(*args, **kwargs)

    return wrapper


_wrappers_undo: Callable[[], None] | None = None


def _push(patch: _Patch) -> Callable[[], None]:
    global _wrappers_undo
    with _lock:
        if _wrappers_undo is None:
            _wrappers_undo = _install_wrappers()
        previous_env = {key: os.environ.get(key) for key in _ENV_KEYS}
        # Env covers clients built outside ServiceClient and worker subprocesses.
        os.environ["TINKER_BASE_URL"] = patch.target.base_url
        os.environ["TINKER_API_KEY"] = patch.target.api_key
        _active.append(patch)

    def pop() -> None:
        global _wrappers_undo
        with _lock:
            _active.remove(patch)
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if not _active and _wrappers_undo is not None:
                _wrappers_undo()
                _wrappers_undo = None

    return pop


def patch_tinker(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    model_map: Mapping[str, str] | None = None,
    check_models: bool = True,
    app_name: str = "spindle",
    environment: str | None = None,
    api_secret: str = "spindle-api",
) -> Callable[[], None]:
    """Redirect Tinker clients to Spindle until the returned callable is invoked."""
    target = resolve_target(
        base_url,
        api_key,
        app_name=app_name,
        environment=environment,
        api_secret=api_secret,
    )
    return _push(_Patch(target, dict(model_map or {}), check_models))


class on_modal:
    """Run Tinker code against Spindle; usable as a decorator or context manager.

    With `engine=`, a scoped single-tenant Spindle server is brought up for the
    duration via `spindle.run`; otherwise a deployed server is used.
    """

    def __init__(
        self,
        func: Callable | None = None,
        /,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_map: Mapping[str, str] | None = None,
        check_models: bool = True,
        app_name: str = "spindle",
        environment: str | None = None,
        api_secret: str = "spindle-api",
        engine: Any = None,
        **run_kwargs: Any,
    ) -> None:
        self._options = dict(
            base_url=base_url,
            api_key=api_key,
            model_map=model_map,
            check_models=check_models,
            app_name=app_name,
            environment=environment,
            api_secret=api_secret,
        )
        self._engine = engine
        self._run_kwargs = run_kwargs
        self._stack: ExitStack | None = None
        self._func = func
        if func is not None:
            functools.update_wrapper(self, func)

    @contextmanager
    def _session(self) -> Iterator[Target]:
        options = dict(self._options)
        with ExitStack() as stack:
            if self._engine is not None:
                from spindle.run import run

                url, key = stack.enter_context(
                    run(engine=self._engine, **self._run_kwargs)
                )
                options["base_url"], options["api_key"] = url, key
            undo = patch_tinker(**options)
            stack.callback(undo)
            yield _current().target

    def __enter__(self) -> Target:
        self._stack = ExitStack()
        return self._stack.enter_context(self._session())

    def __exit__(self, *exc: object) -> None:
        stack, self._stack = self._stack, None
        stack.__exit__(*exc)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._func is None:
            (func,) = args
            return on_modal(
                func, engine=self._engine, **self._options, **self._run_kwargs
            )
        func = self._func
        if inspect.iscoroutinefunction(func):

            async def run_async() -> Any:
                with self._session():
                    return await func(*args, **kwargs)

            return run_async()
        with self._session():
            return func(*args, **kwargs)
