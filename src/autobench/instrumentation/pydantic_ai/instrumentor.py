from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from importlib.util import find_spec
from inspect import getattr_static, signature
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from autobench._version import __version__
from autobench.instrumentation.config import AssetDiscoverySettings
from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import (
    Compatibility,
    CompatibilityStatus,
    InstrumentationHandle,
    InstrumentorCapabilities,
    InstrumentorInfo,
)
from autobench.instrumentation.patching import InstrumentationConflictError
from autobench.protocol.context import get_context
from autobench.protocol.signals import AbstractionLayer, CaptureMechanism
from autobench.tracking import TrackingRegistry, track

if TYPE_CHECKING:
    from autobench.instrumentation.pydantic_ai.capability import AutobenchCapability


@dataclass(slots=True)
class _EntryPointState:
    original: Callable[..., Any]
    installed: Callable[..., Any]


_ENTRY_POINTS: WeakKeyDictionary[type[Any], _EntryPointState] = WeakKeyDictionary()


class PydanticAI:
    """Install automatic ABP capture for Pydantic AI agent runs."""

    def __init__(
        self,
        *,
        assets: Sequence[Any] = (),
        discovery: AssetDiscoverySettings | None = None,
        registry: TrackingRegistry = track,
    ) -> None:
        self._assets = tuple(assets)
        self._discovery = discovery or AssetDiscoverySettings()
        self._registry = registry
        self._info = InstrumentorInfo(
            id="autobench.pydantic_ai",
            version=__version__,
            target_distribution="pydantic-ai-slim",
            supported_versions=">=2.22,<2.23",
            mechanism=CaptureMechanism.HOOK,
            layer=AbstractionLayer.FRAMEWORK,
            span_kinds=("agent", "llm", "tool", "validation"),
            semantic_families=("agent", "llm", "tool", "message", "validation"),
            source_convention="pydantic-ai",
            source_convention_version="2.22",
            capabilities=InstrumentorCapabilities(
                sync=True,
                async_=True,
                streaming=True,
                native_hooks=True,
                asset_discovery=True,
                asset_kinds=(
                    "agent",
                    "capability",
                    "output_schema",
                    "policy",
                    "prompt",
                    "tool",
                    "toolset",
                ),
            ),
        )

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        if find_spec("pydantic_ai") is None:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=(
                    "Pydantic AI is unavailable; install Autobench with the 'pydantic-ai' extra",
                ),
            )
        try:
            from pydantic_ai import Agent
            from pydantic_ai.agent.wrapper import WrapperAgent
        except ImportError as error:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=(f"Pydantic AI could not be imported: {error}",),
            )
        missing = tuple(
            target.__qualname__
            for target in (Agent, WrapperAgent)
            if "capabilities" not in signature(target.iter).parameters
        )
        if missing:
            return Compatibility(
                status=CompatibilityStatus.UNSUPPORTED,
                degraded_features=missing,
                diagnostics=("Pydantic AI agent iteration lacks public capability injection",),
            )
        return Compatibility.compatible()

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        from importlib.metadata import version

        from pydantic_ai import Agent
        from pydantic_ai.agent.wrapper import WrapperAgent

        from autobench.instrumentation.pydantic_ai.capability import AutobenchCapability

        capability = AutobenchCapability(
            runtime,
            self.info,
            target_version=version("pydantic-ai-slim"),
            assets=self._assets,
            registry=self._registry,
            discovery=self._discovery,
        )
        installed: list[type[Any]] = []
        try:
            for target in (Agent, WrapperAgent):
                _install_entry_point(target, capability, self.info.id)
                installed.append(target)
        except BaseException:
            for target in reversed(installed):
                _restore_entry_point(target, runtime, self.info)
            raise

        def close() -> None:
            for target in reversed(installed):
                _restore_entry_point(target, runtime, self.info)

        return InstrumentationHandle(close, info=self.info)


def _install_entry_point(
    target: type[Any],
    capability: AutobenchCapability,
    instrumentor_id: str,
) -> None:
    if target in _ENTRY_POINTS:
        raise InstrumentationConflictError(
            f"{target.__qualname__}.iter is already instrumented by Autobench"
        )
    descriptor: Callable[..., Any] = getattr_static(target, "iter")

    @wraps(descriptor)
    def iter_with_autobench(*args: Any, **kwargs: Any) -> Any:
        active = get_context()
        if active is None or active.is_suppressed(instrumentor_id):
            return descriptor(*args, **kwargs)
        configured = kwargs.get("capabilities")
        capabilities = () if configured is None else tuple(configured)
        from autobench.instrumentation.pydantic_ai.capability import AutobenchCapability

        if not any(isinstance(item, AutobenchCapability) for item in capabilities):
            kwargs["capabilities"] = (
                *capabilities,
                capability.for_entrypoint(kwargs),
            )
        return descriptor(*args, **kwargs)

    target.iter = iter_with_autobench
    _ENTRY_POINTS[target] = _EntryPointState(
        original=descriptor,
        installed=iter_with_autobench,
    )


def _restore_entry_point(
    target: type[Any],
    runtime: InstrumentationRuntime,
    info: InstrumentorInfo,
) -> None:
    state = _ENTRY_POINTS.pop(target)
    if getattr_static(target, "iter") is not state.installed:
        runtime.diagnose(
            info,
            "pydantic_ai_patch_conflict",
            f"{target.__qualname__}.iter changed after Autobench installed its capability injector",
        )
        return
    target.iter = state.original


__all__ = ("PydanticAI",)
