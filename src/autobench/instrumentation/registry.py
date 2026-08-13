from __future__ import annotations as _annotations

from collections.abc import Collection, Sequence
from importlib.util import find_spec

from pydantic import BaseModel, ConfigDict

from autobench._version import __version__
from autobench.instrumentation.config import (
    AutoInstrumentation,
    BuiltinInstrumentationConfig,
    HTTPXInstrumentation,
    InstrumentationConfig,
    InstrumentorName,
    OpenAIAgentsInstrumentation,
    OpenAIInstrumentation,
    PydanticAIInstrumentation,
    PydanticGEPAInstrumentation,
)
from autobench.instrumentation.manager import InstrumentationManager
from autobench.instrumentation.models import (
    Compatibility,
    CompatibilityStatus,
    InstrumentationError,
    Instrumentor,
    InstrumentorCapabilities,
    InstrumentorInfo,
)
from autobench.protocol.signals import AbstractionLayer, CaptureMechanism


class InstrumentorStatus(BaseModel):
    """Dependency and capability report for one built-in instrumentor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: InstrumentorName
    extra: str
    info: InstrumentorInfo
    compatibility: Compatibility
    capture_mode: str


_CONFIGS: tuple[BuiltinInstrumentationConfig, ...] = (
    PydanticAIInstrumentation(),
    PydanticGEPAInstrumentation(),
    OpenAIInstrumentation(),
    OpenAIAgentsInstrumentation(),
    HTTPXInstrumentation(),
)

_EXTRAS: dict[InstrumentorName, str] = {
    "pydantic_ai": "pydantic-ai",
    "pydantic_gepa": "pydantic-gepa",
    "openai": "openai",
    "openai_agents": "openai-agents",
    "httpx": "httpx",
}

_INFO: dict[InstrumentorName, InstrumentorInfo] = {
    "pydantic_ai": InstrumentorInfo(
        id="autobench.pydantic_ai",
        version=__version__,
        target_distribution="pydantic-ai-slim",
        supported_versions=">=2.22,<2.24",
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
    ),
    "pydantic_gepa": InstrumentorInfo(
        id="autobench.pydantic_gepa",
        version=__version__,
        target_distribution="pydantic-gepa",
        supported_versions=">=0.1.0a0,<0.2",
        mechanism=CaptureMechanism.CALLBACK,
        layer=AbstractionLayer.FRAMEWORK,
        span_kinds=(
            "optimization",
            "workflow",
            "candidate",
            "evaluation",
            "reflection",
            "scorer",
        ),
        semantic_families=(
            "optimization",
            "evaluation",
            "candidate",
            "asset",
            "checkpoint",
        ),
        source_convention="pydantic-gepa",
        source_convention_version="1",
        capabilities=InstrumentorCapabilities(
            sync=True,
            async_=True,
            native_hooks=True,
            asset_discovery=True,
            asset_kinds=(
                "prompt",
                "tool",
                "input_schema",
                "output_schema",
                "field_description",
                "schema_description",
                "optimization_component",
            ),
        ),
    ),
    "openai": InstrumentorInfo(
        id="autobench.openai",
        version=__version__,
        target_distribution="openai",
        supported_versions=">=2.52,<2.54",
        mechanism=CaptureMechanism.PATCH,
        layer=AbstractionLayer.CLIENT,
        span_kinds=("llm",),
        semantic_families=("llm", "message", "tool", "stream"),
        source_convention="openai-python",
        source_convention_version="2.52",
        capabilities=InstrumentorCapabilities(
            sync=True,
            async_=True,
            streaming=True,
            asset_discovery=True,
            asset_kinds=("output_schema", "prompt", "tool"),
        ),
    ),
    "openai_agents": InstrumentorInfo(
        id="autobench.openai_agents",
        version=__version__,
        target_distribution="openai-agents",
        supported_versions=">=0.19.2,<0.20",
        mechanism=CaptureMechanism.HOOK,
        layer=AbstractionLayer.FRAMEWORK,
        span_kinds=("workflow", "agent", "tool", "handoff", "validation", "generation"),
        semantic_families=("workflow", "agent", "tool", "llm", "result"),
        source_convention="openai-agents",
        source_convention_version="0.19",
        capabilities=InstrumentorCapabilities(
            sync=True,
            async_=True,
            streaming=True,
            native_hooks=True,
            asset_discovery=True,
            asset_kinds=(
                "agent",
                "guardrail",
                "handoff",
                "output_schema",
                "policy",
                "prompt",
                "tool",
                "toolset",
            ),
        ),
    ),
    "httpx": InstrumentorInfo(
        id="autobench.httpx",
        version=__version__,
        target_distribution="httpx",
        supported_versions=">=0.28,<0.29",
        mechanism=CaptureMechanism.PATCH,
        layer=AbstractionLayer.TRANSPORT,
        span_kinds=("http",),
        semantic_families=("http", "network", "stream", "error"),
        source_convention="httpx",
        source_convention_version="0.28",
        capabilities=InstrumentorCapabilities(sync=True, async_=True, streaming=True),
    ),
}


def resolve_instrumentor(config: BuiltinInstrumentationConfig) -> Instrumentor:
    """Build one configured instrumentor, importing only its installed integration."""

    if find_spec(_module_name(config.kind)) is None:
        raise InstrumentationError(
            f"instrumentation '{config.kind}' is unavailable; "
            f"install autobench[{_EXTRAS[config.kind]}]"
        )
    try:
        if isinstance(config, PydanticAIInstrumentation):
            from autobench.instrumentation.pydantic_ai import PydanticAI

            return PydanticAI(discovery=config.assets)
        if isinstance(config, PydanticGEPAInstrumentation):
            from autobench.instrumentation.pydantic_gepa import PydanticGEPA

            return PydanticGEPA(detail=config.detail, discovery=config.assets)
        if isinstance(config, OpenAIInstrumentation):
            from autobench.instrumentation.openai import OpenAIClient

            return OpenAIClient(discovery=config.assets)
        if isinstance(config, OpenAIAgentsInstrumentation):
            from autobench.instrumentation.openai_agents import OpenAIAgents

            return OpenAIAgents(discovery=config.assets)

        from autobench.instrumentation.httpx import HTTPX, HTTPXCapture

        return HTTPX(capture=HTTPXCapture.model_validate(config.capture.model_dump()))
    except ImportError as error:
        raise InstrumentationError(
            f"instrumentation '{config.kind}' could not be imported: {error}"
        ) from error


def resolve_instrumentors(
    configs: Sequence[InstrumentationConfig],
    *,
    reserved_ids: Collection[str] = (),
) -> tuple[tuple[Instrumentor, ...], tuple[InstrumentorStatus, ...]]:
    """Resolve explicit and automatically discovered built-in instrumentors."""

    explicit_names = {
        config.kind for config in configs if not isinstance(config, AutoInstrumentation)
    }
    selected: list[Instrumentor] = []
    skipped: list[InstrumentorStatus] = []
    manager = InstrumentationManager()
    try:
        for auto in configs:
            if not isinstance(auto, AutoInstrumentation) or not auto.enabled:
                continue
            for config in _CONFIGS:
                if (
                    config.kind in auto.exclude
                    or config.kind in explicit_names
                    or _INFO[config.kind].id in reserved_ids
                ):
                    continue
                selected_config = config
                if auto.assets is not None and isinstance(
                    config,
                    (
                        PydanticAIInstrumentation,
                        PydanticGEPAInstrumentation,
                        OpenAIInstrumentation,
                        OpenAIAgentsInstrumentation,
                    ),
                ):
                    selected_config = config.model_copy(update={"assets": auto.assets})
                instrumentor, status = _inspect_instrumentor(
                    selected_config,
                    manager=manager,
                )
                if status.compatibility.installable and instrumentor is not None:
                    selected.append(instrumentor)
                    continue
                if auto.strict:
                    detail = (
                        "; ".join(status.compatibility.conflicts + status.compatibility.diagnostics)
                        or status.compatibility.status.value
                    )
                    raise InstrumentationError(
                        f"automatic instrumentation '{status.name}' is not installable: {detail}"
                    )
                skipped.append(status)
    finally:
        manager.close()

    selected.extend(
        resolve_instrumentor(config)
        for config in configs
        if not isinstance(config, AutoInstrumentation) and config.enabled
    )
    return tuple(selected), tuple(skipped)


def instrumentor_statuses() -> tuple[InstrumentorStatus, ...]:
    """Inspect every built-in integration without importing unavailable SDKs."""

    manager = InstrumentationManager()
    try:
        return tuple(_inspect_instrumentor(config, manager=manager)[1] for config in _CONFIGS)
    finally:
        manager.close()


def _inspect_instrumentor(
    config: BuiltinInstrumentationConfig,
    *,
    manager: InstrumentationManager,
) -> tuple[Instrumentor | None, InstrumentorStatus]:
    info = _INFO[config.kind]
    distribution = info.target_distribution
    instrumentor: Instrumentor | None = None
    if distribution is None or find_spec(_module_name(config.kind)) is None:
        compatibility = Compatibility(
            status=CompatibilityStatus.UNAVAILABLE,
            diagnostics=(
                f"distribution '{distribution}' is not installed; "
                f"install autobench[{_EXTRAS[config.kind]}]",
            ),
        )
    else:
        try:
            instrumentor = resolve_instrumentor(config)
        except InstrumentationError as error:
            compatibility = Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=(f"integration import failed: {error}",),
            )
        else:
            info = instrumentor.info
            compatibility = manager.check(instrumentor)
    return instrumentor, InstrumentorStatus(
        name=config.kind,
        extra=_EXTRAS[config.kind],
        info=info,
        compatibility=compatibility,
        capture_mode=_capture_mode(config),
    )


def _module_name(name: InstrumentorName) -> str:
    if name == "pydantic_ai":
        return "pydantic_ai"
    if name == "openai_agents":
        return "agents"
    return name


def _capture_mode(config: BuiltinInstrumentationConfig) -> str:
    if isinstance(config, HTTPXInstrumentation):
        capture = config.capture
        bodies = "bodies" if capture.request_body or capture.response_body else "no bodies"
        return f"path={capture.path}, {bodies}, selected headers"
    if isinstance(config, PydanticGEPAInstrumentation):
        return f"detail={config.detail}, optimizer lifecycle, scores, budgets, assets"
    return "metadata, lifecycle, usage"


__all__ = (
    "InstrumentorName",
    "InstrumentorStatus",
    "instrumentor_statuses",
    "resolve_instrumentor",
    "resolve_instrumentors",
)
