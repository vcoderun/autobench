from __future__ import annotations as _annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Literal

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
from autobench.protocol.signals import AbstractionLayer, CaptureMechanism

Detail = Literal["summary", "evaluations", "full"]


class PydanticGEPA:
    """Capture pydantic-gepa optimization evidence through its native event stream."""

    def __init__(
        self,
        *,
        detail: Detail = "full",
        discovery: AssetDiscoverySettings | None = None,
    ) -> None:
        self._detail: Detail = detail
        self._discovery = discovery or AssetDiscoverySettings()
        self._info = InstrumentorInfo(
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
        )

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        if find_spec("pydantic_gepa") is None:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=(
                    "pydantic-gepa is unavailable; install Autobench with the "
                    "'pydantic-gepa' extra",
                ),
            )
        try:
            from pydantic_gepa.events import RunStarted, subscribe
        except ImportError as error:
            return Compatibility(
                status=CompatibilityStatus.UNAVAILABLE,
                diagnostics=(f"pydantic-gepa event API could not be imported: {error}",),
            )
        if not callable(subscribe) or RunStarted.model_fields["event_version"].default != "1":
            return Compatibility(
                status=CompatibilityStatus.UNSUPPORTED,
                diagnostics=("pydantic-gepa event contract v1 is required",),
            )
        try:
            target_version = version("pydantic-gepa")
        except PackageNotFoundError:
            target_version = None
        return Compatibility.compatible(target_version=target_version)

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        from pydantic_gepa.events import subscribe

        from autobench.instrumentation.pydantic_gepa.adapter import EventAdapter
        from autobench.instrumentation.pydantic_gepa.assets import CandidateAssets

        target_version = version("pydantic-gepa")
        adapter = EventAdapter(
            runtime,
            self.info,
            CandidateAssets(
                runtime,
                self.info,
                self._discovery,
                target_version=target_version,
            ),
            detail=self._detail,
        )
        subscription = subscribe(adapter.observe, on_error="ignore")

        def close() -> None:
            subscription.close()
            adapter.close()

        return InstrumentationHandle(close, info=self.info)


__all__ = ("PydanticGEPA",)
