from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autobench.metrics.observations import Direction, ObservationRole
from autobench.metrics.semantics import SemanticType
from autobench.protocol.signals import AbstractionLayer, CaptureMechanism
from autobench.tracking import (
    AssetRepresentation,
    AssetSensitivity,
    SerializedValue,
)

if TYPE_CHECKING:
    from autobench.instrumentation.manager import InstrumentationRuntime


class InstrumentationError(RuntimeError):
    """Raised when native instrumentation cannot be installed safely."""


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    CONFLICT = "conflict"


class InstrumentorCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    sync: bool = True
    async_: bool = Field(default=False, alias="async")
    streaming: bool = False
    native_hooks: bool = False
    asset_discovery: bool = False
    asset_kinds: tuple[str, ...] = ()


class InstrumentorInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    target_distribution: str | None = Field(default=None, min_length=1)
    supported_versions: str | None = Field(default=None, min_length=1)
    mechanism: CaptureMechanism
    layer: AbstractionLayer
    span_kinds: tuple[str, ...] = ()
    semantic_families: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    source_convention: str | None = Field(default=None, min_length=1)
    source_convention_version: str | None = Field(default=None, min_length=1)
    capabilities: InstrumentorCapabilities = Field(default_factory=InstrumentorCapabilities)

    @model_validator(mode="after")
    def validate_target_and_convention(self) -> InstrumentorInfo:
        if self.supported_versions is not None and self.target_distribution is None:
            raise ValueError("supported_versions requires target_distribution")
        if self.source_convention is None and self.source_convention_version is not None:
            raise ValueError("source_convention_version requires source_convention")
        return self


class Compatibility(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: CompatibilityStatus = CompatibilityStatus.COMPATIBLE
    target_version: str | None = Field(default=None, min_length=1)
    degraded_features: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    private_seam_supported: bool | None = None

    @property
    def available(self) -> bool:
        return self.status is not CompatibilityStatus.UNAVAILABLE

    @property
    def supported(self) -> bool:
        return self.status not in {
            CompatibilityStatus.UNAVAILABLE,
            CompatibilityStatus.UNSUPPORTED,
        }

    @property
    def installable(self) -> bool:
        return self.status in {
            CompatibilityStatus.COMPATIBLE,
            CompatibilityStatus.DEGRADED,
        }

    @classmethod
    def compatible(cls, *, target_version: str | None = None) -> Compatibility:
        return cls(target_version=target_version)


class InstrumentationHandle(AbstractContextManager["InstrumentationHandle"]):
    def __init__(
        self,
        close_callback: Callable[[], None],
        *,
        info: InstrumentorInfo | None = None,
    ) -> None:
        self._close_callback = close_callback
        self.info = info
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._close_callback()
        self._closed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None


class Instrumentor(Protocol):
    @property
    def info(self) -> InstrumentorInfo: ...

    def check(self) -> Compatibility: ...

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle: ...


@dataclass(slots=True)
class InstrumentCall:
    instance: Any | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any = None
    error: BaseException | None = None
    stream_item_count: int = 0
    last_stream_item: Any = None


class InstrumentMetricSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = Field(min_length=1)
    semantic_type: SemanticType | None = None
    unit: str | None = None
    direction: Direction | None = None
    role: ObservationRole | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    value_path: str | None = None
    value_factory: Callable[[InstrumentCall], Any] | None = None

    @model_validator(mode="after")
    def validate_extractor(self) -> InstrumentMetricSpec:
        if self.value_path is None and self.value_factory is None:
            raise ValueError("instrument metrics require value_path or value_factory")
        return self


class InstrumentFactorSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = Field(min_length=1)
    semantic_type: SemanticType | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    value_path: str | None = None
    value_factory: Callable[[InstrumentCall], Any] | None = None

    @model_validator(mode="after")
    def validate_extractor(self) -> InstrumentFactorSpec:
        if self.value_path is None and self.value_factory is None:
            raise ValueError("instrument factors require value_path or value_factory")
        return self


class InstrumentAssetSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kind: str = Field(min_length=1)
    local_id: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)
    source_locator: str | None = Field(default=None, min_length=1)
    representation: AssetRepresentation = AssetRepresentation.DEFINITION
    semantic_type: SemanticType | None = None
    scope: str | None = Field(default=None, min_length=1)
    owner_locator: str | None = Field(default=None, min_length=1)
    definition_locator: str | None = Field(default=None, min_length=1)
    aliases: tuple[str, ...] = ()
    metadata: dict[str, SerializedValue] = Field(default_factory=dict)
    sensitivity: AssetSensitivity = AssetSensitivity.INTERNAL
    many: bool = False
    value_path: str | None = None
    value_factory: Callable[[InstrumentCall], Any] | None = None
    extractor_target: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_extractor(self) -> InstrumentAssetSpec:
        extractor_count = sum(
            value is not None
            for value in (self.value_path, self.value_factory, self.extractor_target)
        )
        if extractor_count != 1:
            raise ValueError(
                "instrument assets require exactly one of value_path, value_factory, "
                "or extractor_target"
            )
        return self


__all__ = (
    "Compatibility",
    "CompatibilityStatus",
    "InstrumentationError",
    "InstrumentationHandle",
    "InstrumentCall",
    "InstrumentAssetSpec",
    "InstrumentFactorSpec",
    "InstrumentMetricSpec",
    "Instrumentor",
    "InstrumentorCapabilities",
    "InstrumentorInfo",
)
