from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from autobench.metrics.observations import Direction, ObservationRole
from autobench.metrics.semantics import SemanticType
from autobench.protocol.ids import SignalId, SpanId, TraceId, new_signal_id
from autobench.protocol.values import EvidenceRef, ReferenceKind, SerializedValue

PROTOCOL_NAME = "abp"
PROTOCOL_VERSION = 1


class CaptureMechanism(StrEnum):
    HOOK = "hook"
    CALLBACK = "callback"
    WRAPPER = "wrapper"
    PATCH = "patch"
    MANUAL = "manual"


class AbstractionLayer(StrEnum):
    APPLICATION = "application"
    FRAMEWORK = "framework"
    CLIENT = "client"
    TRANSPORT = "transport"


class CaptureLevel(StrEnum):
    NONE = "none"
    METADATA = "metadata"
    HASH = "hash"
    REDACTED = "redacted"
    FULL = "full"


class KnownSpanKind(StrEnum):
    TASK = "task"
    WORKFLOW = "workflow"
    AGENT = "agent"
    LLM = "llm"
    EMBEDDING = "embedding"
    TOOL = "tool"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    PARSER = "parser"
    VALIDATION = "validation"
    APPROVAL = "approval"
    SCORER = "scorer"
    DERIVER = "deriver"
    HTTP = "http"
    DATABASE = "database"
    CACHE = "cache"
    STORAGE = "storage"
    CUSTOM = "custom"


class SpanStatus(StrEnum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class EndReason(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"
    TIMEOUT = "timeout"
    ABANDONED = "abandoned"


class MeasurementScope(StrEnum):
    DIRECT = "direct"
    AGGREGATE = "aggregate"


class LinkRelation(StrEnum):
    HANDOFF = "handoff"
    DELEGATION = "delegation"
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    BATCH_ITEM = "batch_item"
    CACHE_SOURCE = "cache_source"
    RETRY_OF = "retry_of"
    ARTIFACT_CONSUMPTION = "artifact_consumption"
    RUN_LINEAGE = "run_lineage"


class ExecutionRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    case_id: str | None = Field(default=None, min_length=1)
    variant_id: str | None = Field(default=None, min_length=1)


class InstrumentationScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instrumentor_name: str = Field(min_length=1)
    instrumentor_version: str = Field(min_length=1)
    package_name: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    mechanism: CaptureMechanism
    layer: AbstractionLayer
    source_convention: str | None = Field(default=None, min_length=1)
    source_convention_version: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_source_convention(self) -> InstrumentationScope:
        if self.source_convention is None and self.source_convention_version is not None:
            raise ValueError("source_convention_version requires source_convention")
        return self


class SourceProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str = Field(min_length=1)
    key: str = Field(min_length=1)
    path: tuple[str | int, ...] = ()
    convention_version: str | None = Field(default=None, min_length=1)
    source_map_id: str | None = Field(default=None, min_length=1)
    source_map_version: int | None = Field(default=None, ge=1)
    instrumentor: str | None = Field(default=None, min_length=1)
    instrumented_library_version: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_source_map(self) -> SourceProvenance:
        if (self.source_map_id is None) != (self.source_map_version is None):
            raise ValueError("source_map_id and source_map_version must be provided together")
        return self


class LinkTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: TraceId | None = None
    span_id: SpanId | None = None
    run_id: str | None = Field(default=None, min_length=1)
    reference: EvidenceRef | None = None

    @model_validator(mode="after")
    def validate_target(self) -> LinkTarget:
        target_count = sum(
            target is not None for target in (self.trace_id, self.run_id, self.reference)
        )
        if target_count != 1:
            raise ValueError("links require exactly one trace, run, or reference target")
        if self.span_id is not None and self.trace_id is None:
            raise ValueError("link span_id requires trace_id")
        return self


class SignalBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: Literal["abp"] = PROTOCOL_NAME
    protocol_version: Literal[1] = PROTOCOL_VERSION
    signal_id: SignalId = Field(default_factory=new_signal_id)
    trace_id: TraceId
    emitted_at: AwareDatetime
    monotonic_ns: NonNegativeInt
    sequence: NonNegativeInt
    execution: ExecutionRef | None = None
    scope: InstrumentationScope
    source: SourceProvenance | None = None

    @field_validator("emitted_at")
    @classmethod
    def normalize_emitted_at(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


class SpanStart(SignalBase):
    type: Literal["span_start"] = "span_start"
    span_id: SpanId
    parent_span_id: SpanId | None = None
    operation: str = Field(min_length=1)
    kind: str = Field(default=KnownSpanKind.CUSTOM, min_length=1)
    attributes: dict[str, SerializedValue] = Field(default_factory=dict)
    source_attributes: dict[str, SerializedValue] = Field(default_factory=dict)
    links: tuple[LinkTarget, ...] = ()
    capture: CaptureLevel = CaptureLevel.METADATA


class SpanEnd(SignalBase):
    type: Literal["span_end"] = "span_end"
    span_id: SpanId
    attributes: dict[str, SerializedValue] = Field(default_factory=dict)
    output: SerializedValue = None
    output_reference: EvidenceRef | None = None
    status: SpanStatus = SpanStatus.UNSET
    reason: EndReason = EndReason.COMPLETED
    errors: tuple[EvidenceRef, ...] = ()
    partial: bool = False
    usage: dict[str, SerializedValue] = Field(default_factory=dict)
    stream: dict[str, SerializedValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_output(self) -> SpanEnd:
        if self.output is not None and self.output_reference is not None:
            raise ValueError("span end output and output_reference are mutually exclusive")
        if any(error.kind is not ReferenceKind.ERROR for error in self.errors):
            raise ValueError("span end errors require error references")
        return self


class Event(SignalBase):
    type: Literal["event"] = "event"
    span_id: SpanId
    name: str = Field(min_length=1)
    semantic_type: SemanticType = Field(min_length=1)
    body: SerializedValue = None
    reference: EvidenceRef | None = None
    attributes: dict[str, SerializedValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_body(self) -> Event:
        if self.body is not None and self.reference is not None:
            raise ValueError("event body and reference are mutually exclusive")
        return self


class Measurement(SignalBase):
    type: Literal["measurement"] = "measurement"
    span_id: SpanId
    name: str = Field(min_length=1)
    semantic_type: SemanticType = Field(min_length=1)
    value: StrictBool | StrictInt | StrictFloat
    unit: str | None = Field(default=None, min_length=1)
    direction: Direction | None = None
    role: ObservationRole | None = None
    measurement_scope: MeasurementScope
    layer: AbstractionLayer
    attributes: dict[str, SerializedValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_layer(self) -> Measurement:
        if self.layer is not self.scope.layer:
            raise ValueError("measurement layer must match instrumentation scope layer")
        return self


class Link(SignalBase):
    type: Literal["link"] = "link"
    span_id: SpanId
    relation: LinkRelation
    target: LinkTarget
    attributes: dict[str, SerializedValue] = Field(default_factory=dict)


class Reference(SignalBase):
    type: Literal["reference"] = "reference"
    span_id: SpanId | None = None
    semantic_type: SemanticType | None = Field(default=None, min_length=1)
    name: str | None = Field(default=None, min_length=1)
    reference: EvidenceRef
    attributes: dict[str, SerializedValue] = Field(default_factory=dict)


Signal: TypeAlias = Annotated[
    SpanStart | SpanEnd | Event | Measurement | Link | Reference,
    Field(discriminator="type"),
]


__all__ = (
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "AbstractionLayer",
    "CaptureLevel",
    "CaptureMechanism",
    "EndReason",
    "Event",
    "ExecutionRef",
    "InstrumentationScope",
    "KnownSpanKind",
    "Link",
    "LinkRelation",
    "LinkTarget",
    "Measurement",
    "MeasurementScope",
    "Reference",
    "Signal",
    "SourceProvenance",
    "SpanEnd",
    "SpanStart",
    "SpanStatus",
)
