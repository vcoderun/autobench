from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from autobench.metrics.semantics import SemanticType
from autobench.protocol.ids import SignalId, SpanId, TraceId
from autobench.protocol.signals import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    CaptureLevel,
    EndReason,
    Event,
    ExecutionRef,
    InstrumentationScope,
    Link,
    LinkTarget,
    Measurement,
    Reference,
    Signal,
    SpanEnd,
    SpanStart,
    SpanStatus,
)
from autobench.protocol.values import EvidenceRef, SerializedValue


class DiagnosticSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class Diagnostic(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING
    signal_id: SignalId | None = None
    span_id: SpanId | None = None
    sequence: int | None = Field(default=None, ge=0)
    path: str | None = Field(default=None, min_length=1)
    semantic_type: SemanticType | None = Field(default=None, min_length=1)
    details: dict[str, SerializedValue] = Field(default_factory=dict)


class SpanRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: TraceId
    span_id: SpanId
    parent_span_id: SpanId | None = None
    operation: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    execution: ExecutionRef | None = None
    scope: InstrumentationScope
    capture: CaptureLevel
    started_at: datetime | None = None
    ended_at: datetime | None = None
    start_monotonic_ns: int | None = Field(default=None, ge=0)
    end_monotonic_ns: int | None = Field(default=None, ge=0)
    duration_ns: int | None = Field(default=None, ge=0)
    start_sequence: int | None = Field(default=None, ge=0)
    end_sequence: int | None = Field(default=None, ge=0)
    attributes: dict[str, SerializedValue] = Field(default_factory=dict)
    source_attributes: dict[str, SerializedValue] = Field(default_factory=dict)
    output: SerializedValue = None
    output_reference: EvidenceRef | None = None
    status: SpanStatus = SpanStatus.UNSET
    end_reason: EndReason | None = None
    usage: dict[str, SerializedValue] = Field(default_factory=dict)
    stream: dict[str, SerializedValue] = Field(default_factory=dict)
    errors: tuple[EvidenceRef, ...] = ()
    start_links: tuple[LinkTarget, ...] = ()
    events: tuple[Event, ...] = ()
    measurements: tuple[Measurement, ...] = ()
    links: tuple[Link, ...] = ()
    references: tuple[Reference, ...] = ()
    partial: bool = False

    @property
    def duration_seconds(self) -> float | None:
        if self.duration_ns is None:
            return None
        return self.duration_ns / 1_000_000_000


class Trace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: Literal["abp"] = PROTOCOL_NAME
    protocol_version: Literal[1] = PROTOCOL_VERSION
    trace_id: TraceId
    execution: ExecutionRef | None = None
    root_span_ids: tuple[SpanId, ...] = ()
    spans: tuple[SpanRecord, ...] = ()
    links: tuple[Link, ...] = ()
    references: tuple[Reference, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    signals: tuple[Signal, ...] = ()
    started_at: datetime | None = None
    ended_at: datetime | None = None
    partial: bool = False


def materialize_trace(
    trace_id: TraceId,
    signals: Iterable[Signal],
    *,
    diagnostics: Iterable[Diagnostic] = (),
    diagnostic_limit: int = 100,
) -> Trace:
    if diagnostic_limit < 1:
        raise ValueError("diagnostic_limit must be at least 1")

    ordered = tuple(sorted(signals, key=lambda signal: (signal.sequence, signal.signal_id)))
    trace_signals = tuple(signal for signal in ordered if signal.trace_id == trace_id)
    materialized_diagnostics = list(diagnostics)[:diagnostic_limit]

    def diagnose(
        code: str,
        message: str,
        *,
        signal: Signal | None = None,
        span_id: SpanId | None = None,
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
    ) -> None:
        if len(materialized_diagnostics) >= diagnostic_limit:
            return
        materialized_diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                severity=severity,
                signal_id=None if signal is None else signal.signal_id,
                span_id=span_id,
                sequence=None if signal is None else signal.sequence,
            )
        )

    foreign_count = len(ordered) - len(trace_signals)
    if foreign_count:
        diagnose(
            "foreign_trace_signal",
            f"ignored {foreign_count} signal(s) from another trace",
            severity=DiagnosticSeverity.ERROR,
        )

    starts: dict[SpanId, SpanStart] = {}
    ends: dict[SpanId, SpanEnd] = {}
    events: dict[SpanId, list[Event]] = {}
    measurements: dict[SpanId, list[Measurement]] = {}
    links: dict[SpanId, list[Link]] = {}
    references: dict[SpanId, list[Reference]] = {}
    root_references: list[Reference] = []
    span_order: list[SpanId] = []
    execution: ExecutionRef | None = None

    for signal in trace_signals:
        if signal.execution is not None:
            if execution is None:
                execution = signal.execution
            elif execution != signal.execution:
                diagnose(
                    "execution_mismatch",
                    "signals in one trace contain different execution references",
                    signal=signal,
                    severity=DiagnosticSeverity.ERROR,
                )

        if isinstance(signal, SpanStart):
            if signal.span_id not in span_order:
                span_order.append(signal.span_id)
            if signal.span_id in starts:
                diagnose(
                    "duplicate_span_start",
                    "a span can have only one start signal",
                    signal=signal,
                    span_id=signal.span_id,
                    severity=DiagnosticSeverity.ERROR,
                )
            else:
                starts[signal.span_id] = signal
        elif isinstance(signal, SpanEnd):
            if signal.span_id not in span_order:
                span_order.append(signal.span_id)
            if signal.span_id in ends:
                diagnose(
                    "duplicate_span_end",
                    "a span can have at most one end signal",
                    signal=signal,
                    span_id=signal.span_id,
                    severity=DiagnosticSeverity.ERROR,
                )
            else:
                ends[signal.span_id] = signal
        elif isinstance(signal, Event):
            if signal.span_id not in span_order:
                span_order.append(signal.span_id)
            events.setdefault(signal.span_id, []).append(signal)
        elif isinstance(signal, Measurement):
            if signal.span_id not in span_order:
                span_order.append(signal.span_id)
            measurements.setdefault(signal.span_id, []).append(signal)
        elif isinstance(signal, Link):
            if signal.span_id not in span_order:
                span_order.append(signal.span_id)
            links.setdefault(signal.span_id, []).append(signal)
        elif signal.span_id is None:
            root_references.append(signal)
        else:
            if signal.span_id not in span_order:
                span_order.append(signal.span_id)
            references.setdefault(signal.span_id, []).append(signal)

    records: list[SpanRecord] = []
    for span_id in span_order:
        start = starts.get(span_id)
        end = ends.get(span_id)
        related = tuple(events.get(span_id, ())) + tuple(measurements.get(span_id, ()))
        related += tuple(links.get(span_id, ())) + tuple(references.get(span_id, ()))
        if start is not None:
            basis: Signal = start
        elif end is not None:
            basis = end
        else:
            basis = related[0]

        partial = False
        if start is None:
            diagnose(
                "missing_span_start",
                "span evidence exists without a start signal",
                signal=basis,
                span_id=span_id,
                severity=DiagnosticSeverity.ERROR,
            )
            partial = True
        if end is None:
            diagnose(
                "missing_span_end",
                "span start has no end signal",
                signal=start,
                span_id=span_id,
            )
            partial = True

        duration_ns: int | None = None
        if start is not None and end is not None:
            if end.monotonic_ns < start.monotonic_ns:
                diagnose(
                    "invalid_span_time",
                    "span end monotonic time precedes its start",
                    signal=end,
                    span_id=span_id,
                    severity=DiagnosticSeverity.ERROR,
                )
                partial = True
            else:
                duration_ns = end.monotonic_ns - start.monotonic_ns
            previous_stream_signals = tuple(
                item for item in related if item.sequence < end.sequence
            )
            if previous_stream_signals and end.monotonic_ns < max(
                item.monotonic_ns for item in previous_stream_signals
            ):
                diagnose(
                    "invalid_stream_time",
                    "span end precedes an earlier stream signal on the monotonic clock",
                    signal=end,
                    span_id=span_id,
                    severity=DiagnosticSeverity.ERROR,
                )
                partial = True

        records.append(
            SpanRecord(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None if start is None else start.parent_span_id,
                operation="unknown" if start is None else start.operation,
                kind="custom" if start is None else start.kind,
                execution=basis.execution,
                scope=basis.scope,
                capture=CaptureLevel.METADATA if start is None else start.capture,
                started_at=None if start is None else start.emitted_at,
                ended_at=None if end is None else end.emitted_at,
                start_monotonic_ns=None if start is None else start.monotonic_ns,
                end_monotonic_ns=None if end is None else end.monotonic_ns,
                duration_ns=duration_ns,
                start_sequence=None if start is None else start.sequence,
                end_sequence=None if end is None else end.sequence,
                attributes={
                    **({} if start is None else start.attributes),
                    **({} if end is None else end.attributes),
                },
                source_attributes={} if start is None else start.source_attributes,
                output=None if end is None else end.output,
                output_reference=None if end is None else end.output_reference,
                status=SpanStatus.UNSET if end is None else end.status,
                end_reason=None if end is None else end.reason,
                usage={} if end is None else end.usage,
                stream={} if end is None else end.stream,
                errors=() if end is None else end.errors,
                start_links=() if start is None else start.links,
                events=tuple(events.get(span_id, ())),
                measurements=tuple(measurements.get(span_id, ())),
                links=tuple(links.get(span_id, ())),
                references=tuple(references.get(span_id, ())),
                partial=partial or (False if end is None else end.partial),
            )
        )

    records_by_id = {record.span_id: record for record in records}
    for record in records:
        parent_id = record.parent_span_id
        if parent_id is None:
            continue
        parent = records_by_id.get(parent_id)
        if parent is None:
            diagnose(
                "missing_parent",
                "span parent does not exist in this trace",
                span_id=record.span_id,
                severity=DiagnosticSeverity.ERROR,
            )
            continue
        seen = {record.span_id}
        cursor = parent
        while cursor.parent_span_id is not None:
            if cursor.span_id in seen:
                diagnose(
                    "parent_cycle",
                    "span parent relationships contain a cycle",
                    span_id=record.span_id,
                    severity=DiagnosticSeverity.ERROR,
                )
                break
            seen.add(cursor.span_id)
            next_parent = records_by_id.get(cursor.parent_span_id)
            if next_parent is None:
                break
            cursor = next_parent
        if (
            not record.partial
            and not parent.partial
            and record.start_monotonic_ns is not None
            and parent.start_monotonic_ns is not None
            and record.end_monotonic_ns is not None
            and parent.end_monotonic_ns is not None
            and (
                record.start_monotonic_ns < parent.start_monotonic_ns
                or record.end_monotonic_ns > parent.end_monotonic_ns
            )
        ):
            diagnose(
                "child_outside_parent_time",
                "complete child span falls outside its parent monotonic interval",
                span_id=record.span_id,
            )

    roots = tuple(
        record.span_id
        for record in records
        if record.parent_span_id is None or record.parent_span_id not in records_by_id
    )
    started = tuple(record.started_at for record in records if record.started_at is not None)
    ended = tuple(record.ended_at for record in records if record.ended_at is not None)
    partial = any(record.partial for record in records) or any(
        diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in materialized_diagnostics
    )
    return Trace(
        trace_id=trace_id,
        execution=execution,
        root_span_ids=roots,
        spans=tuple(records),
        links=tuple(signal for signal in trace_signals if isinstance(signal, Link)),
        references=tuple(root_references),
        diagnostics=tuple(materialized_diagnostics),
        signals=trace_signals,
        started_at=min(started) if started else None,
        ended_at=max(ended) if ended and not partial else None,
        partial=partial,
    )


__all__ = (
    "Diagnostic",
    "DiagnosticSeverity",
    "SpanRecord",
    "Trace",
    "materialize_trace",
)
