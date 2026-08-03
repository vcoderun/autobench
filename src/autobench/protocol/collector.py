from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import Token
from datetime import UTC, datetime
from threading import Lock
from time import monotonic_ns
from types import TracebackType

from autobench.metrics.observations import Direction, ObservationRole
from autobench.metrics.semantics import SemanticType
from autobench.protocol.context import ActiveContext, attach_context, get_context, reset_context
from autobench.protocol.ids import SignalId, SpanId, TraceId, new_span_id, new_trace_id
from autobench.protocol.signals import (
    AbstractionLayer,
    CaptureLevel,
    EndReason,
    Event,
    ExecutionRef,
    InstrumentationScope,
    Link,
    LinkRelation,
    LinkTarget,
    Measurement,
    MeasurementScope,
    Reference,
    Signal,
    SourceProvenance,
    SpanEnd,
    SpanStart,
    SpanStatus,
)
from autobench.protocol.traces import Diagnostic, DiagnosticSeverity, Trace, materialize_trace
from autobench.protocol.values import EvidenceRef, SerializedValue


class LocalCollector:
    def __init__(self, *, diagnostic_limit: int = 100) -> None:
        if diagnostic_limit < 1:
            raise ValueError("diagnostic_limit must be at least 1")
        self._diagnostic_limit = diagnostic_limit
        self._lock = Lock()
        self._finish_lock = Lock()
        self._next_sequence = 0
        self._signals: dict[TraceId, list[Signal]] = {}
        self._signal_ids: set[SignalId] = set()
        self._sequences: dict[TraceId, set[int]] = {}
        self._diagnostics: dict[TraceId, list[Diagnostic]] = {}
        self._finished: set[TraceId] = set()
        self._snapshots: dict[TraceId, Trace] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def reserve_sequence(self, trace_id: TraceId) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("collector is closed")
            if trace_id in self._finished:
                raise RuntimeError("trace is finished")
            self._next_sequence += 1
            self._signals.setdefault(trace_id, [])
            return self._next_sequence

    def emit(self, signal: Signal) -> bool:
        with self._lock:
            if signal.signal_id in self._signal_ids:
                return False
            if self._closed or signal.trace_id in self._finished:
                return False
            sequences = self._sequences.setdefault(signal.trace_id, set())
            if signal.sequence in sequences:
                diagnostics = self._diagnostics.setdefault(signal.trace_id, [])
                if len(diagnostics) < self._diagnostic_limit:
                    diagnostics.append(
                        Diagnostic(
                            code="duplicate_sequence",
                            message="a trace cannot admit two signals with one sequence",
                            severity=DiagnosticSeverity.ERROR,
                            signal_id=signal.signal_id,
                            span_id=signal.span_id,
                            sequence=signal.sequence,
                        )
                    )
                return False
            self._signals.setdefault(signal.trace_id, []).append(signal)
            self._signal_ids.add(signal.signal_id)
            sequences.add(signal.sequence)
            return True

    def add_diagnostic(self, trace_id: TraceId, diagnostic: Diagnostic) -> bool:
        with self._lock:
            if self._closed or trace_id in self._finished:
                return False
            diagnostics = self._diagnostics.setdefault(trace_id, [])
            if len(diagnostics) >= self._diagnostic_limit:
                return False
            self._signals.setdefault(trace_id, [])
            diagnostics.append(diagnostic)
            return True

    def snapshot(self, trace_id: TraceId) -> Trace:
        with self._lock:
            cached = self._snapshots.get(trace_id)
            if cached is not None:
                return cached
            signals = tuple(self._signals.get(trace_id, ()))
            diagnostics = tuple(self._diagnostics.get(trace_id, ()))
        return materialize_trace(
            trace_id,
            signals,
            diagnostics=diagnostics,
            diagnostic_limit=self._diagnostic_limit,
        )

    def finish(self, trace_id: TraceId, *, error: bool = False) -> Trace:
        with self._finish_lock:
            with self._lock:
                cached = self._snapshots.get(trace_id)
                if cached is not None:
                    return cached
                self._finished.add(trace_id)
                signals = tuple(self._signals.get(trace_id, ()))
                starts: dict[SpanId, SpanStart] = {}
                ended: set[SpanId] = set()
                for signal in sorted(signals, key=lambda item: (item.sequence, item.signal_id)):
                    if isinstance(signal, SpanStart) and signal.span_id not in starts:
                        starts[signal.span_id] = signal
                    elif isinstance(signal, SpanEnd):
                        ended.add(signal.span_id)
                pending: list[tuple[SpanStart, int]] = []
                for start in starts.values():
                    if start.span_id in ended:
                        continue
                    self._next_sequence += 1
                    pending.append((start, self._next_sequence))

            synthetic = tuple(
                SpanEnd(
                    trace_id=trace_id,
                    span_id=start.span_id,
                    emitted_at=datetime.now(UTC),
                    monotonic_ns=monotonic_ns(),
                    sequence=sequence,
                    execution=start.execution,
                    scope=start.scope,
                    source=start.source,
                    status=SpanStatus.ERROR if error else SpanStatus.UNSET,
                    reason=EndReason.ABANDONED,
                    partial=True,
                )
                for start, sequence in pending
            )
            with self._lock:
                for signal in synthetic:
                    self._signals.setdefault(trace_id, []).append(signal)
                    self._signal_ids.add(signal.signal_id)
                    self._sequences.setdefault(trace_id, set()).add(signal.sequence)
                signals = tuple(self._signals.get(trace_id, ()))
                diagnostics = tuple(self._diagnostics.get(trace_id, ()))

            snapshot = materialize_trace(
                trace_id,
                signals,
                diagnostics=diagnostics,
                diagnostic_limit=self._diagnostic_limit,
            )
            with self._lock:
                self._snapshots[trace_id] = snapshot
            return snapshot

    def flush(self) -> tuple[Trace, ...]:
        with self._lock:
            trace_ids = tuple(sorted(self._signals))
        return tuple(self.snapshot(trace_id) for trace_id in trace_ids)

    def close(self, *, error: bool = False) -> tuple[Trace, ...]:
        with self._lock:
            if self._closed:
                return tuple(self._snapshots[trace_id] for trace_id in sorted(self._snapshots))
            self._closed = True
            trace_ids = tuple(sorted(self._signals))
        return tuple(self.finish(trace_id, error=error) for trace_id in trace_ids)


class Emitter:
    def __init__(
        self,
        collector: LocalCollector,
        scope: InstrumentationScope,
        *,
        trace_id: TraceId | None = None,
        execution: ExecutionRef | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], int] = monotonic_ns,
    ) -> None:
        self.collector = collector
        self.scope = scope
        self.trace_id = new_trace_id() if trace_id is None else trace_id
        self.execution = execution
        self._wall_clock = (lambda: datetime.now(UTC)) if wall_clock is None else wall_clock
        self._monotonic_clock = monotonic_clock

    def start_span(
        self,
        operation: str,
        *,
        span_id: SpanId | None = None,
        parent_span_id: SpanId | None = None,
        kind: str = "custom",
        attributes: dict[str, SerializedValue] | None = None,
        source_attributes: dict[str, SerializedValue] | None = None,
        links: tuple[LinkTarget, ...] = (),
        capture: CaptureLevel = CaptureLevel.METADATA,
        source: SourceProvenance | None = None,
    ) -> SpanStart:
        active = get_context()
        if (
            parent_span_id is None
            and active is not None
            and active.collector is self.collector
            and active.trace_id == self.trace_id
        ):
            parent_span_id = active.current_span_id
        sequence, emitted_at, monotonic = self._stamp()
        signal = SpanStart(
            trace_id=self.trace_id,
            span_id=new_span_id() if span_id is None else span_id,
            parent_span_id=parent_span_id,
            emitted_at=emitted_at,
            monotonic_ns=monotonic,
            sequence=sequence,
            execution=self.execution,
            scope=self.scope,
            source=source,
            operation=operation,
            kind=kind,
            attributes={} if attributes is None else attributes,
            source_attributes={} if source_attributes is None else source_attributes,
            links=links,
            capture=capture,
        )
        self._admit(signal)
        return signal

    def end_span(
        self,
        span_id: SpanId,
        *,
        attributes: dict[str, SerializedValue] | None = None,
        output: SerializedValue = None,
        output_reference: EvidenceRef | None = None,
        status: SpanStatus = SpanStatus.UNSET,
        reason: EndReason = EndReason.COMPLETED,
        errors: tuple[EvidenceRef, ...] = (),
        partial: bool = False,
        usage: dict[str, SerializedValue] | None = None,
        stream: dict[str, SerializedValue] | None = None,
        source: SourceProvenance | None = None,
    ) -> SpanEnd:
        sequence, emitted_at, monotonic = self._stamp()
        signal = SpanEnd(
            trace_id=self.trace_id,
            span_id=span_id,
            emitted_at=emitted_at,
            monotonic_ns=monotonic,
            sequence=sequence,
            execution=self.execution,
            scope=self.scope,
            source=source,
            attributes={} if attributes is None else attributes,
            output=output,
            output_reference=output_reference,
            status=status,
            reason=reason,
            errors=errors,
            partial=partial,
            usage={} if usage is None else usage,
            stream={} if stream is None else stream,
        )
        self._admit(signal)
        return signal

    def event(
        self,
        span_id: SpanId,
        name: str,
        semantic_type: SemanticType,
        *,
        body: SerializedValue = None,
        reference: EvidenceRef | None = None,
        attributes: dict[str, SerializedValue] | None = None,
        source: SourceProvenance | None = None,
    ) -> Event:
        sequence, emitted_at, monotonic = self._stamp()
        signal = Event(
            trace_id=self.trace_id,
            span_id=span_id,
            emitted_at=emitted_at,
            monotonic_ns=monotonic,
            sequence=sequence,
            execution=self.execution,
            scope=self.scope,
            source=source,
            name=name,
            semantic_type=semantic_type,
            body=body,
            reference=reference,
            attributes={} if attributes is None else attributes,
        )
        self._admit(signal)
        return signal

    def measurement(
        self,
        span_id: SpanId,
        name: str,
        semantic_type: SemanticType,
        value: bool | int | float,
        *,
        unit: str | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        measurement_scope: MeasurementScope = MeasurementScope.DIRECT,
        layer: AbstractionLayer | None = None,
        attributes: dict[str, SerializedValue] | None = None,
        source: SourceProvenance | None = None,
    ) -> Measurement:
        sequence, emitted_at, monotonic = self._stamp()
        signal = Measurement(
            trace_id=self.trace_id,
            span_id=span_id,
            emitted_at=emitted_at,
            monotonic_ns=monotonic,
            sequence=sequence,
            execution=self.execution,
            scope=self.scope,
            source=source,
            name=name,
            semantic_type=semantic_type,
            value=value,
            unit=unit,
            direction=direction,
            role=role,
            measurement_scope=measurement_scope,
            layer=self.scope.layer if layer is None else layer,
            attributes={} if attributes is None else attributes,
        )
        self._admit(signal)
        return signal

    def link(
        self,
        span_id: SpanId,
        relation: LinkRelation,
        target: LinkTarget,
        *,
        attributes: dict[str, SerializedValue] | None = None,
        source: SourceProvenance | None = None,
    ) -> Link:
        sequence, emitted_at, monotonic = self._stamp()
        signal = Link(
            trace_id=self.trace_id,
            span_id=span_id,
            emitted_at=emitted_at,
            monotonic_ns=monotonic,
            sequence=sequence,
            execution=self.execution,
            scope=self.scope,
            source=source,
            relation=relation,
            target=target,
            attributes={} if attributes is None else attributes,
        )
        self._admit(signal)
        return signal

    def reference(
        self,
        reference: EvidenceRef,
        *,
        span_id: SpanId | None = None,
        semantic_type: SemanticType | None = None,
        name: str | None = None,
        attributes: dict[str, SerializedValue] | None = None,
        source: SourceProvenance | None = None,
    ) -> Reference:
        sequence, emitted_at, monotonic = self._stamp()
        signal = Reference(
            trace_id=self.trace_id,
            span_id=span_id,
            emitted_at=emitted_at,
            monotonic_ns=monotonic,
            sequence=sequence,
            execution=self.execution,
            scope=self.scope,
            source=source,
            semantic_type=semantic_type,
            name=name,
            reference=reference,
            attributes={} if attributes is None else attributes,
        )
        self._admit(signal)
        return signal

    def diagnostic(
        self,
        code: str,
        message: str,
        *,
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
        span_id: SpanId | None = None,
        path: str | None = None,
        semantic_type: SemanticType | None = None,
        details: dict[str, SerializedValue] | None = None,
    ) -> Diagnostic:
        diagnostic = Diagnostic(
            code=code,
            message=message,
            severity=severity,
            span_id=span_id,
            path=path,
            semantic_type=semantic_type,
            details={} if details is None else details,
        )
        if not self.collector.add_diagnostic(self.trace_id, diagnostic):
            raise RuntimeError("collector rejected diagnostic")
        return diagnostic

    def span(
        self,
        operation: str,
        *,
        kind: str = "custom",
        attributes: dict[str, SerializedValue] | None = None,
        capture: CaptureLevel = CaptureLevel.METADATA,
    ) -> EmittedSpan:
        return EmittedSpan(
            self,
            operation=operation,
            kind=kind,
            attributes=attributes,
            capture=capture,
        )

    def _stamp(self) -> tuple[int, datetime, int]:
        sequence = self.collector.reserve_sequence(self.trace_id)
        return sequence, self._wall_clock().astimezone(UTC), self._monotonic_clock()

    def _admit(self, signal: Signal) -> None:
        if not self.collector.emit(signal):
            raise RuntimeError(f"collector rejected {signal.type}")


class EmittedSpan(
    AbstractContextManager["EmittedSpan"],
    AbstractAsyncContextManager["EmittedSpan"],
):
    def __init__(
        self,
        emitter: Emitter,
        *,
        operation: str,
        kind: str,
        attributes: dict[str, SerializedValue] | None,
        capture: CaptureLevel,
    ) -> None:
        self._emitter = emitter
        self._operation = operation
        self._kind = kind
        self._attributes = attributes
        self._capture = capture
        self._start: SpanStart | None = None
        self._token: Token[ActiveContext | None] | None = None

    @property
    def span_id(self) -> SpanId:
        if self._start is None:
            raise RuntimeError("span has not started")
        return self._start.span_id

    def __enter__(self) -> EmittedSpan:
        if self._start is not None:
            raise RuntimeError("span cannot be entered more than once")
        self._start = self._emitter.start_span(
            self._operation,
            kind=self._kind,
            attributes=self._attributes,
            capture=self._capture,
        )
        active = get_context()
        if (
            active is not None
            and active.collector is self._emitter.collector
            and active.trace_id == self._emitter.trace_id
        ):
            context = active.with_span(self._start.span_id)
        else:
            context = ActiveContext(
                collector=self._emitter.collector,
                trace_id=self._emitter.trace_id,
                current_span_id=self._start.span_id,
                execution=self._emitter.execution,
            )
        self._token = attach_context(context)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._start is None or self._token is None:
            return None
        status = SpanStatus.OK
        reason = EndReason.COMPLETED
        partial = False
        if exc_value is not None:
            status = SpanStatus.ERROR
            reason = EndReason.FAILED
            if isinstance(exc_value, asyncio.CancelledError):
                reason = EndReason.CANCELLED
                partial = True
            elif isinstance(exc_value, TimeoutError):
                reason = EndReason.TIMEOUT
                partial = True
        try:
            self._emitter.end_span(
                self._start.span_id,
                status=status,
                reason=reason,
                partial=partial,
            )
        finally:
            reset_context(self._token)
            self._token = None
        return None

    async def __aenter__(self) -> EmittedSpan:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self.__exit__(exc_type, exc_value, traceback)


__all__ = (
    "Emitter",
    "EmittedSpan",
    "LocalCollector",
)
