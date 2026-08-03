from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from autobench.protocol import (
    AbstractionLayer,
    CaptureMechanism,
    Diagnostic,
    DiagnosticSeverity,
    Event,
    EvidenceRef,
    ExecutionRef,
    InstrumentationScope,
    Link,
    LinkRelation,
    LinkTarget,
    Measurement,
    MeasurementScope,
    Reference,
    ReferenceKind,
    SpanEnd,
    SpanStart,
    SpanStatus,
    materialize_trace,
    new_signal_id,
)

TRACE_ID = "1" * 32
FIRST_SPAN = "2" * 16
SECOND_SPAN = "3" * 16
MISSING_SPAN = "4" * 16
CHAIN_SPAN = "6" * 16


def test_materializer_preserves_out_of_order_valid_signals() -> None:
    now = datetime.now(UTC)
    start = SpanStart(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        emitted_at=now,
        monotonic_ns=10,
        sequence=1,
        execution=execution("run"),
        scope=manual_scope(),
        operation="root",
        attributes={"start": True},
    )
    event = Event(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        emitted_at=now + timedelta(milliseconds=1),
        monotonic_ns=15,
        sequence=2,
        execution=execution("run"),
        scope=manual_scope(),
        name="chunk",
        semantic_type="stream.chunk",
        body="value",
    )
    end = SpanEnd(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        emitted_at=now + timedelta(milliseconds=2),
        monotonic_ns=20,
        sequence=3,
        execution=execution("run"),
        scope=manual_scope(),
        attributes={"end": True},
        status=SpanStatus.OK,
    )

    trace = materialize_trace(TRACE_ID, (end, event, start))
    record = trace.spans[0]
    assert [signal.sequence for signal in trace.signals] == [1, 2, 3]
    assert record.attributes == {"start": True, "end": True}
    assert record.duration_ns == 10
    assert record.duration_seconds == 0.00000001
    assert record.events == (event,)
    assert trace.partial is False


def test_materializer_reports_duplicates_missing_parents_cycles_and_execution_mismatch() -> None:
    now = datetime.now(UTC)
    first = SpanStart(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        parent_span_id=SECOND_SPAN,
        emitted_at=now,
        monotonic_ns=10,
        sequence=1,
        execution=execution("run-1"),
        scope=manual_scope(),
        operation="first",
    )
    second = SpanStart(
        trace_id=TRACE_ID,
        span_id=SECOND_SPAN,
        parent_span_id=FIRST_SPAN,
        emitted_at=now,
        monotonic_ns=11,
        sequence=2,
        execution=execution("run-2"),
        scope=manual_scope(),
        operation="second",
    )
    duplicate_start = first.model_copy(update={"signal_id": new_signal_id(), "sequence": 3})
    duplicate_end = SpanEnd(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        emitted_at=now,
        monotonic_ns=20,
        sequence=4,
        scope=manual_scope(),
    )
    duplicate_end_again = duplicate_end.model_copy(
        update={"signal_id": new_signal_id(), "sequence": 5}
    )
    orphan = SpanStart(
        trace_id=TRACE_ID,
        span_id=MISSING_SPAN,
        parent_span_id="5" * 16,
        emitted_at=now,
        monotonic_ns=12,
        sequence=6,
        scope=manual_scope(),
        operation="orphan",
    )
    chained = SpanStart(
        trace_id=TRACE_ID,
        span_id=CHAIN_SPAN,
        parent_span_id=MISSING_SPAN,
        emitted_at=now,
        monotonic_ns=13,
        sequence=7,
        scope=manual_scope(),
        operation="chained",
    )

    trace = materialize_trace(
        TRACE_ID,
        (first, second, duplicate_start, duplicate_end, duplicate_end_again, orphan, chained),
    )
    codes = {diagnostic.code for diagnostic in trace.diagnostics}
    assert {
        "duplicate_span_start",
        "duplicate_span_end",
        "execution_mismatch",
        "parent_cycle",
        "missing_parent",
        "missing_span_end",
    } <= codes
    assert trace.partial is True
    assert trace.root_span_ids == (MISSING_SPAN,)


def test_materializer_reports_invalid_times_missing_starts_and_foreign_signals() -> None:
    now = datetime.now(UTC)
    start = SpanStart(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        emitted_at=now,
        monotonic_ns=30,
        sequence=1,
        scope=manual_scope(),
        operation="root",
    )
    late_event = Event(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        emitted_at=now,
        monotonic_ns=40,
        sequence=2,
        scope=manual_scope(),
        name="late",
        semantic_type="stream.chunk",
    )
    end = SpanEnd(
        trace_id=TRACE_ID,
        span_id=FIRST_SPAN,
        emitted_at=now,
        monotonic_ns=20,
        sequence=3,
        scope=manual_scope(),
    )
    end_without_start = SpanEnd(
        trace_id=TRACE_ID,
        span_id=SECOND_SPAN,
        emitted_at=now,
        monotonic_ns=50,
        sequence=4,
        scope=manual_scope(),
    )
    foreign = start.model_copy(update={"trace_id": "9" * 32, "signal_id": new_signal_id()})

    trace = materialize_trace(TRACE_ID, (end_without_start, foreign, end, late_event, start))
    codes = {diagnostic.code for diagnostic in trace.diagnostics}
    assert {
        "foreign_trace_signal",
        "invalid_span_time",
        "invalid_stream_time",
        "missing_span_start",
    } <= codes
    assert trace.spans[1].operation == "unknown"
    assert trace.spans[1].duration_seconds is None
    assert trace.ended_at is None


def test_orphan_attached_signals_remain_available_as_partial_span_evidence() -> None:
    now = datetime.now(UTC)
    artifact = EvidenceRef(kind=ReferenceKind.ARTIFACT, id="artifact")
    signals = (
        Event(
            trace_id=TRACE_ID,
            span_id=FIRST_SPAN,
            emitted_at=now,
            monotonic_ns=1,
            sequence=1,
            scope=manual_scope(),
            name="event",
            semantic_type="test.event",
        ),
        Measurement(
            trace_id=TRACE_ID,
            span_id=SECOND_SPAN,
            emitted_at=now,
            monotonic_ns=2,
            sequence=2,
            scope=manual_scope(),
            name="measurement",
            semantic_type="quality.score",
            value=1,
            measurement_scope=MeasurementScope.DIRECT,
            layer=AbstractionLayer.APPLICATION,
        ),
        Link(
            trace_id=TRACE_ID,
            span_id=MISSING_SPAN,
            emitted_at=now,
            monotonic_ns=3,
            sequence=3,
            scope=manual_scope(),
            relation=LinkRelation.RUN_LINEAGE,
            target=LinkTarget(run_id="parent"),
        ),
        Reference(
            trace_id=TRACE_ID,
            span_id=CHAIN_SPAN,
            emitted_at=now,
            monotonic_ns=4,
            sequence=4,
            scope=manual_scope(),
            reference=artifact,
        ),
    )

    trace = materialize_trace(TRACE_ID, signals)
    assert [record.operation for record in trace.spans] == ["unknown"] * 4
    assert trace.spans[0].events == (signals[0],)
    assert trace.spans[1].measurements == (signals[1],)
    assert trace.spans[2].links == (signals[2],)
    assert trace.spans[3].references == (signals[3],)
    assert all(record.partial for record in trace.spans)


def test_complete_child_outside_parent_time_is_diagnosed() -> None:
    now = datetime.now(UTC)
    signals = (
        SpanStart(
            trace_id=TRACE_ID,
            span_id=FIRST_SPAN,
            emitted_at=now,
            monotonic_ns=10,
            sequence=1,
            scope=manual_scope(),
            operation="parent",
        ),
        SpanStart(
            trace_id=TRACE_ID,
            span_id=SECOND_SPAN,
            parent_span_id=FIRST_SPAN,
            emitted_at=now,
            monotonic_ns=5,
            sequence=2,
            scope=manual_scope(),
            operation="child",
        ),
        SpanEnd(
            trace_id=TRACE_ID,
            span_id=SECOND_SPAN,
            emitted_at=now,
            monotonic_ns=25,
            sequence=3,
            scope=manual_scope(),
        ),
        SpanEnd(
            trace_id=TRACE_ID,
            span_id=FIRST_SPAN,
            emitted_at=now,
            monotonic_ns=20,
            sequence=4,
            scope=manual_scope(),
        ),
    )

    trace = materialize_trace(TRACE_ID, signals)
    assert "child_outside_parent_time" in {diagnostic.code for diagnostic in trace.diagnostics}


def test_diagnostics_are_bounded_and_models_are_frozen() -> None:
    diagnostic = Diagnostic(code="existing", message="existing")
    foreign = SpanStart(
        trace_id="9" * 32,
        span_id=FIRST_SPAN,
        emitted_at=datetime.now(UTC),
        monotonic_ns=1,
        sequence=1,
        scope=manual_scope(),
        operation="foreign",
    )
    trace = materialize_trace(
        TRACE_ID,
        (foreign,),
        diagnostics=(diagnostic, diagnostic),
        diagnostic_limit=1,
    )
    assert trace.diagnostics == (diagnostic,)
    assert trace.spans == ()
    assert trace.started_at is None
    assert trace.ended_at is None

    with pytest.raises(ValueError, match="diagnostic_limit"):
        materialize_trace(TRACE_ID, (), diagnostic_limit=0)
    with pytest.raises(ValidationError, match="frozen"):
        trace.partial = True

    assert DiagnosticSeverity.ERROR.value == "error"


def execution(run_id: str) -> ExecutionRef:
    return ExecutionRef(
        benchmark_id="benchmark",
        experiment_id="experiment",
        run_id=run_id,
    )


def manual_scope() -> InstrumentationScope:
    return InstrumentationScope(
        instrumentor_name="autobench.manual",
        instrumentor_version="0.1.0",
        package_name="autobench",
        package_version="0.1.0",
        mechanism=CaptureMechanism.MANUAL,
        layer=AbstractionLayer.APPLICATION,
    )
