from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Event as ThreadEvent

import pytest

from autobench import Direction, ObservationRole
from autobench.protocol import (
    AbstractionLayer,
    ActiveContext,
    CaptureLevel,
    CaptureMechanism,
    Diagnostic,
    DiagnosticSeverity,
    Emitter,
    EndReason,
    EvidenceRef,
    ExecutionRef,
    InstrumentationScope,
    LinkRelation,
    LinkTarget,
    LocalCollector,
    MeasurementScope,
    ReferenceKind,
    SourceProvenance,
    SpanStatus,
    new_signal_id,
    new_trace_id,
    use_context,
)


def test_emitter_materializes_every_signal_with_execution_and_provenance() -> None:
    collector = LocalCollector()
    execution = ExecutionRef(
        benchmark_id="benchmark",
        experiment_id="experiment",
        run_id="run",
        case_id="case",
        variant_id="variant",
    )
    emitter = Emitter(collector, manual_scope(), execution=execution)
    artifact = EvidenceRef(kind=ReferenceKind.ARTIFACT, id="artifact-1")

    with emitter.span("workflow", kind="workflow", attributes={"attempt": 1}) as root:
        emitter.event(root.span_id, "started", "workflow.started", body={"ok": True})
        emitter.measurement(
            root.span_id,
            "latency",
            "time.latency",
            4.5,
            unit="ms",
            direction=Direction.MINIMIZE,
            role=ObservationRole.DIAGNOSTIC,
            measurement_scope=MeasurementScope.DIRECT,
        )
        emitter.link(
            root.span_id,
            LinkRelation.RUN_LINEAGE,
            LinkTarget(run_id="parent-run"),
        )
        emitter.reference(
            artifact,
            span_id=root.span_id,
            semantic_type="artifact.created",
            name="raw output",
        )
        with emitter.span("tool", kind="tool") as child:
            child_id = child.span_id
    emitter.reference(artifact, semantic_type="artifact.trace")

    trace = collector.finish(emitter.trace_id)
    root_record, child_record = trace.spans
    assert trace.execution == execution
    assert trace.root_span_ids == (root.span_id,)
    assert root_record.operation == "workflow"
    assert root_record.events[0].body == {"ok": True}
    assert root_record.measurements[0].value == 4.5
    assert root_record.links[0].relation is LinkRelation.RUN_LINEAGE
    assert root_record.references[0].reference == artifact
    assert child_record.span_id == child_id
    assert child_record.parent_span_id == root.span_id
    assert trace.references[0].reference == artifact
    assert trace.links == root_record.links
    assert trace.partial is False
    assert trace.started_at is not None
    assert trace.ended_at is not None
    assert all(
        earlier.sequence < later.sequence
        for earlier, later in zip(trace.signals, trace.signals[1:], strict=False)
    )


def test_emitter_preserves_explicit_capture_fields_and_complete_end_payload() -> None:
    collector = LocalCollector()
    emitter = Emitter(collector, manual_scope())
    source = SourceProvenance(system="native.test", key="result")
    artifact = EvidenceRef(kind=ReferenceKind.ARTIFACT, id="output")
    error = EvidenceRef(kind=ReferenceKind.ERROR, id="error")
    parent_id = "7" * 16
    span_id = "8" * 16
    start = emitter.start_span(
        "explicit",
        span_id=span_id,
        parent_span_id=parent_id,
        source_attributes={"native.key": "value"},
        links=(LinkTarget(run_id="source-run"),),
        capture=CaptureLevel.FULL,
        source=source,
    )
    emitter.measurement(
        span_id,
        "aggregate",
        "quality.score",
        1,
        layer=AbstractionLayer.APPLICATION,
        attributes={"window": 5},
        source=source,
    )
    emitter.end_span(
        span_id,
        attributes={"finished": True},
        output_reference=artifact,
        status=SpanStatus.ERROR,
        reason=EndReason.FAILED,
        errors=(error,),
        usage={"requests": 1},
        stream={"chunks": 2},
        source=source,
    )

    record = collector.finish(emitter.trace_id).spans[0]
    assert start.parent_span_id == parent_id
    assert record.capture is CaptureLevel.FULL
    assert record.source_attributes == {"native.key": "value"}
    assert record.start_links == (LinkTarget(run_id="source-run"),)
    assert record.output_reference == artifact
    assert record.errors == (error,)
    assert record.attributes == {"finished": True}
    assert record.usage == {"requests": 1}
    assert record.stream == {"chunks": 2}


@pytest.mark.asyncio
async def test_async_siblings_and_inherited_child_tasks_keep_correct_parents() -> None:
    collector = LocalCollector()
    emitter = Emitter(collector, manual_scope())
    both_entered = asyncio.Event()
    entered = 0

    async with emitter.span("root") as root:

        async def sibling(name: str) -> tuple[str, str]:
            nonlocal entered
            async with emitter.span(name) as child:
                entered += 1
                if entered == 2:
                    both_entered.set()
                await both_entered.wait()

                async def inherited() -> str:
                    async with emitter.span(f"{name}.nested") as nested:
                        return nested.span_id

                nested_id = await asyncio.create_task(inherited())
                return child.span_id, nested_id

        first, second = await asyncio.gather(sibling("first"), sibling("second"))

    trace = collector.finish(emitter.trace_id)
    records = {span.span_id: span for span in trace.spans}
    assert records[first[0]].parent_span_id == root.span_id
    assert records[second[0]].parent_span_id == root.span_id
    assert records[first[1]].parent_span_id == first[0]
    assert records[second[1]].parent_span_id == second[0]


@pytest.mark.asyncio
async def test_concurrent_emitters_keep_runs_isolated() -> None:
    first_collector = LocalCollector()
    second_collector = LocalCollector()
    first = Emitter(first_collector, manual_scope())
    second = Emitter(second_collector, manual_scope())

    async def run(emitter: Emitter, name: str) -> None:
        async with emitter.span(name):
            await asyncio.sleep(0)

    await asyncio.gather(run(first, "first"), run(second, "second"))
    first_trace = first_collector.finish(first.trace_id)
    second_trace = second_collector.finish(second.trace_id)

    assert {span.operation for span in first_trace.spans} == {"first"}
    assert {span.operation for span in second_trace.spans} == {"second"}
    assert first_trace.trace_id != second_trace.trace_id


@pytest.mark.asyncio
async def test_cancellation_and_timeout_end_spans_as_partial_errors() -> None:
    collector = LocalCollector()
    emitter = Emitter(collector, manual_scope())

    with pytest.raises(asyncio.CancelledError):
        async with emitter.span("cancelled"):
            raise asyncio.CancelledError
    with pytest.raises(TimeoutError), emitter.span("timeout"):
        raise TimeoutError("deadline")

    trace = collector.finish(emitter.trace_id)
    cancelled, timed_out = trace.spans
    assert cancelled.status is SpanStatus.ERROR
    assert cancelled.end_reason is EndReason.CANCELLED
    assert cancelled.partial is True
    assert timed_out.status is SpanStatus.ERROR
    assert timed_out.end_reason is EndReason.TIMEOUT
    assert timed_out.partial is True


def test_finish_abandons_open_spans_and_close_flush_are_idempotent() -> None:
    collector = LocalCollector()
    first = Emitter(collector, manual_scope(), trace_id="1" * 32)
    second = Emitter(collector, manual_scope(), trace_id="2" * 32)
    open_span = first.start_span("open")
    with second.span("complete"):
        pass

    before_finish = collector.snapshot(first.trace_id)
    assert before_finish.partial is True
    assert before_finish.spans[0].ended_at is None

    finished = collector.finish(first.trace_id, error=True)
    assert finished.spans[0].span_id == open_span.span_id
    assert finished.spans[0].status is SpanStatus.ERROR
    assert finished.spans[0].end_reason is EndReason.ABANDONED
    assert finished.spans[0].partial is True
    assert collector.finish(first.trace_id) is finished
    completed = collector.finish(second.trace_id)
    assert completed.partial is False
    with pytest.raises(RuntimeError, match="trace is finished"):
        collector.reserve_sequence(second.trace_id)

    flushed = collector.flush()
    assert [trace.trace_id for trace in flushed] == [first.trace_id, second.trace_id]
    closed = collector.close()
    assert [trace.trace_id for trace in closed] == [first.trace_id, second.trace_id]
    assert collector.close() == closed
    assert collector.closed is True
    with pytest.raises(RuntimeError, match="collector is closed"):
        collector.reserve_sequence(new_trace_id())

    rejected = open_span.model_copy(
        update={"signal_id": new_signal_id(), "sequence": open_span.sequence + 100}
    )
    assert collector.emit(rejected) is False
    assert LocalCollector().close() == ()


def test_duplicate_ids_are_idempotent_and_duplicate_sequences_are_diagnosed() -> None:
    collector = LocalCollector(diagnostic_limit=1)
    emitter = Emitter(collector, manual_scope())
    start = emitter.start_span("root")

    assert collector.emit(start) is False
    duplicate_sequence = start.model_copy(
        update={"signal_id": new_signal_id(), "span_id": "9" * 16}
    )
    assert collector.emit(duplicate_sequence) is False
    another_duplicate = start.model_copy(update={"signal_id": new_signal_id(), "span_id": "8" * 16})
    assert collector.emit(another_duplicate) is False

    trace = collector.finish(emitter.trace_id)
    assert len(trace.diagnostics) == 1
    assert trace.diagnostics[0].code == "duplicate_sequence"

    with pytest.raises(ValueError, match="diagnostic_limit"):
        LocalCollector(diagnostic_limit=0)


def test_diagnostics_are_bounded_materialized_and_rejected_after_finish() -> None:
    collector = LocalCollector(diagnostic_limit=1)
    emitter = Emitter(collector, manual_scope())

    diagnostic = emitter.diagnostic(
        "capture_omitted",
        "capture policy omitted a value",
        severity=DiagnosticSeverity.WARNING,
        path="task.input",
        semantic_type="operation.input",
        details={"level": "none"},
    )
    assert diagnostic.details == {"level": "none"}
    assert (
        collector.add_diagnostic(
            emitter.trace_id,
            Diagnostic(code="overflow", message="not retained"),
        )
        is False
    )

    trace = collector.finish(emitter.trace_id)
    assert trace.diagnostics == (diagnostic,)
    assert (
        collector.add_diagnostic(
            emitter.trace_id,
            Diagnostic(code="finished", message="too late"),
        )
        is False
    )
    with pytest.raises(RuntimeError, match="collector rejected diagnostic"):
        emitter.diagnostic("finished", "too late")


def test_closed_collector_rejects_direct_diagnostics() -> None:
    collector = LocalCollector()
    collector.close()

    assert (
        collector.add_diagnostic(
            new_trace_id(),
            Diagnostic(code="closed", message="too late"),
        )
        is False
    )


def test_sequence_reservation_is_deterministic_under_thread_contention() -> None:
    collector = LocalCollector()
    emitter = Emitter(collector, manual_scope())
    start = emitter.start_span("root")

    def emit(index: int) -> int:
        return emitter.event(start.span_id, f"event-{index}", "test.event", body=index).sequence

    with ThreadPoolExecutor(max_workers=12) as executor:
        sequences = list(executor.map(emit, range(200)))
    emitter.end_span(start.span_id, status=SpanStatus.OK)

    trace = collector.finish(emitter.trace_id)
    assert len(set(sequences)) == 200
    assert [signal.sequence for signal in trace.signals] == sorted(
        signal.sequence for signal in trace.signals
    )
    assert len(trace.spans[0].events) == 200


def test_clock_callbacks_run_outside_the_collector_lock() -> None:
    collector = LocalCollector()
    callback_finished = ThreadEvent()
    emitter: Emitter

    def wall_clock() -> datetime:
        collector.snapshot(emitter.trace_id)
        callback_finished.set()
        return datetime.now(UTC)

    emitter = Emitter(collector, manual_scope(), wall_clock=wall_clock)
    emitter.start_span("root")

    assert callback_finished.is_set()


def test_emitted_span_rejects_invalid_lifecycle_and_records_generic_failure() -> None:
    collector = LocalCollector()
    emitter = Emitter(collector, manual_scope())
    span = emitter.span("lifecycle")

    with pytest.raises(RuntimeError, match="has not started"):
        _ = span.span_id
    assert span.__exit__(None, None, None) is None
    with pytest.raises(ValueError, match="failed"), span:
        raise ValueError("failed")
    with pytest.raises(RuntimeError, match="more than once"):
        span.__enter__()

    record = collector.finish(emitter.trace_id).spans[0]
    assert record.status is SpanStatus.ERROR
    assert record.end_reason is EndReason.FAILED
    assert record.partial is False


def test_emitter_reports_collector_rejection_after_sequence_reservation() -> None:
    collector = LocalCollector()
    emitter: Emitter

    def close_during_stamp() -> datetime:
        collector.close()
        return datetime.now(UTC)

    emitter = Emitter(collector, manual_scope(), wall_clock=close_during_stamp)
    with pytest.raises(RuntimeError, match="collector rejected span_start"):
        emitter.start_span("rejected")


def test_explicit_parent_wins_over_unrelated_active_context() -> None:
    collector = LocalCollector()
    emitter = Emitter(collector, manual_scope())
    unrelated = LocalCollector()
    explicit_parent = "6" * 16
    with use_context(ActiveContext(collector=unrelated, trace_id=new_trace_id())):
        start = emitter.start_span("child", parent_span_id=explicit_parent)
    emitter.end_span(start.span_id)

    assert collector.finish(emitter.trace_id).spans[0].parent_span_id == explicit_parent


def manual_scope() -> InstrumentationScope:
    return InstrumentationScope(
        instrumentor_name="autobench.manual",
        instrumentor_version="0.1.0",
        package_name="autobench",
        package_version="0.1.0",
        mechanism=CaptureMechanism.MANUAL,
        layer=AbstractionLayer.APPLICATION,
    )
