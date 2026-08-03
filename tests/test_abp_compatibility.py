from __future__ import annotations

from pathlib import Path

import pytest

import autobench
from autobench import (
    Case,
    Direction,
    DurationMetricSpec,
    InstrumentMetricSpec,
    ObservationSource,
    RunContext,
    RunRecord,
    Semantic,
    SpanKind,
    TraceEnvelope,
    Variant,
    instrument_method,
)
from autobench.io import load_yaml
from autobench.records.recording import RunRecord as RecordingRunRecord
from autobench.runtime.context import RunContext as RuntimeRunContext
from autobench.runtime.context import Span as RuntimeSpan
from autobench.runtime.context import SpanRecord as RuntimeSpanRecord
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context
from autobench.runtime.traces import TraceEnvelope as RuntimeTraceEnvelope
from scripts.benchmark_spans import (
    benchmark_httpx_stream,
    benchmark_httpx_transport,
    benchmark_manual_spans,
)

FIXTURES = Path(__file__).parent / "fixtures" / "abp"


def test_public_instrumentation_imports_remain_stable() -> None:
    expected = {
        "ArtifactRef",
        "AssetVersion",
        "DurationMetricSpec",
        "ErrorRecord",
        "InstrumentationHandle",
        "InstrumentCall",
        "InstrumentFactorSpec",
        "InstrumentMetricSpec",
        "Observation",
        "RunContext",
        "RunRecord",
        "Span",
        "SpanKind",
        "SpanRecord",
        "TraceEnvelope",
        "attach_trace",
        "get_active_run_context",
        "instrument_method",
        "trace_to_observations",
    }

    assert expected <= set(autobench.__all__)
    assert autobench.RunContext is RuntimeRunContext
    assert autobench.Span is RuntimeSpan
    assert autobench.SpanRecord is RuntimeSpanRecord
    assert autobench.TraceEnvelope is RuntimeTraceEnvelope
    assert autobench.RunRecord is RecordingRunRecord


def test_nested_manual_spans_preserve_linked_evidence_and_duration() -> None:
    ctx = RunContext(
        benchmark_id="compatibility",
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )

    with ctx.span(
        "workflow",
        kind=SpanKind.WORKFLOW,
        input={"prompt": "hello"},
        duration_metric=DurationMetricSpec(
            name="workflow_latency",
            semantic_type=Semantic.TIME_LATENCY,
            direction=Direction.MINIMIZE,
        ),
    ) as parent:
        parent.set_attribute("route", "direct")
        with ctx.span("tool", kind=SpanKind.TOOL) as child:
            child.set_output({"answer": 42})
            metric = child.metric(
                "correctness",
                1.0,
                semantic_type=Semantic.QUALITY_CORRECTNESS,
                direction=Direction.MAXIMIZE,
            )
            artifact = child.artifact("tool-output", {"answer": 42})
            error = child.error("retained diagnostic")

    parent_record, child_record = ctx.spans
    assert parent_record.parent_id is None
    assert child_record.parent_id == parent.id
    assert child_record.output == {"answer": 42}
    assert parent_record.attributes == {"route": "direct"}
    assert parent_record.ended_at is not None
    assert parent_record.duration_seconds is not None
    assert parent_record.duration_seconds >= 0
    assert child_record.ended_at is not None
    assert child_record.duration_seconds is not None
    assert child_record.duration_seconds >= 0
    assert metric.id in child_record.observations
    assert artifact.id in child_record.artifacts
    assert error.span_id == child.id
    assert child_record.error == error
    duration = next(item for item in ctx.observations if item.name == "workflow_latency")
    assert duration.span_id == parent.id


def test_legacy_run_record_fixture_loads_without_new_protocol_fields() -> None:
    payload = load_yaml(FIXTURES / "legacy_run_record.yaml")
    record = RunRecord.model_validate(payload)

    assert record.run_id == "legacy_run_1"
    assert record.case.id == "legacy_case"
    assert record.task_status.value == "passed"
    assert record.evaluation_status.value == "passed"
    assert record.spans[0].id == "legacy_span_1"


def test_legacy_trace_envelope_fixture_preserves_usage_errors_and_artifact() -> None:
    payload = load_yaml(FIXTURES / "legacy_trace_envelope.yaml")
    trace = TraceEnvelope.model_validate(payload)

    assert trace.trace_id == "legacy_trace_1"
    assert trace.spans[0].usage == {"input_tokens": 4, "output_tokens": 2}
    assert trace.errors[0].error_type == "LegacyWarning"
    assert trace.raw_artifact is not None
    assert trace.raw_artifact.id == "legacy_trace_artifact"


def test_method_instrumentation_restores_original_descriptor_and_records_metric() -> None:
    class Worker:
        def execute(self, value: int) -> dict[str, int]:
            return {"value": value}

    original = Worker.execute
    handle = instrument_method(
        Worker,
        "execute",
        span="worker.execute",
        metrics=[InstrumentMetricSpec(name="value", value_path="result.value")],
    )
    ctx = RunContext(
        benchmark_id="compatibility",
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )
    token = set_active_run_context(ctx)
    try:
        assert Worker().execute(7) == {"value": 7}
    finally:
        reset_active_run_context(token)
        handle.close()

    assert Worker.execute is original
    assert ctx.spans[0].name == "worker.execute"
    assert ctx.observations[0].source is ObservationSource.INSTRUMENTATION
    assert ctx.observations[0].value == 7


def test_manual_span_benchmark_has_reproducible_shape_and_validates_inputs() -> None:
    result = benchmark_manual_spans(iterations=3, repeats=2)

    assert result.iterations == 3
    assert result.repeats == 2
    assert len(result.samples_ns) == 2
    assert result.minimum_ns == min(result.samples_ns)
    assert result.median_ns >= result.minimum_ns
    assert all(sample >= 0 for sample in result.samples_ns)

    with pytest.raises(ValueError, match="iterations must be at least 1"):
        benchmark_manual_spans(iterations=0, repeats=1)
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        benchmark_manual_spans(iterations=1, repeats=0)


def test_httpx_transport_benchmark_reports_timing_and_peak_memory() -> None:
    result = benchmark_httpx_transport(iterations=2, repeats=2)

    assert result.iterations == 2
    assert result.repeats == 2
    assert result.baseline_median_ns > 0
    assert result.instrumented_median_ns > 0
    assert result.overhead_ns == result.instrumented_median_ns - result.baseline_median_ns
    assert result.overhead_ratio == result.instrumented_median_ns / result.baseline_median_ns
    assert result.instrumented_peak_bytes > 0

    with pytest.raises(ValueError, match="iterations must be at least 1"):
        benchmark_httpx_transport(iterations=0, repeats=1)
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        benchmark_httpx_transport(iterations=1, repeats=0)


def test_httpx_stream_benchmark_reports_bounded_peak_memory_and_validates_inputs() -> None:
    result = benchmark_httpx_stream(chunks=3, chunk_size_bytes=4, repeats=2)

    assert result.chunks == 3
    assert result.chunk_size_bytes == 4
    assert result.repeats == 2
    assert result.median_ns_per_chunk > 0
    assert result.peak_bytes > 0
    assert result.peak_bytes_per_chunk == result.peak_bytes / result.chunks

    with pytest.raises(ValueError, match="chunks must be at least 1"):
        benchmark_httpx_stream(chunks=0, chunk_size_bytes=1, repeats=1)
    with pytest.raises(ValueError, match="chunk_size_bytes must be at least 1"):
        benchmark_httpx_stream(chunks=1, chunk_size_bytes=0, repeats=1)
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        benchmark_httpx_stream(chunks=1, chunk_size_bytes=1, repeats=0)
