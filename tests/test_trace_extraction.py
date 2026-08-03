from __future__ import annotations

from datetime import UTC, datetime

import pytest

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    CompositeExtractor,
    ExtractionContext,
    ExtractionResult,
    Observation,
    Semantic,
    SignalExtractor,
    SpanExtractor,
    UsageExtractor,
)
from autobench.protocol import (
    AbstractionLayer,
    CaptureMechanism,
    EndReason,
    Event,
    EvidenceRef,
    InstrumentationScope,
    KnownSpanKind,
    Link,
    LinkRelation,
    LinkTarget,
    Measurement,
    MeasurementScope,
    Reference,
    ReferenceKind,
    SerializedValue,
    SourceProvenance,
    SpanEnd,
    SpanStart,
    SpanStatus,
    Trace,
    materialize_trace,
)

TRACE_ID = "1" * 32
ROOT = "2" * 16
FRAMEWORK = "3" * 16
CLIENT_ONE = "4" * 16
CLIENT_TWO = "5" * 16
OTHER = "6" * 16
NOW = datetime.now(UTC)


def test_signal_extractor_preserves_accounting_and_instrumentation_provenance() -> None:
    measurement = Measurement(
        trace_id=TRACE_ID,
        span_id=ROOT,
        emitted_at=NOW,
        monotonic_ns=2,
        sequence=2,
        scope=_scope(AbstractionLayer.CLIENT),
        name="input",
        semantic_type=Semantic.LLM_TOKENS_INPUT,
        value=12,
        measurement_scope=MeasurementScope.DIRECT,
        layer=AbstractionLayer.CLIENT,
        attributes={"logical_operation_id": "request-1"},
    )
    trace = materialize_trace(
        TRACE_ID,
        (
            *_span(ROOT, 1, 3, layer=AbstractionLayer.CLIENT, kind=KnownSpanKind.LLM),
            measurement,
        ),
    )

    result = SignalExtractor().extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    observation = next(item for item in result.observations if item.name == "input")
    assert observation.tags == {
        "abp.measurement_scope": "direct",
        "abp.abstraction_layer": "client",
        "abp.instrumentor": "test.client",
        "abp.logical_operation_id": "request-1",
    }


def test_usage_extractor_selects_one_layer_deduplicates_and_preserves_models() -> None:
    signals = (
        *_span(
            FRAMEWORK,
            1,
            100,
            layer=AbstractionLayer.FRAMEWORK,
            kind=KnownSpanKind.LLM,
            attributes={"logical_operation_id": "request-1"},
            usage={"input_tokens": 10},
        ),
        *_span(
            CLIENT_ONE,
            10,
            40,
            parent=FRAMEWORK,
            layer=AbstractionLayer.CLIENT,
            kind=KnownSpanKind.LLM,
            attributes={
                "logical_operation_id": "request-1",
                "requested_model": "requested-a",
                "response_model": "served-a",
                "provider": "provider-a",
            },
            usage={"input_tokens": 10, "output_tokens": 4},
        ),
        *_span(
            CLIENT_TWO,
            50,
            90,
            parent=FRAMEWORK,
            layer=AbstractionLayer.CLIENT,
            kind=KnownSpanKind.LLM,
            attributes={
                "logical_operation_id": "request-2",
                "requested_model": "requested-a",
                "response_model": "served-a",
                "provider": "provider-a",
            },
            usage={"input_tokens": 20, "output_tokens": 6},
        ),
        Measurement(
            trace_id=TRACE_ID,
            span_id=FRAMEWORK,
            emitted_at=NOW,
            monotonic_ns=99,
            sequence=99,
            scope=_scope(AbstractionLayer.FRAMEWORK),
            name="input_total",
            semantic_type=Semantic.LLM_TOKENS_INPUT,
            value=30,
            measurement_scope=MeasurementScope.AGGREGATE,
            layer=AbstractionLayer.FRAMEWORK,
        ),
    )
    trace = materialize_trace(TRACE_ID, signals)

    result = UsageExtractor().extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert _summary(result, Semantic.LLM_TOKENS_INPUT).value == 30
    assert _summary(result, Semantic.LLM_TOKENS_OUTPUT).value == 10
    assert _summary(result, Semantic.LLM_REQUEST_COUNT).value == 2
    assert len(_direct(result, Semantic.LLM_TOKENS_INPUT)) == 2
    assert all(
        item.tags["abp.abstraction_layer"] == "client"
        for item in _direct(result, Semantic.LLM_TOKENS_INPUT)
    )
    assert _summary(result, Semantic.LLM_MODEL_REQUESTED).value == "requested-a"
    assert _summary(result, Semantic.LLM_MODEL_RESPONSE).value == "served-a"
    assert _summary(result, Semantic.LLM_PROVIDER_NAME).value == "provider-a"
    assert not result.diagnostics
    assert all(item.semantic_type != Semantic.MONEY_COST for item in result.observations)
    span_result = SpanExtractor().extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )
    assert _summary(span_result, Semantic.OPERATION_COUNT).value == 2


def test_usage_extractor_reports_unresolved_and_authority_resolved_conflicts() -> None:
    ambiguous_trace = materialize_trace(
        TRACE_ID,
        (
            *_span(
                CLIENT_ONE,
                1,
                5,
                layer=AbstractionLayer.CLIENT,
                kind=KnownSpanKind.LLM,
                attributes={"logical_operation_id": "same"},
                usage={"input_tokens": 10},
            ),
            *_span(
                CLIENT_TWO,
                2,
                6,
                layer=AbstractionLayer.CLIENT,
                kind=KnownSpanKind.LLM,
                attributes={"logical_operation_id": "same"},
                usage={"input_tokens": 11},
            ),
        ),
    )

    ambiguous = UsageExtractor().extract(
        ambiguous_trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert not [
        item for item in ambiguous.observations if item.semantic_type == Semantic.LLM_TOKENS_INPUT
    ]
    assert "ambiguous_direct_measurement" in {
        diagnostic.code for diagnostic in ambiguous.diagnostics
    }

    resolved_trace = materialize_trace(
        TRACE_ID,
        (
            *_span(
                CLIENT_ONE,
                1,
                5,
                layer=AbstractionLayer.CLIENT,
                kind=KnownSpanKind.LLM,
                attributes={"logical_operation_id": "same", "usage_authority": "provider"},
                usage={"input_tokens": 10},
            ),
            *_span(
                CLIENT_TWO,
                2,
                6,
                layer=AbstractionLayer.CLIENT,
                kind=KnownSpanKind.LLM,
                attributes={"logical_operation_id": "same", "usage_authority": "estimated"},
                usage={"input_tokens": 11},
            ),
        ),
    )
    resolved = UsageExtractor().extract(
        resolved_trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert _summary(resolved, Semantic.LLM_TOKENS_INPUT).value == 10
    assert "direct_measurement_resolved" in {diagnostic.code for diagnostic in resolved.diagnostics}


def test_usage_extractor_diagnoses_aggregate_mismatch_on_selected_layer() -> None:
    signals = (
        *_span(
            CLIENT_ONE,
            1,
            10,
            layer=AbstractionLayer.CLIENT,
            kind=KnownSpanKind.LLM,
            usage={"input_tokens": 10},
        ),
        Measurement(
            trace_id=TRACE_ID,
            span_id=CLIENT_ONE,
            emitted_at=NOW,
            monotonic_ns=9,
            sequence=99,
            scope=_scope(AbstractionLayer.CLIENT),
            name="input_total",
            semantic_type=Semantic.LLM_TOKENS_INPUT,
            value=12,
            measurement_scope=MeasurementScope.AGGREGATE,
            layer=AbstractionLayer.CLIENT,
        ),
    )

    result = UsageExtractor().extract(
        materialize_trace(TRACE_ID, signals),
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    mismatch = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "aggregate_measurement_mismatch"
    )
    assert mismatch.details == {"aggregate": 12, "direct_total": 10}


def test_span_extractor_derives_graph_workflow_message_and_reference_evidence() -> None:
    artifact = EvidenceRef(kind=ReferenceKind.ARTIFACT, id="artifact")
    prompt = EvidenceRef(kind=ReferenceKind.PROMPT, id="prompt", version="v1")
    retry = Link(
        trace_id=TRACE_ID,
        span_id=CLIENT_TWO,
        emitted_at=NOW,
        monotonic_ns=72,
        sequence=72,
        scope=_scope(AbstractionLayer.APPLICATION),
        relation=LinkRelation.RETRY_OF,
        target=LinkTarget(trace_id=TRACE_ID, span_id=CLIENT_ONE),
    )
    signals = (
        *_span(ROOT, 0, 100, kind=KnownSpanKind.WORKFLOW),
        *_span(
            FRAMEWORK,
            10,
            90,
            parent=ROOT,
            kind=KnownSpanKind.TOOL,
            attributes={"arguments": {"query": "x"}},
        ),
        *_span(
            CLIENT_ONE,
            20,
            60,
            parent=ROOT,
            kind=KnownSpanKind.VALIDATION,
            status=SpanStatus.ERROR,
            reason=EndReason.FAILED,
        ),
        *_span(
            CLIENT_TWO,
            65,
            95,
            parent=ROOT,
            kind=KnownSpanKind.VALIDATION,
        ),
        *_span(
            OTHER,
            30,
            80,
            parent=ROOT,
            kind=KnownSpanKind.APPROVAL,
            status=SpanStatus.UNSET,
            reason=EndReason.DEFERRED,
        ),
        retry,
        Event(
            trace_id=TRACE_ID,
            span_id=ROOT,
            emitted_at=NOW,
            monotonic_ns=5,
            sequence=5,
            scope=_scope(AbstractionLayer.APPLICATION),
            name="input_messages",
            semantic_type=Semantic.MESSAGE_INPUT,
            body=[{"role": "user"}, {"role": "system"}],
        ),
        Event(
            trace_id=TRACE_ID,
            span_id=ROOT,
            emitted_at=NOW,
            monotonic_ns=96,
            sequence=96,
            scope=_scope(AbstractionLayer.APPLICATION),
            name="output_messages",
            semantic_type=Semantic.MESSAGE_OUTPUT,
            body=[{"role": "assistant"}, {"role": "tool"}, {"role": "assistant"}],
        ),
        Reference(
            trace_id=TRACE_ID,
            span_id=ROOT,
            emitted_at=NOW,
            monotonic_ns=6,
            sequence=6,
            scope=_scope(AbstractionLayer.APPLICATION),
            reference=artifact,
        ),
        Reference(
            trace_id=TRACE_ID,
            emitted_at=NOW,
            monotonic_ns=7,
            sequence=7,
            scope=_scope(AbstractionLayer.APPLICATION),
            reference=prompt,
        ),
    )

    result = SpanExtractor().extract(
        materialize_trace(TRACE_ID, signals),
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert _summary(result, Semantic.OPERATION_COUNT).value == 5
    assert _summary(result, Semantic.OPERATION_DEPTH_MAX).value == 2
    assert _summary(result, Semantic.OPERATION_FAN_OUT_MAX).value == 4
    assert _summary(result, Semantic.TIME_CRITICAL_PATH).value == pytest.approx(1e-7)
    assert _summary(result, Semantic.OPERATION_PARALLELISM).value == pytest.approx(2.0)
    assert _summary(result, Semantic.OPERATION_RETRY_COUNT).value == 1
    assert _summary(result, Semantic.OPERATION_RETRY_RECOVERED_COUNT).value == 1
    assert _summary(result, Semantic.OPERATION_FIRST_ATTEMPT_SUCCESS).value == 0
    assert _summary(result, Semantic.VALIDATION_FAILURE_COUNT).value == 1
    assert _summary(result, Semantic.VALIDATION_FAILURE_RATE).value == 0.5
    assert _summary(result, Semantic.APPROVAL_COUNT).value == 1
    assert _summary(result, Semantic.APPROVAL_WAIT).value == pytest.approx(5e-8)
    assert _summary(result, Semantic.TOOL_CALL_COUNT).value == 1
    assert _summary(result, Semantic.TOOL_CALL_SUCCESS_COUNT).value == 1
    assert _summary(result, Semantic.TOOL_CALL_FAILURE_COUNT).value == 0
    assert _summary(result, Semantic.TOOL_CALL_ARGUMENTS_PRESENT_COUNT).value == 1
    assert _summary(result, Semantic.MESSAGE_INPUT_COUNT).value == 2
    assert _summary(result, Semantic.MESSAGE_OUTPUT_COUNT).value == 3
    assert _summary(result, Semantic.MESSAGE_GROWTH).value == 1
    assert _summary(result, Semantic.ARTIFACT_REFERENCE_COUNT).value == 1
    assert _summary(result, Semantic.ASSET_REFERENCE_COUNT).value == 1
    assert not result.diagnostics


def test_span_extractor_handles_empty_partial_and_cyclic_traces_without_false_failures() -> None:
    empty = SpanExtractor().extract(
        Trace(trace_id=TRACE_ID),
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )
    assert _summary(empty, Semantic.OPERATION_COUNT).value == 0
    assert _summary(empty, Semantic.OPERATION_DEPTH_MAX).value == 0
    assert not [
        item for item in empty.observations if item.semantic_type == Semantic.TIME_CRITICAL_PATH
    ]

    partial_trace = materialize_trace(
        TRACE_ID,
        (
            SpanStart(
                trace_id=TRACE_ID,
                span_id=ROOT,
                parent_span_id=OTHER,
                emitted_at=NOW,
                monotonic_ns=1,
                sequence=1,
                scope=_scope(AbstractionLayer.APPLICATION),
                operation="partial",
            ),
            SpanStart(
                trace_id=TRACE_ID,
                span_id=OTHER,
                parent_span_id=ROOT,
                emitted_at=NOW,
                monotonic_ns=2,
                sequence=2,
                scope=_scope(AbstractionLayer.APPLICATION),
                operation="cycle",
            ),
        ),
    )
    partial = SpanExtractor().extract(
        partial_trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert _summary(partial, Semantic.OPERATION_INCOMPLETE_COUNT).value == 2
    assert "incomplete_trace_work" in {diagnostic.code for diagnostic in partial.diagnostics}
    assert _summary(partial, Semantic.APPROVAL_COUNT).value == 0
    assert _summary(partial, Semantic.TOOL_CALL_FAILURE_COUNT).value == 0


def test_span_extractor_uses_event_evidence_and_link_graph_without_fan_out_duplication() -> None:
    delegation = Link(
        trace_id=TRACE_ID,
        span_id=ROOT,
        emitted_at=NOW,
        monotonic_ns=2,
        sequence=20,
        scope=_scope(AbstractionLayer.APPLICATION),
        relation=LinkRelation.DELEGATION,
        target=LinkTarget(trace_id=TRACE_ID, span_id=OTHER),
    )
    duplicate_fan_out = delegation.model_copy(
        update={
            "signal_id": "12345678-1234-4234-8234-123456789abc",
            "relation": LinkRelation.FAN_OUT,
            "target": LinkTarget(trace_id=TRACE_ID, span_id=CLIENT_ONE),
            "sequence": 21,
        }
    )
    signals = (
        *_span(ROOT, 1, 10),
        *_span(CLIENT_ONE, 3, 8, parent=ROOT),
        *_span(OTHER, 2, 9),
        delegation,
        duplicate_fan_out,
        Event(
            trace_id=TRACE_ID,
            span_id=ROOT,
            emitted_at=NOW,
            monotonic_ns=4,
            sequence=22,
            scope=_scope(AbstractionLayer.APPLICATION),
            name="retry",
            semantic_type="operation.retry",
        ),
        Event(
            trace_id=TRACE_ID,
            span_id=ROOT,
            emitted_at=NOW,
            monotonic_ns=5,
            sequence=23,
            scope=_scope(AbstractionLayer.APPLICATION),
            name="validation_failure",
            semantic_type="validation.failure",
        ),
        Event(
            trace_id=TRACE_ID,
            span_id=ROOT,
            emitted_at=NOW,
            monotonic_ns=6,
            sequence=24,
            scope=_scope(AbstractionLayer.APPLICATION),
            name="approval_requested",
            semantic_type="approval.requested",
        ),
    )

    result = SpanExtractor().extract(
        materialize_trace(TRACE_ID, signals),
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert _summary(result, Semantic.OPERATION_DEPTH_MAX).value == 2
    assert _summary(result, Semantic.OPERATION_FAN_OUT_MAX).value == 1
    assert _summary(result, Semantic.OPERATION_RETRY_COUNT).value == 1
    assert _summary(result, Semantic.OPERATION_RETRY_RECOVERED_COUNT).value == 0
    assert _summary(result, Semantic.VALIDATION_COUNT).value == 1
    assert _summary(result, Semantic.VALIDATION_FAILURE_COUNT).value == 1
    assert _summary(result, Semantic.VALIDATION_FAILURE_RATE).value == 1
    assert _summary(result, Semantic.APPROVAL_COUNT).value == 1
    assert _summary(result, Semantic.APPROVAL_WAIT).value == 0


def test_composite_extractor_merges_duplicate_trace_diagnostics_once() -> None:
    diagnostic_trace = materialize_trace(
        TRACE_ID,
        (
            SpanStart(
                trace_id=TRACE_ID,
                span_id=ROOT,
                emitted_at=NOW,
                monotonic_ns=1,
                sequence=1,
                scope=_scope(AbstractionLayer.APPLICATION),
                operation="partial",
            ),
        ),
    )
    extractor = CompositeExtractor(SignalExtractor(), UsageExtractor())

    result = extractor.extract(
        diagnostic_trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert extractor.name == "abp.composite"
    assert extractor.version == "abp.signals@2+abp.llm_usage@1"
    assert len([item for item in result.diagnostics if item.code == "missing_span_end"]) == 1
    assert CompositeExtractor().name == "abp.default"


def test_extractors_cover_explicit_usage_source_models_and_all_reference_locations() -> None:
    error = EvidenceRef(kind=ReferenceKind.ERROR, id="error")
    output = EvidenceRef(kind=ReferenceKind.ARTIFACT, id="output")
    event_reference = EvidenceRef(kind=ReferenceKind.CUSTOM, id="event")
    first_start, first_end = _span(
        CLIENT_ONE,
        1,
        10,
        layer=AbstractionLayer.CLIENT,
        kind=KnownSpanKind.LLM,
        attributes={"logical_operation_id": "same"},
        usage={"requests": 1, "input_tokens": "unavailable", "output_tokens": True},
    )
    first_start = first_start.model_copy(
        update={"source_attributes": {"response_model": "source-model"}}
    )
    first_end = first_end.model_copy(update={"output_reference": output, "errors": (error,)})
    second_start, second_end = _span(
        CLIENT_TWO,
        2,
        9,
        layer=AbstractionLayer.CLIENT,
        kind=KnownSpanKind.LLM,
        attributes={"logical_operation_id": "same", "response_model": "source-model"},
    )
    third_start, third_end = _span(
        OTHER,
        3,
        8,
        layer=AbstractionLayer.CLIENT,
        kind=KnownSpanKind.LLM,
        attributes={"response_model": "other-model"},
    )
    tool_start, tool_end = _span(
        FRAMEWORK,
        3,
        7,
        kind=KnownSpanKind.TOOL,
    )
    signals = (
        first_start,
        first_end,
        second_start,
        second_end,
        third_start,
        third_end,
        tool_start,
        tool_end,
        Measurement(
            trace_id=TRACE_ID,
            span_id=CLIENT_ONE,
            emitted_at=NOW,
            monotonic_ns=4,
            sequence=40,
            scope=_scope(AbstractionLayer.CLIENT),
            source=SourceProvenance(system="provider.sdk", key="usage.input_tokens"),
            name="input",
            semantic_type=Semantic.LLM_TOKENS_INPUT,
            value=10,
            measurement_scope=MeasurementScope.DIRECT,
            layer=AbstractionLayer.CLIENT,
        ),
        Measurement(
            trace_id=TRACE_ID,
            span_id=CLIENT_ONE,
            emitted_at=NOW,
            monotonic_ns=5,
            sequence=41,
            scope=_scope(AbstractionLayer.CLIENT),
            name="input_total",
            semantic_type=Semantic.LLM_TOKENS_INPUT,
            value=10,
            measurement_scope=MeasurementScope.AGGREGATE,
            layer=AbstractionLayer.CLIENT,
        ),
        Measurement(
            trace_id=TRACE_ID,
            span_id=CLIENT_ONE,
            emitted_at=NOW,
            monotonic_ns=6,
            sequence=42,
            scope=_scope(AbstractionLayer.CLIENT),
            name="quality",
            semantic_type=Semantic.QUALITY_SCORE,
            value=1,
            measurement_scope=MeasurementScope.DIRECT,
            layer=AbstractionLayer.CLIENT,
        ),
        Event(
            trace_id=TRACE_ID,
            span_id=FRAMEWORK,
            emitted_at=NOW,
            monotonic_ns=4,
            sequence=43,
            scope=_scope(AbstractionLayer.APPLICATION),
            name="arguments",
            semantic_type=Semantic.TOOL_CALL_ARGUMENTS,
            body={"query": "x"},
        ),
        Event(
            trace_id=TRACE_ID,
            span_id=FRAMEWORK,
            emitted_at=NOW,
            monotonic_ns=5,
            sequence=44,
            scope=_scope(AbstractionLayer.APPLICATION),
            name="event_reference",
            semantic_type=Semantic.ARTIFACT_CONTENT,
            reference=event_reference,
        ),
    )
    trace = materialize_trace(TRACE_ID, signals)

    usage = UsageExtractor().extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )
    spans = SpanExtractor().extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )
    combined = CompositeExtractor(SignalExtractor(), SpanExtractor()).extract(
        trace,
        registry=DEFAULT_SEMANTIC_REGISTRY,
        context=_context(),
    )

    assert _summary(usage, Semantic.LLM_TOKENS_INPUT).value == 10
    assert _summary(usage, Semantic.LLM_REQUEST_COUNT).value == 2
    assert not [
        item
        for item in usage.observations
        if item.semantic_type == Semantic.LLM_MODEL_RESPONSE
        and item.tags.get("abp.summary") is True
    ]
    assert not [
        diagnostic
        for diagnostic in usage.diagnostics
        if diagnostic.code == "aggregate_measurement_mismatch"
    ]
    duration = next(
        item
        for item in spans.observations
        if item.name == "span.duration" and item.span_id == CLIENT_ONE
    )
    assert duration.tags["abp.logical_operation_id"] == "same"
    assert _summary(spans, Semantic.TOOL_CALL_ARGUMENTS_PRESENT_COUNT).value == 1
    assert {reference.id for reference in combined.references} >= {
        "error",
        "output",
        "event",
    }


def _span(
    span_id: str,
    start: int,
    end: int,
    *,
    parent: str | None = None,
    layer: AbstractionLayer = AbstractionLayer.APPLICATION,
    kind: KnownSpanKind = KnownSpanKind.CUSTOM,
    attributes: dict[str, SerializedValue] | None = None,
    usage: dict[str, SerializedValue] | None = None,
    status: SpanStatus = SpanStatus.OK,
    reason: EndReason = EndReason.COMPLETED,
) -> tuple[SpanStart, SpanEnd]:
    scope = _scope(layer)
    return (
        SpanStart(
            trace_id=TRACE_ID,
            span_id=span_id,
            parent_span_id=parent,
            emitted_at=NOW,
            monotonic_ns=start,
            sequence=start,
            scope=scope,
            operation=f"operation-{span_id[-1]}",
            kind=kind,
            attributes={} if attributes is None else attributes,
        ),
        SpanEnd(
            trace_id=TRACE_ID,
            span_id=span_id,
            emitted_at=NOW,
            monotonic_ns=end,
            sequence=end,
            scope=scope,
            status=status,
            reason=reason,
            usage={} if usage is None else usage,
        ),
    )


def _scope(layer: AbstractionLayer) -> InstrumentationScope:
    return InstrumentationScope(
        instrumentor_name=f"test.{layer.value}",
        instrumentor_version="1",
        package_name="test",
        package_version="1",
        mechanism=CaptureMechanism.MANUAL,
        layer=layer,
    )


def _context() -> ExtractionContext:
    return ExtractionContext(
        run_id="run",
        benchmark_id="benchmark",
        experiment_id="experiment",
        case_id="case",
        variant_id="variant",
    )


def _summary(result: ExtractionResult, semantic_type: str) -> Observation:
    return next(
        item
        for item in result.observations
        if item.semantic_type == semantic_type and item.tags.get("abp.summary") is True
    )


def _direct(result: ExtractionResult, semantic_type: str) -> list[Observation]:
    return [
        item
        for item in result.observations
        if item.semantic_type == semantic_type
        and item.tags.get("abp.measurement_scope") == "direct"
    ]
