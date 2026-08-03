from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid1

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from autobench import Direction, ObservationRole
from autobench.protocol import (
    PROTOCOL_VERSION,
    AbstractionLayer,
    CaptureLevel,
    CaptureMechanism,
    EndReason,
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
    SerializedValue,
    Signal,
    SignalId,
    SourceProvenance,
    SpanEnd,
    SpanId,
    SpanStart,
    SpanStatus,
    TraceId,
    new_signal_id,
    new_span_id,
    new_trace_id,
)

TRACE_ID = "1" * 32
SPAN_ID = "2" * 16
TARGET_TRACE_ID = "3" * 32
TARGET_SPAN_ID = "4" * 16


def test_all_signal_variants_round_trip_through_json_and_yaml() -> None:
    emitted_at = datetime(2026, 8, 3, 12, 30, tzinfo=timezone(timedelta(hours=3)))
    scope = InstrumentationScope(
        instrumentor_name="autobench.manual",
        instrumentor_version="0.1.0",
        package_name="autobench",
        package_version="0.1.0",
        mechanism=CaptureMechanism.MANUAL,
        layer=AbstractionLayer.APPLICATION,
        source_convention="autobench.manual",
        source_convention_version="1",
    )
    execution = ExecutionRef(
        benchmark_id="benchmark",
        experiment_id="experiment",
        run_id="run",
        case_id="case",
        variant_id="variant",
    )
    source = SourceProvenance(
        system="autobench.manual",
        key="manual.value",
        convention_version="1",
        instrumentor="autobench.manual",
        instrumented_library_version="0.1.0",
    )
    artifact = EvidenceRef(
        kind=ReferenceKind.ARTIFACT,
        id="artifact_1",
        media_type="application/json",
    )
    signals: tuple[Signal, ...] = (
        SpanStart(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=emitted_at,
            monotonic_ns=100,
            sequence=1,
            execution=execution,
            scope=scope,
            source=source,
            operation="workflow.run",
            kind="domain.custom_span",
            attributes={"attempt": 1},
            source_attributes={"source.key": "value"},
            links=(LinkTarget(trace_id=TARGET_TRACE_ID, span_id=TARGET_SPAN_ID),),
            capture=CaptureLevel.HASH,
        ),
        SpanEnd(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=emitted_at,
            monotonic_ns=200,
            sequence=2,
            execution=execution,
            scope=scope,
            source=source,
            output_reference=artifact,
            status=SpanStatus.OK,
            reason=EndReason.COMPLETED,
            usage={"tokens": 5},
            stream={"chunks": 2},
        ),
        Event(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=emitted_at,
            monotonic_ns=120,
            sequence=3,
            execution=execution,
            scope=scope,
            source=source,
            name="tool.requested",
            semantic_type="tool.requested",
            body={"tool": "search"},
        ),
        Measurement(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=emitted_at,
            monotonic_ns=150,
            sequence=4,
            execution=execution,
            scope=scope,
            source=source,
            name="latency",
            semantic_type="domain.custom_latency",
            value=12.5,
            unit="ms",
            direction=Direction.MINIMIZE,
            role=ObservationRole.DIAGNOSTIC,
            measurement_scope=MeasurementScope.DIRECT,
            layer=AbstractionLayer.APPLICATION,
        ),
        Link(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=emitted_at,
            monotonic_ns=160,
            sequence=5,
            execution=execution,
            scope=scope,
            source=source,
            relation=LinkRelation.DELEGATION,
            target=LinkTarget(run_id="other_run"),
        ),
        Reference(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=emitted_at,
            monotonic_ns=170,
            sequence=6,
            execution=execution,
            scope=scope,
            source=source,
            semantic_type="artifact.created",
            name="raw response",
            reference=artifact,
        ),
    )
    adapter = TypeAdapter(Signal)

    for signal in signals:
        payload = signal.model_dump(mode="json")
        from_json = adapter.validate_json(json.dumps(payload))
        from_yaml = adapter.validate_python(yaml.safe_load(yaml.safe_dump(payload)))
        assert from_json == signal
        assert from_yaml == signal
        assert from_json.emitted_at == datetime(2026, 8, 3, 9, 30, tzinfo=UTC)


def test_id_generators_and_validators_enforce_wire_compatible_shapes() -> None:
    trace_id = new_trace_id()
    span_id = new_span_id()
    signal_id = new_signal_id()

    assert TypeAdapter(TraceId).validate_python(trace_id) == trace_id
    assert TypeAdapter(SpanId).validate_python(span_id) == span_id
    assert TypeAdapter(SignalId).validate_python(signal_id) == signal_id
    assert len(trace_id) == 32
    assert len(span_id) == 16

    for invalid in ("0" * 32, "A" * 32, "1" * 31, "g" * 32):
        with pytest.raises(ValidationError):
            TypeAdapter(TraceId).validate_python(invalid)
    for invalid in ("0" * 16, "A" * 16, "1" * 15, "g" * 16):
        with pytest.raises(ValidationError):
            TypeAdapter(SpanId).validate_python(invalid)
    for invalid in (str(uuid1()), "0" * 36, "not-a-uuid"):
        with pytest.raises(ValidationError):
            TypeAdapter(SignalId).validate_python(invalid)


def test_serialized_values_accept_only_finite_json_boundaries() -> None:
    adapter = TypeAdapter(SerializedValue)
    accepted = (
        None,
        "text",
        True,
        3,
        1.5,
        [1, {"nested": False}],
        {"items": ["a", "b"]},
    )
    for value in accepted:
        assert adapter.validate_python(value) == value

    rejected = (
        (1, 2),
        b"bytes",
        {1: "not a string key"},
        {"set": {1}},
        float("nan"),
        float("inf"),
    )
    for value in rejected:
        with pytest.raises(ValidationError):
            adapter.validate_python(value)


def test_protocol_models_are_frozen_and_reject_unsupported_versions() -> None:
    signal = SpanStart(
        trace_id=TRACE_ID,
        span_id=SPAN_ID,
        emitted_at=datetime.now(UTC),
        monotonic_ns=1,
        sequence=1,
        scope=manual_scope(),
        operation="task",
    )

    with pytest.raises(ValidationError, match="frozen"):
        signal.operation = "changed"

    payload = signal.model_dump(mode="python")
    payload["protocol_version"] = PROTOCOL_VERSION + 1
    with pytest.raises(ValidationError, match="Input should be 1"):
        SpanStart.model_validate(payload)


def test_scope_provenance_and_execution_references_serialize_without_tag_bags() -> None:
    scope = manual_scope()
    source = SourceProvenance(system="otel.genai", key="gen_ai.tool.call.id")
    execution = ExecutionRef(
        benchmark_id="benchmark",
        experiment_id="experiment",
        run_id="run",
    )

    assert scope.model_dump(mode="json") == {
        "instrumentor_name": "autobench.manual",
        "instrumentor_version": "0.1.0",
        "package_name": "autobench",
        "package_version": "0.1.0",
        "mechanism": "manual",
        "layer": "application",
        "source_convention": None,
        "source_convention_version": None,
    }
    assert source.model_dump(mode="json")["system"] == "otel.genai"
    assert execution.case_id is None
    assert execution.variant_id is None

    with pytest.raises(ValidationError, match="requires source_convention"):
        InstrumentationScope(
            instrumentor_name="test",
            instrumentor_version="1",
            package_name="package",
            package_version="1",
            mechanism=CaptureMechanism.HOOK,
            layer=AbstractionLayer.FRAMEWORK,
            source_convention_version="1",
        )


def test_measurement_scope_and_value_are_strictly_typed() -> None:
    direct = measurement(MeasurementScope.DIRECT, value=3)
    aggregate = measurement(MeasurementScope.AGGREGATE, value=True)

    assert direct.measurement_scope is MeasurementScope.DIRECT
    assert aggregate.measurement_scope is MeasurementScope.AGGREGATE
    assert aggregate.value is True

    payload = direct.model_dump(mode="python")
    payload["measurement_scope"] = "recursive"
    with pytest.raises(ValidationError):
        Measurement.model_validate(payload)
    payload = direct.model_dump(mode="python")
    payload["value"] = "3"
    with pytest.raises(ValidationError):
        Measurement.model_validate(payload)
    payload = direct.model_dump(mode="python")
    payload["layer"] = AbstractionLayer.CLIENT
    with pytest.raises(ValidationError, match="must match instrumentation scope layer"):
        Measurement.model_validate(payload)


def test_link_targets_require_one_well_formed_target() -> None:
    assert LinkTarget(trace_id=TARGET_TRACE_ID, span_id=TARGET_SPAN_ID).span_id == TARGET_SPAN_ID
    assert LinkTarget(reference=EvidenceRef(kind=ReferenceKind.ASSET, id="prompt:v2")).reference

    with pytest.raises(ValidationError, match="exactly one"):
        LinkTarget()
    with pytest.raises(ValidationError, match="exactly one"):
        LinkTarget(trace_id=TARGET_TRACE_ID, run_id="run")
    with pytest.raises(ValidationError, match="span_id requires trace_id"):
        LinkTarget(span_id=TARGET_SPAN_ID, run_id="run")


def test_captured_values_and_events_cannot_inline_and_reference_together() -> None:
    artifact = EvidenceRef(kind=ReferenceKind.ARTIFACT, id="artifact")

    with pytest.raises(ValidationError, match="mutually exclusive"):
        SpanEnd(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=datetime.now(UTC),
            monotonic_ns=1,
            sequence=1,
            scope=manual_scope(),
            output={"ok": True},
            output_reference=artifact,
        )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        Event(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=datetime.now(UTC),
            monotonic_ns=1,
            sequence=1,
            scope=manual_scope(),
            name="artifact.created",
            semantic_type="artifact.created",
            body={"ok": True},
            reference=artifact,
        )

    span_end = SpanEnd(
        trace_id=TRACE_ID,
        span_id=SPAN_ID,
        emitted_at=datetime.now(UTC),
        monotonic_ns=1,
        sequence=1,
        scope=manual_scope(),
        output={"ok": True},
    )
    assert span_end.output == {"ok": True}
    assert (
        Event(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=datetime.now(UTC),
            monotonic_ns=1,
            sequence=1,
            scope=manual_scope(),
            name="artifact.created",
            semantic_type="artifact.created",
            reference=artifact,
        ).reference
        == artifact
    )

    with pytest.raises(ValidationError, match="errors require error references"):
        SpanEnd(
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            emitted_at=datetime.now(UTC),
            monotonic_ns=1,
            sequence=1,
            scope=manual_scope(),
            errors=(artifact,),
        )
    error = EvidenceRef(kind=ReferenceKind.ERROR, id="error_1")
    assert SpanEnd(
        trace_id=TRACE_ID,
        span_id=SPAN_ID,
        emitted_at=datetime.now(UTC),
        monotonic_ns=1,
        sequence=1,
        scope=manual_scope(),
        errors=(error,),
    ).errors == (error,)


def test_signal_discriminator_preserves_reference_type_and_rejects_unknown_signals() -> None:
    signal = Reference(
        trace_id=TRACE_ID,
        emitted_at=datetime.now(UTC),
        monotonic_ns=1,
        sequence=1,
        scope=manual_scope(),
        reference=EvidenceRef(kind=ReferenceKind.OUTPUT_SCHEMA, id="schema:v3", version="3"),
    )
    payload = signal.model_dump(mode="json")
    parsed = TypeAdapter(Signal).validate_python(payload)

    assert isinstance(parsed, Reference)
    assert parsed.reference.kind is ReferenceKind.OUTPUT_SCHEMA
    with pytest.raises(ValidationError):
        TypeAdapter(Signal).validate_python({**payload, "type": "unknown"})


def manual_scope() -> InstrumentationScope:
    return InstrumentationScope(
        instrumentor_name="autobench.manual",
        instrumentor_version="0.1.0",
        package_name="autobench",
        package_version="0.1.0",
        mechanism=CaptureMechanism.MANUAL,
        layer=AbstractionLayer.APPLICATION,
    )


def measurement(
    measurement_scope: MeasurementScope,
    *,
    value: bool | int | float,
) -> Measurement:
    return Measurement(
        trace_id=TRACE_ID,
        span_id=SPAN_ID,
        emitted_at=datetime.now(UTC),
        monotonic_ns=1,
        sequence=1,
        scope=manual_scope(),
        name="value",
        semantic_type="domain.value",
        value=value,
        measurement_scope=measurement_scope,
        layer=AbstractionLayer.APPLICATION,
    )
