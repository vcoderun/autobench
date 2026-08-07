from __future__ import annotations as _annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
from pydantic import ValidationError

from autobench import (
    AssetProvenance,
    AssetRepresentation,
    AssetUse,
    BenchmarkPlan,
    Case,
    Direction,
    EndReason,
    EnvironmentMetadata,
    ErrorRecord,
    EvaluationStatus,
    ExecutionCorrelation,
    ExperimentRecord,
    ExperimentStatus,
    ExperimentTermination,
    FactorValue,
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
    OTLPExportError,
    OTLPExportResult,
    OTLPSettings,
    RunRecord,
    RunStatus,
    ScoreRecord,
    Semantic,
    SourceSelector,
    SourceSnapshot,
    TaskStatus,
    export_otlp,
    export_record_otlp,
)
from autobench.cli import cli
from autobench.io import dump_yaml
from autobench.metrics.mappings import RetainedSourceFact
from autobench.protocol.signals import (
    AbstractionLayer,
    CaptureLevel,
    CaptureMechanism,
    Event,
    ExecutionRef,
    InstrumentationScope,
    Link,
    LinkRelation,
    LinkTarget,
    Measurement,
    MeasurementScope,
    Reference,
    SourceProvenance,
    SpanStatus,
)
from autobench.protocol.traces import Diagnostic, DiagnosticSeverity, SpanRecord, Trace
from autobench.protocol.values import EvidenceRef, ReferenceKind
from autobench.records.views import experiment_record_to_yaml_view, run_record_to_yaml_view

TRACE_ID = "1" * 32
PARTIAL_TRACE_ID = "2" * 32
ROOT_SPAN_ID = "a" * 16
CHILD_SPAN_ID = "b" * 16
LINK_TRACE_ID = "3" * 32
LINK_SPAN_ID = "c" * 16


class CapturingExporter(SpanExporter):
    def __init__(
        self,
        *,
        result: SpanExportResult = SpanExportResult.SUCCESS,
        export_error: RuntimeError | None = None,
        shutdown_error: RuntimeError | None = None,
    ) -> None:
        self.result = result
        self.export_error = export_error
        self.shutdown_error = shutdown_error
        self.spans: list[ReadableSpan] = []
        self.shutdown_calls = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        if self.export_error is not None:
            raise self.export_error
        return self.result

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


def test_otlp_export_preserves_hierarchy_semantics_and_immutable_records() -> None:
    experiment, runs = _evidence_records()
    experiment = experiment.model_copy(update={"manifest_path": "manifest.yaml"})
    before_experiment = experiment.model_dump(mode="python")
    before_runs = tuple(run.model_dump(mode="python") for run in runs)
    exporter = CapturingExporter()

    result = export_otlp(
        experiment,
        runs,
        settings=OTLPSettings(
            service_name="routing-benchmark",
            service_namespace="evaluation",
            resource_attributes={
                "deployment.environment": "test",
                "service.name": "must-not-override",
                "autobench.protocol": "must-not-override",
            },
        ),
        exporter=exporter,
    )

    assert result == OTLPExportResult(
        experiment_id="experiment-1",
        benchmark_id="routing",
        record_version=experiment.record_version,
        run_count=3,
        trace_count=2,
        abp_span_count=2,
        exported_span_count=8,
        partial_run_count=1,
        partial_trace_count=1,
        endpoint="OTEL_EXPORTER_OTLP_TRACES_ENDPOINT/environment default",
    )
    assert experiment.model_dump(mode="python") == before_experiment
    assert tuple(run.model_dump(mode="python") for run in runs) == before_runs
    assert exporter.shutdown_calls == 0

    spans = {span.name: span for span in exporter.spans}
    experiment_span = spans["autobench.experiment routing"]
    failed_run_span = spans["autobench.run failed-case / baseline"]
    trace_span = spans[f"autobench.trace {TRACE_ID[:8]}"]
    root_span = spans["agent.run"]
    child_span = spans["tool.lookup"]

    assert experiment_span.parent is None
    assert failed_run_span.parent is not None
    assert experiment_span.context is not None
    assert failed_run_span.parent.span_id == experiment_span.context.span_id
    assert trace_span.parent is not None
    assert failed_run_span.context is not None
    assert trace_span.parent.span_id == failed_run_span.context.span_id
    assert root_span.parent is not None
    assert trace_span.context is not None
    assert root_span.parent.span_id == trace_span.context.span_id
    assert child_span.parent is not None
    assert root_span.context is not None
    assert child_span.parent.span_id == root_span.context.span_id

    assert experiment_span.status.status_code is StatusCode.OK
    assert failed_run_span.status.status_code is StatusCode.ERROR
    assert root_span.status.status_code is StatusCode.ERROR
    assert child_span.status.status_code is StatusCode.OK
    assert root_span.attributes is not None
    assert root_span.attributes["autobench.abp.output_omitted"] is True
    assert root_span.attributes["autobench.abp.attribute.autobench.run.id"] == "collision"
    assert root_span.attributes["autobench.source.attribute.raw.model"] == "provider-model"
    assert root_span.attributes["autobench.usage.input_tokens"] == 12
    assert root_span.attributes["autobench.stream.chunks"] == 2
    assert experiment_span.resource.attributes["service.name"] == "routing-benchmark"
    assert experiment_span.resource.attributes["service.namespace"] == "evaluation"
    assert experiment_span.resource.attributes["deployment.environment"] == "test"
    assert experiment_span.resource.attributes["autobench.protocol"] == "abp"
    assert failed_run_span.attributes is not None
    assert failed_run_span.attributes["autobench.record.run_path"] == "runs/failed.yaml"
    assert experiment_span.attributes is not None
    assert experiment_span.attributes["autobench.record.manifest_path"] == "manifest.yaml"
    assert len(root_span.links) == 2
    link_relations: set[str] = set()
    for link in root_span.links:
        assert link.attributes is not None
        relation = link.attributes["autobench.abp.relation"]
        assert isinstance(relation, str)
        link_relations.add(relation)
    assert link_relations == {"retry_of", "start_link"}

    termination_event = next(
        event
        for event in experiment_span.events
        if event.name == "autobench.experiment.termination"
    )
    termination_attributes = termination_event.attributes
    assert termination_attributes is not None
    assert termination_attributes["status"] == "completed"

    root_events = {event.name: event for event in root_span.events}
    message_attributes = root_events["agent.message"].attributes
    measurement_attributes = root_events["autobench.measurement"].attributes
    assert message_attributes is not None
    assert measurement_attributes is not None
    assert message_attributes["content_omitted"] is True
    assert measurement_attributes["semantic_type"] == (Semantic.LLM_TOKENS_INPUT)
    assert "autobench.abp.start_link" in root_events
    assert "autobench.abp.error_reference" in root_events
    assert "autobench.abp.output_reference" in {event.name for event in child_span.events}
    run_events = list(failed_run_span.events)
    event_observation = next(
        event
        for event in run_events
        if event.name == "autobench.observation"
        and event.attributes is not None
        and event.attributes.get("name") == "conversation"
    )
    event_observation_attributes = event_observation.attributes
    assert event_observation_attributes is not None
    assert event_observation_attributes["name"] == "conversation"
    assert event_observation_attributes["content_omitted"] is True
    score_event = next(event for event in run_events if event.name == "autobench.score")
    score_attributes = score_event.attributes
    assert score_attributes is not None
    assert "actual_value" not in score_attributes
    snapshot_event = next(
        event for event in run_events if event.name == "autobench.source.snapshot"
    )
    snapshot_attributes = snapshot_event.attributes
    assert snapshot_attributes is not None
    assert snapshot_attributes["fact_count"] == 1
    assert "facts" not in snapshot_attributes


def test_otlp_export_can_include_captured_content_and_handles_partial_cycles() -> None:
    experiment, runs = _evidence_records()
    root = runs[0].trace.spans[0] if runs[0].trace is not None else None
    child = runs[0].trace.spans[1] if runs[0].trace is not None else None
    trace = runs[0].trace
    assert root is not None and child is not None and trace is not None
    cyclic_trace = trace.model_copy(
        update={
            "partial": True,
            "started_at": datetime(2026, 8, 7, 9, 0),
            "spans": (
                root.model_copy(update={"parent_span_id": CHILD_SPAN_ID}),
                child.model_copy(update={"parent_span_id": ROOT_SPAN_ID}),
            ),
        }
    )
    cyclic_run = runs[0].model_copy(update={"trace": cyclic_trace})
    direct_experiment = experiment.model_copy(
        update={
            "run_count": 1,
            "passed_count": 0,
            "failed_count": 1,
            "cancelled_count": 0,
            "run_paths": (),
        }
    )
    exporter = CapturingExporter()

    result = export_otlp(
        direct_experiment,
        (cyclic_run,),
        settings=OTLPSettings(
            endpoint="http://collector:4318/v1/traces",
            include_captured_content=True,
        ),
        exporter=exporter,
    )

    assert result.exported_span_count == 5
    spans = {span.name: span for span in exporter.spans}
    root_span = spans["agent.run"]
    assert root_span.attributes is not None
    assert root_span.attributes["autobench.abp.output"] == '{"answer":"secret"}'
    message_event = next(event for event in root_span.events if event.name == "agent.message")
    message_attributes = message_event.attributes
    assert message_attributes is not None
    assert message_attributes["body"] == '{"text":"secret prompt"}'
    trace_span = spans[f"autobench.trace {TRACE_ID[:8]}"]
    assert "autobench.abp.parent_cycle" in {event.name for event in trace_span.events}
    run_span = spans["autobench.run failed-case / baseline"]
    score_event = next(event for event in run_span.events if event.name == "autobench.score")
    score_attributes = score_event.attributes
    assert score_attributes is not None
    assert score_attributes["actual_value"] == "billing"
    exception_event = next(event for event in run_span.events if event.name == "exception")
    exception_attributes = exception_event.attributes
    assert exception_attributes is not None
    assert exception_attributes["traceback"] == "trace"
    snapshot_event = next(
        event for event in run_span.events if event.name == "autobench.source.snapshot"
    )
    snapshot_attributes = snapshot_event.attributes
    assert snapshot_attributes is not None
    assert "facts" in snapshot_attributes


def test_otlp_record_loading_owned_exporter_and_failure_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, runs = _evidence_records()
    record_dir = tmp_path / "record"
    for run_path, run in zip(experiment.run_paths, runs, strict=True):
        path = record_dir / run_path
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_yaml(run_record_to_yaml_view(run), path)
    dump_yaml(experiment_record_to_yaml_view(experiment), record_dir / "experiment.yaml")

    injected = CapturingExporter()
    result = export_record_otlp(record_dir, exporter=injected)
    assert result.run_count == 3
    assert len(injected.spans) == result.exported_span_count

    from autobench.exporters import _otel

    created: list[CapturingExporter] = []
    received: list[tuple[str | None, str | None, dict[str, str] | None, float | None]] = []

    def exporter_factory(
        endpoint: str | None = None,
        certificate_file: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CapturingExporter:
        received.append((endpoint, certificate_file, headers, timeout))
        active = CapturingExporter()
        created.append(active)
        return active

    certificate = tmp_path / "ca.pem"
    certificate.write_text("test certificate", encoding="utf-8")
    monkeypatch.setattr(_otel, "OTLPSpanExporter", exporter_factory)
    owned_result = export_otlp(
        experiment,
        runs,
        settings=OTLPSettings(
            endpoint="https://collector.example/v1/traces",
            certificate_file=certificate,
            headers={"authorization": "test"},
            timeout_seconds=3,
        ),
    )
    assert owned_result.endpoint == "https://collector.example/v1/traces"
    assert received == [
        (
            "https://collector.example/v1/traces",
            str(certificate),
            {"authorization": "test"},
            3.0,
        )
    ]
    assert created[0].shutdown_calls == 1

    for exporter, message in (
        (CapturingExporter(result=SpanExportResult.FAILURE), "reported a failed export"),
        (CapturingExporter(export_error=RuntimeError("offline")), "export failed: offline"),
    ):
        with pytest.raises(OTLPExportError, match=message):
            export_otlp(experiment, runs, exporter=exporter)

    def shutdown_failure_factory(
        endpoint: str | None = None,
        certificate_file: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CapturingExporter:
        del endpoint, certificate_file, headers, timeout
        return CapturingExporter(shutdown_error=RuntimeError("shutdown offline"))

    monkeypatch.setattr(_otel, "OTLPSpanExporter", shutdown_failure_factory)
    with pytest.raises(OTLPExportError, match="shutdown failed: shutdown offline"):
        export_otlp(experiment, runs)

    def export_and_shutdown_failure_factory(
        endpoint: str | None = None,
        certificate_file: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CapturingExporter:
        del endpoint, certificate_file, headers, timeout
        return CapturingExporter(
            result=SpanExportResult.FAILURE,
            shutdown_error=RuntimeError("secondary shutdown failure"),
        )

    monkeypatch.setattr(_otel, "OTLPSpanExporter", export_and_shutdown_failure_factory)
    with pytest.raises(OTLPExportError, match="reported a failed export"):
        export_otlp(experiment, runs)


def test_otlp_export_maps_aborted_and_cancelled_experiment_lifecycle() -> None:
    experiment, runs = _evidence_records()
    one_run = (runs[0].model_copy(update={"errors": ()}),)
    base = experiment.model_copy(
        update={
            "run_count": 1,
            "passed_count": 0,
            "failed_count": 1,
            "cancelled_count": 0,
            "run_paths": (),
        }
    )
    aborted = base.model_copy(
        update={
            "termination": ExperimentTermination(
                status=ExperimentStatus.ABORTED,
                partial=True,
                error=ErrorRecord(
                    error_type="RuntimeError",
                    message="planner failed",
                    traceback="planner trace",
                ),
            )
        }
    )
    aborted_exporter = CapturingExporter()
    export_otlp(aborted, one_run, exporter=aborted_exporter)
    aborted_span = next(
        span for span in aborted_exporter.spans if span.name == "autobench.experiment routing"
    )
    assert aborted_span.status.status_code is StatusCode.ERROR
    assert aborted_span.status.description == "planner failed"
    termination_event = next(
        event for event in aborted_span.events if event.name == "autobench.experiment.termination"
    )
    termination_attributes = termination_event.attributes
    assert termination_attributes is not None
    assert "planner trace" not in str(termination_attributes["error"])
    assert "traceback_omitted" in str(termination_attributes["error"])

    cancelled = base.model_copy(
        update={
            "termination": ExperimentTermination(
                status=ExperimentStatus.CANCELLED,
                partial=True,
            )
        }
    )
    cancelled_exporter = CapturingExporter()
    cancelled_run = one_run[0].model_copy(update={"correlation": None})
    export_otlp(
        cancelled.model_copy(update={"correlation": None}),
        (cancelled_run,),
        exporter=cancelled_exporter,
    )
    cancelled_span = next(
        span for span in cancelled_exporter.spans if span.name == "autobench.experiment routing"
    )
    assert cancelled_span.status.status_code is StatusCode.UNSET


def test_otlp_export_maps_sparse_optional_and_duplicate_trace_evidence() -> None:
    experiment, runs = _evidence_records()
    source_trace = runs[0].trace
    assert source_trace is not None
    sparse_error = ErrorRecord(error_type="ValueError", message="no traceback")
    sparse_span = source_trace.spans[0].model_copy(
        update={
            "end_reason": None,
            "status": SpanStatus.UNSET,
            "links": (
                *source_trace.spans[0].links,
                source_trace.spans[0]
                .links[0]
                .model_copy(update={"target": LinkTarget(run_id="related-run")}),
            ),
        }
    )
    sparse_trace = source_trace.model_copy(
        update={
            "root_span_ids": (sparse_span.span_id,),
            "spans": (sparse_span, sparse_span),
        }
    )
    empty_correlation = ExecutionCorrelation()
    sparse_run = runs[0].model_copy(
        update={
            "trace": sparse_trace,
            "protocol_version": None,
            "semantic_registry_version": None,
            "errors": (sparse_error,),
            "error": sparse_error,
            "correlation": empty_correlation,
        }
    )
    sparse_plan = experiment.plan.model_copy(
        update={"dataset_id": None, "dataset_version": None, "dataset_hash": None}
    )
    sparse_experiment = experiment.model_copy(
        update={
            "plan": sparse_plan,
            "run_count": 1,
            "passed_count": 0,
            "failed_count": 1,
            "cancelled_count": 0,
            "run_paths": (),
            "spec_hash": None,
            "correlation": empty_correlation,
        }
    )
    exporter = CapturingExporter()

    export_otlp(sparse_experiment, (sparse_run,), exporter=exporter)

    trace_span = next(span for span in exporter.spans if span.name.startswith("autobench.trace"))
    assert "autobench.abp.parent_cycle" not in {event.name for event in trace_span.events}
    sparse_otlp_span = next(span for span in exporter.spans if span.name == "agent.run")
    assert sparse_otlp_span.status.status_code is StatusCode.UNSET
    assert len(sparse_otlp_span.links) == 2
    run_span = next(span for span in exporter.spans if span.name.startswith("autobench.run"))
    exception_event = next(event for event in run_span.events if event.name == "exception")
    assert exception_event.attributes is not None
    assert "traceback" not in exception_event.attributes


def test_otlp_validates_settings_record_contract_mapping_and_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, runs = _evidence_records()
    for payload, message in (
        ({"headers": {" ": "value"}}, "header names"),
        ({"resource_attributes": {" ": "value"}}, "attribute names"),
        ({"resource_attributes": {"bad": float("inf")}}, "finite number"),
        ({"endpoint": " "}, "must not be blank"),
    ):
        with pytest.raises(ValidationError, match=message):
            OTLPSettings.model_validate(payload)
    assert OTLPSettings(resource_attributes={"sampling.ratio": 0.5}).resource_attributes == {
        "sampling.ratio": 0.5
    }

    invalid_contracts = (
        (experiment, runs[:1], "declares 3 runs"),
        (experiment.model_copy(update={"run_count": 4}), (*runs, runs[0]), "Duplicate run"),
        (
            experiment,
            (runs[0].model_copy(update={"experiment_id": "other"}), *runs[1:]),
            "another experiment",
        ),
        (
            experiment,
            (runs[0].model_copy(update={"benchmark_id": "other"}), *runs[1:]),
            "another benchmark",
        ),
        (
            experiment,
            (runs[0].model_copy(update={"correlation": None}), *runs[1:]),
            "inconsistent execution correlation",
        ),
    )
    for active_experiment, active_runs, message in invalid_contracts:
        with pytest.raises(OTLPExportError, match=message):
            export_otlp(active_experiment, active_runs, exporter=CapturingExporter())

    from autobench.exporters import _otel

    original = _otel.export_records

    def invalid_mapping(
        active_experiment: ExperimentRecord,
        active_runs: tuple[RunRecord, ...],
        settings: OTLPSettings,
        exporter: SpanExporter | None,
    ) -> OTLPExportResult:
        del active_experiment, active_runs, settings, exporter
        raise ValueError("invalid mapping")

    monkeypatch.setattr(_otel, "export_records", invalid_mapping)
    with pytest.raises(OTLPExportError, match="Could not map.*invalid mapping"):
        export_otlp(experiment, runs, exporter=CapturingExporter())
    monkeypatch.setattr(_otel, "export_records", original)

    with monkeypatch.context() as context:
        context.delitem(sys.modules, "autobench.exporters._otel")
        context.setitem(sys.modules, "opentelemetry.sdk.resources", None)
        with pytest.raises(OTLPExportError, match=r"autobench\[otlp\]"):
            export_otlp(experiment, runs, exporter=CapturingExporter())

    with monkeypatch.context() as context:
        context.setitem(sys.modules, "autobench.exporters._otel", None)
        with pytest.raises(ModuleNotFoundError, match="autobench.exporters._otel"):
            export_otlp(experiment, runs, exporter=CapturingExporter())


def test_otlp_cli_renders_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autobench.cli as cli_module

    record_dir = tmp_path / "record"
    record_dir.mkdir()
    certificate = tmp_path / "ca.pem"
    certificate.write_text("certificate", encoding="utf-8")
    captured_settings: list[OTLPSettings] = []

    def successful_export(
        active_record_dir: Path,
        *,
        settings: OTLPSettings | None = None,
        exporter: SpanExporter | None = None,
    ) -> OTLPExportResult:
        assert active_record_dir == record_dir
        assert exporter is None
        assert settings is not None
        captured_settings.append(settings)
        return OTLPExportResult(
            experiment_id="experiment-1",
            benchmark_id="routing",
            record_version=6,
            run_count=3,
            trace_count=2,
            abp_span_count=2,
            exported_span_count=8,
            partial_run_count=1,
            partial_trace_count=1,
            endpoint=settings.endpoint or "environment default",
        )

    monkeypatch.setattr(cli_module, "export_record_otlp", successful_export)
    runner = CliRunner()
    success = runner.invoke(
        cli,
        [
            "telemetry",
            "export",
            str(record_dir),
            "--endpoint",
            "https://collector.example/v1/traces",
            "--header",
            "authorization",
            "test",
            "--timeout",
            "3",
            "--service-name",
            "routing",
            "--service-namespace",
            "evaluation",
            "--certificate-file",
            str(certificate),
            "--include-captured-content",
        ],
    )
    assert success.exit_code == 0, success.output
    assert "OTLP Export Complete" in success.output
    assert "Telemetry Export Summary" in success.output
    assert captured_settings == [
        OTLPSettings(
            endpoint="https://collector.example/v1/traces",
            headers={"authorization": "test"},
            timeout_seconds=3,
            certificate_file=certificate,
            service_name="routing",
            service_namespace="evaluation",
            include_captured_content=True,
        )
    ]

    def failed_export(
        active_record_dir: Path,
        *,
        settings: OTLPSettings | None = None,
        exporter: SpanExporter | None = None,
    ) -> OTLPExportResult:
        del active_record_dir, settings, exporter
        raise OTLPExportError("collector unavailable")

    monkeypatch.setattr(cli_module, "export_record_otlp", failed_export)
    failure = runner.invoke(cli, ["telemetry", "export", str(record_dir)])
    assert failure.exit_code == 1
    assert "OTLP export failed: collector unavailable" in failure.output


def _evidence_records() -> tuple[ExperimentRecord, tuple[RunRecord, ...]]:
    started_at = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)
    scope = InstrumentationScope(
        instrumentor_name="tests",
        instrumentor_version="1",
        package_name="demo-sdk",
        package_version="2",
        mechanism=CaptureMechanism.CALLBACK,
        layer=AbstractionLayer.FRAMEWORK,
        source_convention="demo",
        source_convention_version="1",
    )
    execution = ExecutionRef(
        benchmark_id="routing",
        experiment_id="experiment-1",
        run_id="run-failed",
        case_id="failed-case",
        variant_id="baseline",
    )
    source = SourceProvenance(
        system="demo",
        key="messages",
        path=(0, "content"),
        convention_version="1",
        source_map_id="demo-map",
        source_map_version=1,
        instrumentor="tests",
        instrumented_library_version="2",
    )
    message = Event(
        trace_id=TRACE_ID,
        span_id=ROOT_SPAN_ID,
        emitted_at=started_at + timedelta(milliseconds=1),
        monotonic_ns=1,
        sequence=1,
        execution=execution,
        scope=scope,
        source=source,
        name="agent.message",
        semantic_type="agent.message",
        body={"text": "secret prompt"},
        attributes={"role": "user"},
    )
    tokens = Measurement(
        trace_id=TRACE_ID,
        span_id=ROOT_SPAN_ID,
        emitted_at=started_at + timedelta(milliseconds=2),
        monotonic_ns=2,
        sequence=2,
        execution=execution,
        scope=scope,
        name="input_tokens",
        semantic_type=Semantic.LLM_TOKENS_INPUT,
        value=12,
        unit="tokens",
        direction=Direction.MINIMIZE,
        role=ObservationRole.DIAGNOSTIC,
        measurement_scope=MeasurementScope.DIRECT,
        layer=AbstractionLayer.FRAMEWORK,
        attributes={"cached": False},
    )
    link = Link(
        trace_id=TRACE_ID,
        span_id=ROOT_SPAN_ID,
        emitted_at=started_at + timedelta(milliseconds=3),
        monotonic_ns=3,
        sequence=3,
        execution=execution,
        scope=scope,
        relation=LinkRelation.RETRY_OF,
        target=LinkTarget(trace_id=LINK_TRACE_ID, span_id=LINK_SPAN_ID),
    )
    reference = Reference(
        trace_id=TRACE_ID,
        span_id=ROOT_SPAN_ID,
        emitted_at=started_at + timedelta(milliseconds=4),
        monotonic_ns=4,
        sequence=4,
        execution=execution,
        scope=scope,
        semantic_type="prompt.version",
        name="prompt",
        reference=EvidenceRef(kind=ReferenceKind.PROMPT, id="routing-prompt", version="v4"),
    )
    root = SpanRecord(
        trace_id=TRACE_ID,
        span_id=ROOT_SPAN_ID,
        operation="agent.run",
        kind="agent",
        execution=execution,
        scope=scope,
        capture=CaptureLevel.FULL,
        started_at=started_at,
        ended_at=started_at + timedelta(milliseconds=10),
        attributes={"autobench.run.id": "collision", "model": "demo"},
        source_attributes={"raw.model": "provider-model"},
        output={"answer": "secret"},
        status=SpanStatus.ERROR,
        end_reason=EndReason.FAILED,
        usage={"input_tokens": 12},
        stream={"chunks": 2},
        errors=(EvidenceRef(kind=ReferenceKind.ERROR, id="error-1"),),
        start_links=(
            LinkTarget(trace_id=LINK_TRACE_ID, span_id=LINK_SPAN_ID),
            LinkTarget(trace_id="4" * 32),
        ),
        events=(message,),
        measurements=(tokens,),
        links=(link,),
        references=(reference,),
    )
    child = SpanRecord(
        trace_id=TRACE_ID,
        span_id=CHILD_SPAN_ID,
        parent_span_id=ROOT_SPAN_ID,
        operation="tool.lookup",
        kind="tool",
        execution=execution,
        scope=scope.model_copy(
            update={"source_convention": None, "source_convention_version": None}
        ),
        capture=CaptureLevel.METADATA,
        started_at=started_at + timedelta(milliseconds=2),
        ended_at=started_at + timedelta(milliseconds=8),
        output_reference=EvidenceRef(kind=ReferenceKind.ARTIFACT, id="tool-output"),
        status=SpanStatus.OK,
        end_reason=EndReason.COMPLETED,
    )
    trace = Trace(
        trace_id=TRACE_ID,
        execution=execution,
        root_span_ids=(ROOT_SPAN_ID,),
        spans=(root, child),
        references=(reference.model_copy(update={"span_id": None}),),
        diagnostics=(
            Diagnostic(
                code="source_warning",
                message="source field was deprecated",
                severity=DiagnosticSeverity.WARNING,
                span_id=ROOT_SPAN_ID,
                details={"field": "old_name"},
            ),
        ),
        started_at=started_at,
        ended_at=started_at + timedelta(milliseconds=10),
    )
    correlation = ExecutionCorrelation(
        group_id="proposal-42",
        attempt=2,
        phase="validation",
        parent_experiment_id="experiment-0",
        resumed_from_experiment_id="experiment-recovery",
        labels={"owner": "evaluation", "seed": 17},
    )
    snapshot = SourceSnapshot(
        system="demo",
        convention_version="1",
        source_map_id="demo-map",
        source_map_version=1,
        facts=(
            RetainedSourceFact(
                selector=SourceSelector(key="messages", path=(0, "content")),
                value="secret prompt",
                available=True,
            ),
        ),
    )
    error = ErrorRecord(
        error_type="RuntimeError",
        message="route failed",
        traceback="trace",
        span_id=ROOT_SPAN_ID,
    )
    failed = RunRecord(
        protocol_version=1,
        semantic_registry_version=1,
        run_id="run-failed",
        experiment_id="experiment-1",
        benchmark_id="routing",
        case_id="failed-case",
        variant_id="baseline",
        status=RunStatus.FAILED,
        evaluation_status=EvaluationStatus.FAILED,
        task_status=TaskStatus.FAILED,
        end_reason=EndReason.FAILED,
        case=Case(id="failed-case"),
        observations=(
            Observation(
                id="metric-1",
                name="accuracy",
                kind=ObservationKind.METRIC,
                semantic_type=Semantic.QUALITY_CORRECTNESS,
                value=0.5,
                unit="ratio",
                direction=Direction.MAXIMIZE,
                role=ObservationRole.OBJECTIVE,
                source=ObservationSource.SCORE,
            ),
            Observation(
                id="event-1",
                name="conversation",
                kind=ObservationKind.EVENT,
                semantic_type="agent.conversation",
                value={"message": "secret"},
                source=ObservationSource.INSTRUMENTATION,
            ),
        ),
        scores=(
            ScoreRecord(
                name="route",
                semantic_type=Semantic.QUALITY_CORRECTNESS,
                value=False,
                role=ObservationRole.OBJECTIVE,
                actual_value="billing",
                expected_value="account",
            ),
        ),
        factors=(
            FactorValue(
                name="model",
                value="demo-model",
                semantic_type=Semantic.LLM_MODEL_NAME,
                optimize=True,
            ),
            FactorValue(name="optional", value=None),
        ),
        asset_uses=(
            AssetUse(
                asset_id="routing-prompt",
                version="v4",
                representation=AssetRepresentation.EFFECTIVE,
                source_locator="Agent.instructions",
                scope="agent:routing",
                span_id=ROOT_SPAN_ID,
                provenance=AssetProvenance(system="pydantic_ai", key="instructions"),
            ),
        ),
        errors=(error,),
        error=error,
        trace=trace,
        source_snapshots=(snapshot,),
        correlation=correlation,
    )
    partial_trace = Trace(
        trace_id=PARTIAL_TRACE_ID,
        started_at=started_at + timedelta(seconds=1),
        ended_at=None,
        partial=True,
    )
    passed = failed.model_copy(
        update={
            "run_id": "run-passed",
            "case_id": "passed-case",
            "case": Case(id="passed-case"),
            "status": RunStatus.PASSED,
            "evaluation_status": EvaluationStatus.PASSED,
            "task_status": TaskStatus.PASSED,
            "end_reason": EndReason.COMPLETED,
            "observations": (),
            "scores": (),
            "factors": (),
            "asset_uses": (),
            "errors": (),
            "error": None,
            "trace": partial_trace,
            "source_snapshots": (),
        }
    )
    cancelled = passed.model_copy(
        update={
            "run_id": "run-cancelled",
            "case_id": "cancelled-case",
            "case": Case(id="cancelled-case"),
            "status": RunStatus.CANCELLED,
            "evaluation_status": EvaluationStatus.NOT_EVALUATED,
            "task_status": TaskStatus.CANCELLED,
            "partial": True,
            "end_reason": EndReason.CANCELLED,
            "trace": None,
            "parent_run_id": "run-parent",
        }
    )
    experiment = ExperimentRecord(
        experiment_id="experiment-1",
        benchmark_id="routing",
        plan=BenchmarkPlan(
            benchmark_id="routing",
            dataset_id="routing-cases",
            dataset_version="v2",
            dataset_hash="f" * 64,
            case_ids=("failed-case", "passed-case", "cancelled-case"),
            case_count=3,
            variant_count=1,
            planned_run_count=3,
        ),
        environment=EnvironmentMetadata(python_version="3.11", platform="test", cwd="."),
        termination=ExperimentTermination(),
        spec_hash="e" * 64,
        run_paths=("runs/failed.yaml", "runs/passed.yaml", "runs/cancelled.yaml"),
        run_count=3,
        passed_count=1,
        failed_count=1,
        errored_count=0,
        skipped_count=0,
        cancelled_count=1,
        correlation=correlation,
    )
    return experiment, (failed, passed, cancelled)
