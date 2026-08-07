from __future__ import annotations as _annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from opentelemetry import trace as otel_trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import (
    Link as OTelLink,
)
from opentelemetry.trace import (
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    Tracer,
)
from opentelemetry.util.types import AttributeValue

from autobench._version import __version__
from autobench.errors import ErrorRecord
from autobench.exporters.otlp import OTLPExportError, OTLPExportResult, OTLPSettings
from autobench.metrics.observations import ObservationKind
from autobench.protocol.signals import SpanStatus
from autobench.protocol.traces import SpanRecord, Trace
from autobench.protocol.values import SerializedValue
from autobench.records.models import ExperimentRecord, RunRecord
from autobench.runtime.models import ExperimentStatus, RunStatus

OTelAttributes = dict[str, AttributeValue]


class _CollectingExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def export_records(
    experiment: ExperimentRecord,
    runs: tuple[RunRecord, ...],
    settings: OTLPSettings,
    exporter: SpanExporter | None,
) -> OTLPExportResult:
    resource_attributes: OTelAttributes = dict(settings.resource_attributes)
    resource_attributes.update(
        {
            "service.name": settings.service_name,
            "autobench.protocol": "abp",
            "autobench.record.version": experiment.record_version,
        }
    )
    if settings.service_namespace is not None:
        resource_attributes["service.namespace"] = settings.service_namespace

    collector = _CollectingExporter()
    provider = TracerProvider(
        resource=Resource.create(resource_attributes),
        shutdown_on_exit=False,
    )
    provider.add_span_processor(SimpleSpanProcessor(collector))
    tracer = provider.get_tracer("autobench.otlp", __version__)

    trace_times = tuple(
        _normalize_datetime(timestamp)
        for run in runs
        if run.trace is not None
        for timestamp in (run.trace.started_at, run.trace.ended_at)
        if timestamp is not None
    )
    exported_at = datetime.now(UTC)
    experiment_start = min(trace_times) if trace_times else exported_at
    experiment_end = max(trace_times) if trace_times else exported_at
    experiment_span = tracer.start_span(
        f"autobench.experiment {experiment.benchmark_id}",
        attributes=_experiment_attributes(experiment),
        start_time=_timestamp_ns(experiment_start),
    )
    experiment_context = otel_trace.set_span_in_context(experiment_span)

    try:
        _add_model_event(
            experiment_span,
            "autobench.experiment.termination",
            _termination_payload(experiment, settings),
        )
        run_paths: tuple[str | None, ...]
        if len(experiment.run_paths) == len(runs):
            run_paths = experiment.run_paths
        else:
            run_paths = (None,) * len(runs)
        for run, run_path in zip(runs, run_paths, strict=True):
            _emit_run(tracer, experiment_context, experiment, run, run_path, settings)
        _set_experiment_status(experiment_span, experiment)
    finally:
        experiment_span.end(end_time=_timestamp_ns(max(experiment_start, experiment_end)))
        provider.force_flush()
        provider.shutdown()

    active_exporter = exporter
    owns_exporter = active_exporter is None
    if active_exporter is None:
        active_exporter = OTLPSpanExporter(
            endpoint=settings.endpoint,
            certificate_file=(
                None if settings.certificate_file is None else str(settings.certificate_file)
            ),
            headers=settings.headers or None,
            timeout=settings.timeout_seconds,
        )
    failure: OTLPExportError | None = None
    failure_cause: Exception | None = None
    try:
        export_result = active_exporter.export(tuple(collector.spans))
        if export_result is not SpanExportResult.SUCCESS:
            raise OTLPExportError("The OTLP span exporter reported a failed export.")
    except OTLPExportError as exc:
        failure = exc
    except Exception as exc:
        failure = OTLPExportError(f"OTLP span export failed: {exc}")
        failure_cause = exc
    if owns_exporter:
        try:
            active_exporter.shutdown()
        except Exception as exc:
            if failure is None:
                failure = OTLPExportError(f"OTLP exporter shutdown failed: {exc}")
                failure_cause = exc
    if failure is not None:
        raise failure from failure_cause

    traces = tuple(run.trace for run in runs if run.trace is not None)
    return OTLPExportResult(
        experiment_id=experiment.experiment_id,
        benchmark_id=experiment.benchmark_id,
        record_version=experiment.record_version,
        run_count=len(runs),
        trace_count=len(traces),
        abp_span_count=sum(len(trace.spans) for trace in traces),
        exported_span_count=len(collector.spans),
        partial_run_count=sum(run.partial for run in runs),
        partial_trace_count=sum(trace.partial for trace in traces),
        endpoint=settings.endpoint or "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT/environment default",
    )


def _emit_run(
    tracer: Tracer,
    parent_context: Context,
    experiment: ExperimentRecord,
    run: RunRecord,
    run_path: str | None,
    settings: OTLPSettings,
) -> None:
    trace = run.trace
    now = datetime.now(UTC)
    started_at = _normalize_datetime(
        now if trace is None or trace.started_at is None else trace.started_at
    )
    ended_at = _normalize_datetime(
        started_at if trace is None or trace.ended_at is None else trace.ended_at
    )
    run_span = tracer.start_span(
        f"autobench.run {run.case_id} / {run.variant_id}",
        context=parent_context,
        attributes=_run_attributes(experiment, run, run_path),
        start_time=_timestamp_ns(started_at),
    )
    run_context = otel_trace.set_span_in_context(run_span)
    try:
        for factor in run.factors:
            _add_model_event(run_span, "autobench.factor", factor.model_dump(mode="json"))
        for observation in run.observations:
            payload = observation.model_dump(mode="json", exclude_none=True)
            if not settings.include_captured_content and observation.kind not in {
                ObservationKind.METRIC,
                ObservationKind.FACTOR,
            }:
                payload.pop("value", None)
                payload["content_omitted"] = True
            _add_model_event(run_span, "autobench.observation", payload)
        for score in run.scores:
            payload = score.model_dump(mode="json", exclude_none=True)
            if not settings.include_captured_content:
                payload.pop("actual_value", None)
                payload.pop("expected_value", None)
            _add_model_event(run_span, "autobench.score", payload)
        for asset_use in run.asset_uses:
            _add_model_event(
                run_span,
                "autobench.asset.use",
                asset_use.model_dump(mode="json", exclude_none=True),
            )
        for source_snapshot in run.source_snapshots:
            payload = source_snapshot.model_dump(mode="json", exclude_none=True)
            if not settings.include_captured_content:
                payload["fact_count"] = len(source_snapshot.facts)
                payload.pop("facts", None)
            _add_model_event(run_span, "autobench.source.snapshot", payload)
        errors = run.errors
        if run.error is not None and run.error not in errors:
            errors = (*errors, run.error)
        for error in errors:
            _add_model_event(
                run_span,
                "exception",
                _error_payload(error, settings),
            )
        if trace is not None:
            _emit_trace(tracer, run_context, experiment, run, trace, settings)
        _set_run_status(run_span, run)
    finally:
        run_span.end(end_time=_timestamp_ns(max(started_at, ended_at)))


def _emit_trace(
    tracer: Tracer,
    parent_context: Context,
    experiment: ExperimentRecord,
    run: RunRecord,
    trace: Trace,
    settings: OTLPSettings,
) -> None:
    now = datetime.now(UTC)
    started_at = _normalize_datetime(trace.started_at or now)
    ended_at = _normalize_datetime(trace.ended_at or started_at)
    trace_span = tracer.start_span(
        f"autobench.trace {trace.trace_id[:8]}",
        context=parent_context,
        attributes={
            **_run_identity_attributes(experiment, run),
            "autobench.abp.trace_id": trace.trace_id,
            "autobench.abp.protocol_version": trace.protocol_version,
            "autobench.abp.partial": trace.partial,
            "autobench.abp.diagnostic_count": len(trace.diagnostics),
        },
        start_time=_timestamp_ns(started_at),
    )
    trace_context = otel_trace.set_span_in_context(trace_span)
    try:
        for diagnostic in trace.diagnostics:
            _add_model_event(
                trace_span,
                "autobench.abp.diagnostic",
                diagnostic.model_dump(mode="json", exclude_none=True),
            )
        for reference in trace.references:
            _add_model_event(
                trace_span,
                "autobench.abp.reference",
                reference.model_dump(mode="json", exclude_none=True),
                timestamp=reference.emitted_at,
            )

        records = {span.span_id: span for span in trace.spans}
        children: dict[str | None, list[SpanRecord]] = {}
        for span in trace.spans:
            parent_id = span.parent_span_id if span.parent_span_id in records else None
            children.setdefault(parent_id, []).append(span)
        visited: set[str] = set()
        active: set[str] = set()

        def emit_span(record: SpanRecord, context: Context) -> None:
            if record.span_id in active:
                trace_span.add_event(
                    "autobench.abp.parent_cycle",
                    {"autobench.abp.span_id": record.span_id},
                )
                return
            if record.span_id in visited:
                return
            active.add(record.span_id)
            visited.add(record.span_id)
            _emit_abp_span(
                tracer,
                context,
                experiment,
                run,
                record,
                children,
                emit_span,
                settings,
            )
            active.remove(record.span_id)

        for root in children.get(None, ()):
            emit_span(root, trace_context)
        for record in trace.spans:
            if record.span_id not in visited:
                emit_span(record, trace_context)
        if trace.partial:
            trace_span.set_status(StatusCode.UNSET)
        else:
            trace_span.set_status(StatusCode.OK)
    finally:
        trace_span.end(end_time=_timestamp_ns(max(started_at, ended_at)))


def _emit_abp_span(
    tracer: Tracer,
    parent_context: Context,
    experiment: ExperimentRecord,
    run: RunRecord,
    record: SpanRecord,
    children: Mapping[str | None, list[SpanRecord]],
    emit_child: Callable[[SpanRecord, Context], None],
    settings: OTLPSettings,
) -> None:
    started_at = _normalize_datetime(record.started_at or record.ended_at or datetime.now(UTC))
    ended_at = _normalize_datetime(record.ended_at or started_at)
    attributes = _run_identity_attributes(experiment, run)
    attributes.update(
        {
            "autobench.abp.trace_id": record.trace_id,
            "autobench.abp.span_id": record.span_id,
            "autobench.abp.kind": record.kind,
            "autobench.abp.capture": record.capture.value,
            "autobench.abp.partial": record.partial,
            "autobench.instrumentation.name": record.scope.instrumentor_name,
            "autobench.instrumentation.version": record.scope.instrumentor_version,
            "autobench.instrumented.package": record.scope.package_name,
            "autobench.instrumented.package_version": record.scope.package_version,
            "autobench.instrumentation.mechanism": record.scope.mechanism.value,
            "autobench.instrumentation.layer": record.scope.layer.value,
        }
    )
    if record.parent_span_id is not None:
        attributes["autobench.abp.parent_span_id"] = record.parent_span_id
    if record.end_reason is not None:
        attributes["autobench.abp.end_reason"] = record.end_reason.value
    if record.scope.source_convention is not None:
        attributes["autobench.source.convention"] = record.scope.source_convention
    if record.scope.source_convention_version is not None:
        attributes["autobench.source.convention_version"] = record.scope.source_convention_version
    _merge_attributes(attributes, record.attributes)
    _merge_attributes(attributes, record.source_attributes, prefix="autobench.source.attribute.")
    _merge_attributes(attributes, record.usage, prefix="autobench.usage.")
    _merge_attributes(attributes, record.stream, prefix="autobench.stream.")
    if record.output is not None:
        if settings.include_captured_content:
            attributes["autobench.abp.output"] = _attribute_value(record.output)
        else:
            attributes["autobench.abp.output_omitted"] = True

    span = tracer.start_span(
        record.operation,
        context=parent_context,
        attributes=attributes,
        links=_otel_links(record),
        start_time=_timestamp_ns(started_at),
    )
    span_context = otel_trace.set_span_in_context(span)
    try:
        for event in record.events:
            payload = event.model_dump(mode="json", exclude_none=True)
            if not settings.include_captured_content and event.body is not None:
                payload.pop("body", None)
                payload["content_omitted"] = True
            _add_model_event(span, event.name, payload, timestamp=event.emitted_at)
        for measurement in record.measurements:
            _add_model_event(
                span,
                "autobench.measurement",
                measurement.model_dump(mode="json", exclude_none=True),
                timestamp=measurement.emitted_at,
            )
        for link in record.links:
            _add_model_event(
                span,
                "autobench.abp.link",
                link.model_dump(mode="json", exclude_none=True),
                timestamp=link.emitted_at,
            )
        for reference in record.references:
            _add_model_event(
                span,
                "autobench.abp.reference",
                reference.model_dump(mode="json", exclude_none=True),
                timestamp=reference.emitted_at,
            )
        for error in record.errors:
            _add_model_event(
                span,
                "autobench.abp.error_reference",
                error.model_dump(mode="json", exclude_none=True),
            )
        if record.output_reference is not None:
            _add_model_event(
                span,
                "autobench.abp.output_reference",
                record.output_reference.model_dump(mode="json", exclude_none=True),
            )
        for target in record.start_links:
            _add_model_event(
                span,
                "autobench.abp.start_link",
                target.model_dump(mode="json", exclude_none=True),
            )
        for child in children.get(record.span_id, ()):
            emit_child(child, span_context)
        if record.status is SpanStatus.ERROR:
            description = (
                "ABP span failed" if record.end_reason is None else record.end_reason.value
            )
            span.set_status(Status(StatusCode.ERROR, description))
        elif record.status is SpanStatus.OK:
            span.set_status(StatusCode.OK)
    finally:
        span.end(end_time=_timestamp_ns(max(started_at, ended_at)))


def _experiment_attributes(experiment: ExperimentRecord) -> OTelAttributes:
    attributes: OTelAttributes = {
        "autobench.experiment.id": experiment.experiment_id,
        "autobench.benchmark.id": experiment.benchmark_id,
        "autobench.record.version": experiment.record_version,
        "autobench.experiment.status": experiment.termination.status.value,
        "autobench.experiment.partial": experiment.termination.partial,
        "autobench.experiment.cross_run_derivation_complete": (
            experiment.termination.cross_run_derivation_complete
        ),
        "autobench.experiment.policies_complete": experiment.termination.policies_complete,
        "autobench.experiment.missing_run_count": len(experiment.termination.missing_run_ids),
    }
    if experiment.spec_hash is not None:
        attributes["autobench.spec.sha256"] = experiment.spec_hash
    if experiment.manifest_path is not None:
        attributes["autobench.record.manifest_path"] = experiment.manifest_path
    if experiment.plan.dataset_id is not None:
        attributes["autobench.dataset.id"] = experiment.plan.dataset_id
    if experiment.plan.dataset_version is not None:
        attributes["autobench.dataset.version"] = experiment.plan.dataset_version
    if experiment.plan.dataset_hash is not None:
        attributes["autobench.dataset.sha256"] = experiment.plan.dataset_hash
    if experiment.correlation is not None:
        correlation = experiment.correlation
        if correlation.group_id is not None:
            attributes["autobench.correlation.group_id"] = correlation.group_id
        if correlation.attempt is not None:
            attributes["autobench.correlation.attempt"] = correlation.attempt
        if correlation.phase is not None:
            attributes["autobench.correlation.phase"] = correlation.phase
        if correlation.parent_experiment_id is not None:
            attributes["autobench.correlation.parent_experiment_id"] = (
                correlation.parent_experiment_id
            )
        if correlation.resumed_from_experiment_id is not None:
            attributes["autobench.correlation.resumed_from_experiment_id"] = (
                correlation.resumed_from_experiment_id
            )
        for key, value in correlation.labels.items():
            attributes[f"autobench.correlation.label.{key}"] = value
    return attributes


def _run_identity_attributes(
    experiment: ExperimentRecord,
    run: RunRecord,
) -> OTelAttributes:
    return {
        "autobench.experiment.id": experiment.experiment_id,
        "autobench.benchmark.id": experiment.benchmark_id,
        "autobench.run.id": run.run_id,
        "autobench.case.id": run.case_id,
        "autobench.variant.id": run.variant_id,
        "autobench.record.version": run.record_version,
    }


def _run_attributes(
    experiment: ExperimentRecord,
    run: RunRecord,
    run_path: str | None,
) -> OTelAttributes:
    attributes = _run_identity_attributes(experiment, run)
    attributes.update(
        {
            "autobench.run.status": run.status.value,
            "autobench.run.task_status": run.task_status.value,
            "autobench.run.evaluation_status": run.evaluation_status.value,
            "autobench.run.partial": run.partial,
            "autobench.run.end_reason": run.end_reason.value,
        }
    )
    if run.parent_run_id is not None:
        attributes["autobench.run.parent_id"] = run.parent_run_id
    if run_path is not None:
        attributes["autobench.record.run_path"] = run_path
    if run.protocol_version is not None:
        attributes["autobench.protocol.version"] = run.protocol_version
    if run.semantic_registry_version is not None:
        attributes["autobench.semantic_registry.version"] = run.semantic_registry_version
    return attributes


def _set_experiment_status(span: Span, experiment: ExperimentRecord) -> None:
    if experiment.termination.status is ExperimentStatus.ABORTED:
        description = (
            "Autobench experiment aborted"
            if experiment.termination.error is None
            else experiment.termination.error.message
        )
        span.set_status(Status(StatusCode.ERROR, description))
    elif experiment.termination.status is ExperimentStatus.COMPLETED:
        span.set_status(StatusCode.OK)


def _set_run_status(span: Span, run: RunRecord) -> None:
    if run.status in {RunStatus.FAILED, RunStatus.ERRORED}:
        description = "Autobench run failed" if run.error is None else run.error.message
        span.set_status(Status(StatusCode.ERROR, description))
    elif run.status is RunStatus.PASSED:
        span.set_status(StatusCode.OK)


def _termination_payload(
    experiment: ExperimentRecord,
    settings: OTLPSettings,
) -> dict[str, SerializedValue]:
    termination = experiment.termination
    payload: dict[str, SerializedValue] = {
        "status": termination.status.value,
        "partial": termination.partial,
        "cross_run_derivation_complete": termination.cross_run_derivation_complete,
        "policies_complete": termination.policies_complete,
        "planned_run_ids": list(termination.planned_run_ids),
        "recorded_run_ids": list(termination.recorded_run_ids),
        "missing_run_ids": list(termination.missing_run_ids),
    }
    if termination.error is not None:
        payload["error"] = _error_payload(termination.error, settings)
    return payload


def _error_payload(
    error: ErrorRecord,
    settings: OTLPSettings,
) -> dict[str, SerializedValue]:
    payload: dict[str, SerializedValue] = {
        "error_type": error.error_type,
        "message": error.message,
    }
    if error.span_id is not None:
        payload["span_id"] = error.span_id
    if error.traceback is not None:
        if settings.include_captured_content:
            payload["traceback"] = error.traceback
        else:
            payload["traceback_omitted"] = True
    return payload


def _otel_links(record: SpanRecord) -> tuple[OTelLink, ...]:
    links: list[OTelLink] = []
    for target in record.start_links:
        if target.trace_id is not None and target.span_id is not None:
            links.append(
                OTelLink(
                    SpanContext(
                        trace_id=int(target.trace_id, 16),
                        span_id=int(target.span_id, 16),
                        is_remote=True,
                        trace_flags=TraceFlags(TraceFlags.SAMPLED),
                    ),
                    {"autobench.abp.relation": "start_link"},
                )
            )
    for link in record.links:
        if link.target.trace_id is not None and link.target.span_id is not None:
            links.append(
                OTelLink(
                    SpanContext(
                        trace_id=int(link.target.trace_id, 16),
                        span_id=int(link.target.span_id, 16),
                        is_remote=True,
                        trace_flags=TraceFlags(TraceFlags.SAMPLED),
                    ),
                    {
                        "autobench.abp.relation": link.relation.value,
                        "autobench.abp.signal_id": link.signal_id,
                    },
                )
            )
    return tuple(links)


def _add_model_event(
    span: Span,
    name: str,
    payload: Mapping[str, SerializedValue],
    *,
    timestamp: datetime | None = None,
) -> None:
    span.add_event(
        name,
        {key: _attribute_value(value) for key, value in payload.items()},
        timestamp=None if timestamp is None else _timestamp_ns(timestamp),
    )


def _merge_attributes(
    target: OTelAttributes,
    values: Mapping[str, SerializedValue],
    *,
    prefix: str = "",
) -> None:
    for key, value in values.items():
        target_key = f"{prefix}{key}"
        if not prefix and target_key in target:
            target_key = f"autobench.abp.attribute.{key}"
        target[target_key] = _attribute_value(value)


def _attribute_value(value: SerializedValue) -> AttributeValue:
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _timestamp_ns(value: datetime) -> int:
    normalized = _normalize_datetime(value)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = normalized - epoch
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1_000


def _normalize_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ("export_records",)
