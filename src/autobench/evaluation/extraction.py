from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from autobench.metrics.observations import (
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import Semantic, SemanticRegistry
from autobench.protocol.signals import (
    AbstractionLayer,
    EndReason,
    Event,
    KnownSpanKind,
    Link,
    LinkRelation,
    Measurement,
    MeasurementScope,
    SpanStatus,
)
from autobench.protocol.traces import Diagnostic, SpanRecord, Trace
from autobench.protocol.values import EvidenceRef, ReferenceKind, SerializedValue

MEASUREMENT_SCOPE_TAG = "abp.measurement_scope"
ABSTRACTION_LAYER_TAG = "abp.abstraction_layer"
LOGICAL_OPERATION_TAG = "abp.logical_operation_id"
INSTRUMENTOR_TAG = "abp.instrumentor"
EXTRACTOR_TAG = "abp.extractor"
EXTRACTOR_VERSION_TAG = "abp.extractor_version"
SUMMARY_TAG = "abp.summary"


class ExtractionContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    benchmark_id: str
    experiment_id: str
    case_id: str
    variant_id: str


class ExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: tuple[Observation, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    references: tuple[EvidenceRef, ...] = ()


class ExtractionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    extractor: str
    version: str
    observation_ids: tuple[str, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    references: tuple[EvidenceRef, ...] = ()


class TraceExtractor(Protocol):
    name: str
    version: str

    def extract(
        self,
        trace: Trace,
        *,
        registry: SemanticRegistry,
        context: ExtractionContext,
    ) -> ExtractionResult: ...


class SignalExtractor:
    name = "abp.signals"
    version = "2"

    def extract(
        self,
        trace: Trace,
        *,
        registry: SemanticRegistry,
        context: ExtractionContext,
    ) -> ExtractionResult:
        observations: list[Observation] = []
        references: dict[tuple[ReferenceKind, str, str | None], EvidenceRef] = {}
        for signal in trace.signals:
            if isinstance(signal, Measurement):
                attributes = signal.attributes
                kind_value = attributes.get("kind", ObservationKind.METRIC.value)
                kind = (
                    ObservationKind(kind_value)
                    if isinstance(kind_value, str)
                    and kind_value in {kind.value for kind in ObservationKind}
                    else ObservationKind.METRIC
                )
                source_value = attributes.get("source", ObservationSource.IMPORTED.value)
                source = (
                    ObservationSource(source_value)
                    if isinstance(source_value, str)
                    and source_value in {source.value for source in ObservationSource}
                    else ObservationSource.IMPORTED
                )
                tags_value = attributes.get("tags", {})
                tags = dict(tags_value) if isinstance(tags_value, dict) else {}
                tags.setdefault(MEASUREMENT_SCOPE_TAG, signal.measurement_scope.value)
                tags.setdefault(ABSTRACTION_LAYER_TAG, signal.layer.value)
                tags.setdefault(INSTRUMENTOR_TAG, signal.scope.instrumentor_name)
                logical_operation_id = _logical_operation_id(attributes)
                if logical_operation_id is not None:
                    tags.setdefault(LOGICAL_OPERATION_TAG, logical_operation_id)
                observations.append(
                    Observation(
                        id=_observation_id(signal.signal_id, attributes),
                        name=signal.name,
                        kind=kind,
                        semantic_type=registry.normalize(signal.semantic_type),
                        value=signal.value,
                        unit=signal.unit,
                        direction=signal.direction,
                        role=signal.role,
                        span_id=_span_id(signal.span_id, attributes),
                        source=source,
                        tags=tags,
                        case_id=context.case_id,
                        variant_id=context.variant_id,
                    )
                )
            elif isinstance(signal, Event):
                attributes = signal.attributes
                kind_value = attributes.get("kind", ObservationKind.EVENT.value)
                kind = (
                    ObservationKind(kind_value)
                    if isinstance(kind_value, str)
                    and kind_value in {kind.value for kind in ObservationKind}
                    else ObservationKind.EVENT
                )
                source_value = attributes.get("source", ObservationSource.IMPORTED.value)
                source = (
                    ObservationSource(source_value)
                    if isinstance(source_value, str)
                    and source_value in {source.value for source in ObservationSource}
                    else ObservationSource.IMPORTED
                )
                tags_value = attributes.get("tags", {})
                tags = dict(tags_value) if isinstance(tags_value, dict) else {}
                tags.setdefault(ABSTRACTION_LAYER_TAG, signal.scope.layer.value)
                tags.setdefault(INSTRUMENTOR_TAG, signal.scope.instrumentor_name)
                role = (
                    ObservationRole.DIAGNOSTIC
                    if registry.normalize(signal.semantic_type) == Semantic.DIAGNOSTIC_EVENT
                    else None
                )
                value = (
                    signal.reference.model_dump(mode="json")
                    if signal.reference is not None
                    else signal.body
                )
                observations.append(
                    Observation(
                        id=_observation_id(signal.signal_id, attributes),
                        name=signal.name,
                        kind=kind,
                        semantic_type=registry.normalize(signal.semantic_type),
                        value=value,
                        role=role,
                        span_id=_span_id(signal.span_id, attributes),
                        source=source,
                        tags=tags,
                        case_id=context.case_id,
                        variant_id=context.variant_id,
                    )
                )
                if signal.reference is not None:
                    reference = signal.reference
                    references[(reference.kind, reference.id, reference.version)] = reference

        for reference_signal in trace.references:
            reference = reference_signal.reference
            references[(reference.kind, reference.id, reference.version)] = reference
        for span in trace.spans:
            for reference_signal in span.references:
                reference = reference_signal.reference
                references[(reference.kind, reference.id, reference.version)] = reference
            for reference in span.errors:
                references[(reference.kind, reference.id, reference.version)] = reference

        return ExtractionResult(
            observations=tuple(observations),
            diagnostics=trace.diagnostics,
            references=tuple(references.values()),
        )


class SpanExtractor:
    """Derive generic operation, topology, timing, and workflow evidence."""

    name = "abp.spans"
    version = "1"

    def extract(
        self,
        trace: Trace,
        *,
        registry: SemanticRegistry,
        context: ExtractionContext,
    ) -> ExtractionResult:
        del registry
        observations: list[Observation] = []
        diagnostics: list[Diagnostic] = []
        spans = trace.spans
        children: dict[str, list[SpanRecord]] = defaultdict(list)
        spans_by_id = {span.span_id: span for span in spans}
        for span in spans:
            if span.parent_span_id in spans_by_id:
                children[span.parent_span_id].append(span)

        observations.extend(self._operation_evidence(trace, context, children))
        observations.extend(self._workflow_evidence(trace, context))
        observations.extend(self._reference_evidence(trace, context))
        incomplete = tuple(span for span in spans if _is_incomplete(span))
        if incomplete:
            diagnostics.append(
                Diagnostic(
                    code="incomplete_trace_work",
                    message=f"trace contains {len(incomplete)} incomplete operation(s)",
                    details={"span_ids": [span.span_id for span in incomplete]},
                )
            )
        return ExtractionResult(
            observations=tuple(observations),
            diagnostics=(*trace.diagnostics, *diagnostics),
            references=_trace_references(trace),
        )

    def _operation_evidence(
        self,
        trace: Trace,
        context: ExtractionContext,
        children: Mapping[str, list[SpanRecord]],
    ) -> list[Observation]:
        spans = trace.spans
        accounted_spans = _accounted_spans(spans)
        kind_counts = Counter(span.kind for span in accounted_spans)
        operation_counts = Counter(span.operation for span in accounted_spans)
        incomplete_count = sum(_is_incomplete(span) for span in spans)
        max_depth = _max_depth(spans, trace.links)
        fan_out = max(
            (_fan_out(span, children.get(span.span_id, [])) for span in spans),
            default=0,
        )
        intervals = tuple(
            (span.start_monotonic_ns, span.end_monotonic_ns)
            for span in spans
            if span.start_monotonic_ns is not None
            and span.end_monotonic_ns is not None
            and span.end_monotonic_ns >= span.start_monotonic_ns
        )
        critical_path_ns = (
            0
            if not intervals
            else max(end for _, end in intervals) - min(start for start, _ in intervals)
        )
        leaves = tuple(span for span in spans if not children.get(span.span_id))
        leaf_work_ns = sum(span.duration_ns for span in leaves if span.duration_ns is not None)
        observations = [
            self._metric(
                context,
                "operation.count",
                Semantic.OPERATION_COUNT,
                len(accounted_spans),
                summary=True,
            ),
            self._metric(
                context,
                "operation.depth.max",
                Semantic.OPERATION_DEPTH_MAX,
                max_depth,
                summary=True,
            ),
            self._metric(
                context,
                "operation.fan_out.max",
                Semantic.OPERATION_FAN_OUT_MAX,
                fan_out,
                summary=True,
            ),
            self._metric(
                context,
                "operation.incomplete.count",
                Semantic.OPERATION_INCOMPLETE_COUNT,
                incomplete_count,
                summary=True,
            ),
        ]
        if critical_path_ns:
            observations.extend(
                (
                    self._metric(
                        context,
                        "time.critical_path",
                        Semantic.TIME_CRITICAL_PATH,
                        critical_path_ns / 1_000_000_000,
                        unit="s",
                        summary=True,
                    ),
                    self._metric(
                        context,
                        "operation.parallelism",
                        Semantic.OPERATION_PARALLELISM,
                        leaf_work_ns / critical_path_ns,
                        unit="ratio",
                        summary=True,
                    ),
                )
            )
        for kind, count in sorted(kind_counts.items()):
            observations.append(
                self._metric(
                    context,
                    f"operation.kind.{kind}.count",
                    Semantic.OPERATION_COUNT,
                    count,
                    tags={"operation.kind": kind},
                )
            )
        for operation, count in sorted(operation_counts.items()):
            observations.append(
                self._metric(
                    context,
                    f"operation.{operation}.count",
                    Semantic.OPERATION_COUNT,
                    count,
                    tags={"operation.name": operation},
                )
            )
        for span in spans:
            if span.duration_seconds is None:
                continue
            observations.append(
                self._metric(
                    context,
                    "span.duration",
                    Semantic.TIME_LATENCY,
                    span.duration_seconds,
                    unit="s",
                    span=span,
                    tags={MEASUREMENT_SCOPE_TAG: MeasurementScope.DIRECT.value},
                )
            )
        return observations

    def _workflow_evidence(
        self,
        trace: Trace,
        context: ExtractionContext,
    ) -> list[Observation]:
        spans = trace.spans
        retry_pairs = {
            (link.span_id, link.target.span_id)
            for link in trace.links
            if link.relation is LinkRelation.RETRY_OF
            and link.target.trace_id == trace.trace_id
            and link.target.span_id is not None
        }
        retry_targets = {target for _, target in retry_pairs}
        by_id = {span.span_id: span for span in spans}
        first_attempts = tuple(by_id[target] for target in retry_targets if target in by_id)
        retry_events = {
            event.signal_id
            for span in spans
            for event in span.events
            if event.semantic_type in {Semantic.OPERATION_RETRY, Semantic.OPERATION_REPAIR}
            or event.name in {"retry", "repair"}
        }
        validations = tuple(span for span in spans if span.kind == KnownSpanKind.VALIDATION)
        validation_event_ids = {
            event.signal_id
            for span in spans
            for event in span.events
            if span.kind != KnownSpanKind.VALIDATION
            if event.semantic_type == Semantic.VALIDATION_FAILURE
            or event.name == "validation_failure"
        }
        validation_ids = {span.span_id for span in validations} | validation_event_ids
        validation_failure_ids = {
            span.span_id for span in validations if _is_failed(span)
        } | validation_event_ids
        approval_spans = tuple(span for span in spans if span.kind == KnownSpanKind.APPROVAL)
        approval_ids = {span.span_id for span in approval_spans} | {
            event.signal_id
            for span in spans
            for event in span.events
            if span.kind != KnownSpanKind.APPROVAL
            if event.semantic_type == Semantic.APPROVAL_REQUESTED
            or event.name == "approval_requested"
        }
        tools = tuple(span for span in spans if span.kind == KnownSpanKind.TOOL)
        validation_failures = len(validation_failure_ids)
        tool_failures = sum(_is_failed(span) for span in tools)
        tool_successes = sum(_is_successful(span) for span in tools)
        tool_arguments = sum(_has_tool_arguments(span) for span in tools)
        recovered_retries = sum(
            _is_failed(by_id[target]) and _is_successful(by_id[retry])
            for retry, target in retry_pairs
            if retry in by_id and target in by_id
        )
        approval_wait_ns = sum(
            span.duration_ns for span in approval_spans if span.duration_ns is not None
        )
        observations = [
            self._metric(
                context,
                "operation.retry.count",
                Semantic.OPERATION_RETRY_COUNT,
                len(retry_pairs) if retry_pairs else len(retry_events),
                summary=True,
            ),
            self._metric(
                context,
                "operation.retry.recovered.count",
                Semantic.OPERATION_RETRY_RECOVERED_COUNT,
                recovered_retries,
                summary=True,
            ),
            self._metric(
                context,
                "validation.count",
                Semantic.VALIDATION_COUNT,
                len(validation_ids),
                summary=True,
            ),
            self._metric(
                context,
                "validation.failure.count",
                Semantic.VALIDATION_FAILURE_COUNT,
                validation_failures,
                summary=True,
            ),
            self._metric(
                context,
                "approval.count",
                Semantic.APPROVAL_COUNT,
                len(approval_ids),
                summary=True,
            ),
            self._metric(
                context,
                "approval.wait",
                Semantic.APPROVAL_WAIT,
                approval_wait_ns / 1_000_000_000,
                unit="s",
                summary=True,
            ),
            self._metric(
                context,
                "tool.call.count",
                Semantic.TOOL_CALL_COUNT,
                len(tools),
                summary=True,
            ),
            self._metric(
                context,
                "tool.call.success.count",
                Semantic.TOOL_CALL_SUCCESS_COUNT,
                tool_successes,
                summary=True,
            ),
            self._metric(
                context,
                "tool.call.failure.count",
                Semantic.TOOL_CALL_FAILURE_COUNT,
                tool_failures,
                summary=True,
            ),
            self._metric(
                context,
                "tool.call.arguments.present.count",
                Semantic.TOOL_CALL_ARGUMENTS_PRESENT_COUNT,
                tool_arguments,
                summary=True,
            ),
        ]
        if first_attempts:
            observations.append(
                self._metric(
                    context,
                    "operation.first_attempt.success",
                    Semantic.OPERATION_FIRST_ATTEMPT_SUCCESS,
                    sum(_is_successful(span) for span in first_attempts) / len(first_attempts),
                    unit="ratio",
                    summary=True,
                )
            )
        if validation_ids:
            observations.append(
                self._metric(
                    context,
                    "validation.failure.rate",
                    Semantic.VALIDATION_FAILURE_RATE,
                    validation_failures / len(validation_ids),
                    unit="ratio",
                    summary=True,
                )
            )
        input_messages = _message_count(spans, Semantic.MESSAGE_INPUT)
        output_messages = _message_count(spans, Semantic.MESSAGE_OUTPUT)
        if input_messages is not None:
            observations.append(
                self._metric(
                    context,
                    "message.input.count",
                    Semantic.MESSAGE_INPUT_COUNT,
                    input_messages,
                    summary=True,
                )
            )
        if output_messages is not None:
            observations.append(
                self._metric(
                    context,
                    "message.output.count",
                    Semantic.MESSAGE_OUTPUT_COUNT,
                    output_messages,
                    summary=True,
                )
            )
        if input_messages is not None and output_messages is not None:
            observations.append(
                self._metric(
                    context,
                    "message.growth",
                    Semantic.MESSAGE_GROWTH,
                    output_messages - input_messages,
                    summary=True,
                )
            )
        return observations

    def _reference_evidence(
        self,
        trace: Trace,
        context: ExtractionContext,
    ) -> list[Observation]:
        references = _trace_references(trace)
        artifact_count = sum(reference.kind is ReferenceKind.ARTIFACT for reference in references)
        asset_count = sum(
            reference.kind
            in {
                ReferenceKind.ASSET,
                ReferenceKind.PROMPT,
                ReferenceKind.TOOL,
                ReferenceKind.OUTPUT_SCHEMA,
            }
            for reference in references
        )
        return [
            self._metric(
                context,
                "artifact.reference.count",
                Semantic.ARTIFACT_REFERENCE_COUNT,
                artifact_count,
                summary=True,
            ),
            self._metric(
                context,
                "asset.reference.count",
                Semantic.ASSET_REFERENCE_COUNT,
                asset_count,
                summary=True,
            ),
        ]

    def _metric(
        self,
        context: ExtractionContext,
        name: str,
        semantic_type: str,
        value: bool | int | float,
        *,
        unit: str | None = None,
        span: SpanRecord | None = None,
        summary: bool = False,
        tags: dict[str, SerializedValue] | None = None,
    ) -> Observation:
        evidence_tags: dict[str, SerializedValue] = {
            EXTRACTOR_TAG: self.name,
            EXTRACTOR_VERSION_TAG: self.version,
        }
        if summary:
            evidence_tags[SUMMARY_TAG] = True
            evidence_tags[MEASUREMENT_SCOPE_TAG] = MeasurementScope.AGGREGATE.value
        if span is not None:
            evidence_tags.update(
                {
                    ABSTRACTION_LAYER_TAG: span.scope.layer.value,
                    INSTRUMENTOR_TAG: span.scope.instrumentor_name,
                    "operation.kind": span.kind,
                    "operation.name": span.operation,
                }
            )
            logical_operation_id = _logical_operation_id(span.attributes)
            if logical_operation_id is not None:
                evidence_tags[LOGICAL_OPERATION_TAG] = logical_operation_id
        if tags is not None:
            evidence_tags.update(tags)
        suffix = "summary" if span is None else span.span_id
        return Observation(
            id=f"abp_span_v{self.version}_{suffix}_{name}",
            name=name,
            kind=ObservationKind.METRIC,
            semantic_type=semantic_type,
            value=value,
            unit=unit,
            role=ObservationRole.DIAGNOSTIC,
            span_id=None if span is None else span.span_id,
            source=ObservationSource.DERIVED,
            tags=evidence_tags,
            case_id=context.case_id,
            variant_id=context.variant_id,
        )


@dataclass(frozen=True, slots=True)
class _UsageEvidence:
    semantic_type: str
    value: int | float
    unit: str
    span: SpanRecord
    logical_operation_id: str
    authority: int
    source_name: str


class UsageExtractor:
    """Derive accounting-safe LLM usage and model evidence."""

    name = "abp.llm_usage"
    version = "1"

    def extract(
        self,
        trace: Trace,
        *,
        registry: SemanticRegistry,
        context: ExtractionContext,
    ) -> ExtractionResult:
        llm_spans = tuple(span for span in trace.spans if span.kind == KnownSpanKind.LLM)
        direct, aggregate = self._usage_evidence(llm_spans, registry)
        observations: list[Observation] = []
        diagnostics: list[Diagnostic] = []
        for semantic_type in _USAGE_KEYS:
            candidates = tuple(item for item in direct if item.semantic_type == semantic_type)
            if not candidates:
                continue
            selected_layer = min(
                (candidate.span.scope.layer for candidate in candidates),
                key=_layer_priority,
            )
            selected = tuple(
                candidate
                for candidate in candidates
                if candidate.span.scope.layer is selected_layer
            )
            grouped: dict[str, list[_UsageEvidence]] = defaultdict(list)
            for candidate in selected:
                grouped[candidate.logical_operation_id].append(candidate)
            resolved: list[_UsageEvidence] = []
            for logical_operation_id, equivalents in grouped.items():
                candidate, diagnostic = _resolve_usage_equivalents(
                    semantic_type,
                    logical_operation_id,
                    equivalents,
                )
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                if candidate is not None:
                    resolved.append(candidate)
            if not resolved:
                continue
            total = sum(candidate.value for candidate in resolved)
            observations.append(
                self._metric(
                    context,
                    f"{semantic_type}.total",
                    semantic_type,
                    total,
                    unit=resolved[0].unit,
                    layer=selected_layer,
                    summary=True,
                    tags={
                        "abp.logical_operation_count": len(resolved),
                        "abp.source_span_ids": [candidate.span.span_id for candidate in resolved],
                    },
                )
            )
            for candidate in resolved:
                observations.append(
                    self._metric(
                        context,
                        f"{semantic_type}.direct",
                        semantic_type,
                        candidate.value,
                        unit=candidate.unit,
                        layer=selected_layer,
                        span=candidate.span,
                        tags={
                            MEASUREMENT_SCOPE_TAG: MeasurementScope.DIRECT.value,
                            LOGICAL_OPERATION_TAG: candidate.logical_operation_id,
                            "abp.usage_source": candidate.source_name,
                        },
                    )
                )
            diagnostics.extend(
                _aggregate_diagnostics(semantic_type, total, aggregate, selected_layer)
            )
        observations.extend(self._model_evidence(llm_spans, registry, context))
        return ExtractionResult(
            observations=tuple(observations),
            diagnostics=(*trace.diagnostics, *diagnostics),
            references=_trace_references(trace),
        )

    def _usage_evidence(
        self,
        spans: tuple[SpanRecord, ...],
        registry: SemanticRegistry,
    ) -> tuple[list[_UsageEvidence], list[_UsageEvidence]]:
        direct: list[_UsageEvidence] = []
        aggregate: list[_UsageEvidence] = []
        for span in spans:
            logical_operation_id = _span_logical_operation_id(span)
            span_semantics: set[str] = set()
            for measurement in span.measurements:
                semantic_type = registry.normalize(measurement.semantic_type)
                if semantic_type not in _USAGE_KEYS:
                    continue
                span_semantics.add(semantic_type)
                evidence = _UsageEvidence(
                    semantic_type=semantic_type,
                    value=measurement.value,
                    unit=measurement.unit or _USAGE_UNITS[semantic_type],
                    span=span,
                    logical_operation_id=logical_operation_id,
                    authority=_authority(
                        measurement.attributes,
                        measurement.source.system if measurement.source else None,
                    ),
                    source_name="measurement",
                )
                (
                    direct
                    if measurement.measurement_scope is MeasurementScope.DIRECT
                    else aggregate
                ).append(evidence)
            for semantic_type, keys in _USAGE_KEYS.items():
                values = [span.usage[key] for key in keys if key in span.usage]
                for value in values:
                    if isinstance(value, bool) or not isinstance(value, int | float):
                        continue
                    span_semantics.add(semantic_type)
                    direct.append(
                        _UsageEvidence(
                            semantic_type=semantic_type,
                            value=value,
                            unit=_USAGE_UNITS[semantic_type],
                            span=span,
                            logical_operation_id=logical_operation_id,
                            authority=_authority(span.attributes, None),
                            source_name="span.usage",
                        )
                    )
            if Semantic.LLM_REQUEST_COUNT not in span_semantics:
                direct.append(
                    _UsageEvidence(
                        semantic_type=Semantic.LLM_REQUEST_COUNT,
                        value=1,
                        unit="requests",
                        span=span,
                        logical_operation_id=logical_operation_id,
                        authority=4,
                        source_name="span.count",
                    )
                )
        return direct, aggregate

    def _model_evidence(
        self,
        spans: tuple[SpanRecord, ...],
        registry: SemanticRegistry,
        context: ExtractionContext,
    ) -> list[Observation]:
        del registry
        model_facts: list[tuple[str, str, SpanRecord]] = []
        for span in spans:
            for semantic_type, keys in _MODEL_KEYS.items():
                value = _first_text(span.attributes, span.source_attributes, keys)
                if value is not None:
                    model_facts.append((semantic_type, value, span))
        observations: list[Observation] = []
        for semantic_type in _MODEL_KEYS:
            candidates = tuple(fact for fact in model_facts if fact[0] == semantic_type)
            if not candidates:
                continue
            selected_layer = min((fact[2].scope.layer for fact in candidates), key=_layer_priority)
            selected = tuple(fact for fact in candidates if fact[2].scope.layer is selected_layer)
            seen: set[tuple[str, str]] = set()
            for _, value, span in selected:
                logical_operation_id = _span_logical_operation_id(span)
                if (logical_operation_id, value) in seen:
                    continue
                seen.add((logical_operation_id, value))
                observations.append(
                    self._factor(
                        context,
                        f"{semantic_type}.direct",
                        semantic_type,
                        value,
                        span=span,
                        layer=selected_layer,
                        tags={LOGICAL_OPERATION_TAG: logical_operation_id},
                    )
                )
            values = {value for _, value, _ in selected}
            if len(values) == 1:
                observations.insert(
                    len(observations) - len(seen),
                    self._factor(
                        context,
                        f"{semantic_type}.summary",
                        semantic_type,
                        values.pop(),
                        layer=selected_layer,
                        summary=True,
                    ),
                )
        return observations

    def _metric(
        self,
        context: ExtractionContext,
        name: str,
        semantic_type: str,
        value: int | float,
        *,
        unit: str,
        layer: AbstractionLayer,
        span: SpanRecord | None = None,
        summary: bool = False,
        tags: dict[str, SerializedValue] | None = None,
    ) -> Observation:
        evidence_tags = self._tags(layer, span=span, summary=summary, tags=tags)
        suffix = "summary" if span is None else span.span_id
        return Observation(
            id=f"abp_usage_v{self.version}_{suffix}_{name}",
            name=name,
            kind=ObservationKind.METRIC,
            semantic_type=semantic_type,
            value=value,
            unit=unit,
            role=ObservationRole.DIAGNOSTIC,
            span_id=None if span is None else span.span_id,
            source=ObservationSource.DERIVED,
            tags=evidence_tags,
            case_id=context.case_id,
            variant_id=context.variant_id,
        )

    def _factor(
        self,
        context: ExtractionContext,
        name: str,
        semantic_type: str,
        value: str,
        *,
        layer: AbstractionLayer,
        span: SpanRecord | None = None,
        summary: bool = False,
        tags: dict[str, SerializedValue] | None = None,
    ) -> Observation:
        evidence_tags = self._tags(layer, span=span, summary=summary, tags=tags)
        suffix = "summary" if span is None else span.span_id
        return Observation(
            id=f"abp_usage_v{self.version}_{suffix}_{name}",
            name=name,
            kind=ObservationKind.FACTOR,
            semantic_type=semantic_type,
            value=value,
            span_id=None if span is None else span.span_id,
            source=ObservationSource.DERIVED,
            tags=evidence_tags,
            case_id=context.case_id,
            variant_id=context.variant_id,
        )

    def _tags(
        self,
        layer: AbstractionLayer,
        *,
        span: SpanRecord | None,
        summary: bool,
        tags: dict[str, SerializedValue] | None,
    ) -> dict[str, SerializedValue]:
        evidence_tags: dict[str, SerializedValue] = {
            EXTRACTOR_TAG: self.name,
            EXTRACTOR_VERSION_TAG: self.version,
            ABSTRACTION_LAYER_TAG: layer.value,
        }
        if summary:
            evidence_tags[SUMMARY_TAG] = True
            evidence_tags[MEASUREMENT_SCOPE_TAG] = MeasurementScope.AGGREGATE.value
        if span is not None:
            evidence_tags[INSTRUMENTOR_TAG] = span.scope.instrumentor_name
            evidence_tags["operation.name"] = span.operation
        if tags is not None:
            evidence_tags.update(tags)
        return evidence_tags


class CompositeExtractor:
    def __init__(self, *extractors: TraceExtractor) -> None:
        self.extractors = extractors or (
            SignalExtractor(),
            SpanExtractor(),
            UsageExtractor(),
        )
        self.name = "abp.default" if not extractors else "abp.composite"
        self.version = "+".join(
            f"{extractor.name}@{extractor.version}" for extractor in self.extractors
        )

    def extract(
        self,
        trace: Trace,
        *,
        registry: SemanticRegistry,
        context: ExtractionContext,
    ) -> ExtractionResult:
        observations: dict[str, Observation] = {}
        diagnostics: list[Diagnostic] = []
        references: dict[tuple[ReferenceKind, str, str | None], EvidenceRef] = {}
        for extractor in self.extractors:
            result = extractor.extract(trace, registry=registry, context=context)
            observations.update(
                (observation.id, observation) for observation in result.observations
            )
            diagnostics.extend(result.diagnostics)
            for reference in result.references:
                references[(reference.kind, reference.id, reference.version)] = reference
        unique_diagnostics = {
            (
                diagnostic.code,
                diagnostic.signal_id,
                diagnostic.span_id,
                diagnostic.sequence,
            ): diagnostic
            for diagnostic in diagnostics
        }
        return ExtractionResult(
            observations=tuple(observations.values()),
            diagnostics=tuple(unique_diagnostics.values()),
            references=tuple(references.values()),
        )


_USAGE_KEYS: dict[str, tuple[str, ...]] = {
    Semantic.LLM_TOKENS_INPUT: (
        Semantic.LLM_TOKENS_INPUT,
        "input_tokens",
        "prompt_tokens",
    ),
    Semantic.LLM_TOKENS_OUTPUT: (
        Semantic.LLM_TOKENS_OUTPUT,
        "output_tokens",
        "completion_tokens",
    ),
    Semantic.LLM_TOKENS_TOTAL: (Semantic.LLM_TOKENS_TOTAL, "total_tokens"),
    Semantic.LLM_TOKENS_CACHED_INPUT: (
        Semantic.LLM_TOKENS_CACHED_INPUT,
        "cached_input_tokens",
        "cache_read_tokens",
    ),
    Semantic.LLM_TOKENS_CACHE_WRITE: (
        Semantic.LLM_TOKENS_CACHE_WRITE,
        "cache_write_tokens",
    ),
    Semantic.LLM_TOKENS_REASONING_OUTPUT: (
        Semantic.LLM_TOKENS_REASONING_OUTPUT,
        "reasoning_tokens",
        "reasoning_output_tokens",
    ),
    Semantic.LLM_REQUEST_COUNT: (
        Semantic.LLM_REQUEST_COUNT,
        "requests",
        "request_count",
    ),
}
_USAGE_UNITS = {
    semantic_type: "requests" if semantic_type == Semantic.LLM_REQUEST_COUNT else "tokens"
    for semantic_type in _USAGE_KEYS
}
_MODEL_KEYS: dict[str, tuple[str, ...]] = {
    Semantic.LLM_MODEL_REQUESTED: (
        Semantic.LLM_MODEL_REQUESTED,
        "requested_model",
        "request_model",
    ),
    Semantic.LLM_MODEL_RESPONSE: (
        Semantic.LLM_MODEL_RESPONSE,
        "response_model",
        "model",
        "model_name",
    ),
    Semantic.LLM_PROVIDER_NAME: (
        Semantic.LLM_PROVIDER_NAME,
        "provider",
        "provider_name",
    ),
}
_CORRELATION_KEYS = (
    LOGICAL_OPERATION_TAG,
    "logical_operation_id",
    "operation_id",
    "request_id",
    "response_id",
)
_AUTHORITY = {"provider": 0, "reported": 1, "measured": 2, "estimated": 3}
_LAYER_ORDER = {
    AbstractionLayer.CLIENT: 0,
    AbstractionLayer.FRAMEWORK: 1,
    AbstractionLayer.APPLICATION: 2,
    AbstractionLayer.TRANSPORT: 3,
}


def _layer_priority(layer: AbstractionLayer) -> int:
    return _LAYER_ORDER[layer]


def _authority(attributes: Mapping[str, SerializedValue], source_system: str | None) -> int:
    value = attributes.get("usage_authority", attributes.get("authority"))
    if isinstance(value, str) and value in _AUTHORITY:
        return _AUTHORITY[value]
    if source_system is not None and "provider" in source_system.lower():
        return _AUTHORITY["provider"]
    return 4


def _resolve_usage_equivalents(
    semantic_type: str,
    logical_operation_id: str,
    candidates: list[_UsageEvidence],
) -> tuple[_UsageEvidence | None, Diagnostic | None]:
    values = {candidate.value for candidate in candidates}
    sorted_values: list[SerializedValue] = []
    for value in sorted(values):
        sorted_values.append(value)
    if len(values) == 1:
        return min(candidates, key=lambda candidate: candidate.authority), None
    best_authority = min(candidate.authority for candidate in candidates)
    authoritative = [candidate for candidate in candidates if candidate.authority == best_authority]
    authoritative_values = {candidate.value for candidate in authoritative}
    if best_authority < 4 and len(authoritative_values) == 1:
        return authoritative[0], Diagnostic(
            code="direct_measurement_resolved",
            message="conflicting direct measurements were resolved by explicit authority",
            semantic_type=semantic_type,
            details={
                "logical_operation_id": logical_operation_id,
                "selected": authoritative[0].value,
                "values": sorted_values,
            },
        )
    return None, Diagnostic(
        code="ambiguous_direct_measurement",
        message="equivalent direct measurements disagree without a unique authority",
        semantic_type=semantic_type,
        details={
            "logical_operation_id": logical_operation_id,
            "span_ids": [candidate.span.span_id for candidate in candidates],
            "values": sorted_values,
        },
    )


def _aggregate_diagnostics(
    semantic_type: str,
    total: int | float,
    aggregate: list[_UsageEvidence],
    selected_layer: AbstractionLayer,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for candidate in aggregate:
        if candidate.semantic_type != semantic_type:
            continue
        if candidate.span.scope.layer is not selected_layer:
            continue
        if abs(float(candidate.value) - float(total)) <= 1e-9:
            continue
        diagnostics.append(
            Diagnostic(
                code="aggregate_measurement_mismatch",
                message="aggregate measurement disagrees with the direct accounting total",
                span_id=candidate.span.span_id,
                semantic_type=semantic_type,
                details={"aggregate": candidate.value, "direct_total": total},
            )
        )
    return diagnostics


def _accounted_spans(spans: tuple[SpanRecord, ...]) -> tuple[SpanRecord, ...]:
    by_kind: dict[str, list[SpanRecord]] = defaultdict(list)
    for span in spans:
        by_kind[span.kind].append(span)
    accounted: list[SpanRecord] = []
    for candidates in by_kind.values():
        selected_layer = min((span.scope.layer for span in candidates), key=_layer_priority)
        seen: set[str] = set()
        for span in candidates:
            if span.scope.layer is not selected_layer:
                continue
            logical_operation_id = _span_logical_operation_id(span)
            if logical_operation_id in seen:
                continue
            seen.add(logical_operation_id)
            accounted.append(span)
    return tuple(accounted)


def _max_depth(spans: tuple[SpanRecord, ...], links: tuple[Link, ...]) -> int:
    span_ids = {span.span_id for span in spans}
    children: dict[str, set[str]] = defaultdict(set)
    for span in spans:
        if span.parent_span_id in span_ids:
            children[span.parent_span_id].add(span.span_id)
    for link in links:
        if (
            link.relation in {LinkRelation.DELEGATION, LinkRelation.HANDOFF, LinkRelation.FAN_OUT}
            and link.target.trace_id == link.trace_id
            and link.target.span_id in span_ids
        ):
            children[link.span_id].add(link.target.span_id)
    max_depth = 0
    for span in spans:
        pending = [(span.span_id, 1, frozenset({span.span_id}))]
        while pending:
            span_id, depth, path = pending.pop()
            max_depth = max(max_depth, depth)
            pending.extend(
                (child_id, depth + 1, path | {child_id})
                for child_id in children.get(span_id, ())
                if child_id not in path
            )
    return max_depth


def _fan_out(span: SpanRecord, children: list[SpanRecord]) -> int:
    targets: set[tuple[str | None, str | None, str | None]] = {
        (child.trace_id, child.span_id, None) for child in children
    }
    targets.update(
        (link.target.trace_id, link.target.span_id, link.target.run_id)
        for link in span.links
        if link.relation is LinkRelation.FAN_OUT
    )
    return len(targets)


def _is_failed(span: SpanRecord) -> bool:
    return span.status is SpanStatus.ERROR or span.end_reason in {
        EndReason.FAILED,
        EndReason.CANCELLED,
        EndReason.TIMEOUT,
        EndReason.ABANDONED,
    }


def _is_successful(span: SpanRecord) -> bool:
    return span.status is SpanStatus.OK and span.end_reason is EndReason.COMPLETED


def _is_incomplete(span: SpanRecord) -> bool:
    return span.partial or span.end_reason is EndReason.ABANDONED


def _has_tool_arguments(span: SpanRecord) -> bool:
    if any(key in span.attributes for key in ("arguments", "tool_arguments")):
        return True
    return any(event.semantic_type == Semantic.TOOL_CALL_ARGUMENTS for event in span.events)


def _message_count(spans: tuple[SpanRecord, ...], semantic_type: str) -> int | None:
    attribute = (
        "message_input_count" if semantic_type == Semantic.MESSAGE_INPUT else "message_output_count"
    )
    counts: list[int] = []
    for span in spans:
        explicit = span.attributes.get(attribute)
        if isinstance(explicit, int) and not isinstance(explicit, bool):
            counts.append(explicit)
            continue
        counts.extend(
            len(event.body)
            for event in span.events
            if event.semantic_type == semantic_type and isinstance(event.body, list)
        )
    return sum(counts) if counts else None


def _trace_references(trace: Trace) -> tuple[EvidenceRef, ...]:
    references: dict[tuple[ReferenceKind, str, str | None], EvidenceRef] = {}
    for signal in trace.references:
        reference = signal.reference
        references[(reference.kind, reference.id, reference.version)] = reference
    for span in trace.spans:
        for signal in span.references:
            reference = signal.reference
            references[(reference.kind, reference.id, reference.version)] = reference
        for reference in span.errors:
            references[(reference.kind, reference.id, reference.version)] = reference
        if span.output_reference is not None:
            reference = span.output_reference
            references[(reference.kind, reference.id, reference.version)] = reference
        for event in span.events:
            if event.reference is not None:
                reference = event.reference
                references[(reference.kind, reference.id, reference.version)] = reference
    return tuple(references.values())


def _span_logical_operation_id(span: SpanRecord) -> str:
    return (
        _logical_operation_id(span.attributes)
        or _logical_operation_id(span.source_attributes)
        or span.span_id
    )


def _logical_operation_id(attributes: Mapping[str, SerializedValue]) -> str | None:
    for key in _CORRELATION_KEYS:
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_text(
    attributes: Mapping[str, SerializedValue],
    source_attributes: Mapping[str, SerializedValue],
    keys: tuple[str, ...],
) -> str | None:
    for values in (attributes, source_attributes):
        for key in keys:
            value = values.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _observation_id(signal_id: str, attributes: dict[str, SerializedValue]) -> str:
    observation_id = attributes.get("observation_id")
    return observation_id if isinstance(observation_id, str) else f"abp_{signal_id}"


def _span_id(span_id: str, attributes: dict[str, SerializedValue]) -> str:
    legacy_span_id = attributes.get("legacy_span_id")
    return legacy_span_id if isinstance(legacy_span_id, str) else span_id


__all__ = (
    "ABSTRACTION_LAYER_TAG",
    "CompositeExtractor",
    "EXTRACTOR_TAG",
    "EXTRACTOR_VERSION_TAG",
    "ExtractionContext",
    "ExtractionEvidence",
    "ExtractionResult",
    "INSTRUMENTOR_TAG",
    "LOGICAL_OPERATION_TAG",
    "MEASUREMENT_SCOPE_TAG",
    "SUMMARY_TAG",
    "SignalExtractor",
    "SpanExtractor",
    "TraceExtractor",
    "UsageExtractor",
)
