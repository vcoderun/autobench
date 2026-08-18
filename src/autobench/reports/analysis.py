from __future__ import annotations as _annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path, PurePosixPath
from statistics import fmean, median
from typing import Any, Literal

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from autobench.errors import ErrorRecord, SpecLoadError
from autobench.io import load_yaml
from autobench.metrics.observations import Observation, ObservationKind, ObservationSource
from autobench.metrics.semantics import SemanticRegistry
from autobench.protocol.traces import Trace
from autobench.records.artifacts import ArtifactRef
from autobench.records.files import hash_and_size
from autobench.records.models import ExperimentRecord, RecordingError, RunRecord
from autobench.reports.models import (
    AggregationFn,
    ArtifactInventoryReport,
    ArtifactReport,
    AssetLineageReport,
    AssetVersionReport,
    BenchmarkReport,
    ErrorSummaryReport,
    EvaluationCaseReport,
    EvaluationMetricReport,
    EvaluationSummaryReport,
    EvidenceAvailability,
    ExecutiveSummary,
    ExperimentDesignReport,
    FailureReport,
    FileHashReport,
    MarkdownReportConfig,
    MetricAggregation,
    MetricDefinitionReport,
    PolicyOutcomeReport,
    ProvenanceReport,
    ReportEvidenceRef,
    ReportFinding,
    ReportNotice,
    ReportSpec,
    RunDetailReport,
    RunHealthReport,
    ScoreEvidenceReport,
    SlowSpanReport,
    SourceIdentityReport,
    TraceSummaryReport,
)
from autobench.runtime.models import ExperimentResult, RunResult
from autobench.tracking import (
    AssetUse,
    AssetVersion,
    load_asset_content,
    load_asset_diff,
)

_JSON_VALUE = TypeAdapter(JsonValue)


@dataclass
class MetricEvidence:
    names: set[str] = field(default_factory=set)
    units: set[str] = field(default_factory=set)
    directions: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    run_ids: set[str] = field(default_factory=set)
    aggregations: set[AggregationFn] = field(default_factory=set)


@dataclass
class AssetEvidence:
    version: AssetVersion
    uses: list[AssetUse] = field(default_factory=list)
    run_ids: set[str] = field(default_factory=set)
    variant_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RecordedAssetEvidence:
    kind: str | None = None
    semantic_type: str | None = None
    sensitivity: str | None = None
    changed_fields: tuple[str, ...] = ()
    content_database: Path | None = None
    diff_database: Path | None = None
    content: dict[str, JsonValue] | None = None
    diff: str | None = None


def build_source_identity(
    result: ExperimentResult,
    *,
    experiment_record: ExperimentRecord | None,
) -> SourceIdentityReport:
    record = experiment_record
    return SourceIdentityReport(
        record_version=None if record is None else record.record_version,
        benchmark_id=result.benchmark_id,
        experiment_id=result.experiment_id,
        spec_hash=result.spec_hash,
        dataset_id=result.plan.dataset_id,
        dataset_version=result.plan.dataset_version,
        dataset_hash=result.plan.dataset_hash,
        manifest_path=None if record is None else record.manifest_path,
        run_paths=() if record is None else record.run_paths,
        file_hashes=(
            ()
            if record is None
            else tuple(
                FileHashReport(path=file_hash.path, sha256=file_hash.sha256)
                for file_hash in record.file_hashes
            )
        ),
        correlation=result.correlation,
    )


def build_experiment_design(result: ExperimentResult) -> ExperimentDesignReport:
    snapshot = result.spec_snapshot or {}
    benchmark = snapshot.get("benchmark")
    description = benchmark.get("description") if isinstance(benchmark, dict) else None
    scoring = snapshot.get("scoring")
    derive = snapshot.get("derive")
    post_derive = snapshot.get("post_derive")
    policies = snapshot.get("policies")
    instrumentation = snapshot.get("instrumentation")
    instrumentation_kinds = (
        tuple(
            sorted(
                str(config.get("kind"))
                for config in instrumentation
                if isinstance(config, dict) and config.get("kind") is not None
            )
        )
        if isinstance(instrumentation, list)
        else ()
    )
    case_tags = {
        run.case_id: tuple(run.case.tags)
        for run in sorted(result.runs, key=lambda item: (item.case_id, item.run_id))
    }
    return ExperimentDesignReport(
        description=description if isinstance(description, str) else None,
        dataset_id=result.plan.dataset_id,
        dataset_version=result.plan.dataset_version,
        dataset_hash=result.plan.dataset_hash,
        case_count=result.plan.case_count,
        case_ids=result.plan.case_ids,
        case_tags=case_tags,
        variant_count=result.plan.variant_count,
        planned_run_count=result.plan.planned_run_count,
        scorer_count=len(scoring) if isinstance(scoring, list) else 0,
        deriver_count=len(derive) if isinstance(derive, list) else 0,
        post_deriver_count=len(post_derive) if isinstance(post_derive, list) else 0,
        policy_count=len(policies) if isinstance(policies, list) else 0,
        instrumentation=instrumentation_kinds,
        capture_enabled=snapshot.get("capture") is not None,
        warnings=tuple(result.plan.warnings),
    )


def build_run_health(result: ExperimentResult) -> RunHealthReport:
    status_by_variant: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors_by_type: dict[str, list[str]] = defaultdict(list)
    for run in result.runs:
        status_by_variant[run.variant_id][run.status.value] += 1
        for error in _run_errors(run):
            errors_by_type[error.error_type].append(run.run_id)

    termination = result.termination
    missing_ids = termination.missing_run_ids
    if not missing_ids and result.plan.planned_run_count > len(result.runs):
        missing_count = result.plan.planned_run_count - len(result.runs)
    else:
        missing_count = len(missing_ids)
    return RunHealthReport(
        experiment_status=termination.status.value,
        partial=termination.partial,
        planned_count=result.plan.planned_run_count,
        recorded_count=len(result.runs),
        missing_count=missing_count,
        partial_run_count=sum(run.partial for run in result.runs),
        status_counts={
            status: sum(run.status.value == status for run in result.runs)
            for status in sorted({run.status.value for run in result.runs})
        },
        status_by_variant={
            variant_id: dict(sorted(counts.items()))
            for variant_id, counts in sorted(status_by_variant.items())
        },
        missing_run_ids=missing_ids,
        cross_run_derivation_complete=termination.cross_run_derivation_complete,
        policies_complete=termination.policies_complete,
        errors=tuple(
            ErrorSummaryReport(
                error_type=error_type,
                count=len(run_ids),
                run_ids=tuple(sorted(run_ids)),
            )
            for error_type, run_ids in sorted(errors_by_type.items())
        ),
    )


def build_evaluation_summary(result: ExperimentResult) -> EvaluationSummaryReport | None:
    cases = tuple(
        _evaluation_case(run)
        for run in sorted(
            result.runs, key=lambda item: (item.case_id, item.variant_id, item.run_id)
        )
    )
    if not any(
        case.quality_pass is not None or case.score is not None or case.metrics or case.feedback
        for case in cases
    ):
        return None

    evaluated = tuple(case for case in cases if case.quality_pass is not None)
    scores = tuple(case.score for case in cases if case.score is not None)
    passed_count = sum(case.quality_pass is True for case in evaluated)
    return EvaluationSummaryReport(
        case_count=len(cases),
        evaluated_count=len(evaluated),
        passed_count=passed_count,
        failed_count=len(evaluated) - passed_count,
        unevaluated_count=len(cases) - len(evaluated),
        pass_rate=(None if not evaluated else passed_count / len(evaluated)),
        score_count=len(scores),
        mean_score=None if not scores else fmean(scores),
        median_score=None if not scores else median(scores),
        minimum_score=None if not scores else min(scores),
        maximum_score=None if not scores else max(scores),
        metrics=_evaluation_metric_summaries(cases),
        cases=cases,
    )


def _evaluation_case(run: RunResult) -> EvaluationCaseReport:
    payload = _evaluation_mapping(run.task_result.output)
    nested = payload.get("evaluation")
    envelope = nested if isinstance(nested, dict) else payload
    quality_pass = _evaluation_pass(envelope)
    score = _finite_number(envelope.get("score"))
    metrics = _evaluation_metrics(envelope)
    feedback = _evaluation_feedback(envelope)

    for score_record in run.scores:
        normalized_name = score_record.name.casefold().replace("-", "_").replace(" ", "_")
        if quality_pass is None and normalized_name in {"hard_pass", "pass", "passed"}:
            quality_pass = _pass_value(score_record.value)
        numeric = _finite_number(score_record.value)
        if (
            score is None
            and score_record.role is not None
            and score_record.role.value == "objective"
        ):
            score = numeric
        if numeric is not None and normalized_name not in {"hard_pass", "pass", "passed"}:
            metrics.setdefault(score_record.name, numeric)

    return EvaluationCaseReport(
        run_id=run.run_id,
        case_id=run.case_id,
        variant_id=run.variant_id,
        quality_pass=quality_pass,
        score=score,
        metrics=dict(sorted(metrics.items())),
        feedback=feedback,
    )


def _evaluation_mapping(value: Any) -> dict[str, JsonValue]:
    candidate = (
        value.model_dump(mode="json") if isinstance(value, BaseModel) else _json_value(value)
    )
    return candidate if isinstance(candidate, dict) else {}


def _evaluation_pass(payload: dict[str, JsonValue]) -> bool | None:
    for name in ("hard_pass", "passed", "pass"):
        if name in payload:
            return _pass_value(payload[name])
    return None


def _pass_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _evaluation_metrics(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    candidate = payload.get("metrics")
    if not isinstance(candidate, dict):
        return {}
    return {
        str(name): value
        for name, value in candidate.items()
        if value is None or isinstance(value, str | int | float | bool)
    }


def _evaluation_feedback(payload: dict[str, JsonValue]) -> tuple[str, ...]:
    candidate = payload.get("feedback")
    if isinstance(candidate, str):
        return (candidate,)
    if not isinstance(candidate, list):
        return ()
    return tuple(item for item in candidate if isinstance(item, str) and item.strip())


def _evaluation_metric_summaries(
    cases: tuple[EvaluationCaseReport, ...],
) -> tuple[EvaluationMetricReport, ...]:
    names = sorted({name for case in cases for name in case.metrics})
    metrics: list[EvaluationMetricReport] = []
    for name in names:
        values = tuple(
            numeric
            for case in cases
            if (numeric := _evaluation_metric_number(case.metrics.get(name))) is not None
        )
        if not values:
            continue
        metrics.append(
            EvaluationMetricReport(
                name=name,
                label=_evaluation_metric_label(name),
                kind=_evaluation_metric_kind(name, values),
                sample_count=len(values),
                missing_count=len(cases) - len(values),
                mean=fmean(values),
                median=median(values),
                minimum=min(values),
                maximum=max(values),
                total=sum(values),
            )
        )
    kind_order = {"score": 0, "value": 1, "count": 2}
    return tuple(sorted(metrics, key=lambda item: (kind_order[item.kind], item.label, item.name)))


def _evaluation_metric_kind(
    name: str,
    values: tuple[float, ...],
) -> Literal["score", "count", "value"]:
    normalized = name.casefold()
    score_terms = (
        "accuracy",
        "correct",
        "fidelity",
        "grounded",
        "latest_wins",
        "pass",
        "quality",
        "rate",
        "ratio",
        "score",
        "success",
    )
    count_terms = ("count", "omission", "leak", "missing", "invented", "mismatch", "chars")
    if any(term in normalized for term in score_terms) and all(0 <= value <= 1 for value in values):
        return "score"
    if any(term in normalized for term in count_terms):
        return "count"
    return "value"


def _evaluation_metric_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return _finite_number(value)


def _evaluation_metric_label(name: str) -> str:
    labels = {
        "active_state_score": "Active state retention",
        "deterministic_fidelity_score": "Deterministic fidelity",
        "episodic_score": "Episodic retention",
        "judge_critical_omissions": "Critical omissions",
        "judge_grounded": "Grounded answers",
        "judge_latest_wins": "Latest-instruction adherence",
        "judge_score": "Evaluator assessment",
        "reference_score": "Reference preservation",
        "semantic_score": "Semantic fidelity",
        "sensitive_leaks": "Sensitive leaks",
        "forbidden_leaks": "Forbidden leaks",
    }
    return labels.get(name, name.replace("_", " ").strip().capitalize())


def build_metric_catalog(
    result: ExperimentResult,
    *,
    report_spec: ReportSpec,
    registry: SemanticRegistry,
) -> tuple[tuple[MetricDefinitionReport, ...], tuple[ReportNotice, ...]]:
    evidence: dict[str, MetricEvidence] = defaultdict(MetricEvidence)
    for metric in _configured_metrics(report_spec):
        semantic_type = registry.normalize(metric.semantic_type) or metric.semantic_type
        item = evidence[semantic_type]
        item.names.add(metric.name)
        item.aggregations.add(metric.fn)

    for run in result.runs:
        for score in run.scores:
            semantic_type = registry.normalize(score.semantic_type) or score.semantic_type
            item = evidence[semantic_type]
            item.names.add(score.name)
            item.run_ids.add(run.run_id)
            item.sources.add("score")
            if score.unit is not None:
                item.units.add(score.unit)
            if score.direction is not None:
                item.directions.add(score.direction.value)
            if score.role is not None:
                item.roles.add(score.role.value)
        for observation in run.task_result.observations:
            _add_observation_evidence(evidence, observation, run_id=run.run_id, registry=registry)

    catalog: list[MetricDefinitionReport] = []
    notices: list[ReportNotice] = []
    for semantic_type, item in sorted(evidence.items()):
        semantic_info = registry.types.get(semantic_type)
        units = set(item.units)
        if not units and semantic_info is not None and semantic_info.unit is not None:
            units.add(semantic_info.unit)
        if len(units) > 1:
            notices.append(
                ReportNotice(
                    code="metric_unit_conflict",
                    severity="warning",
                    message=f"Metric {semantic_type} was recorded with multiple units.",
                    evidence_ids=tuple(sorted(units)),
                )
            )
        if len(item.directions) > 1:
            notices.append(
                ReportNotice(
                    code="metric_direction_conflict",
                    severity="warning",
                    message=f"Metric {semantic_type} was recorded with multiple directions.",
                    evidence_ids=tuple(sorted(item.directions)),
                )
            )
        catalog.append(
            MetricDefinitionReport(
                name=sorted(item.names)[0] if item.names else semantic_type,
                semantic_type=semantic_type,
                unit=sorted(units)[0] if len(units) == 1 else None,
                direction=(sorted(item.directions)[0] if len(item.directions) == 1 else None),
                role=sorted(item.roles)[0] if len(item.roles) == 1 else None,
                aggregation=(sorted(item.aggregations)[0] if len(item.aggregations) == 1 else None),
                sources=tuple(sorted(item.sources)),
                observed_count=len(item.run_ids),
                missing_count=max(0, len(result.runs) - len(item.run_ids)),
                description=None if semantic_info is None else semantic_info.description,
            )
        )
    return tuple(catalog), tuple(notices)


def build_run_details(
    result: ExperimentResult,
    *,
    markdown: MarkdownReportConfig,
) -> tuple[RunDetailReport, ...]:
    details: list[RunDetailReport] = []
    include_values = markdown.profile == "audit" and markdown.content.include_captured
    excerpt_limit = markdown.limits.value_excerpt_chars
    for run in sorted(result.runs, key=lambda item: (item.case_id, item.variant_id, item.run_id)):
        metrics = {
            score.semantic_type: score.value for score in run.scores if score.value is not None
        }
        for observation in run.task_result.observations:
            if observation.kind is ObservationKind.METRIC and observation.semantic_type is not None:
                semantic_type = (
                    result.semantic_registry.normalize(observation.semantic_type)
                    or observation.semantic_type
                )
                metrics.setdefault(semantic_type, observation.value)
        error = run.error or run.task_result.error
        details.append(
            RunDetailReport(
                run_id=run.run_id,
                case_id=run.case_id,
                variant_id=run.variant_id,
                status=run.status.value,
                evaluation_status=run.evaluation_status.value,
                partial=run.partial,
                end_reason=run.end_reason.value,
                metrics=dict(sorted(metrics.items())),
                score_count=len(run.scores),
                span_count=(
                    len(run.trace.spans) if run.trace is not None else len(run.task_result.spans)
                ),
                asset_count=len({version.asset_id for version in run.asset_versions}),
                artifact_count=len(run.task_result.artifacts),
                parent_run_id=run.parent_run_id,
                error_type=None if error is None else error.error_type,
                error_message=None if error is None else error.message,
                input_excerpt=(
                    _optional_value_excerpt(run.case.input, limit=excerpt_limit)
                    if include_values
                    else None
                ),
                expected_excerpt=(
                    _optional_value_excerpt(run.case.expected, limit=excerpt_limit)
                    if include_values
                    else None
                ),
                output_excerpt=(
                    _optional_value_excerpt(run.task_result.output, limit=excerpt_limit)
                    if include_values
                    else None
                ),
                score_evidence=(
                    tuple(
                        ScoreEvidenceReport(
                            name=score.name,
                            actual_excerpt=_value_excerpt(
                                score.actual_value,
                                limit=excerpt_limit,
                            ),
                            expected_excerpt=_value_excerpt(
                                score.expected_value,
                                limit=excerpt_limit,
                            ),
                        )
                        for score in run.scores
                        if score.actual_value is not None or score.expected_value is not None
                    )
                    if include_values
                    else ()
                ),
            )
        )
    return tuple(details)


def build_policy_outcomes(
    result: ExperimentResult,
    *,
    registry: SemanticRegistry,
) -> tuple[PolicyOutcomeReport, ...]:
    outcomes: list[PolicyOutcomeReport] = []
    for run in sorted(result.runs, key=lambda item: (item.case_id, item.variant_id, item.run_id)):
        for observation in run.task_result.observations:
            semantic_type = (
                None
                if observation.semantic_type is None
                else registry.normalize(observation.semantic_type) or observation.semantic_type
            )
            if (
                observation.kind is not ObservationKind.EVENT
                or semantic_type != "policy.result"
                or not isinstance(observation.value, bool)
            ):
                continue
            metric = observation.tags.get("metric")
            reason = observation.tags.get("reason")
            outcomes.append(
                PolicyOutcomeReport(
                    name=observation.name,
                    run_id=run.run_id,
                    case_id=run.case_id,
                    variant_id=run.variant_id,
                    passed=observation.value,
                    metric=metric if isinstance(metric, str) else None,
                    actual=_json_value(observation.tags.get("actual")),
                    reason=reason if isinstance(reason, str) else None,
                )
            )
    return tuple(outcomes)


def build_failures(
    result: ExperimentResult,
    *,
    include_tracebacks: bool,
) -> tuple[FailureReport, ...]:
    failures: list[FailureReport] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for run in sorted(result.runs, key=lambda item: item.run_id):
        staged_errors = [("run", run.error), ("task", run.task_result.error)]
        staged_errors.extend(("task", error) for error in run.task_result.errors)
        staged_errors.extend((f"score:{score.name}", score.error) for score in run.scores)
        for stage, error in staged_errors:
            if error is None:
                continue
            identity = (run.run_id, error.error_type, error.message, error.span_id)
            if identity in seen:
                continue
            seen.add(identity)
            failures.append(
                FailureReport(
                    run_id=run.run_id,
                    case_id=run.case_id,
                    variant_id=run.variant_id,
                    stage=stage,
                    error_type=error.error_type,
                    message=error.message,
                    span_id=error.span_id,
                    traceback=error.traceback if include_tracebacks else None,
                )
            )
    return tuple(failures)


def build_trace_summary(
    result: ExperimentResult,
    *,
    top_slowest: int,
) -> TraceSummaryReport | None:
    traced_runs: list[tuple[RunResult, Trace]] = []
    for run in result.runs:
        if run.trace is not None:
            traced_runs.append((run, run.trace))
    if not traced_runs:
        return None
    spans_by_kind: Counter[str] = Counter()
    spans_by_instrumentor: Counter[str] = Counter()
    diagnostics_by_code: Counter[str] = Counter()
    slow_spans: list[SlowSpanReport] = []
    root_count = 0
    orphan_count = 0
    error_count = 0
    cancelled_count = 0
    partial_trace_count = 0
    span_count = 0
    for run, trace in traced_runs:
        partial_trace_count += trace.partial
        root_count += len(trace.root_span_ids)
        span_ids = {span.span_id for span in trace.spans}
        for diagnostic in trace.diagnostics:
            diagnostics_by_code[diagnostic.code] += 1
        for span in trace.spans:
            span_count += 1
            spans_by_kind[span.kind] += 1
            spans_by_instrumentor[span.scope.instrumentor_name] += 1
            orphan_count += span.parent_span_id is not None and span.parent_span_id not in span_ids
            error_count += span.status.value == "error"
            cancelled_count += span.end_reason is not None and span.end_reason.value == "cancelled"
            if span.duration_ns is not None:
                slow_spans.append(
                    SlowSpanReport(
                        run_id=run.run_id,
                        span_id=span.span_id,
                        operation=span.operation,
                        kind=span.kind,
                        instrumentor=span.scope.instrumentor_name,
                        duration_ns=span.duration_ns,
                        status=span.status.value,
                        partial=span.partial,
                    )
                )
    return TraceSummaryReport(
        traced_run_count=len(traced_runs),
        untraced_run_count=len(result.runs) - len(traced_runs),
        complete_trace_count=len(traced_runs) - partial_trace_count,
        partial_trace_count=partial_trace_count,
        span_count=span_count,
        root_span_count=root_count,
        orphan_span_count=orphan_count,
        error_span_count=error_count,
        cancelled_span_count=cancelled_count,
        spans_by_kind=dict(sorted(spans_by_kind.items())),
        spans_by_instrumentor=dict(sorted(spans_by_instrumentor.items())),
        diagnostics_by_code=dict(sorted(diagnostics_by_code.items())),
        slow_spans=tuple(
            sorted(
                slow_spans,
                key=lambda item: (-item.duration_ns, item.run_id, item.span_id),
            )[:top_slowest]
        ),
    )


def build_asset_lineage(
    result: ExperimentResult,
    *,
    experiment_root: Path | None = None,
    markdown: MarkdownReportConfig | None = None,
) -> AssetLineageReport | None:
    evidence: dict[tuple[str, str], AssetEvidence] = {}
    for run in result.runs:
        uses = {(use.asset_id, use.version): use for use in run.asset_uses}
        for version in run.asset_versions:
            key = (version.asset_id, version.version)
            item = evidence.setdefault(key, AssetEvidence(version=version))
            item.run_ids.add(run.run_id)
            item.variant_ids.add(run.variant_id)
            use = uses.get(key)
            if use is not None and use not in item.uses:
                item.uses.append(use)
    if not evidence:
        return None
    active_markdown = markdown or MarkdownReportConfig()
    recorded = _recorded_asset_evidence(
        experiment_root,
        keys=tuple(sorted(evidence)),
        markdown=active_markdown,
    )
    versions = tuple(
        AssetVersionReport(
            asset_id=item.version.asset_id,
            version=item.version.version,
            parent_version=item.version.parent_version,
            content_hash=item.version.content_hash,
            source_hash=item.version.source_hash,
            source_path=item.version.source_path,
            git_commit=item.version.git_commit,
            kind=recorded_item.kind,
            semantic_type=recorded_item.semantic_type,
            sensitivity=recorded_item.sensitivity,
            representations=tuple(sorted({use.representation.value for use in item.uses})),
            scopes=tuple(sorted({use.scope for use in item.uses if use.scope is not None})),
            source_locators=tuple(sorted({use.source_locator for use in item.uses})),
            definition_asset_ids=tuple(
                sorted(
                    {
                        use.definition_asset_id
                        for use in item.uses
                        if use.definition_asset_id is not None
                    }
                )
            ),
            provenance=tuple(sorted({_asset_provenance_label(use) for use in item.uses})),
            changed_fields=(
                recorded_item.changed_fields if active_markdown.assets.diffs != "none" else ()
            ),
            content_state=_asset_content_state(recorded_item, markdown=active_markdown),
            diff_state=_asset_diff_state(recorded_item, markdown=active_markdown),
            content_excerpt=_asset_content_excerpt(
                recorded_item,
                markdown=active_markdown,
            ),
            diff_excerpt=_asset_diff_excerpt(
                recorded_item,
                markdown=active_markdown,
            ),
            run_ids=tuple(sorted(item.run_ids)),
            variant_ids=tuple(sorted(item.variant_ids)),
        )
        for key, item in sorted(evidence.items())
        for recorded_item in (recorded.get(key, RecordedAssetEvidence()),)
    )
    return AssetLineageReport(
        asset_count=len({version.asset_id for version in versions}),
        version_count=len(versions),
        transition_count=sum(version.parent_version is not None for version in versions),
        versions=versions,
    )


def build_artifact_inventory(
    result: ExperimentResult,
    *,
    experiment_root: Path | None,
    experiment_record: ExperimentRecord | None = None,
) -> tuple[ArtifactInventoryReport | None, tuple[ReportNotice, ...]]:
    artifacts: list[ArtifactReport] = []
    notices: list[ReportNotice] = []
    recorded_artifacts = _recorded_artifacts(
        experiment_root,
        experiment_record=experiment_record,
    )
    for run in sorted(result.runs, key=lambda item: item.run_id):
        run_artifacts = recorded_artifacts.get(run.run_id, tuple(run.task_result.artifacts))
        for artifact in sorted(run_artifacts, key=lambda item: item.id):
            safe_path: str | None = None
            if experiment_root is not None and isinstance(artifact.value, str):
                candidate = PurePosixPath(artifact.value.replace("\\", "/"))
                unsafe = (
                    candidate.is_absolute()
                    or ".." in candidate.parts
                    or (candidate.parts and candidate.parts[0].endswith(":"))
                )
                if unsafe:
                    notices.append(
                        ReportNotice(
                            code="unsafe_artifact_path",
                            severity="warning",
                            message=f"Artifact {artifact.id} has a non-portable path.",
                            evidence_ids=(run.run_id, artifact.id),
                        )
                    )
                else:
                    safe_path = candidate.as_posix()
                    root = experiment_root.resolve()
                    payload = (root / safe_path).resolve()
                    if not payload.is_relative_to(root):
                        safe_path = None
                        notices.append(
                            ReportNotice(
                                code="unsafe_artifact_path",
                                severity="warning",
                                message=f"Artifact {artifact.id} resolves outside the record.",
                                evidence_ids=(run.run_id, artifact.id),
                            )
                        )
                    elif not payload.is_file():
                        safe_path = None
                        notices.append(
                            ReportNotice(
                                code="missing_artifact_file",
                                severity="warning",
                                message=f"Artifact {artifact.id} payload is missing from the record.",
                                evidence_ids=(run.run_id, artifact.id),
                            )
                        )
                    elif artifact.sha256 is not None or artifact.byte_count is not None:
                        digest, byte_count = hash_and_size(payload)
                        if digest != artifact.sha256 or byte_count != artifact.byte_count:
                            safe_path = None
                            notices.append(
                                ReportNotice(
                                    code="artifact_integrity_mismatch",
                                    severity="error",
                                    message=(
                                        f"Artifact {artifact.id} payload does not match recorded "
                                        "hash and size evidence."
                                    ),
                                    evidence_ids=(run.run_id, artifact.id),
                                )
                            )
            artifacts.append(
                ArtifactReport(
                    artifact_id=artifact.id,
                    name=artifact.name,
                    run_id=run.run_id,
                    span_id=artifact.span_id,
                    media_type=artifact.media_type,
                    source=artifact.source.value,
                    state=artifact.state.value,
                    sha256=artifact.sha256,
                    byte_count=artifact.byte_count,
                    path=safe_path,
                )
            )
    if not artifacts:
        return None, tuple(notices)
    states = Counter(artifact.state for artifact in artifacts)
    return (
        ArtifactInventoryReport(
            artifact_count=len(artifacts),
            complete_count=states["complete"],
            partial_count=states["partial"],
            truncated_count=states["truncated"],
            artifacts=tuple(artifacts),
        ),
        tuple(notices),
    )


def build_provenance(
    result: ExperimentResult,
    *,
    markdown: MarkdownReportConfig,
) -> ProvenanceReport:
    include_captured = markdown.profile == "audit" and markdown.content.include_captured
    working_directory = result.environment.cwd
    if not include_captured:
        working_directory = Path(working_directory).name or "."
    source_maps = {
        f"{snapshot.system}:{snapshot.source_map_id}@{snapshot.source_map_version}"
        for run in result.runs
        for snapshot in run.source_snapshots
    }
    return ProvenanceReport(
        python_version=result.environment.python_version,
        platform=result.environment.platform,
        working_directory=working_directory,
        working_directory_redacted=not include_captured,
        spec_hash=result.spec_hash,
        semantic_registry_version=result.semantic_registry.version,
        source_maps=tuple(sorted(source_maps)),
    )


def build_executive_summary(report: BenchmarkReport) -> ExecutiveSummary:
    health = report.health
    evaluation = report.evaluation
    if evaluation is not None and evaluation.evaluated_count:
        health_text = (
            f"{evaluation.passed_count} of {evaluation.evaluated_count} evaluated cases met the "
            "recorded quality gate."
        )
        if evaluation.mean_score is not None:
            health_text += f" The average score was {evaluation.mean_score:.1%}."
    elif health is None:
        health_text = f"{report.run_count} recorded runs; lifecycle metadata unavailable."
    else:
        health_text = (
            f"{health.recorded_count}/{health.planned_count} runs recorded; "
            f"{health.status_counts.get('passed', 0)} passed, "
            f"{health.status_counts.get('failed', 0)} failed, "
            f"{health.status_counts.get('errored', 0)} errored."
        )

    leader_variant, objective_metric = _objective_leader(report)
    findings: list[ReportFinding] = []
    if evaluation is not None and evaluation.evaluated_count:
        findings.append(
            ReportFinding(
                kind="evaluation",
                severity="warning" if evaluation.failed_count else "info",
                title="Quality gate outcome",
                statement=(
                    f"{evaluation.passed_count} of {evaluation.evaluated_count} evaluated cases "
                    f"passed ({evaluation.pass_rate:.1%})."
                ),
                evidence=(ReportEvidenceRef(kind="evaluation", id=report.experiment_id),),
            )
        )
        scored_cases = tuple(
            (case, case.score) for case in evaluation.cases if case.score is not None
        )
        if scored_cases:
            strongest, strongest_score = max(
                scored_cases,
                key=lambda item: (item[1], item[0].case_id),
            )
            weakest, weakest_score = min(
                scored_cases,
                key=lambda item: (item[1], item[0].case_id),
            )
            findings.append(
                ReportFinding(
                    kind="evaluation",
                    severity="info",
                    title="Score range",
                    statement=(
                        f"Scores ranged from {weakest_score:.1%} on {weakest.case_id} to "
                        f"{strongest_score:.1%} on {strongest.case_id}."
                    ),
                    evidence=(
                        ReportEvidenceRef(kind="run", id=weakest.run_id),
                        ReportEvidenceRef(kind="run", id=strongest.run_id),
                    ),
                )
            )
    if health is not None and (
        health.partial
        or health.missing_count
        or not health.cross_run_derivation_complete
        or not health.policies_complete
    ):
        findings.append(
            ReportFinding(
                kind="health",
                severity="error" if health.missing_count else "warning",
                title="Experiment evidence is incomplete",
                statement=(
                    f"{health.recorded_count} of {health.planned_count} planned runs were recorded; "
                    f"{health.missing_count} runs are missing."
                ),
                evidence=(ReportEvidenceRef(kind="experiment", id=report.experiment_id),),
            )
        )
    if (
        len(report.variant_configs) > 1
        and leader_variant is not None
        and objective_metric is not None
    ):
        findings.append(
            ReportFinding(
                kind="leader",
                severity="info",
                title="Objective leader",
                statement=(
                    f"Variant {leader_variant} has the best recorded {objective_metric} value "
                    "among variants without an observed boolean constraint failure."
                ),
                evidence=(
                    ReportEvidenceRef(kind="metric", id=objective_metric),
                    ReportEvidenceRef(kind="run", id=f"variant:{leader_variant}"),
                ),
            )
        )
    elif len(report.variant_configs) > 1:
        findings.append(
            ReportFinding(
                kind="limitation",
                severity="info",
                title="No defensible overall winner",
                statement=(
                    "The recorded evidence does not identify one unique direction-aware objective "
                    "leader with valid observed constraints."
                ),
                evidence=(ReportEvidenceRef(kind="experiment", id=report.experiment_id),),
            )
        )

    for comparison in report.comparisons:
        comparison_id = f"{comparison.baseline}->{comparison.candidate}"
        if comparison.confounded:
            findings.append(
                ReportFinding(
                    kind="comparison",
                    severity="warning",
                    title="Comparison is confounded",
                    statement=(
                        f"{comparison.baseline} and {comparison.candidate} differ across "
                        f"{len(comparison.factor_deltas)} factors; metric changes are associated "
                        "with the combined configuration."
                    ),
                    evidence=(ReportEvidenceRef(kind="comparison", id=comparison_id),),
                )
            )
        for metric in comparison.metric_results:
            delta = metric.delta
            if delta is None or metric.outcome not in {"improved", "regressed"}:
                continue
            findings.append(
                ReportFinding(
                    kind="comparison",
                    severity="warning" if metric.outcome == "regressed" else "info",
                    title=f"{metric.name} {metric.outcome}",
                    statement=(
                        f"{comparison.candidate} is {metric.outcome} versus "
                        f"{comparison.baseline} for {metric.name} by {delta:+.6g}."
                    ),
                    evidence=(
                        ReportEvidenceRef(kind="comparison", id=comparison_id),
                        ReportEvidenceRef(kind="metric", id=metric.semantic_type),
                    ),
                )
            )

    if health is not None and health.planned_count:
        unhealthy = sum(
            health.status_counts.get(status, 0) for status in ("failed", "errored", "cancelled")
        )
        if unhealthy / health.planned_count >= 0.2:
            findings.append(
                ReportFinding(
                    kind="health",
                    severity="warning",
                    title="High unsuccessful run rate",
                    statement=(
                        f"{unhealthy} of {health.planned_count} planned runs failed, errored, "
                        "or were cancelled."
                    ),
                    evidence=(ReportEvidenceRef(kind="experiment", id=report.experiment_id),),
                )
            )

    if report.markdown.profile == "audit":
        for metric in report.metric_catalog:
            if not metric.missing_count:
                continue
            findings.append(
                ReportFinding(
                    kind="coverage",
                    severity="warning",
                    title=f"Missing {metric.name} evidence",
                    statement=(
                        f"{metric.semantic_type} is missing from {metric.missing_count} of "
                        f"{report.run_count} recorded runs."
                    ),
                    evidence=(ReportEvidenceRef(kind="metric", id=metric.semantic_type),),
                )
            )

    failed_policies = tuple(policy for policy in report.policies if not policy.passed)
    if failed_policies:
        findings.append(
            ReportFinding(
                kind="constraint",
                severity="warning",
                title="Policy constraints failed",
                statement=(
                    f"{len(failed_policies)} of {len(report.policies)} recorded policy outcomes "
                    "failed."
                ),
                evidence=tuple(
                    ReportEvidenceRef(kind="run", id=policy.run_id) for policy in failed_policies
                ),
            )
        )

    if report.markdown.profile == "audit" and report.traces is not None:
        trace = report.traces
        if trace.partial_trace_count:
            findings.append(
                ReportFinding(
                    kind="trace",
                    severity="warning",
                    title="Partial trace evidence",
                    statement=f"{trace.partial_trace_count} recorded traces are partial.",
                    evidence=(ReportEvidenceRef(kind="trace", id=report.experiment_id),),
                )
            )
        accounting_codes = tuple(code for code in trace.diagnostics_by_code if "account" in code)
        if accounting_codes:
            findings.append(
                ReportFinding(
                    kind="trace",
                    severity="warning",
                    title="Trace accounting disagreement",
                    statement="ABP diagnostics report inconsistent accounting evidence.",
                    evidence=tuple(
                        ReportEvidenceRef(kind="trace", id=code) for code in accounting_codes
                    ),
                )
            )

    if (
        report.markdown.profile == "audit"
        and report.assets is not None
        and report.assets.transition_count
    ):
        findings.append(
            ReportFinding(
                kind="asset",
                severity="info",
                title="Tracked assets changed",
                statement=(
                    f"{report.assets.transition_count} asset version transitions appear in the "
                    "recorded evidence; associated metric changes are not causal attribution."
                ),
                evidence=tuple(
                    ReportEvidenceRef(kind="asset", id=version.asset_id)
                    for version in report.assets.versions
                    if version.parent_version is not None
                ),
            )
        )
    if report.optimization_warnings:
        findings.append(
            ReportFinding(
                kind="optimization",
                severity="warning",
                title="Optimization evidence warnings",
                statement=f"{len(report.optimization_warnings)} optimizer evidence warnings exist.",
                evidence=(ReportEvidenceRef(kind="optimization", id=report.experiment_id),),
            )
        )

    severity_order = {"error": 0, "warning": 1, "info": 2}
    unique: dict[tuple[str, str, tuple[tuple[str, str], ...]], ReportFinding] = {}
    for finding in findings:
        identity = (
            finding.kind,
            finding.title,
            tuple((evidence.kind, evidence.id) for evidence in finding.evidence),
        )
        unique.setdefault(identity, finding)
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda finding: (
                severity_order[finding.severity],
                finding.kind,
                finding.title,
                tuple((item.kind, item.id) for item in finding.evidence),
            ),
        )
    )
    return ExecutiveSummary(
        health=health_text,
        leader_variant=leader_variant,
        objective_metric=objective_metric,
        findings=ordered,
    )


def _objective_leader(report: BenchmarkReport) -> tuple[str | None, str | None]:
    objective_names = {
        detail.name
        for row in report.leaderboard
        for detail in row.metric_details
        if detail.role == "objective" and detail.direction in {"maximize", "minimize"}
    }
    for objective_name in sorted(objective_names):
        leaders = [
            row.variant_id
            for row in report.leaderboard
            if not any(
                detail.role == "constraint" and detail.value is False
                for detail in row.metric_details
            )
            and any(detail.name == objective_name and detail.best for detail in row.metric_details)
        ]
        if len(leaders) == 1:
            return leaders[0], objective_name
    return None, None


def _configured_metrics(report_spec: ReportSpec) -> tuple[MetricAggregation, ...]:
    metrics = list(report_spec.leaderboard_metrics())
    for comparison in report_spec.comparisons:
        metrics.extend(comparison.resolved_metrics())
    metrics.extend(
        MetricAggregation(
            name=distribution.name,
            semantic_type=distribution.semantic_type,
            fn=distribution.summaries[0] if distribution.summaries else "count",
        )
        for distribution in report_spec.distributions
    )
    metrics.append(
        MetricAggregation(
            name=report_spec.case_matrix.semantic_type,
            semantic_type=report_spec.case_matrix.semantic_type,
            fn="mean",
        )
    )
    return tuple(metrics)


def _add_observation_evidence(
    evidence: dict[str, MetricEvidence],
    observation: Observation,
    *,
    run_id: str,
    registry: SemanticRegistry,
) -> None:
    if observation.kind is not ObservationKind.METRIC or observation.semantic_type is None:
        return
    semantic_type = registry.normalize(observation.semantic_type) or observation.semantic_type
    item = evidence[semantic_type]
    item.names.add(observation.name)
    item.run_ids.add(run_id)
    if observation.source is not None:
        source = observation.source
        item.sources.add(source.value if isinstance(source, ObservationSource) else source)
    if observation.unit is not None:
        item.units.add(observation.unit)
    if observation.direction is not None:
        item.directions.add(observation.direction.value)
    if observation.role is not None:
        item.roles.add(observation.role.value)


def _value_excerpt(value: Any, *, limit: int) -> str:
    normalized = _json_value(value)
    if normalized is None and value is not None:
        value_type = type(value)
        public_module = value_type.__module__.split("._", maxsplit=1)[0]
        return f"<unavailable:{public_module}.{value_type.__qualname__}>"
    rendered = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(rendered) <= limit:
        return rendered
    omitted = len(rendered) - limit
    return f"{rendered[:limit]}... [truncated {omitted} chars]"


def _optional_value_excerpt(value: Any, *, limit: int) -> str | None:
    return None if value is None else _value_excerpt(value, limit=limit)


def _json_value(value: Any) -> JsonValue:
    try:
        return _JSON_VALUE.validate_python(value)
    except ValidationError:
        return None


def _asset_provenance_label(use: AssetUse) -> str:
    provenance = use.provenance
    instrumentor = "" if provenance.instrumentor is None else f"@{provenance.instrumentor}"
    return f"{provenance.system}:{provenance.key}{instrumentor}"


def _recorded_asset_evidence(
    experiment_root: Path | None,
    *,
    keys: tuple[tuple[str, str], ...],
    markdown: MarkdownReportConfig,
) -> dict[tuple[str, str], RecordedAssetEvidence]:
    if experiment_root is None:
        return {}
    root = experiment_root.resolve()
    assets_root = (root / "assets").resolve()
    index_path = assets_root / "index.yaml"
    try:
        index_payload = load_yaml(index_path)
    except (OSError, SpecLoadError):
        return {}
    if not isinstance(index_payload, dict):
        return {}
    raw_index = index_payload.get("assets")
    if not isinstance(raw_index, dict):
        return {}

    recorded: dict[tuple[str, str], RecordedAssetEvidence] = {}
    for asset_id, version in keys:
        index_entry = raw_index.get(asset_id)
        if not isinstance(index_entry, dict):
            continue
        relative_file = index_entry.get("file")
        if not isinstance(relative_file, str):
            continue
        asset_path = (assets_root / relative_file).resolve()
        if not asset_path.is_relative_to(assets_root) or not asset_path.is_file():
            continue
        try:
            asset_payload = load_yaml(asset_path)
        except (OSError, SpecLoadError):
            continue
        if not isinstance(asset_payload, dict):
            continue
        raw_asset = asset_payload.get("asset")
        raw_versions = asset_payload.get("versions")
        if not isinstance(raw_asset, dict) or not isinstance(raw_versions, list):
            continue
        raw_version = next(
            (
                candidate
                for candidate in raw_versions
                if isinstance(candidate, dict) and candidate.get("version") == version
            ),
            None,
        )
        if raw_version is None:
            continue
        content_database = _asset_database_path(root, raw_version.get("content_ref"))
        changes = raw_version.get("changes")
        raw_fields = changes.get("fields") if isinstance(changes, dict) else None
        changed_fields = (
            tuple(field for field in raw_fields if isinstance(field, str))
            if isinstance(raw_fields, list)
            else ()
        )
        diff_database = (
            _asset_database_path(root, changes.get("diff_ref"))
            if isinstance(changes, dict)
            else None
        )
        kind = raw_asset.get("kind")
        semantic_type = raw_asset.get("semantic")
        sensitivity = raw_asset.get("sensitivity")
        can_inline = (
            markdown.profile == "audit"
            and markdown.content.include_captured
            and sensitivity != "sensitive"
        )
        content = (
            _load_recorded_asset_content(content_database, asset_id=asset_id, version=version)
            if can_inline
            else None
        )
        parent_version = raw_version.get("parent")
        diff = (
            _load_recorded_asset_diff(
                diff_database,
                asset_id=asset_id,
                version=version,
                parent_version=parent_version,
            )
            if can_inline and markdown.assets.diffs == "full" and isinstance(parent_version, str)
            else None
        )
        recorded[(asset_id, version)] = RecordedAssetEvidence(
            kind=kind if isinstance(kind, str) else None,
            semantic_type=semantic_type if isinstance(semantic_type, str) else None,
            sensitivity=sensitivity if isinstance(sensitivity, str) else None,
            changed_fields=changed_fields,
            content_database=content_database,
            diff_database=diff_database,
            content=content,
            diff=diff,
        )
    return recorded


def _recorded_artifacts(
    experiment_root: Path | None,
    *,
    experiment_record: ExperimentRecord | None,
) -> dict[str, tuple[ArtifactRef, ...]]:
    if experiment_root is None or experiment_record is None:
        return {}
    from autobench.records.views import run_record_payload_from_yaml_view

    root = experiment_root.resolve()
    artifacts: dict[str, tuple[ArtifactRef, ...]] = {}
    for relative_path in experiment_record.run_paths:
        run_path = (root / relative_path).resolve()
        if not run_path.is_relative_to(root) or not run_path.is_file():
            continue
        try:
            payload = load_yaml(run_path)
            record = RunRecord.model_validate(run_record_payload_from_yaml_view(payload))
        except (OSError, RecordingError, SpecLoadError, ValueError):
            continue
        artifacts[record.run_id] = record.artifacts
    return artifacts


def _asset_database_path(root: Path, raw_reference: Any) -> Path | None:
    if not isinstance(raw_reference, dict):
        return None
    raw_path = raw_reference.get("path")
    if not isinstance(raw_path, str):
        return None
    relative = PurePosixPath(raw_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    path = (root / relative.as_posix()).resolve()
    return path if path.is_relative_to(root) else None


def _load_recorded_asset_content(
    database: Path | None,
    *,
    asset_id: str,
    version: str,
) -> dict[str, JsonValue] | None:
    if database is None:
        return None
    try:
        return load_asset_content(database, asset_id=asset_id, version=version)
    except (FileNotFoundError, KeyError, sqlite3.DatabaseError):
        return None


def _load_recorded_asset_diff(
    database: Path | None,
    *,
    asset_id: str,
    version: str,
    parent_version: str,
) -> str | None:
    if database is None:
        return None
    try:
        return load_asset_diff(
            database,
            asset_id=asset_id,
            version=version,
            parent_version=parent_version,
        )
    except (FileNotFoundError, KeyError, sqlite3.DatabaseError):
        return None


def _asset_content_state(
    recorded: RecordedAssetEvidence,
    *,
    markdown: MarkdownReportConfig,
) -> EvidenceAvailability:
    if recorded.content_database is None or not recorded.content_database.is_file():
        return "unavailable"
    if recorded.sensitivity == "sensitive":
        return "omitted"
    if markdown.profile == "audit" and markdown.content.include_captured:
        return "available" if recorded.content is not None else "unavailable"
    return "available"


def _asset_diff_state(
    recorded: RecordedAssetEvidence,
    *,
    markdown: MarkdownReportConfig,
) -> EvidenceAvailability:
    if markdown.assets.diffs == "none" or recorded.sensitivity == "sensitive":
        return "omitted"
    if recorded.diff_database is None or not recorded.diff_database.is_file():
        return "unavailable"
    if (
        markdown.assets.diffs == "full"
        and markdown.profile == "audit"
        and markdown.content.include_captured
    ):
        return "available" if recorded.diff is not None else "unavailable"
    return "available"


def _asset_content_excerpt(
    recorded: RecordedAssetEvidence,
    *,
    markdown: MarkdownReportConfig,
) -> str | None:
    if (
        markdown.profile != "audit"
        or not markdown.content.include_captured
        or recorded.sensitivity == "sensitive"
        or recorded.content is None
    ):
        return None
    return _value_excerpt(recorded.content, limit=markdown.limits.value_excerpt_chars)


def _asset_diff_excerpt(
    recorded: RecordedAssetEvidence,
    *,
    markdown: MarkdownReportConfig,
) -> str | None:
    if (
        markdown.assets.diffs != "full"
        or markdown.profile != "audit"
        or not markdown.content.include_captured
        or recorded.sensitivity == "sensitive"
        or recorded.diff is None
    ):
        return None
    limit = markdown.limits.value_excerpt_chars
    if len(recorded.diff) <= limit:
        return recorded.diff
    omitted = len(recorded.diff) - limit
    return f"{recorded.diff[:limit]}... [truncated {omitted} chars]"


def _run_errors(run: RunResult) -> tuple[ErrorRecord, ...]:
    errors = list(run.task_result.errors)
    if run.task_result.error is not None and run.task_result.error not in errors:
        errors.append(run.task_result.error)
    if run.error is not None and run.error not in errors:
        errors.append(run.error)
    errors.extend(score.error for score in run.scores if score.error is not None)
    return tuple(errors)


__all__ = (
    "build_evaluation_summary",
    "build_experiment_design",
    "build_metric_catalog",
    "build_run_health",
    "build_source_identity",
)
