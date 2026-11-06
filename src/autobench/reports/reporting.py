from __future__ import annotations as _annotations

from collections import defaultdict
from collections.abc import Sequence
from math import isclose
from pathlib import Path
from typing import Any

from autobench.instrumentation.pydantic_gepa.projection import (
    EXTENSION_KEY as PYDANTIC_GEPA_EXTENSION,
)
from autobench.instrumentation.pydantic_gepa.projection import PydanticGEPAEvidence
from autobench.metrics.observations import Direction, ObservationKind
from autobench.metrics.projection import observation_priority
from autobench.metrics.semantics import SemanticRegistry
from autobench.records.models import ExperimentRecord
from autobench.reports.analysis import (
    build_artifact_inventory,
    build_asset_lineage,
    build_evaluation_summary,
    build_executive_summary,
    build_experiment_design,
    build_failures,
    build_metric_catalog,
    build_policy_outcomes,
    build_provenance,
    build_run_details,
    build_run_health,
    build_source_identity,
    build_trace_summary,
)
from autobench.reports.comparison import compare_variants
from autobench.reports.inference import build_hypothesis_results
from autobench.reports.markdown import render_markdown_report
from autobench.reports.models import (
    DEFAULT_LEADERBOARD_METRICS,
    AggregationFn,
    BenchmarkReport,
    CaseMatrix,
    CaseMatrixReportSpec,
    ComparisonReport,
    ComparisonReportSpec,
    CorrelatedReportGroup,
    DistributionReportSpec,
    EvaluationCaseReport,
    EvaluationMetricReport,
    EvaluationSummaryReport,
    FactorReport,
    LeaderboardMetricReport,
    LeaderboardReportSpec,
    LeaderboardRow,
    MarkdownAssetConfig,
    MarkdownContentConfig,
    MarkdownReportConfig,
    MarkdownReportLimits,
    MarkdownTraceConfig,
    MetricAggregation,
    MetricDistribution,
    OptimizationRunReport,
    RegressionReport,
    ReportLayout,
    ReportProfile,
    ReportSpec,
    RunMetricRow,
    VariantConfigRow,
)
from autobench.reports.trajectory import build_asset_trajectory, compare_asset_versions
from autobench.reports.values import (
    aggregate_values,
    metric_metadata,
    metric_observation,
    metric_value,
    numeric_value,
)
from autobench.runtime.models import ExecutionCorrelation, ExperimentResult, RunResult


def build_report(
    result: ExperimentResult,
    *,
    registry: SemanticRegistry | None = None,
    report_spec: ReportSpec | None = None,
    experiment_record: ExperimentRecord | None = None,
    experiment_root: Path | None = None,
) -> BenchmarkReport:
    active_registry = registry or result.semantic_registry
    active_report_spec = report_spec
    if active_report_spec is None and result.report_spec_data is not None:
        active_report_spec = ReportSpec.model_validate(result.report_spec_data)
    if active_report_spec is None:
        active_report_spec = ReportSpec()
    optimizations, optimization_warnings = build_optimization_runs(result)
    metric_catalog, report_notices = build_metric_catalog(
        result,
        report_spec=active_report_spec,
        registry=active_registry,
    )
    artifact_inventory, artifact_notices = build_artifact_inventory(
        result,
        experiment_root=experiment_root,
        experiment_record=experiment_record,
    )
    include_tracebacks = (
        active_report_spec.markdown.profile == "audit"
        and active_report_spec.markdown.content.include_captured
    )
    comparisons = [
        _build_comparison(result, comparison, registry=active_registry)
        for comparison in active_report_spec.comparisons
    ]
    hypotheses = build_hypothesis_results(active_report_spec.comparisons, comparisons)
    trajectories = tuple(
        build_asset_trajectory(
            result,
            asset_id=trajectory.asset,
            metrics=trajectory.resolved_metrics(),
            registry=active_registry,
            case_tags=trajectory.case_tags,
            variants=trajectory.variants,
        )
        for trajectory in active_report_spec.trajectories
    )
    report = BenchmarkReport(
        benchmark_id=result.benchmark_id,
        experiment_id=result.experiment_id,
        run_count=result.total_count,
        markdown=active_report_spec.markdown,
        source=build_source_identity(result, experiment_record=experiment_record),
        evaluation=build_evaluation_summary(result),
        design=build_experiment_design(result),
        health=build_run_health(result),
        metric_catalog=metric_catalog,
        notices=(*report_notices, *artifact_notices),
        status_counts=build_status_counts(result),
        variant_configs=build_variant_configs(result),
        leaderboard=build_leaderboard(
            result,
            metrics=active_report_spec.leaderboard_metrics(),
            registry=active_registry,
        ),
        run_metrics=build_run_metric_rows(result, registry=active_registry),
        run_details=build_run_details(result, markdown=active_report_spec.markdown),
        failures=build_failures(result, include_tracebacks=include_tracebacks),
        traces=build_trace_summary(
            result,
            top_slowest=active_report_spec.markdown.traces.top_slowest,
        ),
        assets=build_asset_lineage(
            result,
            experiment_root=experiment_root,
            markdown=active_report_spec.markdown,
        ),
        artifacts=artifact_inventory,
        policies=build_policy_outcomes(result, registry=active_registry),
        provenance=build_provenance(result, markdown=active_report_spec.markdown),
        case_matrix=build_case_matrix(
            result,
            semantic_type=active_report_spec.case_matrix.semantic_type,
            registry=active_registry,
        ),
        comparisons=comparisons,
        hypotheses=hypotheses,
        trajectories=trajectories,
        regressions=build_regressions(comparisons),
        distributions=[
            build_metric_distribution(
                result,
                name=distribution.name,
                semantic_type=distribution.semantic_type,
                summaries=distribution.summaries,
                registry=active_registry,
            )
            for distribution in active_report_spec.distributions
        ],
        optimizations=optimizations,
        optimization_warnings=optimization_warnings,
        correlation=result.correlation,
    )
    return report.model_copy(update={"summary": build_executive_summary(report)})


def build_optimization_runs(
    result: ExperimentResult,
) -> tuple[list[OptimizationRunReport], list[str]]:
    reports: list[OptimizationRunReport] = []
    warnings: list[str] = []
    for run in result.runs:
        payload = run.extensions.get(PYDANTIC_GEPA_EXTENSION)
        if payload is None:
            continue
        try:
            evidence = PydanticGEPAEvidence.model_validate(payload)
        except ValueError as error:
            warnings.append(f"run {run.run_id}: invalid pydantic-gepa evidence: {error}")
            continue
        reports.extend(
            OptimizationRunReport(
                benchmark_run_id=run.run_id,
                case_id=run.case_id,
                variant_id=run.variant_id,
                execution=execution,
            )
            for execution in evidence.executions
        )
    return reports, warnings


def correlation_matches(
    actual: ExecutionCorrelation | None,
    expected: ExecutionCorrelation,
) -> bool:
    supplied = expected.model_fields_set
    if actual is None:
        return not supplied
    if "group_id" in supplied and actual.group_id != expected.group_id:
        return False
    if "attempt" in supplied and actual.attempt != expected.attempt:
        return False
    if "phase" in supplied and actual.phase != expected.phase:
        return False
    if (
        "parent_experiment_id" in supplied
        and actual.parent_experiment_id != expected.parent_experiment_id
    ):
        return False
    if (
        "resumed_from_experiment_id" in supplied
        and actual.resumed_from_experiment_id != expected.resumed_from_experiment_id
    ):
        return False
    return "labels" not in supplied or all(
        actual.labels.get(key) == value for key, value in expected.labels.items()
    )


def filter_experiments(
    results: Sequence[ExperimentResult],
    *,
    correlation: ExecutionCorrelation,
) -> list[ExperimentResult]:
    return [result for result in results if correlation_matches(result.correlation, correlation)]


def build_grouped_reports(
    results: Sequence[ExperimentResult],
    *,
    correlation: ExecutionCorrelation | None = None,
) -> list[CorrelatedReportGroup]:
    selected = (
        list(results)
        if correlation is None
        else filter_experiments(
            results,
            correlation=correlation,
        )
    )
    grouped: dict[str | None, list[ExperimentResult]] = {}
    for result in selected:
        group_id = None if result.correlation is None else result.correlation.group_id
        grouped.setdefault(group_id, []).append(result)
    return [
        CorrelatedReportGroup(
            group_id=group_id,
            attempts=tuple(
                sorted(
                    {
                        result.correlation.attempt
                        for result in group_results
                        if result.correlation is not None and result.correlation.attempt is not None
                    }
                )
            ),
            phases=tuple(
                sorted(
                    {
                        result.correlation.phase
                        for result in group_results
                        if result.correlation is not None and result.correlation.phase is not None
                    }
                )
            ),
            reports=[build_report(result) for result in group_results],
        )
        for group_id, group_results in grouped.items()
    ]


def build_status_counts(result: ExperimentResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in result.runs:
        counts[run.status.value] = counts.get(run.status.value, 0) + 1
    return dict(sorted(counts.items()))


def build_variant_configs(result: ExperimentResult) -> list[VariantConfigRow]:
    grouped: dict[str, dict[str, Any]] = {}
    labels = _variant_labels(result)
    for run in result.runs:
        factor_values = grouped.setdefault(run.variant_id, {})
        for factor in run.factors:
            factor_values[factor.name] = factor.value
    return [
        VariantConfigRow(
            variant_id=variant_id,
            label=labels.get(variant_id),
            factors=grouped[variant_id],
            factor_details=tuple(
                FactorReport(
                    name=factor.name,
                    value=factor.value,
                    semantic_type=factor.semantic_type,
                    optimize=factor.optimize,
                )
                for factor in next(
                    run.factors for run in result.runs if run.variant_id == variant_id
                )
            ),
        )
        for variant_id in sorted(grouped)
    ]


def build_run_metric_rows(
    result: ExperimentResult,
    *,
    registry: SemanticRegistry | None = None,
) -> list[RunMetricRow]:
    active_registry = registry or result.semantic_registry
    return [
        RunMetricRow(
            case_id=run.case_id,
            variant_id=run.variant_id,
            status=run.status.value,
            metrics=_run_metric_values(run, registry=active_registry),
        )
        for run in result.runs
    ]


def build_leaderboard(
    result: ExperimentResult,
    *,
    metrics: tuple[MetricAggregation, ...] = DEFAULT_LEADERBOARD_METRICS,
    registry: SemanticRegistry | None = None,
) -> list[LeaderboardRow]:
    active_registry = registry or result.semantic_registry
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for run in result.runs:
        grouped[run.variant_id].append(run)

    rows: list[LeaderboardRow] = []
    for variant_id in sorted(grouped):
        runs = grouped[variant_id]
        values: dict[str, Any] = {}
        details: list[LeaderboardMetricReport] = []
        for metric in metrics:
            samples = [
                value
                for run in runs
                if (value := metric_value(run, metric.semantic_type, registry=active_registry))
                is not None
            ]
            value = aggregate_values(samples, metric.fn)
            values[metric.name] = value
            direction, unit, role = metric_metadata(
                runs,
                metric.semantic_type,
                registry=active_registry,
            )
            details.append(
                LeaderboardMetricReport(
                    name=metric.name,
                    semantic_type=active_registry.normalize(metric.semantic_type)
                    or metric.semantic_type,
                    value=value,
                    sample_count=len(samples),
                    missing_count=len(runs) - len(samples),
                    unit=unit,
                    direction=direction,
                    role=role,
                )
            )
        rows.append(
            LeaderboardRow(
                variant_id=variant_id,
                run_count=len(runs),
                metrics=values,
                metric_details=tuple(details),
            )
        )
    return _mark_leaderboard_bests(rows)


def build_case_matrix(
    result: ExperimentResult,
    *,
    semantic_type: str,
    registry: SemanticRegistry | None = None,
) -> CaseMatrix:
    active_registry = registry or result.semantic_registry
    rows: dict[str, dict[str, Any]] = defaultdict(dict)
    for run in result.runs:
        rows[run.case_id][run.variant_id] = metric_value(
            run,
            semantic_type,
            registry=active_registry,
        )
    return CaseMatrix(
        metric=semantic_type, rows={case_id: dict(values) for case_id, values in rows.items()}
    )


def build_metric_distribution(
    result: ExperimentResult,
    *,
    name: str,
    semantic_type: str,
    summaries: tuple[AggregationFn, ...] = ("min", "median", "p95", "max"),
    registry: SemanticRegistry | None = None,
) -> MetricDistribution:
    active_registry = registry or result.semantic_registry
    by_variant: dict[str, list[Any]] = defaultdict(list)
    for run in result.runs:
        value = metric_value(run, semantic_type, registry=active_registry)
        if value is None:
            continue
        by_variant[run.variant_id].append(value)
    summary_by_variant: dict[str, dict[str, Any]] = {
        variant_id: {
            str(summary_name): aggregate_values(values, summary_name) for summary_name in summaries
        }
        for variant_id, values in sorted(by_variant.items())
    }
    return MetricDistribution(
        name=name,
        semantic_type=semantic_type,
        by_variant=dict(sorted(by_variant.items())),
        summaries=summary_by_variant,
    )


def _run_metric_values(
    run: RunResult,
    *,
    registry: SemanticRegistry,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    observations = sorted(
        run.task_result.observations,
        key=observation_priority,
    )
    for observation in observations:
        if observation.kind is not ObservationKind.METRIC:
            continue
        metric_key = observation.name
        semantic_type = observation.normalized_semantic_type(registry)
        if semantic_type is not None:
            metric_key = f"{observation.name} ({semantic_type})"
        values.setdefault(metric_key, observation.value)
    return values


def _variant_labels(result: ExperimentResult) -> dict[str, str | None]:
    spec_snapshot = result.spec_snapshot
    if not isinstance(spec_snapshot, dict):
        return {}
    raw_variants = _snapshot_variants(spec_snapshot)
    if not isinstance(raw_variants, list):
        return {}
    labels: dict[str, str | None] = {}
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            continue
        variant_id = raw_variant.get("id")
        if not isinstance(variant_id, str):
            continue
        label = raw_variant.get("label")
        labels[variant_id] = label if isinstance(label, str) else None
    return labels


def _snapshot_variants(spec_snapshot: dict[str, Any]) -> Any:
    raw_variants = spec_snapshot.get("variants")
    if isinstance(raw_variants, list):
        return raw_variants
    nested_spec = spec_snapshot.get("spec")
    if isinstance(nested_spec, dict):
        return nested_spec.get("variants")
    return None


def _build_comparison(
    result: ExperimentResult,
    spec: ComparisonReportSpec,
    *,
    registry: SemanticRegistry,
) -> ComparisonReport:
    if spec.asset is None:
        return compare_variants(
            result,
            baseline=spec.baseline,
            candidate=spec.candidate,
            metrics=spec.resolved_metrics(),
            registry=registry,
            case_tags=spec.case_tags,
            name=spec.name,
        )
    # Version selectors such as root and leaf say nothing about which asset they belong to, so
    # an asset comparison is always reported under its label.
    return compare_asset_versions(
        result,
        asset_id=spec.asset,
        baseline=spec.baseline,
        candidate=spec.candidate,
        metrics=spec.resolved_metrics(),
        registry=registry,
        case_tags=spec.case_tags,
        variants=spec.variants,
        name=spec.label(),
    )


def build_regressions(
    comparisons: Sequence[ComparisonReport],
) -> tuple[RegressionReport, ...]:
    reports = [
        RegressionReport(
            baseline=comparison.baseline,
            candidate=comparison.candidate,
            metric=metric.name,
            semantic_type=metric.semantic_type,
            outcome=metric.outcome,
            delta=metric.delta,
            relative_delta=metric.relative_delta,
        )
        for comparison in comparisons
        for metric in comparison.metric_results
        if metric.delta is not None and metric.outcome in {"improved", "regressed"}
    ]
    return tuple(
        sorted(
            reports,
            key=lambda report: (
                report.outcome != "regressed",
                -abs(report.delta),
                report.baseline,
                report.candidate,
                report.metric,
            ),
        )
    )


def _mark_leaderboard_bests(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    best_values: dict[str, float] = {}
    metric_directions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for metric in row.metric_details:
            if metric.direction is not None:
                metric_directions[metric.name].add(metric.direction)
    for row in rows:
        for metric in row.metric_details:
            numeric = numeric_value(metric.value)
            directions = metric_directions[metric.name]
            if (
                numeric is None
                or len(directions) != 1
                or metric.direction
                not in {
                    Direction.MAXIMIZE.value,
                    Direction.MINIMIZE.value,
                }
            ):
                continue
            current = best_values.get(metric.name)
            if current is None:
                best_values[metric.name] = numeric
            elif metric.direction == Direction.MAXIMIZE.value:
                best_values[metric.name] = max(current, numeric)
            else:
                best_values[metric.name] = min(current, numeric)

    return [
        row.model_copy(
            update={
                "metric_details": tuple(
                    metric.model_copy(
                        update={
                            "best": (
                                (numeric := numeric_value(metric.value)) is not None
                                and metric.name in best_values
                                and isclose(numeric, best_values[metric.name], abs_tol=1e-12)
                            )
                        }
                    )
                    for metric in row.metric_details
                )
            }
        )
        for row in rows
    ]


__all__ = (
    "AggregationFn",
    "BenchmarkReport",
    "CaseMatrix",
    "CaseMatrixReportSpec",
    "ComparisonReportSpec",
    "ComparisonReport",
    "DEFAULT_LEADERBOARD_METRICS",
    "DistributionReportSpec",
    "EvaluationCaseReport",
    "EvaluationMetricReport",
    "EvaluationSummaryReport",
    "LeaderboardReportSpec",
    "LeaderboardRow",
    "MarkdownAssetConfig",
    "MarkdownContentConfig",
    "MarkdownReportConfig",
    "MarkdownReportLimits",
    "MarkdownTraceConfig",
    "MetricDistribution",
    "MetricAggregation",
    "OptimizationRunReport",
    "ReportLayout",
    "ReportProfile",
    "ReportSpec",
    "RunMetricRow",
    "VariantConfigRow",
    "aggregate_values",
    "build_case_matrix",
    "build_leaderboard",
    "build_metric_distribution",
    "build_optimization_runs",
    "build_report",
    "build_run_metric_rows",
    "build_status_counts",
    "build_variant_configs",
    "compare_variants",
    "metric_observation",
    "metric_value",
    "render_markdown_report",
)
