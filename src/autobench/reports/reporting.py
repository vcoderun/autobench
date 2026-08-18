from __future__ import annotations as _annotations

from collections import defaultdict
from collections.abc import Sequence
from math import isclose, prod
from pathlib import Path
from statistics import median, pstdev
from typing import Any

from autobench.instrumentation.pydantic_gepa.projection import (
    EXTENSION_KEY as PYDANTIC_GEPA_EXTENSION,
)
from autobench.instrumentation.pydantic_gepa.projection import PydanticGEPAEvidence
from autobench.metrics.observations import (
    Direction,
    Observation,
    ObservationKind,
)
from autobench.metrics.projection import observation_priority
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
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
from autobench.reports.markdown import render_markdown_report
from autobench.reports.models import (
    DEFAULT_LEADERBOARD_METRICS,
    AggregationFn,
    BenchmarkReport,
    CaseMatrix,
    CaseMatrixReportSpec,
    ComparisonOutcome,
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
    MetricComparisonReport,
    MetricDistribution,
    OptimizationRunReport,
    RegressionReport,
    ReportLayout,
    ReportProfile,
    ReportSpec,
    RunMetricRow,
    VariantConfigRow,
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
        compare_variants(
            result,
            baseline=comparison.baseline,
            candidate=comparison.candidate,
            metrics=comparison.resolved_metrics(),
            registry=active_registry,
        )
        for comparison in active_report_spec.comparisons
    ]
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
            direction, unit, role = _metric_metadata(
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


def compare_variants(
    result: ExperimentResult,
    *,
    baseline: str,
    candidate: str,
    metrics: tuple[MetricAggregation, ...] = DEFAULT_LEADERBOARD_METRICS,
    registry: SemanticRegistry | None = None,
) -> ComparisonReport:
    active_registry = registry or result.semantic_registry
    baseline_runs = [run for run in result.runs if run.variant_id == baseline]
    candidate_runs = [run for run in result.runs if run.variant_id == candidate]
    factor_deltas = _factor_deltas(baseline_runs, candidate_runs)
    metric_deltas: dict[str, dict[str, Any]] = {}
    metric_results: list[MetricComparisonReport] = []
    baseline_by_case = {run.case_id: run for run in baseline_runs}
    candidate_by_case = {run.case_id: run for run in candidate_runs}
    paired_case_ids = tuple(sorted(set(baseline_by_case) & set(candidate_by_case)))
    all_case_ids = set(baseline_by_case) | set(candidate_by_case)

    for metric in metrics:
        baseline_samples = [
            value
            for run in baseline_runs
            if (value := metric_value(run, metric.semantic_type, registry=active_registry))
            is not None
        ]
        candidate_samples = [
            value
            for run in candidate_runs
            if (value := metric_value(run, metric.semantic_type, registry=active_registry))
            is not None
        ]
        baseline_value = aggregate_values(
            baseline_samples,
            metric.fn,
        )
        candidate_value = aggregate_values(
            candidate_samples,
            metric.fn,
        )
        delta = _numeric_delta(baseline_value, candidate_value)
        metric_deltas[metric.name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
        direction, unit, _ = _metric_metadata(
            [*baseline_runs, *candidate_runs],
            metric.semantic_type,
            registry=active_registry,
        )
        wins, ties, losses, paired_metric_count = _paired_outcomes(
            paired_case_ids,
            baseline_by_case=baseline_by_case,
            candidate_by_case=candidate_by_case,
            semantic_type=metric.semantic_type,
            direction=direction,
            registry=active_registry,
        )
        metric_results.append(
            MetricComparisonReport(
                name=metric.name,
                semantic_type=active_registry.normalize(metric.semantic_type)
                or metric.semantic_type,
                aggregation=metric.fn,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=delta,
                relative_delta=_relative_delta(baseline_value, delta),
                direction=direction,
                unit=unit,
                outcome=_comparison_outcome(delta, direction),
                baseline_count=len(baseline_samples),
                candidate_count=len(candidate_samples),
                paired_count=paired_metric_count,
                missing_pair_count=len(paired_case_ids) - paired_metric_count,
                wins=wins,
                ties=ties,
                losses=losses,
            )
        )

    return ComparisonReport(
        baseline=baseline,
        candidate=candidate,
        run_count=min(len(baseline_runs), len(candidate_runs)),
        factor_deltas=factor_deltas,
        metric_deltas=metric_deltas,
        confounded=len(factor_deltas) > 1,
        baseline_factors=_factor_map(baseline_runs),
        candidate_factors=_factor_map(candidate_runs),
        paired_count=len(paired_case_ids),
        missing_pair_count=len(all_case_ids) - len(paired_case_ids),
        metric_results=tuple(metric_results),
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


def metric_value(
    run: RunResult,
    semantic_type: str,
    *,
    registry: SemanticRegistry | None = None,
) -> Any | None:
    observation = metric_observation(run, semantic_type, registry=registry)
    return observation.value if observation is not None else None


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


def metric_observation(
    run: RunResult,
    semantic_type: str,
    *,
    registry: SemanticRegistry | None = None,
) -> Observation | None:
    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    normalized = active_registry.normalize(semantic_type)
    candidates = [
        observation
        for observation in run.task_result.observations
        if observation.kind is ObservationKind.METRIC
        and observation.normalized_semantic_type(active_registry) == normalized
    ]
    candidates.extend(
        score.to_observation(
            observation_id=f"{run.run_id}_report_score_{index}",
            case_id=run.case_id,
            variant_id=run.variant_id,
        )
        for index, score in enumerate(run.scores)
        if active_registry.normalize(score.semantic_type) == normalized
    )
    if not candidates:
        return None
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (*observation_priority(item[1]), item[0]),
    )
    return ordered[0][1]


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
            numeric = _numeric_value(metric.value)
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
                                (numeric := _numeric_value(metric.value)) is not None
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


def _metric_metadata(
    runs: Sequence[RunResult],
    semantic_type: str,
    *,
    registry: SemanticRegistry,
) -> tuple[str | None, str | None, str | None]:
    observations = [
        observation
        for run in runs
        if (observation := metric_observation(run, semantic_type, registry=registry)) is not None
    ]
    directions = {item.direction.value for item in observations if item.direction is not None}
    units = {item.unit for item in observations if item.unit is not None}
    roles = {item.role.value for item in observations if item.role is not None}
    semantic_info = registry.types.get(registry.normalize(semantic_type) or semantic_type)
    if not units and semantic_info is not None and semantic_info.unit is not None:
        units.add(semantic_info.unit)
    return (
        next(iter(directions)) if len(directions) == 1 else None,
        next(iter(units)) if len(units) == 1 else None,
        next(iter(roles)) if len(roles) == 1 else None,
    )


def _paired_outcomes(
    case_ids: Sequence[str],
    *,
    baseline_by_case: dict[str, RunResult],
    candidate_by_case: dict[str, RunResult],
    semantic_type: str,
    direction: str | None,
    registry: SemanticRegistry,
) -> tuple[int, int, int, int]:
    wins = 0
    ties = 0
    losses = 0
    paired_count = 0
    for case_id in case_ids:
        baseline = _numeric_value(
            metric_value(baseline_by_case[case_id], semantic_type, registry=registry)
        )
        candidate = _numeric_value(
            metric_value(candidate_by_case[case_id], semantic_type, registry=registry)
        )
        if baseline is None or candidate is None:
            continue
        paired_count += 1
        outcome = _comparison_outcome(candidate - baseline, direction)
        if outcome == "improved":
            wins += 1
        elif outcome == "regressed":
            losses += 1
        elif outcome == "unchanged":
            ties += 1
    return wins, ties, losses, paired_count


def _numeric_delta(baseline: Any, candidate: Any) -> float | None:
    baseline_number = _numeric_value(baseline)
    candidate_number = _numeric_value(candidate)
    if baseline_number is None or candidate_number is None:
        return None
    return candidate_number - baseline_number


def _relative_delta(baseline: Any, delta: float | None) -> float | None:
    baseline_number = _numeric_value(baseline)
    if baseline_number is None or delta is None or isclose(baseline_number, 0.0, abs_tol=1e-12):
        return None
    return delta / abs(baseline_number)


def _comparison_outcome(delta: float | None, direction: str | None) -> ComparisonOutcome:
    if delta is None or direction not in {
        Direction.MAXIMIZE.value,
        Direction.MINIMIZE.value,
    }:
        return "indeterminate"
    if isclose(delta, 0.0, abs_tol=1e-12):
        return "unchanged"
    if direction == Direction.MAXIMIZE.value:
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def aggregate_values(values: list[Any], fn: AggregationFn) -> Any | None:
    if not values:
        return None
    if fn == "count":
        return len(values)
    if fn == "ratio_true":
        return sum(1 for value in values if bool(value)) / len(values)

    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if not numeric_values:
        return None
    if fn == "mean":
        return sum(numeric_values) / len(numeric_values)
    if fn == "sum":
        return sum(numeric_values)
    if fn == "min":
        return min(numeric_values)
    if fn == "max":
        return max(numeric_values)
    if fn == "median":
        return median(numeric_values)
    if fn == "p95":
        return _percentile(numeric_values, 95.0)
    if fn == "stddev":
        return pstdev(numeric_values)
    if fn == "geomean":
        if any(value <= 0.0 for value in numeric_values):
            return None
        return prod(numeric_values) ** (1.0 / len(numeric_values))
    raise ValueError(f"Unsupported aggregation: {fn}")


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * weight
    )


def _factor_deltas(
    baseline_runs: list[RunResult],
    candidate_runs: list[RunResult],
) -> dict[str, dict[str, Any]]:
    baseline_factors = _factor_map(baseline_runs)
    candidate_factors = _factor_map(candidate_runs)
    deltas: dict[str, dict[str, Any]] = {}
    for name in sorted(set(baseline_factors) | set(candidate_factors)):
        baseline_value = baseline_factors.get(name)
        candidate_value = candidate_factors.get(name)
        if baseline_value != candidate_value:
            deltas[name] = {"baseline": baseline_value, "candidate": candidate_value}
    return deltas


def _factor_map(runs: list[RunResult]) -> dict[str, Any]:
    if not runs:
        return {}
    return {factor.name: factor.value for factor in runs[0].factors}


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
