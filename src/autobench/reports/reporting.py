from __future__ import annotations as _annotations

from collections import defaultdict
from math import prod
from statistics import median, pstdev
from typing import Any, Literal

from pydantic import BaseModel, Field

from autobench.metrics.observations import Observation, ObservationKind
from autobench.metrics.projection import source_priority
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, Semantic, SemanticRegistry
from autobench.runtime.pipeline import ExperimentResult, RunResult

AggregationFn = Literal[
    "count",
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "p95",
    "stddev",
    "geomean",
    "ratio_true",
]


class MetricAggregation(BaseModel):
    name: str
    semantic_type: str
    fn: AggregationFn


class LeaderboardReportSpec(BaseModel):
    metrics: tuple[MetricAggregation, ...] = ()


class CaseMatrixReportSpec(BaseModel):
    semantic_type: str = Semantic.COVERAGE_RATIO


class ComparisonReportSpec(BaseModel):
    baseline: str
    candidate: str
    metrics: tuple[MetricAggregation, ...] = ()

    def resolved_metrics(self) -> tuple[MetricAggregation, ...]:
        if self.metrics:
            return self.metrics
        return DEFAULT_LEADERBOARD_METRICS


class DistributionReportSpec(BaseModel):
    name: str
    semantic_type: str
    summaries: tuple[AggregationFn, ...] = ("min", "median", "p95", "max")


class ReportSpec(BaseModel):
    leaderboard: LeaderboardReportSpec = Field(default_factory=LeaderboardReportSpec)
    case_matrix: CaseMatrixReportSpec = Field(default_factory=CaseMatrixReportSpec)
    comparisons: tuple[ComparisonReportSpec, ...] = ()
    distributions: tuple[DistributionReportSpec, ...] = ()

    def leaderboard_metrics(self) -> tuple[MetricAggregation, ...]:
        if self.leaderboard.metrics:
            return self.leaderboard.metrics
        return DEFAULT_LEADERBOARD_METRICS


class LeaderboardRow(BaseModel):
    variant_id: str
    run_count: int
    metrics: dict[str, Any] = Field(default_factory=dict)


class VariantConfigRow(BaseModel):
    variant_id: str
    label: str | None = None
    factors: dict[str, Any] = Field(default_factory=dict)


class RunMetricRow(BaseModel):
    case_id: str
    variant_id: str
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class CaseMatrix(BaseModel):
    metric: str
    rows: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ComparisonReport(BaseModel):
    baseline: str
    candidate: str
    run_count: int
    factor_deltas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metric_deltas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    confounded: bool = False


class MetricDistribution(BaseModel):
    name: str
    semantic_type: str
    by_variant: dict[str, list[Any]] = Field(default_factory=dict)
    summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    benchmark_id: str
    experiment_id: str
    run_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    variant_configs: list[VariantConfigRow] = Field(default_factory=list)
    leaderboard: list[LeaderboardRow]
    run_metrics: list[RunMetricRow] = Field(default_factory=list)
    case_matrix: CaseMatrix
    comparisons: list[ComparisonReport] = Field(default_factory=list)
    distributions: list[MetricDistribution] = Field(default_factory=list)


DEFAULT_LEADERBOARD_METRICS: tuple[MetricAggregation, ...] = (
    MetricAggregation(
        name="pass_rate",
        semantic_type=Semantic.RESULT_SUCCESS,
        fn="ratio_true",
    ),
    MetricAggregation(
        name="avg_coverage",
        semantic_type=Semantic.COVERAGE_RATIO,
        fn="mean",
    ),
    MetricAggregation(
        name="total_cost",
        semantic_type=Semantic.MONEY_COST,
        fn="sum",
    ),
    MetricAggregation(
        name="avg_input_tokens",
        semantic_type=Semantic.LLM_TOKENS_INPUT,
        fn="mean",
    ),
)


def build_report(
    result: ExperimentResult,
    *,
    registry: SemanticRegistry | None = None,
    report_spec: ReportSpec | None = None,
) -> BenchmarkReport:
    active_registry = registry or result.semantic_registry
    active_report_spec = report_spec
    if active_report_spec is None and result.report_spec_data is not None:
        active_report_spec = ReportSpec.model_validate(result.report_spec_data)
    if active_report_spec is None:
        active_report_spec = ReportSpec()
    return BenchmarkReport(
        benchmark_id=result.benchmark_id,
        experiment_id=result.experiment_id,
        run_count=result.total_count,
        status_counts=build_status_counts(result),
        variant_configs=build_variant_configs(result),
        leaderboard=build_leaderboard(
            result,
            metrics=active_report_spec.leaderboard_metrics(),
            registry=active_registry,
        ),
        run_metrics=build_run_metric_rows(result, registry=active_registry),
        case_matrix=build_case_matrix(
            result,
            semantic_type=active_report_spec.case_matrix.semantic_type,
            registry=active_registry,
        ),
        comparisons=[
            compare_variants(
                result,
                baseline=comparison.baseline,
                candidate=comparison.candidate,
                metrics=comparison.resolved_metrics(),
                registry=active_registry,
            )
            for comparison in active_report_spec.comparisons
        ],
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
    )


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
        values = {
            metric.name: aggregate_values(
                [
                    value
                    for run in runs
                    if (value := metric_value(run, metric.semantic_type, registry=active_registry))
                    is not None
                ],
                metric.fn,
            )
            for metric in metrics
        }
        rows.append(
            LeaderboardRow(
                variant_id=variant_id,
                run_count=len(runs),
                metrics=values,
            )
        )
    return rows


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

    for metric in metrics:
        baseline_value = aggregate_values(
            [
                value
                for run in baseline_runs
                if (value := metric_value(run, metric.semantic_type, registry=active_registry))
                is not None
            ],
            metric.fn,
        )
        candidate_value = aggregate_values(
            [
                value
                for run in candidate_runs
                if (value := metric_value(run, metric.semantic_type, registry=active_registry))
                is not None
            ],
            metric.fn,
        )
        metric_deltas[metric.name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": (
                float(candidate_value) - float(baseline_value)
                if isinstance(candidate_value, int | float)
                and isinstance(baseline_value, int | float)
                else None
            ),
        }

    return ComparisonReport(
        baseline=baseline,
        candidate=candidate,
        run_count=min(len(baseline_runs), len(candidate_runs)),
        factor_deltas=factor_deltas,
        metric_deltas=metric_deltas,
        confounded=len(factor_deltas) > 1,
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
        key=lambda observation: source_priority(observation.source),
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
    if not candidates:
        return None
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (source_priority(item[1].source), item[0]),
    )
    return ordered[0][1]


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


def render_markdown_report(report: BenchmarkReport) -> str:
    lines = [
        f"# {report.benchmark_id}",
        "",
        f"experiment: `{report.experiment_id}`",
        f"runs: `{report.run_count}`",
        "",
        "## Leaderboard",
        "",
    ]
    metric_names = list(dict.fromkeys(name for row in report.leaderboard for name in row.metrics))
    lines.append("| variant | runs | " + " | ".join(metric_names) + " |")
    lines.append("| --- | ---: | " + " | ".join("---:" for _ in metric_names) + " |")
    for row in report.leaderboard:
        lines.append(
            f"| {row.variant_id} | {row.run_count} | "
            + " | ".join(_format_value(row.metrics.get(name)) for name in metric_names)
            + " |"
        )

    lines.extend(["", "## Case Matrix", ""])
    variants = sorted({variant for row in report.case_matrix.rows.values() for variant in row})
    lines.append("| case | " + " | ".join(variants) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in variants) + " |")
    for case_id, values in sorted(report.case_matrix.rows.items()):
        lines.append(
            f"| {case_id} | "
            + " | ".join(_format_value(values.get(variant)) for variant in variants)
            + " |"
        )

    if report.comparisons:
        lines.extend(["", "## Comparisons", ""])
        for comparison in report.comparisons:
            lines.append(f"### {comparison.baseline} vs {comparison.candidate}")
            lines.append("")
            lines.append(f"runs: `{comparison.run_count}`")
            lines.append(f"confounded: `{comparison.confounded}`")
            lines.append("")
            lines.append("| metric | baseline | candidate | delta |")
            lines.append("| --- | ---: | ---: | ---: |")
            for metric_name, delta in sorted(comparison.metric_deltas.items()):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            metric_name,
                            _format_value(delta.get("baseline")),
                            _format_value(delta.get("candidate")),
                            _format_value(delta.get("delta")),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    if report.distributions:
        lines.extend(["", "## Distributions", ""])
        for distribution in report.distributions:
            lines.append(f"### {distribution.name} (`{distribution.semantic_type}`)")
            lines.append("")
            summary_names = list(
                dict.fromkeys(
                    name for summaries in distribution.summaries.values() for name in summaries
                )
            )
            lines.append("| variant | samples | " + " | ".join(summary_names) + " |")
            lines.append("| --- | ---: | " + " | ".join("---:" for _ in summary_names) + " |")
            for variant_id, values in sorted(distribution.by_variant.items()):
                summaries = distribution.summaries.get(variant_id, {})
                lines.append(
                    f"| {variant_id} | {len(values)} | "
                    + " | ".join(_format_value(summaries.get(name)) for name in summary_names)
                    + " |"
                )
    return "\n".join(lines) + "\n"


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


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = (
    "AggregationFn",
    "BenchmarkReport",
    "CaseMatrix",
    "CaseMatrixReportSpec",
    "ComparisonReportSpec",
    "ComparisonReport",
    "DEFAULT_LEADERBOARD_METRICS",
    "DistributionReportSpec",
    "LeaderboardReportSpec",
    "LeaderboardRow",
    "MetricDistribution",
    "MetricAggregation",
    "ReportSpec",
    "RunMetricRow",
    "VariantConfigRow",
    "aggregate_values",
    "build_case_matrix",
    "build_leaderboard",
    "build_metric_distribution",
    "build_report",
    "build_run_metric_rows",
    "build_status_counts",
    "build_variant_configs",
    "compare_variants",
    "metric_observation",
    "metric_value",
    "render_markdown_report",
)
