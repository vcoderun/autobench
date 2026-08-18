from __future__ import annotations as _annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, field_validator

from autobench.instrumentation.pydantic_gepa.projection import OptimizationExecution
from autobench.runtime.models import ExecutionCorrelation

REPORT_VERSION = 3

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
ReportProfile = Literal["summary", "full", "audit"]
ReportLayout = Literal["single", "bundle", "auto"]
AssetDiffMode = Literal["none", "summary", "full"]
NoticeSeverity = Literal["info", "warning", "error"]
ComparisonOutcome = Literal["improved", "regressed", "unchanged", "indeterminate"]
EvidenceAvailability = Literal["available", "omitted", "unavailable"]
FindingKind = Literal[
    "health",
    "evaluation",
    "leader",
    "comparison",
    "constraint",
    "coverage",
    "trace",
    "asset",
    "optimization",
    "limitation",
]
ReportEvidenceKind = Literal[
    "experiment",
    "evaluation",
    "run",
    "metric",
    "comparison",
    "trace",
    "asset",
    "artifact",
    "optimization",
]


class MetricAggregation(BaseModel):
    name: str
    semantic_type: str
    fn: AggregationFn


class LeaderboardReportSpec(BaseModel):
    metrics: tuple[MetricAggregation, ...] = ()


class CaseMatrixReportSpec(BaseModel):
    semantic_type: str = "coverage.ratio"


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


class MarkdownReportLimits(BaseModel):
    table_rows: int = Field(default=200, ge=1)
    run_details: int = Field(default=100, ge=1)
    failure_details: int = Field(default=100, ge=1)
    value_excerpt_chars: int = Field(default=2_000, ge=1)


class MarkdownTraceConfig(BaseModel):
    top_slowest: int = Field(default=20, ge=1)


class MarkdownAssetConfig(BaseModel):
    diffs: AssetDiffMode = "summary"


class MarkdownContentConfig(BaseModel):
    include_captured: bool = False


class MarkdownReportConfig(BaseModel):
    profile: ReportProfile = "full"
    layout: ReportLayout = "auto"
    output: Path | None = None
    limits: MarkdownReportLimits = Field(default_factory=MarkdownReportLimits)
    traces: MarkdownTraceConfig = Field(default_factory=MarkdownTraceConfig)
    assets: MarkdownAssetConfig = Field(default_factory=MarkdownAssetConfig)
    content: MarkdownContentConfig = Field(default_factory=MarkdownContentConfig)

    @field_validator("output")
    @classmethod
    def validate_output(cls, output: Path | None) -> Path | None:
        if output is None:
            return None
        if output.is_absolute() or ".." in output.parts:
            raise ValueError("markdown report output must be a portable relative path")
        return output


class PublishedReportFile(BaseModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)


class MarkdownReportPublication(BaseModel):
    profile: ReportProfile
    requested_layout: ReportLayout
    layout: Literal["single", "bundle"]
    destination: Path
    files: tuple[PublishedReportFile, ...]


class ReportSpec(BaseModel):
    leaderboard: LeaderboardReportSpec = Field(default_factory=LeaderboardReportSpec)
    case_matrix: CaseMatrixReportSpec = Field(default_factory=CaseMatrixReportSpec)
    comparisons: tuple[ComparisonReportSpec, ...] = ()
    distributions: tuple[DistributionReportSpec, ...] = ()
    markdown: MarkdownReportConfig = Field(default_factory=MarkdownReportConfig)

    def leaderboard_metrics(self) -> tuple[MetricAggregation, ...]:
        if self.leaderboard.metrics:
            return self.leaderboard.metrics
        return DEFAULT_LEADERBOARD_METRICS


class ReportNotice(BaseModel):
    code: str
    severity: NoticeSeverity
    message: str
    evidence_ids: tuple[str, ...] = ()


class ReportEvidenceRef(BaseModel):
    kind: ReportEvidenceKind
    id: str


class ReportFinding(BaseModel):
    kind: FindingKind
    severity: NoticeSeverity
    title: str
    statement: str
    evidence: tuple[ReportEvidenceRef, ...]


class ExecutiveSummary(BaseModel):
    health: str
    leader_variant: str | None = None
    objective_metric: str | None = None
    findings: tuple[ReportFinding, ...] = ()


class EvaluationMetricReport(BaseModel):
    name: str
    label: str
    kind: Literal["score", "count", "value"]
    sample_count: int
    missing_count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    total: float


class EvaluationCaseReport(BaseModel):
    run_id: str
    case_id: str
    variant_id: str
    quality_pass: bool | None = None
    score: float | None = None
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    feedback: tuple[str, ...] = ()


class EvaluationSummaryReport(BaseModel):
    case_count: int
    evaluated_count: int
    passed_count: int
    failed_count: int
    unevaluated_count: int
    pass_rate: float | None = None
    score_count: int
    mean_score: float | None = None
    median_score: float | None = None
    minimum_score: float | None = None
    maximum_score: float | None = None
    metrics: tuple[EvaluationMetricReport, ...] = ()
    cases: tuple[EvaluationCaseReport, ...] = ()


class FileHashReport(BaseModel):
    path: str
    sha256: str


class SourceIdentityReport(BaseModel):
    report_version: int = REPORT_VERSION
    record_version: int | None = None
    benchmark_id: str
    experiment_id: str
    spec_hash: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    dataset_hash: str | None = None
    manifest_path: str | None = None
    run_paths: tuple[str, ...] = ()
    file_hashes: tuple[FileHashReport, ...] = ()
    correlation: ExecutionCorrelation | None = None


class FactorReport(BaseModel):
    name: str
    value: JsonValue
    semantic_type: str | None = None
    optimize: bool = False


class ExperimentDesignReport(BaseModel):
    description: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    dataset_hash: str | None = None
    case_count: int
    case_ids: tuple[str, ...] = ()
    case_tags: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    variant_count: int
    planned_run_count: int
    scorer_count: int = 0
    deriver_count: int = 0
    post_deriver_count: int = 0
    policy_count: int = 0
    instrumentation: tuple[str, ...] = ()
    capture_enabled: bool = False
    warnings: tuple[str, ...] = ()


class ErrorSummaryReport(BaseModel):
    error_type: str
    count: int
    run_ids: tuple[str, ...]


class RunHealthReport(BaseModel):
    experiment_status: str
    partial: bool
    planned_count: int
    recorded_count: int
    missing_count: int
    partial_run_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    status_by_variant: dict[str, dict[str, int]] = Field(default_factory=dict)
    missing_run_ids: tuple[str, ...] = ()
    cross_run_derivation_complete: bool
    policies_complete: bool
    errors: tuple[ErrorSummaryReport, ...] = ()


class MetricDefinitionReport(BaseModel):
    name: str
    semantic_type: str
    unit: str | None = None
    direction: str | None = None
    role: str | None = None
    aggregation: AggregationFn | None = None
    sources: tuple[str, ...] = ()
    observed_count: int = 0
    missing_count: int = 0
    description: str | None = None


class LeaderboardMetricReport(BaseModel):
    name: str
    semantic_type: str
    value: JsonValue
    sample_count: int
    missing_count: int
    unit: str | None = None
    direction: str | None = None
    role: str | None = None
    best: bool = False


class LeaderboardRow(BaseModel):
    variant_id: str
    run_count: int
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    metric_details: tuple[LeaderboardMetricReport, ...] = ()


class VariantConfigRow(BaseModel):
    variant_id: str
    label: str | None = None
    factors: dict[str, JsonValue] = Field(default_factory=dict)
    factor_details: tuple[FactorReport, ...] = ()


class RunMetricRow(BaseModel):
    case_id: str
    variant_id: str
    status: str
    metrics: dict[str, JsonValue] = Field(default_factory=dict)


class CaseMatrix(BaseModel):
    metric: str
    rows: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)


class ComparisonReport(BaseModel):
    baseline: str
    candidate: str
    run_count: int
    factor_deltas: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    metric_deltas: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    confounded: bool = False
    baseline_factors: dict[str, JsonValue] = Field(default_factory=dict)
    candidate_factors: dict[str, JsonValue] = Field(default_factory=dict)
    paired_count: int = 0
    missing_pair_count: int = 0
    metric_results: tuple[MetricComparisonReport, ...] = ()


class MetricComparisonReport(BaseModel):
    name: str
    semantic_type: str
    aggregation: AggregationFn
    baseline: JsonValue
    candidate: JsonValue
    delta: float | None = None
    relative_delta: float | None = None
    direction: str | None = None
    unit: str | None = None
    outcome: ComparisonOutcome = "indeterminate"
    baseline_count: int = 0
    candidate_count: int = 0
    paired_count: int = 0
    missing_pair_count: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0


class RegressionReport(BaseModel):
    baseline: str
    candidate: str
    metric: str
    semantic_type: str
    outcome: ComparisonOutcome
    delta: float
    relative_delta: float | None = None


class ScoreEvidenceReport(BaseModel):
    name: str
    actual_excerpt: str | None = None
    expected_excerpt: str | None = None


class RunDetailReport(BaseModel):
    run_id: str
    case_id: str
    variant_id: str
    status: str
    evaluation_status: str
    partial: bool
    end_reason: str
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    score_count: int = 0
    span_count: int = 0
    asset_count: int = 0
    artifact_count: int = 0
    parent_run_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    input_excerpt: str | None = None
    expected_excerpt: str | None = None
    output_excerpt: str | None = None
    score_evidence: tuple[ScoreEvidenceReport, ...] = ()


class FailureReport(BaseModel):
    run_id: str
    case_id: str
    variant_id: str
    stage: str
    error_type: str
    message: str
    span_id: str | None = None
    traceback: str | None = None


class SlowSpanReport(BaseModel):
    run_id: str
    span_id: str
    operation: str
    kind: str
    instrumentor: str
    duration_ns: int
    status: str
    partial: bool


class TraceSummaryReport(BaseModel):
    traced_run_count: int
    untraced_run_count: int
    complete_trace_count: int
    partial_trace_count: int
    span_count: int
    root_span_count: int
    orphan_span_count: int
    error_span_count: int
    cancelled_span_count: int
    spans_by_kind: dict[str, int] = Field(default_factory=dict)
    spans_by_instrumentor: dict[str, int] = Field(default_factory=dict)
    diagnostics_by_code: dict[str, int] = Field(default_factory=dict)
    slow_spans: tuple[SlowSpanReport, ...] = ()


class AssetVersionReport(BaseModel):
    asset_id: str
    version: str
    parent_version: str | None = None
    content_hash: str
    source_hash: str | None = None
    source_path: str | None = None
    git_commit: str | None = None
    kind: str | None = None
    semantic_type: str | None = None
    sensitivity: str | None = None
    representations: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    source_locators: tuple[str, ...] = ()
    definition_asset_ids: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    changed_fields: tuple[str, ...] = ()
    content_state: EvidenceAvailability = "unavailable"
    diff_state: EvidenceAvailability = "unavailable"
    content_excerpt: str | None = None
    diff_excerpt: str | None = None
    run_ids: tuple[str, ...] = ()
    variant_ids: tuple[str, ...] = ()


class AssetLineageReport(BaseModel):
    asset_count: int
    version_count: int
    transition_count: int
    versions: tuple[AssetVersionReport, ...] = ()


class ArtifactReport(BaseModel):
    artifact_id: str
    name: str
    run_id: str
    span_id: str | None = None
    media_type: str | None = None
    source: str
    state: str
    sha256: str | None = None
    byte_count: int | None = None
    path: str | None = None


class ArtifactInventoryReport(BaseModel):
    artifact_count: int
    complete_count: int
    partial_count: int
    truncated_count: int
    artifacts: tuple[ArtifactReport, ...] = ()


class PolicyOutcomeReport(BaseModel):
    name: str
    run_id: str
    case_id: str
    variant_id: str
    passed: bool
    metric: str | None = None
    actual: JsonValue = None
    reason: str | None = None


class ProvenanceReport(BaseModel):
    python_version: str
    platform: str
    working_directory: str
    working_directory_redacted: bool
    spec_hash: str | None = None
    semantic_registry_version: int
    source_maps: tuple[str, ...] = ()


class MetricDistribution(BaseModel):
    name: str
    semantic_type: str
    by_variant: dict[str, list[JsonValue]] = Field(default_factory=dict)
    summaries: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)


class OptimizationRunReport(BaseModel):
    benchmark_run_id: str
    case_id: str
    variant_id: str
    execution: OptimizationExecution


class BenchmarkReport(BaseModel):
    report_version: int = REPORT_VERSION
    markdown: MarkdownReportConfig = Field(default_factory=MarkdownReportConfig)
    benchmark_id: str
    experiment_id: str
    run_count: int
    source: SourceIdentityReport | None = None
    summary: ExecutiveSummary | None = None
    evaluation: EvaluationSummaryReport | None = None
    design: ExperimentDesignReport | None = None
    health: RunHealthReport | None = None
    metric_catalog: tuple[MetricDefinitionReport, ...] = ()
    notices: tuple[ReportNotice, ...] = ()
    status_counts: dict[str, int] = Field(default_factory=dict)
    variant_configs: list[VariantConfigRow] = Field(default_factory=list)
    leaderboard: list[LeaderboardRow]
    run_metrics: list[RunMetricRow] = Field(default_factory=list)
    run_details: tuple[RunDetailReport, ...] = ()
    failures: tuple[FailureReport, ...] = ()
    traces: TraceSummaryReport | None = None
    assets: AssetLineageReport | None = None
    artifacts: ArtifactInventoryReport | None = None
    policies: tuple[PolicyOutcomeReport, ...] = ()
    provenance: ProvenanceReport | None = None
    case_matrix: CaseMatrix
    comparisons: list[ComparisonReport] = Field(default_factory=list)
    regressions: tuple[RegressionReport, ...] = ()
    distributions: list[MetricDistribution] = Field(default_factory=list)
    optimizations: list[OptimizationRunReport] = Field(default_factory=list)
    optimization_warnings: list[str] = Field(default_factory=list)
    correlation: ExecutionCorrelation | None = None


class CorrelatedReportGroup(BaseModel):
    group_id: str | None = None
    attempts: tuple[int, ...] = ()
    phases: tuple[str, ...] = ()
    reports: list[BenchmarkReport] = Field(default_factory=list)


DEFAULT_LEADERBOARD_METRICS: tuple[MetricAggregation, ...] = (
    MetricAggregation(
        name="pass_rate",
        semantic_type="result.success",
        fn="ratio_true",
    ),
    MetricAggregation(
        name="avg_coverage",
        semantic_type="coverage.ratio",
        fn="mean",
    ),
    MetricAggregation(
        name="total_cost",
        semantic_type="money.cost",
        fn="sum",
    ),
    MetricAggregation(
        name="avg_input_tokens",
        semantic_type="llm.tokens.input",
        fn="mean",
    ),
)


__all__ = (
    "AggregationFn",
    "AssetDiffMode",
    "ArtifactInventoryReport",
    "ArtifactReport",
    "AssetLineageReport",
    "AssetVersionReport",
    "BenchmarkReport",
    "CaseMatrix",
    "CaseMatrixReportSpec",
    "ComparisonReport",
    "ComparisonReportSpec",
    "ComparisonOutcome",
    "CorrelatedReportGroup",
    "DEFAULT_LEADERBOARD_METRICS",
    "DistributionReportSpec",
    "EvidenceAvailability",
    "ErrorSummaryReport",
    "EvaluationCaseReport",
    "EvaluationMetricReport",
    "EvaluationSummaryReport",
    "ExperimentDesignReport",
    "ExecutiveSummary",
    "FactorReport",
    "FailureReport",
    "FileHashReport",
    "FindingKind",
    "LeaderboardReportSpec",
    "LeaderboardMetricReport",
    "LeaderboardRow",
    "MarkdownAssetConfig",
    "MarkdownContentConfig",
    "MarkdownReportConfig",
    "MarkdownReportLimits",
    "MarkdownReportPublication",
    "MarkdownTraceConfig",
    "MetricAggregation",
    "MetricComparisonReport",
    "MetricDefinitionReport",
    "MetricDistribution",
    "NoticeSeverity",
    "OptimizationRunReport",
    "PolicyOutcomeReport",
    "ProvenanceReport",
    "PublishedReportFile",
    "REPORT_VERSION",
    "ReportLayout",
    "ReportEvidenceKind",
    "ReportEvidenceRef",
    "ReportFinding",
    "ReportNotice",
    "ReportProfile",
    "ReportSpec",
    "RegressionReport",
    "RunDetailReport",
    "RunMetricRow",
    "RunHealthReport",
    "ScoreEvidenceReport",
    "SourceIdentityReport",
    "SlowSpanReport",
    "TraceSummaryReport",
    "VariantConfigRow",
)
