from __future__ import annotations as _annotations

import csv
from io import StringIO
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest
from rich.console import Console

from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    Observation,
    ObservationKind,
    ObservationSource,
    OutputMetricScorer,
    PassFailScorer,
    Semantic,
    TaskSpec,
    Variant,
    record_experiment,
    replay_experiment,
    run_benchmark_spec,
)
from autobench.data.variants import FactorValue
from autobench.io import load_yaml
from autobench.reports.exporting import (
    export_markdown_report,
    export_runs_csv,
    export_summary_yaml,
)
from autobench.reports.reporting import (
    DEFAULT_LEADERBOARD_METRICS,
    AggregationFn,
    ComparisonReport,
    MetricAggregation,
    MetricDistribution,
    ReportSpec,
    aggregate_values,
    build_case_matrix,
    build_leaderboard,
    build_metric_distribution,
    build_report,
    compare_variants,
    metric_observation,
    metric_value,
    render_markdown_report,
)
from autobench.reports.rich import (
    render_comparison,
    render_model_configurations,
    render_recorded_runs,
    render_report,
)


async def test_report_replays_the_same_semantic_experiment_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _reporting_spec()

    fresh = await run_benchmark_spec(spec, experiment_id="exp_reporting")
    record_dir = tmp_path / "recorded"
    record_experiment(fresh, record_dir)
    replayed = replay_experiment(record_dir)

    assert build_report(fresh).model_dump(mode="json") == build_report(replayed).model_dump(
        mode="json"
    )


async def test_leaderboard_case_matrix_and_comparison_use_semantic_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")

    leaderboard = build_leaderboard(result)
    baseline = leaderboard[0]
    candidate = leaderboard[1]
    assert baseline.variant_id == "baseline"
    assert baseline.metrics == {
        "pass_rate": 0.5,
        "avg_coverage": pytest.approx(0.55),
        "total_cost": pytest.approx(0.2),
        "avg_input_tokens": 10.0,
    }
    assert candidate.variant_id == "candidate"
    assert candidate.metrics == {
        "pass_rate": 1.0,
        "avg_coverage": pytest.approx(0.8),
        "total_cost": pytest.approx(0.4),
        "avg_input_tokens": 20.0,
    }

    matrix = build_case_matrix(result, semantic_type=Semantic.COVERAGE_RATIO)
    assert matrix.metric == Semantic.COVERAGE_RATIO
    assert matrix.rows == {
        "case_easy": {"baseline": 0.5, "candidate": 0.75},
        "case_hard": {"baseline": 0.6, "candidate": 0.85},
    }

    comparison = compare_variants(result, baseline="baseline", candidate="candidate")
    assert comparison.run_count == 2
    assert comparison.confounded is True
    assert comparison.factor_deltas == {
        "model.name": {"baseline": "model-a", "candidate": "model-b"},
        "temperature": {"baseline": 0.1, "candidate": 0.2},
    }
    assert comparison.metric_deltas["avg_coverage"] == {
        "baseline": pytest.approx(0.55),
        "candidate": pytest.approx(0.8),
        "delta": pytest.approx(0.25),
    }

    report = build_report(result)
    assert report.status_counts == {"failed": 1, "passed": 3}
    assert report.variant_configs[0].factors["model.name"] == "model-a"
    assert report.variant_configs[1].factors["model.name"] == "model-b"
    assert report.run_metrics[0].case_id == "case_easy"
    assert "coverage (coverage.ratio)" in report.run_metrics[0].metrics

    first_run = result.runs[0]
    assert metric_value(first_run, Semantic.MONEY_COST) == 0.1
    assert metric_observation(first_run, Semantic.MONEY_COST) is not None
    assert metric_value(first_run, "unknown.metric") is None
    direct = Observation(
        id="direct_tokens",
        name="tokens.direct",
        kind=ObservationKind.METRIC,
        semantic_type=Semantic.LLM_TOKENS_INPUT,
        value=10,
        source=ObservationSource.DERIVED,
        tags={"abp.measurement_scope": "direct"},
    )
    aggregate = direct.model_copy(
        update={
            "id": "aggregate_tokens",
            "name": "tokens.total",
            "value": 30,
            "tags": {"abp.measurement_scope": "aggregate", "abp.summary": True},
        }
    )
    accounted_run = first_run.model_copy(
        update={
            "task_result": first_run.task_result.model_copy(
                update={
                    "observations": [
                        *first_run.task_result.observations,
                        direct,
                        aggregate,
                    ]
                }
            )
        }
    )
    assert metric_value(accounted_run, Semantic.LLM_TOKENS_INPUT) == 30


async def test_report_variant_labels_support_nested_spec_snapshots_and_bad_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")
    nested_result = result.model_copy(
        update={
            "spec_snapshot": {
                "spec": {
                    "variants": [
                        {"id": "baseline", "label": "Baseline model"},
                        "not-a-variant",
                        {"id": 42, "label": "bad-id"},
                        {"id": "candidate", "label": 99},
                    ]
                }
            }
        }
    )
    invalid_snapshot_result = result.model_copy(update={"spec_snapshot": {"spec": "invalid"}})

    nested_report = build_report(nested_result)
    invalid_snapshot_report = build_report(invalid_snapshot_result)

    assert nested_report.variant_configs[0].label == "Baseline model"
    assert nested_report.variant_configs[1].label is None
    assert all(config.label is None for config in invalid_snapshot_report.variant_configs)


async def test_exporters_write_yaml_csv_and_markdown_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")

    summary_path = tmp_path / "summary.yaml"
    csv_path = tmp_path / "runs.csv"
    markdown_path = tmp_path / "report.md"

    summary = export_summary_yaml(result, summary_path)
    runs_csv = export_runs_csv(result, csv_path)
    markdown = export_markdown_report(result, markdown_path)

    summary_view = load_yaml(summary_path)
    assert summary_view["report"]["benchmark"] == "reporting-demo"
    assert "leaderboard" not in summary_view
    assert summary_view["report"]["leaderboard"]["baseline"]["metrics"]["avg_coverage"] == 0.55
    assert summary_view["report"]["matrix"]["cases"]["case_easy"]["baseline"] == 0.5
    assert summary_view["report"]["cases"]["case_easy"]["baseline"]["status"] == "failed"
    assert summary_view["report"]["compare"] == {}
    assert summary == summary_path.read_text(encoding="utf-8")
    assert export_summary_yaml(result).startswith("# yaml-language-server: $schema=")

    rows = list(csv.DictReader(StringIO(runs_csv)))
    assert rows[0]["case_id"] == "case_easy"
    assert rows[0]["variant_id"] == "baseline"
    assert rows[0]["success"] == "False"
    assert rows[0]["coverage"] == "0.5"
    assert rows[0]["cost"] == "0.1"
    assert rows[0]["input_tokens"] == "10"
    assert runs_csv == csv_path.read_bytes().decode("utf-8")
    assert export_runs_csv(result).startswith("run_id,case_id,variant_id,status")

    assert markdown.startswith("# reporting-demo\n")
    assert "| baseline | 2 | 0.5 | 0.55 | 0.2 | 10 |" in markdown
    assert "| case_hard | 0.6 | 0.85 |" in markdown
    assert markdown == markdown_path.read_text(encoding="utf-8")
    assert export_markdown_report(result).startswith("# reporting-demo\n")


async def test_report_spec_controls_leaderboard_and_case_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")
    report_spec = ReportSpec.model_validate(
        {
            "leaderboard": {
                "metrics": [
                    {
                        "name": "max_cost",
                        "semantic_type": Semantic.MONEY_COST,
                        "fn": "max",
                    }
                ]
            },
            "case_matrix": {"semantic_type": Semantic.MONEY_COST},
            "comparisons": [
                {
                    "baseline": "baseline",
                    "candidate": "candidate",
                    "metrics": [
                        {
                            "name": "coverage_delta",
                            "semantic_type": Semantic.COVERAGE_RATIO,
                            "fn": "mean",
                        }
                    ],
                }
            ],
            "distributions": [
                {
                    "name": "cost_distribution",
                    "semantic_type": Semantic.MONEY_COST,
                    "summaries": ["min", "median", "max"],
                }
            ],
        }
    )

    report = build_report(result, report_spec=report_spec)
    summary = export_summary_yaml(result, report_spec=report_spec)
    markdown = export_markdown_report(result, report_spec=report_spec)

    assert report.leaderboard[0].metrics == {"max_cost": 0.1}
    assert report.leaderboard[1].metrics == {"max_cost": 0.2}
    assert report.case_matrix.metric == Semantic.MONEY_COST
    assert report.case_matrix.rows == {
        "case_easy": {"baseline": 0.1, "candidate": 0.2},
        "case_hard": {"baseline": 0.1, "candidate": 0.2},
    }
    assert report.comparisons[0].baseline == "baseline"
    assert report.comparisons[0].candidate == "candidate"
    assert report.comparisons[0].metric_deltas["coverage_delta"]["delta"] == pytest.approx(0.25)
    assert report.distributions[0].name == "cost_distribution"
    assert report.distributions[0].summaries["baseline"] == {
        "min": 0.1,
        "median": 0.1,
        "max": 0.1,
    }
    assert "leaderboard:\n" in summary
    assert "baseline:\n" in summary
    assert "max_cost:" in summary
    assert "baseline -> candidate:" in summary
    assert "| baseline | 2 | 0.1 |" in markdown
    assert "## Comparisons" in markdown
    assert "## Distributions" in markdown
    assert "### cost_distribution (`money.cost`)" in markdown


async def test_rich_renderers_cover_no_factor_deltas_and_runs_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")
    report = build_report(result)
    report.distributions.append(MetricDistribution(name="empty", semantic_type="unknown.metric"))
    report.distributions.append(
        MetricDistribution(
            name="samples",
            semantic_type=Semantic.COVERAGE_RATIO,
            by_variant={"baseline": [0.5, 0.6], "candidate": [0.75, 0.85]},
            summaries={
                "baseline": {"min": 0.5, "max": 0.6},
                "candidate": {"min": 0.75, "max": 0.85},
            },
        )
    )
    comparison = ComparisonReport(
        baseline="baseline",
        candidate="candidate",
        run_count=0,
        factor_deltas={},
        metric_deltas={"avg_coverage": {"baseline": 0.5, "candidate": 0.75, "delta": 0.25}},
        confounded=False,
    )
    report.comparisons.append(comparison)
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)

    render_comparison(console, comparison)
    render_model_configurations(
        console,
        [("openrouter:provider/spec-model", "openrouter:provider/explore-model")],
    )
    render_recorded_runs(console, result.runs)
    render_report(console, report)

    rendered = output.getvalue()
    assert "No factor differences" in rendered
    assert "Recorded Runs Preview" in rendered
    assert "No samples" in rendered
    assert "model_pair_1" in rendered
    assert "Distribution: samples" in rendered
    assert "Leaderboard: Effectiveness" in rendered


async def test_replay_preserves_recorded_report_spec_and_semantic_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    base_spec = _reporting_spec()
    custom_registry = base_spec.semantic_registry.model_copy(
        update={
            "aliases": dict(base_spec.semantic_registry.aliases)
            | {"answer.score": Semantic.COVERAGE_RATIO}
        }
    )
    spec = base_spec.model_copy(
        update={
            "reports": ReportSpec.model_validate(
                {
                    "leaderboard": {
                        "metrics": [
                            {
                                "name": "alias_score",
                                "semantic_type": "answer.score",
                                "fn": "mean",
                            }
                        ]
                    },
                    "case_matrix": {"semantic_type": "answer.score"},
                }
            ),
            "semantic_registry": custom_registry,
        }
    )
    result = await run_benchmark_spec(spec, experiment_id="exp_reporting")
    record_dir = tmp_path / "recorded"
    record_experiment(result, record_dir)

    replayed = replay_experiment(record_dir)
    report = build_report(replayed)

    assert report.leaderboard[0].metrics == {"alias_score": 0.55}
    assert report.leaderboard[1].metrics == {"alias_score": 0.8}
    assert report.case_matrix.metric == "answer.score"


def test_reporting_aggregation_contracts_cover_empty_and_non_numeric_values() -> None:
    assert aggregate_values([], "mean") is None
    assert aggregate_values(["no-number"], "mean") is None
    assert aggregate_values([1, False, 3.0], "count") == 3
    assert aggregate_values([True, False, 1], "ratio_true") == pytest.approx(2 / 3)
    assert aggregate_values([1, 2.5, "skip"], "sum") == 3.5
    assert aggregate_values([3, 1, 2], "min") == 1.0
    assert aggregate_values([3, 1, 2], "max") == 3.0
    assert aggregate_values([3, 1, 2], "median") == 2.0
    assert aggregate_values([1, 2, 3], "p95") == pytest.approx(2.9)
    assert aggregate_values([7], "p95") == pytest.approx(7.0)
    assert aggregate_values(list(range(1, 22)), "p95") == pytest.approx(20.0)
    assert aggregate_values([1, 2, 3], "max") == 3.0
    assert aggregate_values([1, 2, 3], "stddev") == pytest.approx(0.8164965809)
    assert aggregate_values([1, 4], "geomean") == pytest.approx(2.0)
    assert aggregate_values([0, 4], "geomean") is None
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        aggregate_values([1], cast(AggregationFn, "unsupported"))
    report_spec = ReportSpec.model_validate(
        {"comparisons": [{"baseline": "baseline", "candidate": "candidate"}]}
    )
    assert report_spec.comparisons[0].resolved_metrics() == DEFAULT_LEADERBOARD_METRICS


async def test_build_metric_distribution_summarizes_values_by_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")

    distribution = build_metric_distribution(
        result,
        name="coverage_distribution",
        semantic_type=Semantic.COVERAGE_RATIO,
        summaries=("min", "median", "p95", "max"),
    )

    assert distribution.by_variant == {
        "baseline": [0.5, 0.6],
        "candidate": [0.75, 0.85],
    }
    assert distribution.summaries["baseline"]["median"] == pytest.approx(0.55)
    assert distribution.summaries["candidate"]["p95"] == pytest.approx(0.845)

    empty_distribution = build_metric_distribution(
        result,
        name="missing_distribution",
        semantic_type="unknown.metric",
    )
    assert empty_distribution.by_variant == {}
    assert empty_distribution.summaries == {}


async def test_markdown_report_handles_empty_leaderboard_and_missing_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")
    sparse_report = build_report(result)
    sparse_report.leaderboard = []
    sparse_report.case_matrix.rows["case_sparse"] = {"baseline": None}
    sparse_report.case_matrix.rows["case_label"] = {"baseline": "manual-pass"}

    markdown = render_markdown_report(sparse_report)

    assert "| variant | runs |  |" in markdown
    assert "| case_sparse |  |  |" in markdown
    assert "| case_label | manual-pass |  |" in markdown


async def test_custom_metric_aggregation_preserves_non_numeric_delta_as_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")
    same_variant = compare_variants(result, baseline="baseline", candidate="baseline")
    metrics = (
        MetricAggregation(
            name="winning_answer",
            semantic_type=Semantic.RESULT_SUCCESS,
            fn="count",
        ),
    )

    assert same_variant.factor_deltas == {}
    comparison = compare_variants(
        result,
        baseline="missing-baseline",
        candidate="candidate",
        metrics=metrics,
    )

    assert comparison.factor_deltas == {
        "model.name": {"baseline": None, "candidate": "model-b"},
        "temperature": {"baseline": None, "candidate": 0.2},
    }
    assert comparison.metric_deltas["winning_answer"] == {
        "baseline": None,
        "candidate": 2,
        "delta": None,
    }


async def test_build_report_falls_back_to_default_report_spec_when_not_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(_reporting_spec(), experiment_id="exp_reporting")
    manual = result.model_copy(update={"report_spec_data": None})

    report = build_report(manual)

    assert report.leaderboard[0].metrics["pass_rate"] == 0.5


def _reporting_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="reporting-demo"),
        dataset=DatasetSpec(
            cases=[
                Case(id="case_easy"),
                Case(id="case_hard"),
            ]
        ),
        task=TaskSpec(kind="python", target="reporting_tasks:run"),
        variants=[
            Variant(
                id="baseline",
                factors=[
                    FactorValue(
                        name="model.name",
                        value="model-a",
                        semantic_type=Semantic.LLM_MODEL_NAME,
                    ),
                    FactorValue(name="temperature", value=0.1),
                ],
            ),
            Variant(
                id="candidate",
                factors=[
                    FactorValue(
                        name="model.name",
                        value="model-b",
                        semantic_type=Semantic.LLM_MODEL_NAME,
                    ),
                    FactorValue(name="temperature", value=0.2),
                ],
            ),
        ],
        scoring=[
            PassFailScorer(
                name="success",
                path="output.success",
                semantic_type=Semantic.RESULT_SUCCESS,
            ),
            OutputMetricScorer(
                name="coverage",
                path="output.coverage",
                semantic_type=Semantic.COVERAGE_RATIO,
            ),
        ],
    )


def _write_module(tmp_path: Path) -> None:
    source = """
    def run(ctx, case):
        is_candidate = ctx.variant.id == "candidate"
        is_hard = case.id == "case_hard"
        coverage = (0.75 if is_candidate else 0.5) + (0.1 if is_hard else 0.0)
        cost = 0.2 if is_candidate else 0.1
        tokens = 20 if is_candidate else 10
        ctx.metric("cost", cost, semantic_type="money.cost")
        ctx.metric("input_tokens", tokens, semantic_type="llm.tokens.input")
        ctx.metric("raw_latency", 12.5)
        return {
            "success": is_candidate or is_hard,
            "coverage": coverage,
        }
    """
    (tmp_path / "reporting_tasks.py").write_text(
        dedent(source).strip() + "\n",
        encoding="utf-8",
    )
