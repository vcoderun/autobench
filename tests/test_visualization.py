from __future__ import annotations as _annotations

from math import isnan
from pathlib import Path

from autobench.records.storage import EnvironmentMetadata
from autobench.reports.reporting import (
    BenchmarkReport,
    CaseMatrix,
    CaseMatrixVisualSpec,
    ComparisonReport,
    ComparisonVisualSpec,
    DistributionVisualSpec,
    LeaderboardRow,
    LeaderboardVisualSpec,
    MetricDistribution,
    StatusVisualSpec,
    VariantConfigVisualSpec,
)
from autobench.reports.visualization import (
    _case_matrix_is_numeric,
    _display_value,
    _find_comparison,
    _find_distribution,
    _nan_if_none,
    _numeric_metric_delta,
    _numeric_metric_delta_names,
    _numeric_value,
    _ordered_metric_names,
    _ordered_summary_names,
    _report_metric_is_numeric,
    _select_numeric_metric_name,
    default_visual_specs,
    export_png_report,
    render_report_figure,
)
from autobench.runtime.pipeline import BenchmarkPlan, ExperimentResult


def test_export_png_report_supports_default_and_embedded_visual_specs(tmp_path: Path) -> None:
    result = _empty_result()
    default_path = tmp_path / "default-report.png"
    embedded_path = tmp_path / "embedded-report.png"

    export_png_report(result, default_path)
    export_png_report(
        result.model_copy(
            update={
                "report_spec_data": {
                    "visuals": [{"kind": "leaderboard", "render_as": "table", "title": "Empty"}]
                }
            }
        ),
        embedded_path,
    )

    assert default_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert embedded_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_export_png_report_set_supports_default_visual_specs(tmp_path: Path) -> None:
    from autobench.reports.visualization import export_png_report_set

    exported = export_png_report_set(_empty_result(), tmp_path / "empty-set")

    assert exported
    assert all(Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in exported)


def test_default_visual_specs_and_render_report_figure_cover_numeric_paths() -> None:
    report = _numeric_report()
    visuals = default_visual_specs(report)
    figure = render_report_figure(report, visuals=visuals)

    assert any(isinstance(visual, VariantConfigVisualSpec) for visual in visuals)
    assert any(isinstance(visual, StatusVisualSpec) for visual in visuals)
    assert any(visual.kind == "leaderboard" for visual in visuals)
    assert any(visual.kind == "case_matrix" for visual in visuals)
    assert any(visual.kind == "comparison" for visual in visuals)
    assert any(visual.kind == "distribution" for visual in visuals)
    assert len(figure.axes) >= 4


def test_visualization_fallback_paths_and_helpers() -> None:
    report = _nonnumeric_report()
    figure = render_report_figure(
        report,
        visuals=(
            LeaderboardVisualSpec(render_as="bar", metric="missing_metric"),
            LeaderboardVisualSpec(render_as="line", metric="label"),
            LeaderboardVisualSpec(render_as="pie", metric="label"),
            ComparisonVisualSpec(
                baseline="baseline",
                candidate="candidate",
                render_as="bar",
                metric="label_delta",
            ),
            ComparisonVisualSpec(
                baseline="empty",
                candidate="candidate",
                render_as="table",
            ),
            ComparisonVisualSpec(
                baseline="missing",
                candidate="candidate",
                render_as="table",
            ),
            DistributionVisualSpec(name="missing_distribution", render_as="boxplot"),
            DistributionVisualSpec(name="label_distribution", render_as="boxplot"),
            DistributionVisualSpec(name="label_distribution", render_as="line"),
            DistributionVisualSpec(name="label_distribution", render_as="table"),
        ),
    )
    empty_figure = render_report_figure(report, visuals=())

    assert len(figure.axes) >= 5
    assert len(empty_figure.axes) == 1
    assert (
        _find_comparison(
            report,
            ComparisonVisualSpec(baseline="baseline", candidate="candidate", render_as="table"),
        )
        == report.comparisons[0]
    )
    assert (
        _find_comparison(
            report,
            ComparisonVisualSpec(baseline="missing", candidate="candidate", render_as="table"),
        )
        is None
    )
    assert _find_distribution(report, "label_distribution") == report.distributions[0]
    assert _find_distribution(report, "missing_distribution") is None
    assert _ordered_metric_names(report) == ["label"]
    assert _ordered_summary_names(report.distributions[0]) == ["mode"]
    assert _select_numeric_metric_name(report, None) is None
    assert _select_numeric_metric_name(report, "label") is None
    assert _report_metric_is_numeric(report, "label") is False
    assert _numeric_metric_delta_names(report.comparisons[0], selected_metric="label_delta") == []
    assert _numeric_metric_delta_names(report.comparisons[1], selected_metric=None) == []
    assert _numeric_metric_delta({"baseline": 1.0, "candidate": 2.0}) is True
    assert _numeric_metric_delta({"baseline": "a", "candidate": 2.0}) is False
    assert _case_matrix_is_numeric(report) is False
    assert _display_value(None) == "-"
    assert _display_value(True) == "true"
    assert _display_value(False) == "false"
    assert _display_value(1.23456789) == "1.23457"
    assert _display_value("ready") == "ready"
    assert _numeric_value(True) is None
    assert _numeric_value("x") is None
    assert _numeric_value(3) == 3.0
    assert _numeric_value(2.5) == 2.5
    assert _numeric_value(float("nan")) is None
    assert isnan(_nan_if_none(None))
    assert _nan_if_none(2.5) == 2.5


def test_visualization_additional_chart_modes_cover_numeric_paths() -> None:
    report = _numeric_report()
    zero_report = _zero_metric_report()

    figure = render_report_figure(
        report,
        visuals=(
            StatusVisualSpec(render_as="bar"),
            LeaderboardVisualSpec(render_as="line", metric="avg_quality"),
            LeaderboardVisualSpec(render_as="pie", metric="avg_quality"),
            CaseMatrixVisualSpec(render_as="line"),
            DistributionVisualSpec(name="latency_distribution", render_as="line"),
        ),
    )
    zero_figure = render_report_figure(
        zero_report,
        visuals=(LeaderboardVisualSpec(render_as="pie", metric="zero_score"),),
    )

    assert len(figure.axes) == 5
    assert len(zero_figure.axes) == 1


def test_visualization_numeric_helper_paths() -> None:
    report = _numeric_report()

    assert _select_numeric_metric_name(report, None) == "avg_quality"
    assert _select_numeric_metric_name(report, "avg_quality") == "avg_quality"
    assert _report_metric_is_numeric(report, "avg_quality") is True
    assert _numeric_metric_delta_names(report.comparisons[0], selected_metric=None) == [
        "avg_quality"
    ]
    assert _numeric_metric_delta_names(
        report.comparisons[0],
        selected_metric="avg_quality",
    ) == ["avg_quality"]
    assert _case_matrix_is_numeric(report) is True


def _empty_result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp-empty",
        benchmark_id="viz-empty",
        plan=BenchmarkPlan(
            benchmark_id="viz-empty",
            case_count=0,
            variant_count=0,
            planned_run_count=0,
        ),
        runs=[],
        environment=EnvironmentMetadata(
            python_version="3.11.13",
            platform="test-platform",
            cwd="/tmp/autobench",
        ),
    )


def _numeric_report() -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_id="viz-demo",
        experiment_id="exp-1",
        run_count=4,
        status_counts={"failed": 1, "passed": 3},
        leaderboard=[
            LeaderboardRow(
                variant_id="baseline",
                run_count=2,
                metrics={"avg_quality": 0.5, "avg_cost": 0.2},
            ),
            LeaderboardRow(
                variant_id="candidate",
                run_count=2,
                metrics={"avg_quality": 0.8, "avg_cost": 0.4},
            ),
        ],
        case_matrix=CaseMatrix(
            metric="quality.score",
            rows={
                "case_a": {"baseline": 0.45, "candidate": 0.75},
                "case_b": {"baseline": 0.55, "candidate": 0.85},
            },
        ),
        comparisons=[
            ComparisonReport(
                baseline="baseline",
                candidate="candidate",
                run_count=2,
                factor_deltas={},
                metric_deltas={"avg_quality": {"baseline": 0.5, "candidate": 0.8, "delta": 0.3}},
            )
        ],
        distributions=[
            MetricDistribution(
                name="latency_distribution",
                semantic_type="time.latency",
                by_variant={"baseline": [12.0, 13.0], "candidate": [9.0, 10.0]},
                summaries={
                    "baseline": {"min": 12.0, "median": 12.5, "max": 13.0},
                    "candidate": {"min": 9.0, "median": 9.5, "max": 10.0},
                },
            )
        ],
    )


def _nonnumeric_report() -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_id="viz-nonnumeric",
        experiment_id="exp-2",
        run_count=2,
        leaderboard=[
            LeaderboardRow(
                variant_id="baseline",
                run_count=1,
                metrics={"label": "slow"},
            ),
            LeaderboardRow(
                variant_id="candidate",
                run_count=1,
                metrics={"label": "fast"},
            ),
        ],
        case_matrix=CaseMatrix(
            metric="result.label",
            rows={"case_a": {"baseline": "slow", "candidate": "fast"}},
        ),
        comparisons=[
            ComparisonReport(
                baseline="baseline",
                candidate="candidate",
                run_count=1,
                factor_deltas={},
                metric_deltas={
                    "label_delta": {"baseline": "slow", "candidate": "fast", "delta": "n/a"}
                },
            ),
            ComparisonReport(
                baseline="empty",
                candidate="candidate",
                run_count=0,
                factor_deltas={},
                metric_deltas={},
            ),
        ],
        distributions=[
            MetricDistribution(
                name="label_distribution",
                semantic_type="result.label",
                by_variant={"baseline": ["slow"], "candidate": ["fast"]},
                summaries={"baseline": {"mode": "slow"}, "candidate": {"mode": "fast"}},
            )
        ],
    )


def _zero_metric_report() -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_id="viz-zero",
        experiment_id="exp-zero",
        run_count=1,
        leaderboard=[
            LeaderboardRow(
                variant_id="baseline",
                run_count=1,
                metrics={"zero_score": 0.0},
            )
        ],
        case_matrix=CaseMatrix(metric="quality.score", rows={}),
    )
