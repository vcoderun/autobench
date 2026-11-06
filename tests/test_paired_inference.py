from __future__ import annotations as _annotations

from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from rich.console import Console

from autobench import (
    BenchmarkPlan,
    Case,
    Direction,
    EvaluationStatus,
    ExperimentResult,
    ObservationRole,
    RunResult,
    RunStatus,
    ScoreRecord,
    Semantic,
    TaskResult,
    TaskStatus,
)
from autobench.cli import cli
from autobench.data.variants import FactorValue
from autobench.records.storage import EnvironmentMetadata
from autobench.reports.inference import (
    build_hypothesis_results,
    exact_sign_test,
    holm_adjusted,
    paired_success_difference,
)
from autobench.reports.markdown import render_markdown_report
from autobench.reports.models import (
    ComparisonReportSpec,
    MetricAggregation,
    PairedBinaryTable,
    ReportSpec,
)
from autobench.reports.reporting import build_report, compare_variants
from autobench.reports.rich import render_comparison, render_report
from autobench.spec import load_benchmark_spec

SUCCESS = MetricAggregation(name="success", semantic_type=Semantic.RESULT_SUCCESS, fn="ratio_true")
QUALITY = MetricAggregation(name="quality", semantic_type=Semantic.QUALITY_SCORE, fn="mean")


def _hypothesis_comparison(
    tag: str,
    family: str,
    *,
    name: str | None = None,
    alternative: str = "candidate_better",
    alpha: float = 0.05,
) -> dict[str, object]:
    return {
        "name": name or tag,
        "baseline": "baseline",
        "candidate": "candidate",
        "case_tags": [tag],
        "metrics": [SUCCESS.model_dump()],
        "hypothesis": {
            "metric": "success",
            "family": family,
            "alternative": alternative,
            "alpha": alpha,
        },
    }


@pytest.mark.parametrize(
    ("wins", "losses", "candidate_better_p", "two_sided_p"),
    [
        (7, 0, 0.0078125, 0.015625),
        (1, 1, 0.75, 1.0),
        (3, 1, 0.3125, 0.625),
        (0, 0, 1.0, 1.0),
        (0, 4, 1.0, 0.125),
    ],
)
def test_exact_sign_test_matches_the_exact_mcnemar_values(
    wins: int,
    losses: int,
    candidate_better_p: float,
    two_sided_p: float,
) -> None:
    result = exact_sign_test(wins, losses)

    assert result.discordant_pairs == wins + losses
    assert result.candidate_better_p == pytest.approx(candidate_better_p)
    assert result.two_sided_p == pytest.approx(two_sided_p)


def test_exact_sign_test_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        exact_sign_test(-1, 2)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (PairedBinaryTable(both=8, candidate_only=7, baseline_only=0, neither=0), (47, 17, 70)),
        (PairedBinaryTable(both=0, candidate_only=1, baseline_only=1, neither=22), (0, -17, 17)),
        (PairedBinaryTable(both=10, candidate_only=3, baseline_only=1, neither=1), (13, -13, 38)),
    ],
)
def test_paired_success_difference_reproduces_published_newcombe_intervals(
    table: PairedBinaryTable,
    expected: tuple[int, int, int],
) -> None:
    interval = paired_success_difference(table)

    assert interval is not None
    assert interval.method == "newcombe_paired_wilson"
    assert interval.confidence == 0.95
    assert (
        round(interval.estimate * 100),
        round(interval.low * 100),
        round(interval.high * 100),
    ) == expected


def test_paired_success_difference_handles_empty_tables_and_degenerate_margins() -> None:
    assert (
        paired_success_difference(
            PairedBinaryTable(both=0, candidate_only=0, baseline_only=0, neither=0)
        )
        is None
    )

    all_passed = paired_success_difference(
        PairedBinaryTable(both=5, candidate_only=0, baseline_only=0, neither=0)
    )
    assert all_passed is not None
    assert all_passed.estimate == 0.0
    assert all_passed.low < 0.0 < all_passed.high

    with pytest.raises(ValueError, match="confidence"):
        paired_success_difference(
            PairedBinaryTable(both=1, candidate_only=0, baseline_only=0, neither=0),
            confidence=1.0,
        )


def test_holm_adjustment_is_monotone_and_keeps_input_order() -> None:
    assert holm_adjusted([0.0078125, 0.75, 0.3125]) == pytest.approx((0.0234375, 0.75, 0.625))
    assert holm_adjusted([0.04, 0.03]) == pytest.approx((0.06, 0.06))
    assert holm_adjusted([]) == ()
    with pytest.raises(ValueError, match="between zero and one"):
        holm_adjusted([1.5])


def test_comparison_pairs_boolean_metrics_and_filters_by_case_tags() -> None:
    result = _result(
        {
            "h1-1": (("h1",), (False, True), (0.1, 0.5)),
            "h1-2": (("h1",), (False, True), (0.2, 0.4)),
            "h1-3": (("h1",), (True, True), (0.3, 0.3)),
            "h1-4": (("h1",), (True, None), (0.4, 0.4)),
            "h2-1": (("h2",), (True, False), (0.9, 0.1)),
        }
    )

    comparison = compare_variants(
        result,
        baseline="baseline",
        candidate="candidate",
        metrics=(SUCCESS, QUALITY),
        case_tags=("h1",),
        name="first",
    )
    metrics = {metric.name: metric for metric in comparison.metric_results}

    assert comparison.name == "first"
    assert comparison.case_tags == ("h1",)
    assert comparison.paired_count == 4
    success = metrics["success"]
    assert (success.paired_count, success.missing_pair_count) == (3, 1)
    assert (success.wins, success.ties, success.losses) == (2, 1, 0)
    assert success.binary_table == PairedBinaryTable(
        both=1, candidate_only=2, baseline_only=0, neither=0
    )
    assert success.sign_test is not None
    assert success.sign_test.candidate_better_p == pytest.approx(0.25)
    assert success.success_difference is not None
    assert success.success_difference.estimate == pytest.approx(2 / 3)

    quality = metrics["quality"]
    assert quality.binary_table is None
    assert quality.success_difference is None
    assert quality.sign_test is not None
    assert quality.sign_test.discordant_pairs == 2
    assert quality.paired_count == 4


def test_comparison_without_pairs_reports_no_test() -> None:
    comparison = compare_variants(
        _result({"only-1": (("x",), (True, True), (0.5, 0.5))}),
        baseline="baseline",
        candidate="candidate",
        metrics=(SUCCESS,),
        case_tags=("absent",),
    )

    assert comparison.paired_count == 0
    assert comparison.metric_results[0].sign_test is None
    assert comparison.metric_results[0].binary_table is None


def test_hypotheses_are_holm_adjusted_within_families_and_keep_untestable_members() -> None:
    result = _result(
        {
            **{f"a-{index}": (("a",), (False, True), (0.1, 0.2)) for index in range(7)},
            "b-1": (("b",), (True, False), (0.5, 0.4)),
            "b-2": (("b",), (False, True), (0.4, 0.5)),
        }
    )
    spec = ReportSpec.model_validate(
        {
            "comparisons": [
                _hypothesis_comparison("a", "primary"),
                _hypothesis_comparison("b", "primary", alternative="two_sided"),
                _hypothesis_comparison("missing", "primary"),
                _hypothesis_comparison("a", "secondary", name="a-secondary"),
            ]
        }
    )

    report = build_report(result, report_spec=spec)
    hypotheses = {hypothesis.comparison: hypothesis for hypothesis in report.hypotheses}

    first = hypotheses["a"]
    assert first.p_value == pytest.approx(0.0078125)
    assert first.adjusted_p_value == pytest.approx(0.0234375)
    assert first.status == "supported"
    assert first.family_size == 3
    second = hypotheses["b"]
    assert second.alternative == "two_sided"
    assert second.p_value == pytest.approx(1.0)
    assert second.status == "not_supported"
    untestable = hypotheses["missing"]
    assert untestable.status == "untestable"
    assert untestable.p_value is None
    assert untestable.adjusted_p_value is None
    secondary = hypotheses["a-secondary"]
    assert secondary.family_size == 1
    assert secondary.adjusted_p_value == pytest.approx(0.0078125)


def test_hypothesis_results_require_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        build_hypothesis_results((ComparisonReportSpec(baseline="a", candidate="b"),), ())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "comparisons": [
                    {"baseline": "a", "candidate": "b", "hypothesis": {"metric": "nope"}}
                ]
            },
            "is not one of the comparison metrics",
        ),
        (
            {
                "comparisons": [
                    {"baseline": "a", "candidate": "b"},
                    {"baseline": "a", "candidate": "b"},
                ]
            },
            "comparison labels must be unique",
        ),
        (
            {
                "comparisons": [
                    _hypothesis_comparison("a", "primary"),
                    _hypothesis_comparison("b", "primary", alpha=0.01),
                ]
            },
            "mixes alpha levels",
        ),
    ],
)
def test_report_spec_rejects_ambiguous_comparison_declarations(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ReportSpec.model_validate(payload)


def test_comparison_labels_default_to_the_variant_pair() -> None:
    assert ComparisonReportSpec(baseline="a", candidate="b").label() == "a -> b"
    assert ComparisonReportSpec(name="named", baseline="a", candidate="b").label() == "named"


def test_yaml_compare_entries_accept_names_tags_and_hypotheses(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        dedent(
            """
            benchmark:
              paired:
                cases:
                  - id: one
                    tags: [h1]
                run:
                  python: tasks:run
                variants:
                  base: {}
                  next: {}
                report:
                  compare:
                    base -> next:
                      tags: [h1]
                    h1:
                      baseline: base
                      candidate: next
                      tags: [h1]
                      show:
                        success:
                          metric: result.success
                          aggregate: ratio_true
                      hypothesis:
                        metric: success
                        alternative: two_sided
            """
        ),
        encoding="utf-8",
    )

    comparisons = load_benchmark_spec(path).reports.comparisons

    assert comparisons[0].name is None
    assert comparisons[0].case_tags == ("h1",)
    assert comparisons[1].name == "h1"
    assert comparisons[1].hypothesis is not None
    assert comparisons[1].hypothesis.alternative == "two_sided"


def test_reports_render_paired_tests_and_hypotheses() -> None:
    result = _result(
        {
            **{f"a-{index}": (("a",), (False, True), (0.1, 0.2)) for index in range(3)},
            "a-tie": (("a",), (True, True), (0.3, 0.3)),
        }
    )
    spec = ReportSpec.model_validate(
        {
            "comparisons": [
                _hypothesis_comparison("a", "primary"),
                _hypothesis_comparison("missing", "primary"),
                {
                    "baseline": "baseline",
                    "candidate": "candidate",
                    "metrics": [QUALITY.model_dump()],
                },
            ]
        }
    )
    report = build_report(result, report_spec=spec)

    markdown = render_markdown_report(report)
    assert "### a: baseline vs candidate" in markdown
    assert "case tags: `a`" in markdown
    assert "### baseline vs candidate" in markdown
    assert "## Hypotheses" in markdown
    assert "+75 pts (" in markdown
    assert "| untestable |" in markdown

    console = Console(record=True, width=200)
    render_report(console, report)
    text = console.export_text()
    assert "a: baseline vs candidate" in text
    assert "Paired Tests" in text
    assert "Hypotheses (Holm-adjusted within each family)" in text
    assert "untestable" in text

    untagged = Console(record=True, width=200)
    render_comparison(untagged, report.comparisons[2])
    assert "Case tags" not in untagged.export_text()


def test_cli_compare_accepts_tags_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "paired_tasks.py").write_text(
        dedent(
            """
            def run(ctx, case):
                return {"passed": ctx.variant.id == "candidate" or case.id == "both"}
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              paired-cli:
                cases:
                  - id: first
                    tags: [selected]
                  - id: both
                    tags: [selected]
                  - id: other
                run:
                  python: paired_tasks:run
                variants:
                  baseline: {}
                  candidate: {}
                score:
                  passed:
                    pass: output.passed
                    semantic: result.success
                    goal: maximize
            """
        ),
        encoding="utf-8",
    )
    record_dir = tmp_path / "runs"
    runner = CliRunner()
    assert runner.invoke(cli, ["run", str(spec_path), "--record", str(record_dir)]).exit_code == 0

    compared = runner.invoke(
        cli,
        [
            "compare",
            str(record_dir),
            "--baseline",
            "baseline",
            "--candidate",
            "candidate",
            "--tag",
            "selected",
            "--metric",
            "result.success:ratio_true",
        ],
        terminal_width=200,
    )
    assert compared.exit_code == 0, compared.output
    assert "Paired Tests" in compared.output
    assert "1/1/0" in compared.output

    for bad_metric, message in (
        ("result.success", "expected SEMANTIC:AGGREGATE"),
        ("result.success:average", "unknown aggregate"),
    ):
        rejected = runner.invoke(
            cli,
            [
                "compare",
                str(record_dir),
                "--baseline",
                "baseline",
                "--candidate",
                "candidate",
                "--metric",
                bad_metric,
            ],
        )
        assert rejected.exit_code != 0
        assert message in rejected.output


def _result(
    cases: dict[str, tuple[tuple[str, ...], tuple[bool, bool | None], tuple[float, float]]],
) -> ExperimentResult:
    runs: list[RunResult] = []
    for case_id, (tags, successes, qualities) in cases.items():
        for variant_id, success, quality in zip(
            ("baseline", "candidate"), successes, qualities, strict=True
        ):
            runs.append(
                RunResult(
                    run_id=f"{case_id}_{variant_id}",
                    benchmark_id="paired",
                    experiment_id="exp_paired",
                    case_id=case_id,
                    variant_id=variant_id,
                    status=RunStatus.PASSED,
                    evaluation_status=EvaluationStatus.PASSED,
                    case=Case(id=case_id, tags=list(tags)),
                    task_result=TaskResult(output={}, status=TaskStatus.PASSED),
                    scores=[
                        ScoreRecord(
                            name="success",
                            semantic_type=Semantic.RESULT_SUCCESS,
                            value=success,
                            direction=Direction.MAXIMIZE,
                            role=ObservationRole.OBJECTIVE,
                        ),
                        ScoreRecord(
                            name="quality",
                            semantic_type=Semantic.QUALITY_SCORE,
                            value=quality,
                            direction=Direction.MAXIMIZE,
                            role=ObservationRole.DIAGNOSTIC,
                        ),
                    ],
                    factors=[FactorValue(name="variant", value=variant_id)],
                )
            )
    return ExperimentResult(
        experiment_id="exp_paired",
        benchmark_id="paired",
        plan=BenchmarkPlan(
            benchmark_id="paired",
            case_ids=tuple(cases),
            case_count=len(cases),
            variant_count=2,
            planned_run_count=2 * len(cases),
        ),
        runs=runs,
        environment=EnvironmentMetadata(
            python_version="3.11.13", platform="test", cwd="/workspace"
        ),
    )


def test_metrics_without_direction_keep_counts_but_have_no_sign_test() -> None:
    result = _result({"case-1": ((), (False, True), (0.1, 0.2))})
    for run in result.runs:
        run.scores.append(
            ScoreRecord(
                name="flag", semantic_type="custom.flag", value=run.variant_id == "candidate"
            )
        )

    comparison = compare_variants(
        result,
        baseline="baseline",
        candidate="candidate",
        metrics=(MetricAggregation(name="flag", semantic_type="custom.flag", fn="ratio_true"),),
    )
    flag = comparison.metric_results[0]

    assert flag.direction is None
    assert flag.paired_count == 1
    assert flag.sign_test is None
    assert flag.binary_table == PairedBinaryTable(
        both=0, candidate_only=1, baseline_only=0, neither=0
    )
