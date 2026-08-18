from __future__ import annotations as _annotations

from autobench.reports.charts import (
    render_case_score_chart,
    render_dimension_chart,
    render_quality_gate_chart,
)
from autobench.reports.models import (
    EvaluationCaseReport,
    EvaluationMetricReport,
    EvaluationSummaryReport,
)


def test_quality_gate_chart_requires_evaluated_cases_and_shows_outcome() -> None:
    empty = _summary()
    evaluation = _summary(
        evaluated_count=4,
        passed_count=3,
        failed_count=1,
        pass_rate=0.75,
    )

    assert render_quality_gate_chart(empty) is None
    chart = render_quality_gate_chart(evaluation)
    assert chart is not None
    assert "75% passed" in chart
    assert "Passed 3" in chart
    assert "Failed 1" in chart

    missing_rate = render_quality_gate_chart(_summary(evaluated_count=1, failed_count=1))
    assert missing_rate is not None and "0% passed" in missing_rate


def test_case_score_chart_ranks_escapes_and_bounds_visible_cases() -> None:
    cases = tuple(
        EvaluationCaseReport(
            run_id=f"run-{index}",
            case_id="unsafe <script>" if index == 34 else f"case-{index:02d}",
            variant_id="candidate",
            quality_pass=True if index == 34 else False if index == 33 else None,
            score=index / 40,
        )
        for index in range(35)
    )
    chart = render_case_score_chart(_summary(cases=cases))

    assert chart is not None
    assert "Case score ranking" in chart
    assert "Showing the highest-scoring 30 of 35 cases" in chart
    assert "unsafe <script>" not in chart
    assert "unsafe &lt;script&gt;" in chart
    assert "case-33" in chart
    assert "case-01" not in chart
    assert "#2E7D5B" in chart
    assert "#C44E52" in chart
    assert "#2F6F73" in chart

    assert render_case_score_chart(_summary()) is None
    negative = render_case_score_chart(
        _summary(
            cases=(
                EvaluationCaseReport(
                    run_id="negative",
                    case_id="negative",
                    variant_id="candidate",
                    score=-1,
                ),
            )
        )
    )
    assert negative is not None and ">-1<" in negative


def test_dimension_chart_only_renders_normalized_score_metrics() -> None:
    ignored = EvaluationMetricReport(
        name="latency",
        label="Latency",
        kind="value",
        sample_count=2,
        missing_count=0,
        mean=120,
        median=120,
        minimum=100,
        maximum=140,
        total=240,
    )
    assert render_dimension_chart(_summary(metrics=(ignored,))) is None

    metrics = tuple(
        EvaluationMetricReport(
            name=f"score_{index}",
            label="unsafe <dimension>" if index == 0 else f"Dimension {index}",
            kind="score",
            sample_count=3,
            missing_count=0,
            mean=index / 10,
            median=index / 10,
            minimum=0,
            maximum=index / 10,
            total=3 * index / 10,
        )
        for index in range(11)
    )
    chart = render_dimension_chart(_summary(metrics=metrics))

    assert chart is not None
    assert "Average quality by dimension" in chart
    assert "unsafe &lt;dimension&gt;" in chart
    assert "Dimension 9" in chart
    assert "Dimension 10" not in chart
    assert "n=3" in chart


def _summary(
    *,
    evaluated_count: int = 0,
    passed_count: int = 0,
    failed_count: int = 0,
    pass_rate: float | None = None,
    cases: tuple[EvaluationCaseReport, ...] = (),
    metrics: tuple[EvaluationMetricReport, ...] = (),
) -> EvaluationSummaryReport:
    return EvaluationSummaryReport(
        case_count=len(cases),
        evaluated_count=evaluated_count,
        passed_count=passed_count,
        failed_count=failed_count,
        unevaluated_count=max(0, len(cases) - evaluated_count),
        pass_rate=pass_rate,
        score_count=sum(case.score is not None for case in cases),
        metrics=metrics,
        cases=cases,
    )
