from __future__ import annotations as _annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from autobench import (
    BenchmarkInfo,
    BenchmarkPlan,
    BenchmarkSpec,
    BetweenRequirement,
    Case,
    DatasetSpec,
    EvaluationStatus,
    ExperimentResult,
    ObservationSource,
    PolicySpec,
    RunContext,
    RunResult,
    RunStatus,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    apply_policies,
    capture_environment,
    evaluate_policies,
    evaluate_run_policies,
    run_benchmark_spec,
)


@pytest.mark.parametrize(
    ("policy", "value", "expected"),
    [
        (PolicySpec(name="equal", metric="metric.score", must_equal=False), False, True),
        (PolicySpec(name="not_equal", metric="metric.score", must_not_equal=0), 1, True),
        (PolicySpec(name="greater", metric="metric.score", must_greater=1.0), 2.0, True),
        (
            PolicySpec(name="greater_equal", metric="metric.score", must_greater_equal=1.0),
            1.0,
            True,
        ),
        (PolicySpec(name="less", metric="metric.score", must_less=1.0), 0.5, True),
        (PolicySpec(name="less_equal", metric="metric.score", must_less_equal=1.0), 1.0, True),
        (PolicySpec(name="in", metric="metric.score", must_in=("a", "b")), "a", True),
        (
            PolicySpec(name="not_in", metric="metric.score", must_not_in=("a", "b")),
            "c",
            True,
        ),
        (
            PolicySpec(
                name="between",
                metric="metric.score",
                must_between=BetweenRequirement(min=1.0, max=2.0),
            ),
            1.0,
            True,
        ),
        (
            PolicySpec(
                name="between_open",
                metric="metric.score",
                must_between=BetweenRequirement(min=1.0, max=2.0, inclusive=False),
            ),
            1.0,
            False,
        ),
        (
            PolicySpec(name="numeric_on_text", metric="metric.score", must_greater=1.0),
            "fast",
            False,
        ),
        (
            PolicySpec(name="numeric_on_bool", metric="metric.score", must_greater=0.0),
            True,
            False,
        ),
    ],
)
def test_policy_requirements_evaluate_run_metrics(
    policy: PolicySpec,
    value: object,
    expected: bool,
) -> None:
    run = _run_with_metric(value)

    result = evaluate_run_policies([policy], run=run)[0]

    assert result.policy_name == policy.name
    assert result.actual == value
    assert result.passed is expected
    assert result.reason is (None if expected else "requirement_failed")


def test_policies_handle_missing_metrics_and_append_policy_observations() -> None:
    run = _run_with_metric(0.75)
    result = _experiment([run])
    policies = [
        PolicySpec(name="quality_gate", metric="metric.score", must_greater_equal=0.5),
        PolicySpec(name="missing_gate", metric="missing.metric", must_equal=True),
    ]

    policy_results = evaluate_policies(policies, result=result)
    updated = apply_policies(policies, result=result)
    empty = apply_policies([], result=result)

    assert empty is result
    assert [policy.passed for policy in policy_results] == [True, False]
    assert policy_results[1].reason == "missing_metric"
    observations = updated.runs[0].task_result.observations
    policy_observations = [
        observation for observation in observations if observation.semantic_type == "policy.result"
    ]
    assert [observation.name for observation in policy_observations] == [
        "quality_gate",
        "missing_gate",
    ]
    assert [observation.value for observation in policy_observations] == [True, False]
    assert {observation.source for observation in policy_observations} == {
        ObservationSource.DERIVED
    }


def test_policy_specs_validate_requirement_shapes() -> None:
    with pytest.raises(ValidationError, match="exactly one requirement"):
        PolicySpec(name="missing", metric="metric.score")
    with pytest.raises(ValidationError, match="exactly one requirement"):
        PolicySpec(
            name="too_many",
            metric="metric.score",
            must_greater=1.0,
            must_less=2.0,
        )
    with pytest.raises(ValidationError, match="between min"):
        BetweenRequirement(min=2.0, max=1.0)


async def test_benchmark_pipeline_applies_policy_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "policy_tasks.py").write_text(
        dedent(
            """
            def run(ctx, case):
                ctx.metric("score", 0.9, semantic_type="metric.score")
                return {"ok": True}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="policy-pipeline"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="policy_tasks:run"),
        variants=[Variant(id="variant_1")],
        policies=[PolicySpec(name="score_gate", metric="metric.score", must_greater=0.5)],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_policy_pipeline")

    policy_observation = next(
        observation
        for observation in result.runs[0].task_result.observations
        if observation.name == "score_gate"
    )
    assert policy_observation.value is True


def _run_with_metric(value: object) -> RunResult:
    ctx = RunContext(benchmark_id="policy-demo", case=Case(id="case_1"), variant=Variant(id="v1"))
    ctx.metric("score", value, semantic_type="metric.score")
    return RunResult(
        run_id="run_1",
        benchmark_id="policy-demo",
        experiment_id="exp_policy",
        case_id="case_1",
        variant_id="v1",
        status=RunStatus.PASSED,
        evaluation_status=EvaluationStatus.PASSED,
        case=Case(id="case_1"),
        task_result=TaskResult(
            status=TaskStatus.PASSED,
            observations=list(ctx.observations),
        ),
    )


def _experiment(runs: list[RunResult]) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp_policy",
        benchmark_id="policy-demo",
        plan=BenchmarkPlan(
            benchmark_id="policy-demo",
            case_count=1,
            variant_count=1,
            planned_run_count=1,
        ),
        runs=runs,
        environment=capture_environment(),
    )
