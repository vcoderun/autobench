from __future__ import annotations as _annotations

from pathlib import Path
from textwrap import dedent

import pytest

from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    ComparisonVerdictSpec,
    DatasetSpec,
    DerivedMetricOutput,
    Direction,
    ExperimentResult,
    FactorValue,
    ObservationKind,
    ObservationRole,
    ObservationSource,
    PairedBaselineDeriverSpec,
    PairedBaselineFormula,
    PostDerivationMissingPolicy,
    RelativeThreshold,
    RunMatchKey,
    RunResult,
    Semantic,
    TaskSpec,
    Variant,
    classify_metric_comparison,
    derive_experiment_observations,
    load_benchmark_spec,
    run_benchmark_spec,
)
from autobench.reports.reporting import metric_value


async def test_paired_baseline_derivation_adds_candidate_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _spec(
        target="post_derive_tasks:latency",
        variants=[Variant(id="baseline"), Variant(id="candidate")],
        post_derive=[
            _post("speedup", "baseline_over_candidate"),
            _post("relative_latency", "candidate_over_baseline"),
            _post("latency_delta", "candidate_minus_baseline"),
            _post("latency_saved", "baseline_minus_candidate"),
            _post("latency_change_pct", "percent_change_from_baseline"),
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_post")

    baseline_easy = _run(result, case_id="case_easy", variant_id="baseline")
    candidate_easy = _run(result, case_id="case_easy", variant_id="candidate")
    values = _derived_metric_values(candidate_easy)

    assert _derived_metric_values(baseline_easy) == {}
    assert values == {
        "speedup": pytest.approx(2.0),
        "relative_latency": pytest.approx(0.5),
        "latency_delta": pytest.approx(-5.0),
        "latency_saved": pytest.approx(5.0),
        "latency_change_pct": pytest.approx(-50.0),
    }
    speedup = next(
        observation
        for observation in candidate_easy.task_result.observations
        if observation.name == "speedup"
    )
    assert speedup.kind is ObservationKind.METRIC
    assert speedup.source is ObservationSource.DERIVED
    assert speedup.semantic_type == "performance.speedup"
    assert speedup.direction is Direction.MAXIMIZE
    assert speedup.role is ObservationRole.OBJECTIVE
    assert speedup.tags["baseline_variant"] == "baseline"
    assert speedup.tags["source_metric"] == Semantic.TIME_LATENCY
    assert metric_value(candidate_easy, "performance.speedup") == pytest.approx(2.0)


async def test_paired_baseline_can_include_baseline_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _spec(
        target="post_derive_tasks:latency",
        variants=[Variant(id="baseline"), Variant(id="candidate")],
        post_derive=[
            _post("self_ratio", "baseline_over_candidate", include_baseline=True),
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_include_baseline")

    baseline = _run(result, case_id="case_easy", variant_id="baseline")
    candidate = _run(result, case_id="case_easy", variant_id="candidate")
    assert _derived_metric_values(baseline)["self_ratio"] == pytest.approx(1.0)
    assert _derived_metric_values(candidate)["self_ratio"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("target", "variants", "expected_reason"),
    [
        (
            "post_derive_tasks:latency",
            [Variant(id="candidate")],
            "missing_baseline_run",
        ),
        (
            "post_derive_tasks:candidate_only_latency",
            [Variant(id="baseline"), Variant(id="candidate")],
            "missing_metric",
        ),
        (
            "post_derive_tasks:text_latency",
            [Variant(id="baseline"), Variant(id="candidate")],
            "non_numeric_metric",
        ),
        (
            "post_derive_tasks:bool_latency",
            [Variant(id="baseline"), Variant(id="candidate")],
            "non_numeric_metric",
        ),
        (
            "post_derive_tasks:zero_candidate_latency",
            [Variant(id="baseline"), Variant(id="candidate")],
            "zero_division",
        ),
    ],
)
async def test_paired_baseline_derivation_emits_diagnostics_for_unavailable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    variants: list[Variant],
    expected_reason: str,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    result = await run_benchmark_spec(
        _spec(
            target=target,
            variants=variants,
            post_derive=[_post("speedup", "baseline_over_candidate")],
        ),
        experiment_id=f"exp_{expected_reason}",
    )

    candidate = _run(result, case_id="case_easy", variant_id="candidate")
    diagnostic = next(
        observation
        for observation in candidate.task_result.observations
        if observation.name == "paired_baseline_unavailable"
    )
    assert diagnostic.kind is ObservationKind.EVENT
    assert diagnostic.source is ObservationSource.DERIVED
    assert diagnostic.role is ObservationRole.DIAGNOSTIC
    assert diagnostic.value == expected_reason


async def test_paired_baseline_derivation_can_skip_unavailable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    missing_spec = _spec(
        target="post_derive_tasks:zero_candidate_latency",
        variants=[Variant(id="candidate")],
        post_derive=[
            _post("missing_speedup", "baseline_over_candidate", missing="skip"),
        ],
    )
    zero_spec = _spec(
        target="post_derive_tasks:zero_candidate_latency",
        variants=[Variant(id="baseline"), Variant(id="candidate")],
        post_derive=[
            _post("zero_speedup", "baseline_over_candidate", zero_division="skip"),
        ],
    )

    missing_result = await run_benchmark_spec(missing_spec, experiment_id="exp_missing_skip")
    zero_result = await run_benchmark_spec(zero_spec, experiment_id="exp_zero_skip")

    missing_candidate = _run(missing_result, case_id="case_easy", variant_id="candidate")
    zero_candidate = _run(zero_result, case_id="case_easy", variant_id="candidate")
    for candidate in (missing_candidate, zero_candidate):
        assert _derived_metric_values(candidate) == {}
        assert not any(
            observation.name == "paired_baseline_unavailable"
            for observation in candidate.task_result.observations
        )


async def test_empty_comparison_derivation_returns_original_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    result = await run_benchmark_spec(
        _spec(
            target="post_derive_tasks:latency",
            variants=[Variant(id="baseline")],
            post_derive=[],
        ),
        experiment_id="exp_no_post_derive",
    )

    assert derive_experiment_observations([], result=result) is result


def test_load_benchmark_spec_accepts_post_derive_and_report_config(tmp_path: Path) -> None:
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: post-derive-config
            task:
              kind: python
              target: app.tasks.run
            dataset:
              cases:
                - id: case_1
            variants:
              - id: baseline
              - id: candidate
            post_derive:
              - kind: paired_baseline
                baseline_variant: baseline
                match_on:
                  - kind: case_id
                  - kind: factor
                    name: workload.size
                metric: time.latency
                output:
                  name: speedup
                  semantic_type: performance.speedup
                formula: baseline_over_candidate
                threshold:
                  kind: relative_noise
                  pct: 2.0
                verdict:
                  output:
                    name: latency_verdict
                    semantic_type: comparison.verdict
                  threshold:
                    kind: relative_noise
                    pct: 2.0
            reports:
              leaderboard:
                metrics:
                  - name: avg_speedup
                    semantic_type: performance.speedup
                    fn: mean
              case_matrix:
                semantic_type: performance.speedup
              visuals:
                - kind: leaderboard
                  render_as: bar
                  metric: avg_speedup
                - kind: case_matrix
                  render_as: heatmap
            policies:
              - name: minimum_speedup
                metric: performance.speedup
                must_greater_equal: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    spec = load_benchmark_spec(spec_path)

    assert spec.post_derive[0].baseline_variant == "baseline"
    assert spec.post_derive[0].match_on[1].name == "workload.size"
    assert spec.post_derive[0].threshold is not None
    assert spec.post_derive[0].threshold.pct == 2.0
    assert spec.post_derive[0].verdict is not None
    assert spec.post_derive[0].verdict.output.name == "latency_verdict"
    assert spec.reports.leaderboard.metrics[0].name == "avg_speedup"
    assert spec.reports.case_matrix.semantic_type == "performance.speedup"
    assert spec.reports.visuals[0].kind == "leaderboard"
    assert spec.reports.visuals[1].kind == "case_matrix"
    assert spec.policies[0].name == "minimum_speedup"


async def test_paired_baseline_supports_factor_matching_threshold_and_verdicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="post-derive-match-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_easy")]),
        task=TaskSpec(kind="python", target="post_derive_tasks:latency_by_factor"),
        variants=[
            Variant(
                id="baseline_small",
                factors=[
                    FactorValue(name="role", value="baseline"),
                    FactorValue(name="workload.size", value="small"),
                ],
            ),
            Variant(
                id="candidate_small",
                factors=[
                    FactorValue(name="role", value="candidate"),
                    FactorValue(name="workload.size", value="small"),
                ],
            ),
            Variant(
                id="candidate_large",
                factors=[
                    FactorValue(name="role", value="candidate"),
                    FactorValue(name="workload.size", value="large"),
                ],
            ),
        ],
        post_derive=[
            PairedBaselineDeriverSpec(
                baseline_variant="baseline_small",
                match_on=(
                    RunMatchKey(kind="case_id"),
                    RunMatchKey(kind="factor", name="workload.size"),
                ),
                metric=Semantic.TIME_LATENCY,
                output=DerivedMetricOutput(
                    name="speedup",
                    semantic_type="performance.speedup",
                    direction=Direction.MAXIMIZE,
                ),
                formula="baseline_over_candidate",
                threshold=RelativeThreshold(pct=2.0),
                verdict=ComparisonVerdictSpec(
                    output=DerivedMetricOutput(
                        name="latency_verdict",
                        semantic_type="comparison.verdict",
                    ),
                    threshold=RelativeThreshold(pct=2.0),
                ),
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_factor_match")

    candidate_small = _run(result, case_id="case_easy", variant_id="candidate_small")
    candidate_large = _run(result, case_id="case_easy", variant_id="candidate_large")
    speedup = next(
        observation
        for observation in candidate_small.task_result.observations
        if observation.name == "speedup"
    )
    verdict = next(
        observation
        for observation in candidate_small.task_result.observations
        if observation.name == "latency_verdict"
    )
    diagnostic = next(
        observation
        for observation in candidate_large.task_result.observations
        if observation.name == "paired_baseline_unavailable"
    )

    assert speedup.value == pytest.approx(2.0)
    assert speedup.tags["significant"] is True
    assert verdict.value == "improved"
    assert diagnostic.value == "missing_baseline_run"


async def test_paired_baseline_reports_missing_candidate_match_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="post-derive-missing-match-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_easy")]),
        task=TaskSpec(kind="python", target="post_derive_tasks:latency_by_factor"),
        variants=[
            Variant(
                id="baseline_small",
                factors=[
                    FactorValue(name="role", value="baseline"),
                    FactorValue(name="workload.size", value="small"),
                ],
            ),
            Variant(
                id="candidate_missing_size", factors=[FactorValue(name="role", value="candidate")]
            ),
        ],
        post_derive=[
            PairedBaselineDeriverSpec(
                baseline_variant="baseline_small",
                match_on=(
                    RunMatchKey(kind="case_id"),
                    RunMatchKey(kind="factor", name="workload.size"),
                ),
                metric=Semantic.TIME_LATENCY,
                output=DerivedMetricOutput(name="speedup", semantic_type="performance.speedup"),
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_missing_match_key")

    candidate = _run(result, case_id="case_easy", variant_id="candidate_missing_size")
    diagnostic = next(
        observation
        for observation in candidate.task_result.observations
        if observation.name == "paired_baseline_unavailable"
    )
    assert diagnostic.value == "missing_match_key"


def test_comparison_classifier_handles_directions_thresholds_and_inconclusive() -> None:
    assert (
        classify_metric_comparison(
            baseline=10.0,
            candidate=9.9,
            direction=Direction.MINIMIZE,
            threshold_pct=2.0,
        )
        == "unchanged"
    )
    assert (
        classify_metric_comparison(
            baseline=10.0,
            candidate=9.0,
            direction=Direction.MINIMIZE,
        )
        == "improved"
    )
    assert (
        classify_metric_comparison(
            baseline=10.0,
            candidate=11.0,
            direction=Direction.MAXIMIZE,
        )
        == "improved"
    )
    assert (
        classify_metric_comparison(
            baseline=10.0,
            candidate=9.0,
            direction=Direction.MAXIMIZE,
        )
        == "regressed"
    )
    assert (
        classify_metric_comparison(
            baseline=0.0,
            candidate=1.0,
            direction=Direction.TARGET,
        )
        == "inconclusive"
    )


def test_match_key_validation_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="require name"):
        RunMatchKey(kind="factor")
    with pytest.raises(ValueError, match="cannot declare name"):
        RunMatchKey(kind="case_id", name="case")


def _post(
    name: str,
    formula: PairedBaselineFormula,
    *,
    include_baseline: bool = False,
    missing: PostDerivationMissingPolicy = "diagnostic",
    zero_division: PostDerivationMissingPolicy = "diagnostic",
) -> PairedBaselineDeriverSpec:
    return PairedBaselineDeriverSpec(
        baseline_variant="baseline",
        metric=Semantic.TIME_LATENCY,
        output=DerivedMetricOutput(
            name=name,
            semantic_type="performance.speedup",
            unit="x",
            direction=Direction.MAXIMIZE,
            role=ObservationRole.OBJECTIVE,
        ),
        formula=formula,
        include_baseline=include_baseline,
        missing=missing,
        zero_division=zero_division,
    )


def _spec(
    *,
    target: str,
    variants: list[Variant],
    post_derive: list[PairedBaselineDeriverSpec],
) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="post-derive-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_easy"), Case(id="case_hard")]),
        task=TaskSpec(kind="python", target=target),
        variants=variants,
        post_derive=post_derive,
    )


def _run(result: ExperimentResult, *, case_id: str, variant_id: str) -> RunResult:
    return next(
        run for run in result.runs if run.case_id == case_id and run.variant_id == variant_id
    )


def _derived_metric_values(run: RunResult) -> dict[str, float]:
    return {
        observation.name: float(observation.value)
        for observation in run.task_result.observations
        if observation.source is ObservationSource.DERIVED
        and observation.kind is ObservationKind.METRIC
    }


def _write_module(tmp_path: Path) -> None:
    source = """
    def latency(ctx, case):
        base = 20.0 if case.id == "case_hard" else 10.0
        value = base / 2.0 if ctx.variant.id == "candidate" else base
        ctx.metric("median_ms", value, semantic_type="time.latency", unit="ms")
        return {"ok": True}

    def candidate_only_latency(ctx, case):
        if ctx.variant.id == "candidate":
            ctx.metric("median_ms", 5.0, semantic_type="time.latency", unit="ms")
        return {"ok": True}

    def text_latency(ctx, case):
        ctx.metric("median_ms", "fast", semantic_type="time.latency", unit="ms")
        return {"ok": True}

    def bool_latency(ctx, case):
        ctx.metric("median_ms", True, semantic_type="time.latency", unit="ms")
        return {"ok": True}

    def zero_candidate_latency(ctx, case):
        value = 0.0 if ctx.variant.id == "candidate" else 10.0
        ctx.metric("median_ms", value, semantic_type="time.latency", unit="ms")
        return {"ok": True}

    def latency_by_factor(ctx, case):
        role = ctx.factor("role")
        size = ctx.factor("workload.size")
        if role == "baseline":
            value = 10.0
        elif size == "small":
            value = 5.0
        else:
            value = 4.0
        ctx.metric(
            "median_ms",
            value,
            semantic_type="time.latency",
            unit="ms",
            direction="minimize",
        )
        return {"ok": True}
    """
    (tmp_path / "post_derive_tasks.py").write_text(
        dedent(source).strip() + "\n",
        encoding="utf-8",
    )
