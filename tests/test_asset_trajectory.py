from __future__ import annotations as _annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError
from rich.console import Console

from autobench import (
    AssetVersion,
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
    SpecValidationError,
    TaskResult,
    TaskStatus,
    TrackedAssetSummary,
    benchmark_spec_to_yaml_view,
    build_asset_trajectory,
    summarize_tracked_assets,
)
from autobench.data.variants import FactorValue
from autobench.records.storage import EnvironmentMetadata
from autobench.reports.inference import holm_adjusted
from autobench.reports.markdown import render_markdown_report
from autobench.reports.models import MetricAggregation, ReportSpec
from autobench.reports.reporting import build_report
from autobench.reports.rich import render_report, render_tracked_assets, render_trajectory
from autobench.spec import load_benchmark_spec

ASSET = "kedi:form:planner"
SUCCESS = MetricAggregation(name="success", semantic_type=Semantic.RESULT_SUCCESS, fn="ratio_true")
FLAG = MetricAggregation(name="flag", semantic_type="custom.flag", fn="ratio_true")
CASES = ("c1", "c2", "c3", "c4")


def test_versions_follow_lineage_and_pair_with_their_parent() -> None:
    result = _result(
        # The child is recorded first: lineage order, not run order, decides the steps.
        _runs("v2", "v1", (True, True, True, False))
        + _runs("v1", None, (False, False, True, False))
        + _runs("v3", "v2", (True, True, False, False))
        + _runs("branch", "v1", (False, False, True, False))
    )

    trajectory = build_asset_trajectory(result, asset_id=ASSET, metrics=(SUCCESS,))

    assert trajectory.experiment_ids == ("exp_trajectory",)
    assert [(step.version, step.parent_version) for step in trajectory.steps] == [
        ("v1", None),
        ("v2", "v1"),
        ("v3", "v2"),
        ("branch", "v1"),
    ]
    root, improved, regressed, unchanged = trajectory.steps
    assert root.comparison is None
    assert root.run_count == len(CASES)
    assert improved.comparison is not None
    assert (improved.comparison.baseline, improved.comparison.candidate) == ("v1", "v2")
    success = improved.comparison.metric_results[0]
    assert (success.wins, success.ties, success.losses) == (2, 2, 0)
    assert success.sign_test is not None
    assert success.sign_test.two_sided_p == 0.5
    assert regressed.comparison is not None
    assert regressed.comparison.metric_results[0].losses == 1
    assert unchanged.comparison is not None
    assert unchanged.comparison.metric_results[0].ties == len(CASES)


def test_step_p_values_are_holm_adjusted_across_the_trajectory() -> None:
    result = _result(
        _runs("v1", None, (False,) * 4)
        + _runs("v2", "v1", (True,) * 4)
        + _runs("v3", "v2", (True, True, True, False))
    )

    root, second, third = build_asset_trajectory(
        result, asset_id=ASSET, metrics=(SUCCESS, FLAG)
    ).steps

    raw = []
    for step in (second, third):
        assert step.comparison is not None
        sign_test = step.comparison.metric_results[0].sign_test
        assert sign_test is not None
        raw.append(sign_test.two_sided_p)
    assert raw == [0.125, 1.0]
    adjusted = [second.adjusted_p_values["success"], third.adjusted_p_values["success"]]
    assert adjusted == list(holm_adjusted(raw)) == [0.25, 1.0]
    assert root.adjusted_p_values == {}
    # The flag metric has no recorded values, so it has no test to adjust.
    assert set(second.adjusted_p_values) == {"success"}


def test_runs_without_a_single_version_are_left_out_of_the_trajectory() -> None:
    ambiguous = _run("c1", "both", None, True, extra=_version("v2", "v1"))
    unrelated = _run("c2", "other", None, True)
    unrelated.asset_versions = [
        AssetVersion(asset_id="kedi:form:critic", version="w1", content_hash="w1")
    ]
    result = _result(
        _runs("v1", None, (True, False, True, False)) + [ambiguous, unrelated],
    )

    trajectory = build_asset_trajectory(result, asset_id=ASSET, metrics=(SUCCESS,))

    assert trajectory.ambiguous_run_count == 1
    assert [step.version for step in trajectory.steps] == ["v1"]
    assert trajectory.steps[0].run_count == len(CASES)


def test_cases_with_several_runs_of_a_version_are_never_paired() -> None:
    small, large, evolved = _shared_version_runs()

    # Whatever order the variants were recorded in, no run is picked to stand for a case.
    for runs in (small + large + evolved, large + small + evolved):
        step = build_asset_trajectory(_result(runs), asset_id=ASSET, metrics=(SUCCESS,)).steps[1]
        comparison = step.comparison
        assert comparison is not None
        assert comparison.paired_count == 0
        assert comparison.duplicate_case_count == len(CASES)
        assert comparison.missing_pair_count == 0
        assert comparison.metric_results[0].sign_test is None
        assert comparison.mixed_factors == ("model",)
        assert comparison.confounded
        assert comparison.baseline_factors == {}
        assert comparison.candidate_factors == {"model": "large"}
        assert comparison.factor_deltas == {}
        assert step.adjusted_p_values == {}

    stratum = build_asset_trajectory(
        _result(small + large + evolved),
        asset_id=ASSET,
        metrics=(SUCCESS,),
        variants=("large", "large-v2"),
    )

    assert stratum.variants == ("large", "large-v2")
    assert [step.run_count for step in stratum.steps] == [len(CASES), len(CASES)]
    comparison = stratum.steps[1].comparison
    assert comparison is not None
    assert (comparison.paired_count, comparison.duplicate_case_count) == (len(CASES), 0)
    assert comparison.metric_results[0].ties == len(CASES)
    assert not comparison.confounded
    assert comparison.factor_deltas == {}


def test_a_factor_only_some_runs_declare_is_mixed() -> None:
    declared = _runs("v1", None, (True, True), cases=("c1", "c2"), factors={"model": "large"})
    undeclared = _runs("v1", None, (True, True), cases=("c3", "c4"), variant="bare")
    evolved = _runs("v2", "v1", (True,) * 4, factors={"model": "large"})

    comparison = (
        build_asset_trajectory(
            _result(declared + undeclared + evolved), asset_id=ASSET, metrics=(SUCCESS,)
        )
        .steps[1]
        .comparison
    )

    assert comparison is not None
    assert comparison.paired_count == len(CASES)
    assert comparison.mixed_factors == ("model",)
    assert comparison.confounded


def test_a_version_whose_parent_was_not_evaluated_has_no_comparison() -> None:
    trajectory = build_asset_trajectory(
        _result(_runs("v5", "v4", (True, True, False, False))),
        asset_id=ASSET,
    )

    assert [(step.version, step.parent_version) for step in trajectory.steps] == [("v5", "v4")]
    assert trajectory.steps[0].comparison is None


def test_case_tags_restrict_the_trajectory() -> None:
    result = _result(
        _runs("v1", None, (False, False, False, False))
        + _runs("v2", "v1", (True, True, True, True))
    )

    trajectory = build_asset_trajectory(
        result, asset_id=ASSET, metrics=(SUCCESS,), case_tags=("odd",)
    )

    assert trajectory.case_tags == ("odd",)
    assert [step.run_count for step in trajectory.steps] == [2, 2]
    comparison = trajectory.steps[1].comparison
    assert comparison is not None
    assert comparison.case_tags == ("odd",)
    assert comparison.metric_results[0].wins == 2


def test_a_parent_cycle_lists_each_version_once() -> None:
    result = _result(_runs("a", "b", (True,) * 4) + _runs("b", "a", (False,) * 4))

    trajectory = build_asset_trajectory(result, asset_id=ASSET, metrics=(SUCCESS,))

    assert sorted(step.version for step in trajectory.steps) == ["a", "b"]
    assert all(step.comparison is not None for step in trajectory.steps)


def test_records_of_one_benchmark_pool_into_one_trajectory() -> None:
    first = _result(
        _runs("v1", None, (False, False, True, False), variant="agent"),
        experiment_id="gen-1",
    )
    second = _result(
        _runs("v2", "v1", (True, True, True, False), variant="agent"),
        experiment_id="gen-2",
    )

    trajectory = build_asset_trajectory([first, second], asset_id=ASSET, metrics=(SUCCESS,))

    assert trajectory.experiment_ids == ("gen-1", "gen-2")
    comparison = trajectory.steps[1].comparison
    assert comparison is not None
    assert comparison.paired_count == len(CASES)
    assert comparison.metric_results[0].wins == 2
    assert summarize_tracked_assets([first, second]) == (
        TrackedAssetSummary(asset_id=ASSET, version_count=2, run_count=2 * len(CASES)),
    )


def test_pooled_records_must_share_the_benchmark_and_its_cases() -> None:
    first = _result(_runs("v1", None, (True,) * 4))
    other = _result(_runs("v2", "v1", (True,) * 4), benchmark_id="other")
    changed = _result(_runs("v2", "v1", (True,) * 4))
    changed.runs[0].case = Case(id="c1", input="changed")

    with pytest.raises(ValueError, match="at least one experiment result"):
        build_asset_trajectory([], asset_id=ASSET)
    with pytest.raises(ValueError, match="share one benchmark, got: other, trajectory"):
        build_asset_trajectory([first, other], asset_id=ASSET)
    with pytest.raises(ValueError, match="case 'c1' differs between the pooled records"):
        summarize_tracked_assets([first, changed])


def test_reports_build_and_render_declared_trajectories() -> None:
    result = _result(
        _runs("v1", None, (False, False, True, False))
        + _runs("v2", "v1", (True, True, True, False))
        + [_run("c1", "both", None, True, extra=_version("v2", "v1"))]
    )
    for run in result.runs:
        run.scores.append(
            ScoreRecord(name="flag", semantic_type="custom.flag", value=run.variant_id == "v2")
        )
    spec = ReportSpec.model_validate(
        {
            "trajectories": [
                {"asset": ASSET, "metrics": [SUCCESS.model_dump(), FLAG.model_dump()]},
                {"asset": ASSET, "case_tags": ["odd"], "metrics": [SUCCESS.model_dump()]},
                {"asset": "kedi:form:missing"},
                {"asset": ASSET, "variants": ["v2"], "metrics": [SUCCESS.model_dump()]},
            ]
        }
    )

    report = build_report(result, report_spec=spec)

    assert [trajectory.asset_id for trajectory in report.trajectories] == [
        ASSET,
        ASSET,
        "kedi:form:missing",
        ASSET,
    ]
    assert report.trajectories[2].steps == ()
    assert [step.version for step in report.trajectories[3].steps] == ["v2"]
    markdown = render_markdown_report(report)
    assert "## Asset Trajectories" in markdown
    assert "Holm p adjusts them across the steps of a trajectory" in markdown
    assert "runs using several versions at once: `1`" in markdown
    assert "case tags: `odd`" in markdown
    assert "variants: `v2`" in markdown
    assert "| `v2` | `v1` | 4 | success |" in markdown
    assert "| `v1` | - | 4 | - |" in markdown
    assert "| Holm p | difference (95% CI) |" in markdown
    assert "No selected run recorded a version of this asset." in markdown
    console = Console(record=True, width=200)
    render_report(console, report)
    text = console.export_text()
    assert f"Asset Trajectory: {ASSET} (tags: odd)" in text
    assert f"Asset Trajectory: {ASSET} (variants: v2)" in text
    assert "kedi:form:missing" not in text
    assert "0.5000" in text
    assert "Runs using several versions at once: 1" in text


def test_trajectory_renderers_explain_pooling_and_unpaired_cases() -> None:
    small, large, evolved = _shared_version_runs()
    first = _result(small + large, experiment_id="gen-1")
    second = _result(evolved, experiment_id="gen-2")
    pooled = build_asset_trajectory([first, second], asset_id=ASSET, metrics=(SUCCESS,))
    report = build_report(first).model_copy(update={"trajectories": (pooled,)})

    markdown = render_markdown_report(report)
    assert "pooled records: `gen-1, gen-2`" in markdown
    assert "- `v2`: 4 cases have several runs of one version and were not paired." in markdown
    assert "- `v2`: confounded by mixed factors `model`." in markdown

    console = Console(record=True, width=200)
    render_trajectory(console, pooled)
    text = console.export_text()
    assert "Pooled records: gen-1, gen-2" in text
    assert "v2: 4 cases have several runs of one version and were not paired" in text
    assert "v2: confounded by mixed factors model" in text

    empty = Console(record=True, width=200)
    render_trajectory(empty, build_asset_trajectory(first, asset_id="kedi:form:missing"))
    assert "No selected run recorded a version of kedi:form:missing." in empty.export_text()


def test_tracked_assets_are_listed_to_choose_a_trajectory() -> None:
    result = _result(_runs("v1", None, (True,) * 4) + _runs("v2", "v1", (True,) * 4))

    console = Console(record=True, width=120)
    render_tracked_assets(console, summarize_tracked_assets(result))
    text = console.export_text()
    assert "Recorded Assets" in text
    assert ASSET in text

    empty = Console(record=True, width=120)
    render_tracked_assets(empty, ())
    assert "No run recorded a tracked asset." in empty.export_text()


def test_trajectory_reports_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="declared twice"):
        ReportSpec.model_validate({"trajectories": [{"asset": ASSET}, {"asset": ASSET}]})

    spec = ReportSpec.model_validate(
        {
            "trajectories": [
                {"asset": ASSET},
                {"asset": ASSET, "case_tags": ["odd"]},
                {"asset": ASSET, "variants": ["base"]},
            ]
        }
    )
    assert spec.trajectories[0].resolved_metrics()
    assert spec.trajectories[1].resolved_metrics() == spec.trajectories[0].resolved_metrics()


def test_yaml_reports_declare_trajectories_and_round_trip(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        """
        compare:
          base -> next: {}
          h1:
            baseline: base
            candidate: next
            tags: [odd]
            show:
              success:
                metric: result.success
                aggregate: ratio_true
            hypothesis:
              metric: success
              family: confirmatory
        trajectories:
          - kedi:form:planner
          - asset: kedi:form:critic
            tags: [odd]
            variants: [base]
            show:
              success:
                metric: result.success
                aggregate: ratio_true
        """,
    )

    spec = load_benchmark_spec(path)

    trajectories = spec.reports.trajectories
    assert [(item.asset, item.case_tags, item.variants) for item in trajectories] == [
        (ASSET, (), ()),
        ("kedi:form:critic", ("odd",), ("base",)),
    ]
    view = benchmark_spec_to_yaml_view(spec)["benchmark"]["paired"]["report"]
    assert view["compare"] == {
        "base -> next": {},
        "h1": {
            "baseline": "base",
            "candidate": "next",
            "show": {"success": {"metric": "result.success", "aggregate": "ratio_true"}},
            "tags": ["odd"],
            "hypothesis": {"metric": "success", "family": "confirmatory"},
        },
    }
    assert view["trajectories"] == [
        {"asset": ASSET},
        {
            "asset": "kedi:form:critic",
            "show": {"success": {"metric": "result.success", "aggregate": "ratio_true"}},
            "tags": ["odd"],
            "variants": ["base"],
        },
    ]


@pytest.mark.parametrize(
    ("trajectories", "message"),
    [
        ("trajectories: kedi:form:planner", "must be a list"),
        ("trajectories: [[planner]]", "entries must be asset ids or mappings"),
    ],
)
def test_yaml_trajectories_must_be_a_list_of_assets(
    tmp_path: Path, trajectories: str, message: str
) -> None:
    path = _write_spec(tmp_path, trajectories)

    with pytest.raises(SpecValidationError, match=message):
        load_benchmark_spec(path)


def _write_spec(directory: Path, report: str) -> Path:
    path = directory / "autobench.yaml"
    body = dedent(
        """
        benchmark:
          paired:
            cases:
              - id: one
                tags: [odd]
            run:
              python: tasks:run
            variants:
              base: {}
              next: {}
            report:
        """
    )
    indented = "\n".join(f"      {line}" for line in dedent(report).strip().splitlines())
    path.write_text(f"{body}{indented}\n", encoding="utf-8")
    return path


def _shared_version_runs() -> tuple[list[RunResult], list[RunResult], list[RunResult]]:
    # Two models share the first version; only the large model evolves it.
    small = _runs("v1", None, (False,) * 4, variant="small", factors={"model": "small"})
    large = _runs("v1", None, (True,) * 4, variant="large", factors={"model": "large"})
    evolved = _runs("v2", "v1", (True,) * 4, variant="large-v2", factors={"model": "large"})
    return small, large, evolved


def _version(version: str, parent: str | None) -> AssetVersion:
    return AssetVersion(
        asset_id=ASSET,
        version=version,
        content_hash=f"hash-{version}",
        parent_version=parent,
    )


def _runs(
    version: str,
    parent: str | None,
    successes: tuple[bool, ...],
    *,
    cases: tuple[str, ...] = CASES,
    variant: str | None = None,
    factors: dict[str, str] | None = None,
) -> list[RunResult]:
    return [
        _run(case_id, version, parent, success, variant=variant, factors=factors)
        for case_id, success in zip(cases, successes, strict=True)
    ]


def _run(
    case_id: str,
    version: str,
    parent: str | None,
    success: bool,
    *,
    extra: AssetVersion | None = None,
    variant: str | None = None,
    factors: dict[str, str] | None = None,
) -> RunResult:
    tags = ["odd"] if CASES.index(case_id) % 2 == 0 else []
    variant_id = version if variant is None else variant
    return RunResult(
        run_id=f"{case_id}_{variant_id}_{version}",
        benchmark_id="trajectory",
        experiment_id="exp_trajectory",
        case_id=case_id,
        variant_id=variant_id,
        status=RunStatus.PASSED,
        evaluation_status=EvaluationStatus.PASSED,
        case=Case(id=case_id, tags=tags),
        task_result=TaskResult(output={}, status=TaskStatus.PASSED),
        scores=[
            ScoreRecord(
                name="success",
                semantic_type=Semantic.RESULT_SUCCESS,
                value=success,
                direction=Direction.MAXIMIZE,
                role=ObservationRole.OBJECTIVE,
            )
        ],
        factors=[FactorValue(name=name, value=value) for name, value in (factors or {}).items()],
        asset_versions=[
            _version(version, parent),
            *([] if extra is None else [extra]),
        ],
    )


def _result(
    runs: list[RunResult],
    *,
    experiment_id: str = "exp_trajectory",
    benchmark_id: str = "trajectory",
) -> ExperimentResult:
    variants = {run.variant_id for run in runs}
    return ExperimentResult(
        experiment_id=experiment_id,
        benchmark_id=benchmark_id,
        plan=BenchmarkPlan(
            benchmark_id=benchmark_id,
            case_ids=CASES,
            case_count=len(CASES),
            variant_count=len(variants),
            planned_run_count=len(runs),
        ),
        runs=runs,
        environment=EnvironmentMetadata(
            python_version="3.11.13", platform="test", cwd="/workspace"
        ),
    )
