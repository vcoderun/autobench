from __future__ import annotations as _annotations

from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from rich.console import Console

from autobench import (
    AssetVersion,
    BenchmarkPlan,
    Case,
    ComparisonReport,
    ComparisonReportSpec,
    Direction,
    EvaluationStatus,
    ExperimentResult,
    ObservationRole,
    ReportSpec,
    RunResult,
    RunStatus,
    ScoreRecord,
    Semantic,
    TaskResult,
    TaskStatus,
    benchmark_spec_to_yaml_view,
    build_report,
    compare_asset_versions,
)
from autobench.cli import cli
from autobench.data.variants import FactorValue
from autobench.records.storage import EnvironmentMetadata
from autobench.reports.markdown import render_markdown_report
from autobench.reports.models import MetricAggregation
from autobench.reports.rich import render_comparison
from autobench.spec import load_benchmark_spec

ASSET = "kedi:form:planner"
SUCCESS = MetricAggregation(name="success", semantic_type=Semantic.RESULT_SUCCESS, fn="ratio_true")
CASES = ("c1", "c2", "c3", "c4")


def test_root_and_leaf_select_the_ends_of_a_lineage() -> None:
    result = _result(
        _runs("v2", "v1", (True, True, False, False))
        + _runs("v1", None, (False, False, False, False))
        + _runs("v3", "v2", (True, True, True, True))
    )

    evolved = compare_asset_versions(
        result, asset_id=ASSET, baseline="root", candidate="leaf", metrics=(SUCCESS,)
    )
    explicit = compare_asset_versions(
        result, asset_id=ASSET, baseline="v2", candidate="v3", metrics=(SUCCESS,)
    )

    assert (evolved.asset_id, evolved.baseline, evolved.candidate) == (ASSET, "v1", "v3")
    assert evolved.diagnostics == ()
    assert evolved.metric_results[0].wins == len(CASES)
    assert (explicit.baseline, explicit.candidate) == ("v2", "v3")
    assert explicit.metric_results[0].wins == 2


def test_unresolved_versions_leave_the_comparison_without_runs() -> None:
    branched = _result(
        _runs("v1", None, (False,) * 4)
        + _runs("a", "v1", (True,) * 4)
        + _runs("b", "v1", (True,) * 4)
    )
    cycle = _result(_runs("a", "b", (True,) * 4) + _runs("b", "a", (False,) * 4))
    single = _result(_runs("v1", None, (True,) * 4))

    two_leaves = compare_asset_versions(
        branched, asset_id=ASSET, baseline="root", candidate="leaf", metrics=(SUCCESS,)
    )
    no_root = compare_asset_versions(cycle, asset_id=ASSET, baseline="root", candidate="b")
    missing = compare_asset_versions(branched, asset_id=ASSET, baseline="v9", candidate="a")
    same = compare_asset_versions(single, asset_id=ASSET, baseline="root", candidate="leaf")

    assert (two_leaves.baseline, two_leaves.candidate) == ("v1", "leaf")
    assert two_leaves.diagnostics == (
        f"asset {ASSET!r} has 2 leaf versions among the selected runs: a, b",
    )
    assert two_leaves.paired_count == 0
    assert two_leaves.metric_results[0].sign_test is None
    assert no_root.diagnostics == (f"asset {ASSET!r} has no root version among the selected runs",)
    assert missing.diagnostics == (f"version 'v9' of asset {ASSET!r} has no selected runs",)
    assert same.diagnostics == (
        f"baseline and candidate both resolve to version 'v1' of asset {ASSET!r}",
    )
    assert (same.baseline, same.candidate, same.paired_count) == ("root", "leaf", 0)


def test_variants_select_the_runs_of_an_asset_comparison() -> None:
    result = _result(
        _runs("v1", None, (False,) * 4, variant="small", model="small")
        + _runs("v1", None, (True,) * 4, variant="large", model="large")
        + _runs("v2", "v1", (True,) * 4, variant="large-v2", model="large")
    )

    def compare(variants: tuple[str, ...]) -> ComparisonReport:
        return compare_asset_versions(
            result,
            asset_id=ASSET,
            baseline="root",
            candidate="leaf",
            metrics=(SUCCESS,),
            variants=variants,
        )

    pooled = compare(())
    stratum = compare(("large", "large-v2"))
    crossed = compare(("small", "large-v2"))

    assert pooled.duplicate_case_count == len(CASES)
    assert pooled.mixed_factors == ("model",)
    assert pooled.confounded
    assert (stratum.duplicate_case_count, stratum.paired_count) == (0, len(CASES))
    assert stratum.metric_results[0].ties == len(CASES)
    assert stratum.factor_deltas == {}
    assert not stratum.confounded
    # Pairing the small model's first version with the large model's second one changes the
    # model too, and the report says so.
    assert crossed.metric_results[0].wins == len(CASES)
    assert crossed.factor_deltas == {"model": {"baseline": "small", "candidate": "large"}}


def test_asset_comparisons_join_the_declared_hypothesis_families() -> None:
    result = _result(
        _runs("v1", None, (False,) * 4)
        + _runs("v2", "v1", (True, True, True, False))
        + _runs("v3", "v2", (True,) * 4)
    )
    hypothesis = {"metric": "success", "family": "confirmatory"}
    spec = ReportSpec.model_validate(
        {
            "comparisons": [
                {
                    "name": "evolved",
                    "asset": ASSET,
                    "baseline": "root",
                    "candidate": "leaf",
                    "metrics": [SUCCESS.model_dump()],
                    "hypothesis": hypothesis,
                },
                {
                    "asset": "kedi:form:missing",
                    "baseline": "root",
                    "candidate": "leaf",
                    "metrics": [SUCCESS.model_dump()],
                    "hypothesis": hypothesis,
                },
            ]
        }
    )

    report = build_report(result, report_spec=spec)

    evolved, missing = report.comparisons
    assert (evolved.name, evolved.baseline, evolved.candidate) == ("evolved", "v1", "v3")
    assert missing.name == "root -> leaf"
    assert missing.diagnostics == (
        "asset 'kedi:form:missing' has no root version among the selected runs",
        "asset 'kedi:form:missing' has no leaf version among the selected runs",
    )
    assert [
        (item.comparison, item.status, item.p_value, item.adjusted_p_value, item.family_size)
        for item in report.hypotheses
    ] == [
        ("evolved", "not_supported", 0.0625, 0.125, 2),
        ("root -> leaf", "untestable", None, None, 2),
    ]

    markdown = render_markdown_report(report)
    assert "### evolved: v1 vs v3" in markdown
    assert "asset: `kedi:form:planner`" in markdown
    assert "note: asset 'kedi:form:missing' has no root version among the selected runs" in markdown

    console = Console(record=True, width=200)
    render_comparison(console, missing)
    text = console.export_text()
    assert "Asset Comparison" in text
    assert "kedi:form:missing" in text
    assert "has no leaf version among the selected runs" in text


def test_renderers_flag_cases_with_several_runs_and_mixed_factors() -> None:
    result = _result(
        _runs("v1", None, (False,) * 4, variant="small", model="small")
        + _runs("v1", None, (True,) * 4, variant="large", model="large")
        + _runs("v2", "v1", (True,) * 4, variant="large-v2", model="large")
    )
    spec = ReportSpec.model_validate(
        {"comparisons": [{"asset": ASSET, "baseline": "root", "candidate": "leaf"}]}
    )

    report = build_report(result, report_spec=spec)

    markdown = render_markdown_report(report)
    assert "cases with several runs: `4`" in markdown
    assert "mixed factors: `model`" in markdown
    console = Console(record=True, width=200)
    render_comparison(console, report.comparisons[0])
    text = console.export_text()
    assert "Cases with several runs" in text
    assert "Mixed factors" in text


def test_variants_filter_requires_an_asset_comparison() -> None:
    with pytest.raises(ValidationError, match="apply only to asset version comparisons"):
        ComparisonReportSpec(baseline="base", candidate="next", variants=("base",))


def test_yaml_asset_comparisons_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "autobench.yaml"
    path.write_text(
        dedent(
            """
            benchmark:
              paired:
                cases:
                  - id: one
                run:
                  python: tasks:run
                variants:
                  base: {}
                  next: {}
                report:
                  compare:
                    root -> leaf:
                      asset: kedi:form:critic
                    evolved:
                      asset: kedi:form:planner
                      baseline: root
                      candidate: leaf
                      variants: [base, next]
                      show:
                        success:
                          metric: result.success
                          aggregate: ratio_true
                      hypothesis:
                        metric: success
                        family: confirmatory
            """
        ),
        encoding="utf-8",
    )

    spec = load_benchmark_spec(path)

    critic, evolved = spec.reports.comparisons
    assert (critic.asset, critic.baseline, critic.candidate, critic.name) == (
        "kedi:form:critic",
        "root",
        "leaf",
        None,
    )
    assert (evolved.asset, evolved.variants, evolved.label()) == (
        ASSET,
        ("base", "next"),
        "evolved",
    )
    view = benchmark_spec_to_yaml_view(spec)["benchmark"]["paired"]["report"]
    assert view["compare"] == {
        "root -> leaf": {"asset": "kedi:form:critic"},
        "evolved": {
            "baseline": "root",
            "candidate": "leaf",
            "asset": ASSET,
            "variants": ["base", "next"],
            "show": {"success": {"metric": "result.success", "aggregate": "ratio_true"}},
            "hypothesis": {"metric": "success", "family": "confirmatory"},
        },
    }


def test_cli_lists_assets_and_renders_trajectories_and_asset_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lineage = _result(_runs("v1", None, (False,) * 4) + _runs("v2", "v1", (True,) * 4))
    first = _result(_runs("v1", None, (False,) * 4, variant="agent"), experiment_id="gen-1")
    second = _result(_runs("v2", "v1", (True,) * 4, variant="agent"), experiment_id="gen-2")
    other = _result(_runs("v1", None, (True,) * 4), benchmark_id="other")
    records = {
        tmp_path / name: result
        for name, result in (
            ("lineage", lineage),
            ("gen-1", first),
            ("gen-2", second),
            ("other", other),
        )
    }
    for record_dir in records:
        record_dir.mkdir()
    monkeypatch.setattr("autobench.cli.replay_experiment", lambda run_dir: records[run_dir])
    runner = CliRunner()

    listed = runner.invoke(
        cli, ["trajectory", str(tmp_path / "gen-1"), str(tmp_path / "gen-2")], terminal_width=200
    )
    assert listed.exit_code == 0, listed.output
    assert "Recorded Assets" in listed.output
    assert ASSET in listed.output

    rendered = runner.invoke(
        cli,
        [
            "trajectory",
            str(tmp_path / "gen-1"),
            str(tmp_path / "gen-2"),
            "--asset",
            ASSET,
            "--variant",
            "agent",
            "--tag",
            "odd",
            "--metric",
            "result.success:ratio_true",
        ],
        terminal_width=200,
    )
    assert rendered.exit_code == 0, rendered.output
    assert f"Asset Trajectory: {ASSET} (tags: odd; variants: agent)" in rendered.output
    assert "Pooled records: gen-1, gen-2" in rendered.output

    rejected = runner.invoke(
        cli, ["trajectory", str(tmp_path / "gen-1"), str(tmp_path / "other"), "--asset", ASSET]
    )
    assert rejected.exit_code != 0
    assert "share one benchmark" in rejected.output

    compared = runner.invoke(
        cli,
        [
            "compare",
            str(tmp_path / "lineage"),
            "--asset",
            ASSET,
            "--baseline",
            "root",
            "--candidate",
            "leaf",
            "--variant",
            "v1",
            "--variant",
            "v2",
        ],
        terminal_width=200,
    )
    assert compared.exit_code == 0, compared.output
    assert "Asset Comparison" in compared.output
    assert "4/0/0" in compared.output

    misused = runner.invoke(
        cli,
        [
            "compare",
            str(tmp_path / "lineage"),
            "--baseline",
            "v1",
            "--candidate",
            "v2",
            "--variant",
            "v1",
        ],
    )
    assert misused.exit_code != 0
    assert "--variant requires --asset." in misused.output


def _runs(
    version: str,
    parent: str | None,
    successes: tuple[bool, ...],
    *,
    variant: str | None = None,
    model: str | None = None,
) -> list[RunResult]:
    variant_id = version if variant is None else variant
    return [
        RunResult(
            run_id=f"{case_id}_{variant_id}_{version}",
            benchmark_id="assets",
            experiment_id="exp_assets",
            case_id=case_id,
            variant_id=variant_id,
            status=RunStatus.PASSED,
            evaluation_status=EvaluationStatus.PASSED,
            case=Case(id=case_id, tags=["odd"] if index % 2 == 0 else []),
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
            factors=[] if model is None else [FactorValue(name="model", value=model)],
            asset_versions=[
                AssetVersion(
                    asset_id=ASSET,
                    version=version,
                    content_hash=f"hash-{version}",
                    parent_version=parent,
                )
            ],
        )
        for index, (case_id, success) in enumerate(zip(CASES, successes, strict=True))
    ]


def _result(
    runs: list[RunResult],
    *,
    experiment_id: str = "exp_assets",
    benchmark_id: str = "assets",
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        benchmark_id=benchmark_id,
        plan=BenchmarkPlan(
            benchmark_id=benchmark_id,
            case_ids=CASES,
            case_count=len(CASES),
            variant_count=len({run.variant_id for run in runs}),
            planned_run_count=len(runs),
        ),
        runs=runs,
        environment=EnvironmentMetadata(
            python_version="3.11.13", platform="test", cwd="/workspace"
        ),
    )
