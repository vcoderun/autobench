from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from rich.console import Console

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    Benchmark,
    BenchmarkInfo,
    BenchmarkPlan,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    EnvironmentMetadata,
    ExecutionCorrelation,
    ExecutionSpec,
    ExperimentStart,
    FactorValue,
    FileRecorder,
    MatrixRunSpec,
    TaskSpec,
    Variant,
    benchmark_spec_to_yaml_view,
    build_grouped_reports,
    build_report,
    correlation_matches,
    export_runs_csv,
    filter_experiments,
    load_benchmark_spec,
    load_experiment_record,
    load_run_record,
    merge_execution_correlation,
    render_markdown_report,
    replay_experiment,
    run_benchmark_spec,
)
from autobench.cli import cli
from autobench.io import benchmark_schema, dump_yaml, load_yaml, record_schema, staging_schema
from autobench.records.staging import (
    PartialRunSnapshot,
    partial_snapshot_from_yaml_view,
    partial_snapshot_to_yaml_view,
)
from autobench.records.views import (
    experiment_record_payload_from_yaml_view,
    run_record_payload_from_yaml_view,
)
from autobench.reports.exporting import report_to_yaml_view
from autobench.reports.rich import render_experiment_result, render_report
from autobench.runtime.lifecycle import RunPhase


def test_execution_correlation_validates_and_merges_explicit_fields() -> None:
    base = ExecutionCorrelation(
        group_id="proposal-17",
        attempt=1,
        phase="train",
        parent_experiment_id="exp-parent",
        labels={"owner": "platform", "priority": 2},
    )
    override = ExecutionCorrelation(attempt=2, labels={"owner": "evaluation"})

    merged = merge_execution_correlation(base, override)

    assert merged == ExecutionCorrelation(
        group_id="proposal-17",
        attempt=2,
        phase="train",
        parent_experiment_id="exp-parent",
        labels={"owner": "evaluation", "priority": 2},
    )
    assert merge_execution_correlation(base, None) == base
    assert merge_execution_correlation(None, ExecutionCorrelation()) is None
    assert merge_execution_correlation(
        base,
        ExecutionCorrelation(group_id=None),
    ) == base.model_copy(update={"group_id": None})

    for payload, message in (
        ({"group_id": " "}, "must not be blank"),
        ({"attempt": 0}, "greater than or equal to 1"),
        ({"labels": {" ": "value"}}, "names must not be blank"),
        ({"labels": {"loss": math.inf}}, "must be finite"),
    ):
        with pytest.raises(ValidationError, match=message):
            ExecutionCorrelation.model_validate(payload)

    planned_run = MatrixRunSpec(
        run_id="run-1",
        benchmark_id="benchmark",
        experiment_id="experiment",
        case_index=0,
        variant_index=0,
        case=Case(id="one"),
        variant=Variant(id="baseline"),
        correlation=base,
    )
    with pytest.raises(ValidationError, match="planned run correlation"):
        ExperimentStart(
            experiment_id="experiment",
            benchmark_id="benchmark",
            plan=BenchmarkPlan(
                benchmark_id="benchmark",
                case_count=1,
                variant_count=1,
                planned_run_count=1,
            ),
            runs=(planned_run,),
            environment=EnvironmentMetadata(python_version="3.11", platform="test", cwd="."),
            semantic_registry=DEFAULT_SEMANTIC_REGISTRY,
            correlation=ExecutionCorrelation(group_id="different"),
        )


async def test_python_yaml_builder_and_durable_replay_share_one_correlation_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_path = tmp_path / "correlation_task.py"
    task_path.write_text(
        "def run(ctx, case):\n    return {'case': case.id, 'attempt': ctx.factor('value')}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yaml_correlation = ExecutionCorrelation(
        group_id="candidate-a",
        attempt=1,
        phase="validation",
        labels={"owner": "autobench", "seed": 7},
    )
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="correlation"),
        execution=ExecutionSpec(correlation=yaml_correlation),
        dataset=DatasetSpec(id="cases", cases=[Case(id="one")]),
        variants=[Variant(id="baseline", factors=[FactorValue(name="value", value=3)])],
        task=TaskSpec(kind="python", target="correlation_task:run"),
    )
    spec_path = tmp_path / "autobench.yaml"
    dump_yaml(benchmark_spec_to_yaml_view(spec), spec_path, schema_name="benchmark")

    loaded = load_benchmark_spec(spec_path)
    assert loaded.execution.correlation == yaml_correlation
    assert load_yaml(spec_path)["benchmark"]["correlation"]["execution"]["correlation"] == {
        "group_id": "candidate-a",
        "attempt": 1,
        "phase": "validation",
        "labels": {"owner": "autobench", "seed": 7},
    }

    override = ExecutionCorrelation(attempt=2, labels={"owner": "cli-or-python"})
    record_path = tmp_path / "record"
    result = await run_benchmark_spec(
        loaded,
        experiment_id="exp-correlation",
        correlation=override,
        recorder=FileRecorder(record_path),
    )
    expected = ExecutionCorrelation(
        group_id="candidate-a",
        attempt=2,
        phase="validation",
        labels={"owner": "cli-or-python", "seed": 7},
    )
    assert result.correlation == expected
    assert result.runs[0].correlation == expected

    experiment_record = load_experiment_record(record_path)
    run_record = load_run_record(
        record_path / experiment_record.run_paths[0],
        root_dir=record_path,
    )
    replayed = replay_experiment(record_path)
    assert experiment_record.correlation == expected
    assert run_record.correlation == expected
    assert replayed.correlation == expected
    assert replayed.runs[0].correlation == expected
    assert run_record.parent_run_id is None
    assert run_record.lineage is None

    built = await (
        Benchmark("builder-correlation")
        .correlation(yaml_correlation)
        .dataset([Case(id="one")])
        .variants([Variant(id="baseline", factors=[FactorValue(name="value", value=3)])])
        .task("correlation_task:run")
        .run_async(
            experiment_id="exp-builder-correlation",
            correlation=ExecutionCorrelation(attempt=3),
        )
    )
    assert built.correlation == yaml_correlation.model_copy(update={"attempt": 3})
    assert built.runs[0].correlation == built.correlation


def test_cli_overrides_only_supplied_correlation_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "cli_task.py").write_text(
        "def run(ctx, case):\n    return {'ok': True}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="cli-correlation"),
        execution=ExecutionSpec(
            correlation=ExecutionCorrelation(
                group_id="yaml-group",
                attempt=1,
                parent_experiment_id="yaml-parent",
                labels={"owner": "yaml", "retained": True},
            )
        ),
        dataset=DatasetSpec(cases=[Case(id="one")]),
        variants=[Variant(id="baseline")],
        task=TaskSpec(kind="python", target="cli_task:run"),
    )
    spec_path = tmp_path / "autobench.yaml"
    record_path = tmp_path / "record"
    dump_yaml(benchmark_spec_to_yaml_view(spec), spec_path, schema_name="benchmark")

    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(spec_path),
            "--record",
            str(record_path),
            "--attempt",
            "4",
            "--phase",
            "holdout",
            "--correlation-label",
            "owner",
            "cli",
        ],
    )

    assert result.exit_code == 0, result.output
    replayed = replay_experiment(record_path)
    assert replayed.correlation == ExecutionCorrelation(
        group_id="yaml-group",
        attempt=4,
        phase="holdout",
        parent_experiment_id="yaml-parent",
        labels={"owner": "cli", "retained": True},
    )
    invalid = CliRunner().invoke(cli, ["run", str(spec_path), "--attempt", "0"])
    assert invalid.exit_code == 2
    assert "not in the range" in invalid.output

    full_override_path = tmp_path / "full-override-record"
    full_override = CliRunner().invoke(
        cli,
        [
            "run",
            str(spec_path),
            "--record",
            str(full_override_path),
            "--group-id",
            "cli-group",
            "--parent-experiment-id",
            "cli-parent",
            "--resumed-from-experiment-id",
            "cli-resumed",
        ],
    )
    assert full_override.exit_code == 0, full_override.output
    assert replay_experiment(full_override_path).correlation == ExecutionCorrelation(
        group_id="cli-group",
        attempt=1,
        parent_experiment_id="cli-parent",
        resumed_from_experiment_id="cli-resumed",
        labels={"owner": "yaml", "retained": True},
    )


async def test_checkpoint_records_exports_and_grouped_reports_preserve_correlation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "group_task.py").write_text(
        "def run(ctx, case):\n    return {'ok': True}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="grouped"),
        dataset=DatasetSpec(cases=[Case(id="one")]),
        variants=[Variant(id="baseline")],
        task=TaskSpec(kind="python", target="group_task:run"),
    )
    first = await run_benchmark_spec(
        spec,
        experiment_id="exp-1",
        correlation=ExecutionCorrelation(
            group_id="proposal",
            attempt=1,
            phase="train",
            labels={"owner": "search"},
        ),
    )
    second = await run_benchmark_spec(
        spec,
        experiment_id="exp-2",
        correlation=ExecutionCorrelation(
            group_id="proposal",
            attempt=2,
            phase="validation",
            labels={"owner": "search"},
        ),
    )
    unrelated = await run_benchmark_spec(spec, experiment_id="exp-3")

    selected = filter_experiments(
        [first, second, unrelated],
        correlation=ExecutionCorrelation(phase="validation", labels={"owner": "search"}),
    )
    assert [result.experiment_id for result in selected] == ["exp-2"]
    assert correlation_matches(None, ExecutionCorrelation()) is True
    assert correlation_matches(None, ExecutionCorrelation(group_id="proposal")) is False
    assert correlation_matches(first.correlation, ExecutionCorrelation(attempt=2)) is False
    assert (
        correlation_matches(first.correlation, ExecutionCorrelation(group_id="different")) is False
    )
    assert (
        correlation_matches(
            first.correlation,
            ExecutionCorrelation(parent_experiment_id="different"),
        )
        is False
    )
    assert (
        correlation_matches(
            first.correlation,
            ExecutionCorrelation(resumed_from_experiment_id="different"),
        )
        is False
    )
    assert (
        correlation_matches(
            first.correlation,
            ExecutionCorrelation(labels={"owner": "different"}),
        )
        is False
    )

    groups = build_grouped_reports([first, second, unrelated])
    assert [(group.group_id, group.attempts, group.phases) for group in groups] == [
        ("proposal", (1, 2), ("train", "validation")),
        (None, (), ()),
    ]
    filtered_groups = build_grouped_reports(
        [first, second, unrelated],
        correlation=ExecutionCorrelation(attempt=2),
    )
    assert [report.experiment_id for report in filtered_groups[0].reports] == ["exp-2"]

    report = build_report(second)
    report_view = report_to_yaml_view(report)
    assert report_view["report"]["correlation"]["attempt"] == 2
    audit_report = report.model_copy(
        update={"markdown": report.markdown.model_copy(update={"profile": "audit"})}
    )
    audit_markdown = render_markdown_report(audit_report)
    assert "## Technical Evidence" in audit_markdown
    assert "| Group id | proposal |" in audit_markdown
    csv_output = export_runs_csv(second)
    assert "correlation_group_id,correlation_attempt,correlation_phase" in csv_output
    assert "proposal,2,validation" in csv_output

    console = Console(record=True, width=120)
    render_experiment_result(console, second, title="Correlation")
    render_report(console, report)
    rendered = console.export_text()
    assert "Correlation group" in rendered
    assert "proposal" in rendered
    assert "Execution phase" in rendered

    parent_only = ExecutionCorrelation(parent_experiment_id="parent-only")
    parent_only_result = type(second).model_validate(
        second.model_dump(mode="python")
        | {
            "correlation": parent_only,
            "runs": [
                run.model_dump(mode="python") | {"correlation": parent_only} for run in second.runs
            ],
        }
    )
    parent_console = Console(record=True, width=120)
    render_experiment_result(parent_console, parent_only_result, title="Parent correlation")
    assert "parent-only" in parent_console.export_text()

    with pytest.raises(ValidationError, match="run correlation must match"):
        type(second).model_validate(
            second.model_dump(mode="python")
            | {"correlation": ExecutionCorrelation(group_id="different")}
        )

    checkpoint = PartialRunSnapshot(
        run_id="run-1",
        experiment_id="exp-2",
        benchmark_id="grouped",
        case_id="one",
        variant_id="baseline",
        name="state",
        phase=RunPhase.EXECUTING,
        correlation=second.correlation,
    )
    checkpoint_view = partial_snapshot_to_yaml_view(checkpoint)
    assert partial_snapshot_from_yaml_view(checkpoint_view).correlation == second.correlation


def test_record_loaders_default_old_records_to_no_execution_correlation(
    tmp_path: Path,
) -> None:
    old_run = {
        "record": {"type": "run", "version": 5},
        "run": {
            "id": "run-1",
            "experiment": "exp-old",
            "benchmark": "old",
            "case": "one",
            "variant": "baseline",
            "status": "passed",
            "outcome": {"evaluation": "passed", "task": "passed"},
        },
        "case": {"id": "one"},
        "variant": {"id": "baseline"},
    }
    run_payload = run_record_payload_from_yaml_view(old_run)
    assert "correlation" not in run_payload
    run_path = tmp_path / "run.yaml"
    dump_yaml(old_run, run_path, schema_name="run_record")
    assert load_run_record(run_path).correlation is None

    old_experiment = {
        "record": {"type": "experiment", "version": 5},
        "experiment": {
            "id": "exp-old",
            "benchmark": "old",
            "termination": {
                "status": "completed",
                "partial": False,
                "post_processing": {"cross_run_derivation": True, "policies": True},
            },
        },
        "benchmark": {
            "id": "old",
            "cases": ["one"],
            "counts": {"cases": 1, "variants": 1, "runs": 1},
        },
        "runs": {"count": 1, "passed": 1, "paths": ["run.yaml"]},
        "environment": {"python": "3.11", "platform": "test", "cwd": "."},
    }
    experiment_payload = experiment_record_payload_from_yaml_view(old_experiment)
    assert "correlation" not in experiment_payload
    experiment_path = tmp_path / "experiment.yaml"
    dump_yaml(old_experiment, experiment_path, schema_name="experiment")
    assert load_experiment_record(tmp_path).correlation is None

    benchmark_properties = benchmark_schema()["properties"]["benchmark"]["additionalProperties"][
        "properties"
    ]
    assert "execution" in benchmark_properties
    assert "correlation" in record_schema("experiment")["properties"]["experiment"]["properties"]
    assert "correlation" in record_schema("run_record")["properties"]["run"]["properties"]
    assert "correlation" in staging_schema("checkpoint")["properties"]["run"]["properties"]


@pytest.fixture(autouse=True)
def clear_dynamic_modules() -> Iterator[None]:
    yield
    for module_name in ("correlation_task", "cli_task", "group_task"):
        sys.modules.pop(module_name, None)
