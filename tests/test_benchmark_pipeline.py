from __future__ import annotations as _annotations

from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

import autobench.runtime.pipeline as pipeline_module
from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    EvaluationStatus,
    PassFailScorer,
    PythonScorer,
    RunStatus,
    Semantic,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    build_benchmark_plan,
    expand_matrix,
    load_benchmark_spec,
    run_benchmark_path,
    run_benchmark_spec,
    stable_run_id,
)
from autobench.cli import cli
from autobench.data.variants import FactorValue


async def test_run_benchmark_spec_executes_full_matrix_in_deterministic_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "matrix_tasks.py",
        """
        def run(ctx, case):
            return {
                "case": case.id,
                "model": ctx.factor("model"),
            }
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _matrix_spec("matrix_tasks:run")

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.total_count == 4
    assert result.passed_count == 4
    assert result.failed_count == 0
    assert [run.case_id for run in result.runs] == [
        "case_1",
        "case_1",
        "case_2",
        "case_2",
    ]
    assert [run.variant_id for run in result.runs] == [
        "variant_1",
        "variant_2",
        "variant_1",
        "variant_2",
    ]
    assert [run.run_id for run in result.runs] == [
        "run_0001_0001_case_1__variant_1",
        "run_0001_0002_case_1__variant_2",
        "run_0002_0001_case_2__variant_1",
        "run_0002_0002_case_2__variant_2",
    ]
    assert result.plan.planned_run_count == build_benchmark_plan(spec).planned_run_count


async def test_pipeline_failure_isolation_keeps_later_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "failure_tasks.py",
        """
        def run(ctx, case):
            if case.id == "case_1":
                raise RuntimeError("case failed")
            return {"ok": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _matrix_spec("failure_tasks:run")

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.total_count == 4
    assert result.failed_count == 2
    assert result.passed_count == 2
    assert [run.status for run in result.runs] == [
        RunStatus.FAILED,
        RunStatus.FAILED,
        RunStatus.PASSED,
        RunStatus.PASSED,
    ]
    assert all(run.evaluation_status is EvaluationStatus.NOT_EVALUATED for run in result.runs[:2])
    assert all(run.evaluation_status is EvaluationStatus.PASSED for run in result.runs[2:])


async def test_async_task_matrix_with_concurrency_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "async_matrix_tasks.py",
        """
        async def run(ctx, case):
            return {"case": case.id, "variant": ctx.variant.id}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = _matrix_spec("async_matrix_tasks:run")

    result = await run_benchmark_spec(
        spec,
        experiment_id="exp_fixed",
        concurrency_limit=2,
    )

    assert result.total_count == 4
    assert result.passed_count == 4
    assert [run.run_id for run in result.runs] == [
        "run_0001_0001_case_1__variant_1",
        "run_0001_0002_case_1__variant_2",
        "run_0002_0001_case_2__variant_1",
        "run_0002_0002_case_2__variant_2",
    ]


def test_expand_matrix_matches_plan() -> None:
    spec = _matrix_spec("app.tasks:run")
    matrix = expand_matrix(spec, experiment_id="exp_fixed")
    plan = build_benchmark_plan(spec)

    assert len(matrix) == plan.planned_run_count


async def test_pipeline_marks_runs_without_task_as_skipped() -> None:
    spec = BenchmarkSpec.model_construct(
        benchmark=BenchmarkInfo(id="skip-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=None,
        variants=[Variant(id="variant_1")],
        scoring=[],
        derive=[],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.skipped_count == 1
    assert result.runs[0].status is RunStatus.SKIPPED
    assert result.runs[0].evaluation_status is EvaluationStatus.SKIPPED
    assert result.runs[0].task_result.status is TaskStatus.SKIPPED


async def test_pipeline_marks_unsupported_task_kind_as_errored() -> None:
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="unsupported-task-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="shell", target="run.sh"),
        variants=[Variant(id="variant_1")],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.errored_count == 1
    assert result.runs[0].evaluation_status is EvaluationStatus.NOT_EVALUATED
    assert result.runs[0].error is not None
    assert result.runs[0].error.error_type == "UnsupportedTaskKind"


async def test_pipeline_wraps_task_runner_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def crash_runner(*args: object, **kwargs: object) -> TaskResult:
        raise RuntimeError("runner crashed")

    monkeypatch.setattr(pipeline_module, "run_python_task", crash_runner)
    spec = _matrix_spec("unused:run")

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.errored_count == 4
    assert all(run.evaluation_status is EvaluationStatus.NOT_EVALUATED for run in result.runs)
    assert all(run.error is not None for run in result.runs)
    assert {run.error.error_type for run in result.runs if run.error is not None} == {
        "RuntimeError"
    }


async def test_pipeline_marks_constraint_failures_as_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "constraint_tasks.py",
        """
        def run(ctx, case):
            ctx.check("tool_call", passed=False)
            return {"success": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="constraint-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="constraint_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            PassFailScorer(
                name="success",
                path="output.success",
                semantic_type=Semantic.RESULT_SUCCESS,
                optional=True,
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.failed_count == 1
    assert result.runs[0].status is RunStatus.FAILED
    assert result.runs[0].evaluation_status is EvaluationStatus.FAILED


def test_run_benchmark_path_resolves_spec_relative_task_and_scorer_modules(tmp_path: Path) -> None:
    package_name = "local_import_pkg"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    _write_module(package_dir, "__init__.py", "")
    _write_module(
        package_dir,
        "benchmark_helpers.py",
        """
        from autobench import Direction, ObservationRole, ScoreRecord, Semantic

        def run(ctx, case):
            return {"answer": case.expected["answer"]}

        def score(call):
            return ScoreRecord(
                name="quality",
                semantic_type=Semantic.QUALITY_SCORE,
                value=1.0 if call.output["answer"] == call.case.expected["answer"] else 0.0,
                direction=Direction.MAXIMIZE,
                role=ObservationRole.OBJECTIVE,
            )
        """,
    )
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        (
            dedent(
                """
            benchmark:
              id: local-imports
            dataset:
              cases:
                - id: case_1
                  expected:
                    answer: ok
            task:
              kind: python
              target: {package_name}.benchmark_helpers:run
            variants:
              - id: variant_1
            scoring:
              - kind: exact
                name: exact_answer
                actual: output.answer
                expected: case.expected.answer
                semantic_type: quality.correctness
              - kind: python
                name: quality
                target: {package_name}.benchmark_helpers:score
                semantic_type: quality.score
            """
            )
            .strip()
            .format(package_name=package_name)
            + "\n"
        ),
        encoding="utf-8",
    )

    result = run_benchmark_path(spec_path, experiment_id="exp_local_imports")
    loaded_spec = load_benchmark_spec(spec_path)

    assert result.passed_count == 1
    assert result.runs[0].scores[0].value == 1.0
    assert result.runs[0].scores[1].value == 1.0
    assert loaded_spec.task is not None
    assert str(tmp_path.resolve()) in loaded_spec.task.module_search_paths
    python_scorer = next(
        scorer for scorer in loaded_spec.scoring if isinstance(scorer, PythonScorer)
    )
    assert str(tmp_path.resolve()) in python_scorer.module_search_paths


def test_run_status_helper_covers_skipped_evaluation_branch() -> None:
    task_result = TaskResult(status=TaskStatus.PASSED)

    status = pipeline_module._run_status_from_task_result(
        task_result,
        evaluation_status=EvaluationStatus.SKIPPED,
    )

    assert status is RunStatus.SKIPPED


def test_stable_run_id_falls_back_to_unnamed_slug() -> None:
    assert (
        stable_run_id(
            case=Case(id="!!!"),
            variant=Variant(id="???"),
            case_index=0,
            variant_index=0,
        )
        == "run_0001_0001_unnamed__unnamed"
    )


def test_cli_run_executes_spec_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "cli_tasks.py",
        """
        def run(ctx, case):
            return {"case": case.id}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: cli-demo
            dataset:
              cases:
                - id: case_1
                - id: case_2
            task:
              kind: python
              target: cli_tasks:run
            variants:
              - id: variant_1
                factors: []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["run", str(spec_path)])

    assert result.exit_code == 0
    assert "Benchmark Run Complete" in result.output
    assert "cli-demo" in result.output
    assert "Planned runs" in result.output
    assert "Runs" in result.output
    assert "Passed" in result.output
    assert "Recorded to" in result.output
    assert next((tmp_path / ".autobench").rglob("experiment.yaml")).exists()


def test_cli_run_can_skip_default_recording(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_module(
        tmp_path,
        "cli_no_record_tasks.py",
        """
        def run(ctx, case):
            return {"case": case.id}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: cli-no-record
            dataset:
              cases:
                - id: case_1
            task:
              kind: python
              target: cli_no_record_tasks:run
            variants:
              - id: variant_1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["run", str(spec_path), "--no-record"])

    assert result.exit_code == 0
    assert "recorded:" not in result.output
    assert not (tmp_path / ".autobench").exists()


def _matrix_spec(target: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="matrix-demo"),
        dataset=DatasetSpec(
            cases=[
                Case(id="case_1", input={"message": "one"}),
                Case(id="case_2", input={"message": "two"}),
            ]
        ),
        task=TaskSpec(kind="python", target=target),
        variants=[
            Variant(
                id="variant_1",
                factors=[FactorValue(name="model", value="model-a")],
            ),
            Variant(
                id="variant_2",
                factors=[FactorValue(name="model", value="model-b")],
            ),
        ],
    )


def _write_module(tmp_path: Path, filename: str, source: str) -> None:
    (tmp_path / filename).write_text(dedent(source).strip() + "\n", encoding="utf-8")
