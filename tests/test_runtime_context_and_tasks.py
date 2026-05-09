from __future__ import annotations as _annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

import autobench.runtime.tasks as tasks_module
from autobench import (
    Case,
    Direction,
    DurationMetricSpec,
    ErrorRecord,
    Measurement,
    ObservationKind,
    ObservationRole,
    RunContext,
    Semantic,
    TaskResolutionError,
    TaskStatus,
    Variant,
    resolve_python_callable,
    run_python_task,
)
from autobench.data.variants import FactorValue


async def test_sync_task_returns_output_and_can_read_case_and_factor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "demo_tasks.py",
        """
        def run(ctx, case):
            return {
                "message": case.input["message"],
                "model": ctx.factor("model"),
            }
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    case = Case(id="case_1", input={"message": "hello"})
    variant = Variant(
        id="model_pair_1",
        factors=[FactorValue(name="model", value="demo-model")],
    )
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    result = await run_python_task("demo_tasks:run", ctx=ctx, case=case)

    assert result.status is TaskStatus.PASSED
    assert result.output == {"message": "hello", "model": "demo-model"}


async def test_async_task_returns_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "async_tasks.py",
        """
        async def run(ctx, case):
            return {"case_id": case.id, "variant_id": ctx.variant.id}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    result = await run_python_task("async_tasks:run", ctx=ctx, case=case)

    assert result.status is TaskStatus.PASSED
    assert result.output == {"case_id": "case_1", "variant_id": "variant_1"}


def test_span_observations_artifacts_and_duration_metric() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    with ctx.span(
        "task",
        duration_metric={
            "name": "latency",
            "semantic_type": Semantic.TIME_LATENCY,
            "direction": Direction.MINIMIZE,
        },
    ) as span:
        span.metric(
            "coverage",
            0.75,
            semantic_type=Semantic.COVERAGE_RATIO,
            direction=Direction.MAXIMIZE,
        )
        span.factor("model", "demo-model", semantic_type=Semantic.LLM_MODEL_NAME)
        span.event("generated")
        artifact = span.artifact(
            "spec",
            "cases: []",
            media_type="application/x-yaml",
        )

    assert artifact.id == "artifact_1"
    assert ctx.spans[0].duration_seconds is not None
    assert ctx.spans[0].duration_seconds >= 0
    assert {observation.span_id for observation in ctx.observations} == {"span_1"}
    assert any(
        observation.semantic_type == Semantic.TIME_LATENCY
        and observation.name == "latency"
        and observation.value >= 0
        for observation in ctx.observations
    )
    assert any(
        observation.kind is ObservationKind.ARTIFACT and observation.value == "artifact_1"
        for observation in ctx.observations
    )
    assert ctx.spans[0].artifacts == ["artifact_1"]


def test_context_helpers_record_measurements_bundles_checks_and_outcomes() -> None:
    ctx = RunContext(benchmark_id="demo", case=Case(id="case_1"), variant=Variant(id="variant_1"))
    measurement = Measurement(
        samples_seconds=(0.001, 0.002, 0.003),
        warmup=1,
        requested_repetitions=3,
        elapsed_seconds=0.01,
    )

    with ctx.span("helpers") as span:
        check = span.check("correctness", False, reason="wrong answer")
        no_reason_check = span.check("format", True)
        skip_payload = check.skip("timing skipped")
        span.outcome(False)
        span.skip_reason("correctness failed")
        span.diagnostic("note", "kept for debugging")
        span.metrics(
            "quality",
            {"score": 0.25, "label": "bad"},
            semantic_types={"score": Semantic.QUALITY_SCORE},
            units={"score": "ratio"},
            direction=Direction.MAXIMIZE,
        )
        recorded = span.record_measurement("latency", measurement)

    assert skip_payload == {
        "skipped": True,
        "check": "correctness",
        "passed": False,
        "reason": "timing skipped",
    }
    assert check.observation.tags["reason"] == "wrong answer"
    assert no_reason_check.observation.tags == {}
    assert recorded.samples_artifact is not None
    assert recorded.samples_artifact.name == "latency.samples_ms"
    names = {observation.name for observation in ctx.observations}
    assert {
        "correctness",
        "success",
        "skip_reason",
        "note",
        "quality.score",
        "quality.label",
        "latency.median_ms",
        "latency.p95_ms",
        "latency.samples_ms",
    } <= names
    assert any(
        observation.name == "note" and observation.role is ObservationRole.DIAGNOSTIC
        for observation in ctx.observations
    )


def test_context_can_record_measurement_without_sample_artifact() -> None:
    ctx = RunContext(benchmark_id="demo", case=Case(id="case_1"), variant=Variant(id="variant_1"))
    measurement = Measurement(
        samples_seconds=(0.001,),
        warmup=0,
        requested_repetitions=1,
        elapsed_seconds=0.001,
    )

    recorded = ctx.record_measurement(
        "latency",
        measurement,
        include_samples_artifact=False,
    )

    assert recorded.samples_artifact is None
    assert len(recorded.metrics) == 9
    assert not ctx.artifacts


async def test_failing_task_returns_structured_error_and_preserves_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "failing_tasks.py",
        """
        def run(ctx, case):
            with ctx.span("task") as span:
                span.metric("coverage", 0.1, semantic_type="coverage.ratio")
                raise RuntimeError("boom")
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    result = await run_python_task("failing_tasks:run", ctx=ctx, case=case)

    assert result.status is TaskStatus.FAILED
    assert result.error is not None
    assert result.error.error_type == "RuntimeError"
    assert len(result.errors) == 1
    assert result.observations[0].name == "coverage"
    assert result.spans[0].error is not None


async def test_failing_task_preserves_preexisting_context_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "manual_error_tasks.py",
        """
        def run(ctx, case):
            ctx.error("manual warning")
            raise RuntimeError("different failure")
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    result = await run_python_task("manual_error_tasks:run", ctx=ctx, case=case)

    assert result.status is TaskStatus.FAILED
    assert [error.message for error in result.errors] == ["manual warning", "different failure"]


async def test_missing_task_target_returns_structured_error() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    result = await run_python_task("missing_task_module:run", ctx=ctx, case=case)

    assert result.status is TaskStatus.ERRORED
    assert result.error is not None
    assert result.error.error_type == "TaskResolutionError"


def test_resolve_python_callable_reports_bad_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "callable_targets.py",
        """
        not_callable = 42

        def run(ctx, case):
            return {"ok": True}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(TaskResolutionError, match="module:function"):
        resolve_python_callable("callable_targets")
    with pytest.raises(TaskResolutionError, match="does not define"):
        resolve_python_callable("callable_targets:missing")
    with pytest.raises(TaskResolutionError, match="not callable"):
        resolve_python_callable("callable_targets:not_callable")


def test_resolve_python_callable_uses_explicit_search_paths(tmp_path: Path) -> None:
    package_dir = tmp_path / "localpkg"
    package_dir.mkdir()
    _write_module(package_dir, "__init__.py", "")
    _write_module(
        package_dir,
        "tasks.py",
        """
        def run(ctx, case):
            return {"ok": True}
        """,
    )

    callable_target = resolve_python_callable(
        "localpkg.tasks:run",
        search_paths=(str(tmp_path),),
    )

    assert callable(callable_target)


def test_temporary_sys_path_is_noop_for_existing_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_path = str(tmp_path.resolve())
    monkeypatch.syspath_prepend(existing_path)
    before = list(sys.path)

    with tasks_module._temporary_sys_path((existing_path,)):
        assert list(sys.path) == before

    assert list(sys.path) == before


def test_temporary_sys_path_tolerates_manual_removal(tmp_path: Path) -> None:
    inserted_path = str(tmp_path.resolve())
    assert inserted_path not in sys.path

    with tasks_module._temporary_sys_path((inserted_path,)):
        assert sys.path[0] == inserted_path
        sys.path.remove(inserted_path)

    assert inserted_path not in sys.path


def test_run_context_reports_unknown_factors_and_unstarted_span_access() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1", factors=[FactorValue(name="model", value="gpt-x")])
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)
    span = ctx.span("not-started")

    with pytest.raises(KeyError, match="Unknown variant factor"):
        ctx.factor("missing")
    with pytest.raises(RuntimeError, match="Span has not started"):
        _ = span.id
    with pytest.raises(RuntimeError, match="Span has not started"):
        _ = span.record

    assert span.__exit__(None, None, None) is None


def test_context_can_attach_errors_and_orphan_span_ids_do_not_crash() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    with ctx.span("tracked") as span:
        manual = span.error("manual failure")
        copied = ctx.error(ErrorRecord(error_type="Custom", message="custom"), span_id=span.id)

    orphan_error = ctx.error("orphan failure", span_id="missing-span")
    orphan_metric = ctx.metric("orphan_metric", 1, span_id="missing-span")
    orphan_artifact = ctx.artifact("orphan_artifact", {"ok": True}, span_id="missing-span")

    assert manual.error_type == "Error"
    assert copied.span_id == "span_1"
    assert orphan_error.span_id == "missing-span"
    assert orphan_metric.span_id == "missing-span"
    assert orphan_artifact.span_id == "missing-span"
    assert ctx.spans[0].error == copied


def test_error_record_can_skip_traceback_capture() -> None:
    record = ErrorRecord.from_exception(ValueError("bad value"), include_traceback=False)

    assert record.error_type == "ValueError"
    assert record.message == "bad value"
    assert record.traceback is None


def test_span_iteration_duration_spec_and_out_of_order_finish_are_supported() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    with ctx.span("iterable", duration_metric=DurationMetricSpec(name="elapsed")) as span:
        assert list(span) == [span.record]

    outer, outer_started_at = ctx._start_span("outer", tags={})
    inner, inner_started_at = ctx._start_span("inner", tags={})
    ctx._finish_span(outer, started_at=outer_started_at, duration_metric=None)
    ctx._finish_span(inner, started_at=inner_started_at, duration_metric=None)
    detached, detached_started_at = ctx._start_span("detached", tags={})
    ctx._span_stack.clear()
    ctx._finish_span(detached, started_at=detached_started_at, duration_metric=None)

    assert outer.ended_at is not None
    assert inner.ended_at is not None
    assert detached.ended_at is not None
    assert ctx._span_stack == []
    assert any(observation.name == "elapsed" for observation in ctx.observations)


def _write_module(tmp_path: Path, filename: str, source: str) -> None:
    (tmp_path / filename).write_text(dedent(source).strip() + "\n", encoding="utf-8")
