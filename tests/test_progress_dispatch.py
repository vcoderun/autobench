from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import Collection, Sequence
from io import StringIO
from pathlib import Path
from textwrap import dedent
from typing import Never

import pytest
from rich.console import Console

import autobench.cli as cli_module
import autobench.runtime.pipeline as pipeline_module
from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    ExperimentStatus,
    FileRecorder,
    MatrixRunSpec,
    PolicySpec,
    ProgressDispatchError,
    ProgressErrorPolicy,
    ProgressEvent,
    ProgressEventKind,
    ProgressHandlerFailure,
    RecordingError,
    RunContext,
    RunResult,
    RunStatus,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    progress_event,
    replay_experiment,
    run_benchmark_path,
    run_benchmark_spec,
)
from autobench.instrumentation.config import InstrumentationConfig
from autobench.instrumentation.registry import InstrumentorStatus
from autobench.records.staging import RecordSession


def benchmark_spec(*, cases: tuple[str, ...] = ("one", "two")) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="progress-demo"),
        dataset=DatasetSpec(cases=[Case(id=case_id) for case_id in cases]),
        task=TaskSpec(kind="python", target="progress_tasks:run"),
        variants=[Variant(id="baseline"), Variant(id="candidate")],
    )


async def passing_task(
    target: str,
    *,
    ctx: RunContext,
    case: Case,
    search_paths: tuple[str, ...] = (),
) -> TaskResult:
    del target, search_paths
    await asyncio.sleep(0 if case.id == "one" else 0.001)
    return TaskResult(
        output={"case": case.id, "variant": ctx.variant.id},
        status=TaskStatus.PASSED,
    )


async def test_progress_dispatch_is_serialized_and_status_bearing_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    events: list[ProgressEvent] = []
    handler_active = False

    async def handler(event: ProgressEvent) -> None:
        nonlocal handler_active
        assert handler_active is False
        handler_active = True
        await asyncio.sleep(0.001)
        events.append(event)
        handler_active = False

    result = await run_benchmark_spec(
        benchmark_spec(),
        experiment_id="exp_progress",
        concurrency_limit=4,
        progress_handlers=(handler,),
    )

    assert result.passed_count == 4
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].kind is ProgressEventKind.BENCHMARK_STARTED
    assert events[-1].kind is ProgressEventKind.BENCHMARK_FINISHED
    assert events[-1].experiment_status is ExperimentStatus.COMPLETED
    run_started = [event for event in events if event.kind is ProgressEventKind.RUN_STARTED]
    run_finished = [event for event in events if event.kind is ProgressEventKind.RUN_FINISHED]
    assert {event.run_id for event in run_started} == {event.run_id for event in run_finished}
    assert [event.run_id for event in run_finished] == [run.run_id for run in result.runs]
    assert all(event.run_status is RunStatus.PASSED for event in run_finished)


async def test_progress_terminal_statuses_cover_failed_errored_and_skipped_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def mixed_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        del target, ctx, search_paths
        if case.id == "error":
            raise RuntimeError("subject failed")
        return TaskResult(status=TaskStatus.FAILED)

    monkeypatch.setattr(pipeline_module, "run_python_task", mixed_task)
    events: list[ProgressEvent] = []
    result = await run_benchmark_spec(
        benchmark_spec(cases=("failed", "error")),
        experiment_id="exp_statuses",
        progress_handlers=(events.append,),
    )
    terminals = [event for event in events if event.kind is ProgressEventKind.RUN_FINISHED]

    assert [run.status for run in result.runs] == [
        RunStatus.FAILED,
        RunStatus.FAILED,
        RunStatus.ERRORED,
        RunStatus.ERRORED,
    ]
    assert [event.run_status for event in terminals] == [run.status for run in result.runs]

    skipped_events: list[ProgressEvent] = []
    skipped_spec = benchmark_spec(cases=("skipped",)).model_copy(update={"task": None})
    skipped = await run_benchmark_spec(
        skipped_spec,
        experiment_id="exp_skipped",
        progress_handlers=(skipped_events.append,),
    )
    skipped_terminal = next(
        event for event in skipped_events if event.kind is ProgressEventKind.RUN_FINISHED
    )
    assert skipped.runs[0].status is RunStatus.SKIPPED
    assert skipped_terminal.run_status is RunStatus.SKIPPED


async def test_policy_violations_precede_the_final_failed_run_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def metric_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        del target, case, search_paths
        ctx.metric("quality", 0.2, semantic_type="quality.custom")
        return TaskResult(status=TaskStatus.PASSED)

    monkeypatch.setattr(pipeline_module, "run_python_task", metric_task)
    spec = benchmark_spec(cases=("policy",)).model_copy(
        update={
            "policies": [
                PolicySpec(
                    name="quality-floor",
                    metric="quality.custom",
                    must_greater_equal=0.8,
                ),
                PolicySpec(
                    name="quality-present",
                    metric="quality.custom",
                    must_greater=0,
                ),
            ]
        }
    )
    events: list[ProgressEvent] = []

    result = await run_benchmark_spec(
        spec,
        experiment_id="exp_policy_progress",
        progress_handlers=(events.append,),
    )

    violations = [event for event in events if event.kind is ProgressEventKind.POLICY_VIOLATION]
    terminals = [event for event in events if event.kind is ProgressEventKind.RUN_FINISHED]
    assert len(violations) == 2
    assert {event.data["policy_name"] for event in violations} == {"quality-floor"}
    assert all(event.run_status is RunStatus.FAILED for event in terminals)
    assert all(run.status is RunStatus.FAILED for run in result.runs)
    assert max(event.sequence for event in violations) < min(event.sequence for event in terminals)


async def test_cancellation_delivers_paired_terminal_events_and_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()

    async def blocking_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        del target, ctx, case, search_paths
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(pipeline_module, "run_python_task", blocking_task)
    events: list[ProgressEvent] = []
    execution = asyncio.create_task(
        run_benchmark_spec(
            benchmark_spec(cases=("cancel",)),
            experiment_id="exp_cancel_progress",
            progress_handlers=(events.append,),
        )
    )
    await entered.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    assert [event.kind for event in events] == [
        ProgressEventKind.BENCHMARK_STARTED,
        ProgressEventKind.RUN_STARTED,
        ProgressEventKind.RUN_FINISHED,
        ProgressEventKind.BENCHMARK_FINISHED,
    ]
    assert events[-2].run_status is RunStatus.CANCELLED
    assert events[-1].experiment_status is ExperimentStatus.CANCELLED


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signals are required")
def test_hard_process_death_does_not_claim_terminal_progress(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    events_path = tmp_path / "events.jsonl"
    (tmp_path / "hard_death_task.py").write_text(
        dedent(
            """
            import asyncio
            import os
            from pathlib import Path

            async def run(ctx, case):
                Path(os.environ["AUTOBENCH_PROGRESS_READY"]).write_text("ready")
                await asyncio.Event().wait()
            """
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "hard-death.yaml"
    spec_path.write_text(
        dedent(
            """
            benchmark:
              id: hard-death
            dataset:
              cases:
                - id: one
            task:
              kind: python
              target: hard_death_task:run
            variants:
              - id: baseline
            """
        ),
        encoding="utf-8",
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        dedent(
            """
            import asyncio
            import os
            from pathlib import Path

            from autobench import load_benchmark_spec, run_benchmark_spec

            events = Path(os.environ["AUTOBENCH_PROGRESS_EVENTS"])

            def observe(event):
                with events.open("a", encoding="utf-8") as stream:
                    stream.write(event.model_dump_json() + "\\n")
                    stream.flush()
                    os.fsync(stream.fileno())

            asyncio.run(
                run_benchmark_spec(
                    load_benchmark_spec(Path(os.environ["AUTOBENCH_PROGRESS_SPEC"])),
                    experiment_id="exp_hard_death",
                    progress_handlers=(observe,),
                )
            )
            """
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    python_paths = [str(tmp_path), str(Path(__file__).resolve().parents[1] / "src")]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path is not None:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["AUTOBENCH_PROGRESS_READY"] = str(ready)
    environment["AUTOBENCH_PROGRESS_EVENTS"] = str(events_path)
    environment["AUTOBENCH_PROGRESS_SPEC"] = str(spec_path)
    process = subprocess.Popen(
        [sys.executable, str(worker)],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not ready.exists():
        stdout, stderr = process.communicate(timeout=5)
        raise AssertionError(f"worker failed to start: stdout={stdout!r} stderr={stderr!r}")

    process.kill()
    process.communicate(timeout=10)

    assert process.returncode == -signal.SIGKILL
    events = [
        ProgressEvent.model_validate_json(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event.kind for event in events] == [
        ProgressEventKind.BENCHMARK_STARTED,
        ProgressEventKind.RUN_STARTED,
    ]


async def test_strict_handler_failure_is_raised_after_durable_terminal_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    failing_events: list[ProgressEventKind] = []
    healthy_events: list[ProgressEvent] = []

    def failing_handler(event: ProgressEvent) -> None:
        failing_events.append(event.kind)
        if event.kind is ProgressEventKind.RUN_STARTED:
            raise RuntimeError("renderer failed")

    output = tmp_path / "record"
    with pytest.raises(ProgressDispatchError) as caught:
        await run_benchmark_spec(
            benchmark_spec(cases=("one",)),
            experiment_id="exp_strict_progress",
            recorder=FileRecorder(output),
            progress_handlers=(failing_handler, healthy_events.append),
        )

    assert failing_events == [
        ProgressEventKind.BENCHMARK_STARTED,
        ProgressEventKind.RUN_STARTED,
    ]
    assert healthy_events[-1].kind is ProgressEventKind.BENCHMARK_FINISHED
    assert healthy_events[-1].experiment_status is ExperimentStatus.COMPLETED
    assert len(caught.value.failures) == 1
    assert caught.value.failures[0].event_kind is ProgressEventKind.RUN_STARTED
    assert replay_experiment(output).passed_count == 2


async def test_benchmark_start_is_not_emitted_before_recorder_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "foreign.txt").write_text("keep", encoding="utf-8")
    events: list[ProgressEvent] = []

    with pytest.raises(RecordingError):
        await run_benchmark_spec(
            benchmark_spec(cases=("one",)),
            experiment_id="exp_recorder_ownership",
            recorder=FileRecorder(output),
            progress_handlers=(events.append,),
        )

    assert events == []


async def test_best_effort_handler_failure_is_reported_without_failing_the_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    failures: list[ProgressHandlerFailure] = []

    def failing_handler(event: ProgressEvent) -> None:
        raise ValueError(event.kind)

    result = await run_benchmark_spec(
        benchmark_spec(cases=("one",)),
        experiment_id="exp_best_effort_progress",
        progress_handlers=(failing_handler,),
        progress_error_policy=ProgressErrorPolicy.BEST_EFFORT,
        progress_error_handler=failures.append,
    )

    assert result.passed_count == 2
    assert len(failures) == 1
    assert failures[0].event_kind is ProgressEventKind.BENCHMARK_STARTED


@pytest.mark.parametrize("reporter_fails", [False, True])
async def test_best_effort_progress_failures_are_never_silent(
    monkeypatch: pytest.MonkeyPatch,
    reporter_fails: bool,
) -> None:
    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)

    def failing_handler(event: ProgressEvent) -> None:
        raise ValueError(event.kind)

    def failing_reporter(failure: ProgressHandlerFailure) -> None:
        raise RuntimeError(failure.event_kind)

    with pytest.warns(UserWarning, match="Progress"):
        result = await run_benchmark_spec(
            benchmark_spec(cases=("one",)),
            experiment_id=f"exp_warning_{reporter_fails}",
            progress_handlers=(failing_handler,),
            progress_error_policy=ProgressErrorPolicy.BEST_EFFORT,
            progress_error_handler=failing_reporter if reporter_fails else None,
        )

    assert result.passed_count == 2


async def test_pipeline_error_emits_one_aborted_benchmark_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[ProgressEvent] = []

    def fail_resolution(
        configs: Sequence[InstrumentationConfig],
        *,
        reserved_ids: Collection[str] = (),
    ) -> Never:
        del configs, reserved_ids
        raise RuntimeError("resolution failed")

    monkeypatch.setattr(pipeline_module, "resolve_instrumentors", fail_resolution)

    with pytest.raises(RuntimeError, match="resolution failed"):
        await run_benchmark_spec(
            benchmark_spec(cases=("one",)),
            experiment_id="exp_aborted_progress",
            progress_handlers=(events.append,),
        )

    assert [event.kind for event in events] == [
        ProgressEventKind.BENCHMARK_STARTED,
        ProgressEventKind.BENCHMARK_FINISHED,
    ]
    assert events[-1].experiment_status is ExperimentStatus.ABORTED
    assert events[-1].data == {"partial": True, "error_type": "RuntimeError"}


async def test_internal_run_crash_pairs_started_run_with_errored_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[ProgressEvent] = []

    async def crash(
        spec: BenchmarkSpec,
        run_spec: MatrixRunSpec,
        *,
        instrumentation_diagnostics: Sequence[InstrumentorStatus] = (),
        record_session: RecordSession | None = None,
    ) -> RunResult:
        del spec, run_spec, instrumentation_diagnostics, record_session
        raise RuntimeError("runtime crashed")

    monkeypatch.setattr(pipeline_module, "_run_matrix_item", crash)

    with pytest.raises(RuntimeError, match="runtime crashed"):
        await run_benchmark_spec(
            benchmark_spec(cases=("one",)),
            experiment_id="exp_run_crash",
            progress_handlers=(events.append,),
        )

    assert [event.kind for event in events] == [
        ProgressEventKind.BENCHMARK_STARTED,
        ProgressEventKind.RUN_STARTED,
        ProgressEventKind.RUN_FINISHED,
        ProgressEventKind.BENCHMARK_FINISHED,
    ]
    assert events[-2].run_status is RunStatus.ERRORED
    assert events[-2].data["partial"] is True
    assert events[-1].experiment_status is ExperimentStatus.ABORTED


async def test_primary_pipeline_error_retains_strict_progress_failure_as_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_handler(event: ProgressEvent) -> None:
        raise ValueError(event.kind)

    def fail_resolution(
        configs: Sequence[InstrumentationConfig],
        *,
        reserved_ids: Collection[str] = (),
    ) -> Never:
        del configs, reserved_ids
        raise RuntimeError("primary failure")

    monkeypatch.setattr(pipeline_module, "resolve_instrumentors", fail_resolution)

    with pytest.raises(RuntimeError, match="primary failure") as caught:
        await run_benchmark_spec(
            benchmark_spec(cases=("one",)),
            experiment_id="exp_primary_progress",
            progress_handlers=(failing_handler,),
        )

    assert any("Progress delivery failed" in note for note in caught.value.__notes__)


async def test_progress_cancellation_preserves_primary_error_and_reports_terminal_failure() -> None:
    async def cancelled_handler(event: ProgressEvent) -> None:
        raise asyncio.CancelledError(event.kind)

    with pytest.raises(asyncio.CancelledError) as caught:
        await run_benchmark_spec(
            benchmark_spec(cases=("one",)),
            experiment_id="exp_cancelled_handler",
            progress_handlers=(cancelled_handler,),
        )

    assert any("progress terminal delivery failed" in note for note in caught.value.__notes__)


def test_progress_event_keeps_extension_data_and_typed_terminal_fields() -> None:
    event = progress_event(
        ProgressEventKind.RUN_FINISHED,
        "done",
        sequence=9,
        run_status=RunStatus.PASSED,
        benchmark_id="demo",
        run_id="run-1",
        custom="value",
    )

    assert event.sequence == 9
    assert event.benchmark_id == "demo"
    assert event.run_id == "run-1"
    assert event.run_status is RunStatus.PASSED
    assert event.data == {"custom": "value"}
    assert "candidate_decision" not in {kind.value for kind in ProgressEventKind}


def test_cli_progress_tolerates_out_of_order_events_and_reports_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = StringIO()
    progress = cli_module._CliProgress(Console(file=output, force_terminal=False))
    with progress:
        progress(progress_event(ProgressEventKind.RUN_STARTED, "early"))
        progress(
            progress_event(
                ProgressEventKind.BENCHMARK_STARTED,
                "started",
                benchmark_id="demo",
                planned_run_count="unknown",
            )
        )
        progress(
            progress_event(
                ProgressEventKind.RUN_FINISHED,
                "done",
                run_status=RunStatus.PASSED,
            )
        )
        progress(
            progress_event(
                ProgressEventKind.BENCHMARK_FINISHED,
                "finished",
                experiment_status=ExperimentStatus.COMPLETED,
            )
        )

    cli_module._report_progress_failure(
        ProgressHandlerFailure(
            handler_index=0,
            sequence=2,
            event_kind=ProgressEventKind.RUN_STARTED,
            error=RuntimeError("display failed"),
        )
    )
    assert "Progress renderer failed during run_started" in capsys.readouterr().err


def test_run_benchmark_path_forwards_progress_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    spec_path = tmp_path / "autobench.yaml"
    spec_path.write_text(
        """benchmark:\n  path-progress:\n    cases:\n      - id: one\n    run:\n      python: progress_tasks:run\n    variants:\n      baseline: {}\n""",
        encoding="utf-8",
    )
    events: list[ProgressEvent] = []

    result = run_benchmark_path(
        spec_path,
        experiment_id="exp_path_progress",
        progress_handlers=(events.append,),
    )

    assert result.passed_count == 1
    assert events[-1].experiment_status is ExperimentStatus.COMPLETED
