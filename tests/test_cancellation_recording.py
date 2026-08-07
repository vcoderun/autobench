from __future__ import annotations as _annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from textwrap import dedent
from typing import Any, NoReturn

import pytest
from click.testing import CliRunner

import autobench.cli as cli_module
import autobench.runtime.pipeline as pipeline_module
from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    EndReason,
    EvaluationStatus,
    ExecutionSnapshot,
    ExperimentResult,
    ExperimentStart,
    ExperimentStatus,
    ExperimentTermination,
    FileRecorder,
    PartialRunSnapshot,
    RecordingError,
    RunContext,
    RunPhase,
    RunResult,
    RunStatus,
    Semantic,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    finalize_staging,
    inspect_staging,
    recover_staging,
    replay_experiment,
    run_benchmark_spec,
)
from autobench.cli import cli
from autobench.evaluation.derivation import DeriverSpec
from autobench.evaluation.scoring import ScoreRecord, ScoringSpec
from autobench.metrics.observations import Observation
from autobench.metrics.semantics import SemanticRegistry
from autobench.records.models import ExperimentRecord
from autobench.runtime.awaitables import ProcessSignalInterrupt, _run_cooperatively


class MemoryRecorder:
    def __init__(self, session: MemorySession) -> None:
        self.session = session

    async def open(self, start: ExperimentStart) -> MemorySession:
        self.session.start = start
        return self.session


class MemorySession:
    def __init__(
        self,
        *,
        block_first_checkpoint: bool = False,
        fail_checkpoints: bool = False,
    ) -> None:
        self.start: ExperimentStart | None = None
        self.block_first_checkpoint = block_first_checkpoint
        self.fail_checkpoints = fail_checkpoints
        self.checkpoint_started = asyncio.Event()
        self.release_checkpoint = asyncio.Event()
        self.snapshots: list[PartialRunSnapshot] = []
        self.staged: list[ExecutionSnapshot] = []
        self.termination: ExperimentTermination | None = None
        self.closed = False

    async def stage(self, snapshot: ExecutionSnapshot) -> None:
        self.staged.append(snapshot)

    async def checkpoint(self, snapshot: PartialRunSnapshot) -> None:
        if self.block_first_checkpoint and not self.snapshots:
            self.checkpoint_started.set()
            await self.release_checkpoint.wait()
        if self.fail_checkpoints:
            raise RecordingError("checkpoint unavailable")
        self.snapshots.append(snapshot)

    async def finish(self, result: ExperimentResult) -> ExperimentRecord:
        raise AssertionError("Cancellation tests must not finish the recording session.")

    async def abort(self, termination: ExperimentTermination) -> None:
        self.termination = termination

    async def close(self) -> None:
        self.closed = True


class CancellingSession(MemorySession):
    async def checkpoint(self, snapshot: PartialRunSnapshot) -> None:
        raise asyncio.CancelledError("session cancelled")


def one_run_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark=BenchmarkInfo(id="cancellation"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="unused:run"),
        variants=[Variant(id="variant_1")],
    )


async def test_run_context_checkpoint_requires_durable_recording_and_valid_name() -> None:
    ctx = RunContext(
        benchmark_id="benchmark",
        experiment_id="experiment",
        run_id="run",
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        await ctx.checkpoint("  ")
    with pytest.raises(ValueError, match="reserved"):
        await ctx.checkpoint("autobench.internal")
    with pytest.raises(RuntimeError, match="durable recording"):
        await ctx.checkpoint("ready")


async def test_run_context_checkpoint_captures_phase_output_and_evidence() -> None:
    ctx = RunContext(
        benchmark_id="benchmark",
        experiment_id="experiment",
        run_id="run",
        case=Case(id="case"),
        variant=Variant(id="variant"),
    )
    captured: list[tuple[str, RunPhase, Any, tuple[Observation, ...]]] = []

    async def capture(
        name: str,
        phase: RunPhase,
        output: Any,
        evidence: pipeline_module.ContextEvidence,
    ) -> None:
        captured.append((name, phase, output, evidence.observations))

    ctx.bind_checkpoint(capture)
    with pytest.raises(RuntimeError, match="already bound"):
        ctx.bind_checkpoint(capture)
    ctx.set_phase(RunPhase.SCORING)
    ctx.retain_task_output({"answer": 42})
    ctx.metric("quality", 1.0, semantic_type=Semantic.QUALITY_SCORE)

    await ctx.checkpoint(" scored ")

    assert ctx.phase is RunPhase.SCORING
    assert ctx.checkpoint_output == {"answer": 42}
    assert captured[0][:3] == ("scored", RunPhase.SCORING, {"answer": 42})
    assert captured[0][3][0].name == "quality"


async def test_cancellation_waits_for_in_flight_explicit_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def checkpointing_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        ctx.metric("step", 1)
        await ctx.checkpoint("after-step")
        return TaskResult(output={"done": True}, status=TaskStatus.PASSED)

    monkeypatch.setattr(pipeline_module, "run_python_task", checkpointing_task)
    session = MemorySession(block_first_checkpoint=True)
    execution = asyncio.create_task(
        run_benchmark_spec(
            one_run_spec(),
            experiment_id="exp_checkpoint",
            recorder=MemoryRecorder(session),
        )
    )
    await session.checkpoint_started.wait()

    execution.cancel()
    await asyncio.sleep(0)
    assert not execution.done()
    session.release_checkpoint.set()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert [snapshot.name for snapshot in session.snapshots] == [
        "after-step",
        "autobench.cancelled",
    ]
    assert session.snapshots[0].phase is RunPhase.EXECUTING
    assert session.snapshots[0].observations[0].name == "step"
    assert session.termination is not None
    assert session.closed is True


async def test_task_cancellation_preserves_original_exception_and_execution_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = asyncio.CancelledError("stop task")

    async def cancel_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        ctx.metric("before_cancel", 1)
        raise original

    monkeypatch.setattr(pipeline_module, "run_python_task", cancel_task)
    session = MemorySession()

    with pytest.raises(asyncio.CancelledError) as raised:
        await run_benchmark_spec(
            one_run_spec(),
            experiment_id="exp_task_cancel",
            recorder=MemoryRecorder(session),
        )

    assert raised.value is original
    assert session.snapshots[0].phase is RunPhase.EXECUTING
    assert session.snapshots[0].task_output is None
    assert session.snapshots[0].end_reason is EndReason.CANCELLED
    assert session.snapshots[0].trace is not None
    assert session.snapshots[0].trace.partial is True


async def test_scoring_cancellation_preserves_completed_task_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passing_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        return TaskResult(output={"answer": 42}, status=TaskStatus.PASSED)

    async def cancel_scoring(
        scoring: list[ScoringSpec],
        *,
        ctx: RunContext,
        task_result: TaskResult,
    ) -> list[ScoreRecord]:
        raise asyncio.CancelledError("stop scoring")

    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    monkeypatch.setattr(pipeline_module, "evaluate_scoring_specs", cancel_scoring)
    session = MemorySession()

    with pytest.raises(asyncio.CancelledError):
        await run_benchmark_spec(
            one_run_spec(),
            experiment_id="exp_scoring_cancel",
            recorder=MemoryRecorder(session),
        )

    snapshot = session.snapshots[0]
    assert snapshot.phase is RunPhase.SCORING
    assert snapshot.task_output == {"answer": 42}
    assert snapshot.task_status is TaskStatus.CANCELLED


async def test_derivation_cancellation_preserves_completed_task_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passing_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        return TaskResult(output={"answer": 42}, status=TaskStatus.PASSED)

    def cancel_derivation(
        specs: list[DeriverSpec],
        *,
        ctx: RunContext,
        observations: list[Observation],
        registry: SemanticRegistry,
    ) -> list[Observation]:
        raise asyncio.CancelledError("stop derivation")

    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    monkeypatch.setattr(pipeline_module, "derive_observations", cancel_derivation)
    session = MemorySession()

    with pytest.raises(asyncio.CancelledError):
        await run_benchmark_spec(
            one_run_spec(),
            experiment_id="exp_derivation_cancel",
            recorder=MemoryRecorder(session),
        )

    snapshot = session.snapshots[0]
    assert snapshot.phase is RunPhase.DERIVING
    assert snapshot.task_output == {"answer": 42}


async def test_concurrent_cancellation_checkpoints_active_sibling_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_started = asyncio.Event()

    async def concurrent_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        if case.id == "case_1":
            await sibling_started.wait()
            raise asyncio.CancelledError("cancel matrix")
        sibling_started.set()
        await asyncio.Event().wait()
        return TaskResult(status=TaskStatus.PASSED)

    monkeypatch.setattr(pipeline_module, "run_python_task", concurrent_task)
    spec = one_run_spec().model_copy(
        update={"dataset": DatasetSpec(cases=[Case(id="case_1"), Case(id="case_2")])}
    )
    session = MemorySession()

    with pytest.raises(asyncio.CancelledError):
        await run_benchmark_spec(
            spec,
            experiment_id="exp_concurrent_cancel",
            concurrency_limit=2,
            recorder=MemoryRecorder(session),
        )

    assert {snapshot.case_id for snapshot in session.snapshots} == {"case_1", "case_2"}
    assert all(snapshot.phase is RunPhase.EXECUTING for snapshot in session.snapshots)
    assert session.termination is not None
    assert session.termination.recorded_run_ids == ()
    assert set(session.termination.missing_run_ids) == {
        "run_0001_0001_case_1__variant_1",
        "run_0002_0001_case_2__variant_1",
    }


async def test_checkpoint_failure_is_not_allowed_to_replace_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = asyncio.CancelledError("original cancellation")

    async def cancel_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        raise original

    monkeypatch.setattr(pipeline_module, "run_python_task", cancel_task)
    session = MemorySession(fail_checkpoints=True)

    with pytest.raises(asyncio.CancelledError) as raised:
        await run_benchmark_spec(
            one_run_spec(),
            experiment_id="exp_failed_checkpoint",
            recorder=MemoryRecorder(session),
        )

    assert raised.value is original
    assert any("checkpoint unavailable" in note for note in original.__notes__)
    assert session.termination is not None
    assert session.closed is True


async def test_explicit_checkpoint_failure_during_cancellation_is_attached_as_note() -> None:
    session = MemorySession(block_first_checkpoint=True)
    snapshot = PartialRunSnapshot(
        run_id="run",
        experiment_id="experiment",
        benchmark_id="benchmark",
        case_id="case",
        variant_id="variant",
        name="checkpoint",
        phase=RunPhase.EXECUTING,
    )
    persistence = asyncio.create_task(
        pipeline_module._persist_explicit_checkpoint(session, snapshot)
    )
    await session.checkpoint_started.wait()

    persistence.cancel()
    await asyncio.sleep(0)
    persistence.cancel()
    await asyncio.sleep(0)
    session.fail_checkpoints = True
    session.release_checkpoint.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await persistence
    assert any("checkpoint unavailable" in note for note in raised.value.__notes__)


async def test_explicit_checkpoint_propagates_session_cancellation() -> None:
    session = CancellingSession()
    snapshot = PartialRunSnapshot(
        run_id="run",
        experiment_id="experiment",
        benchmark_id="benchmark",
        case_id="case",
        variant_id="variant",
        name="checkpoint",
        phase=RunPhase.EXECUTING,
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline_module._persist_explicit_checkpoint(session, snapshot)


async def test_timeout_is_a_failed_partial_run_not_an_explicit_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def timed_out_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        error = ctx.error(TimeoutError("deadline"))
        return TaskResult(
            status=TaskStatus.FAILED,
            error=error,
            end_reason=EndReason.FAILED,
        )

    monkeypatch.setattr(pipeline_module, "run_python_task", timed_out_task)
    result = await run_benchmark_spec(one_run_spec(), experiment_id="exp_timeout")

    run = result.runs[0]
    assert run.status.value == "failed"
    assert run.evaluation_status is EvaluationStatus.NOT_EVALUATED
    assert run.end_reason is EndReason.TIMEOUT
    assert run.partial is True


async def test_cleanup_wait_is_bounded_and_reports_child_cancellation() -> None:
    never_finishes = asyncio.create_task(asyncio.Event().wait())
    error = await pipeline_module._finish_cleanup_task(never_finishes, timeout_seconds=0.001)
    assert isinstance(error, TimeoutError)
    await asyncio.sleep(0)
    assert never_finishes.cancelled()

    cancelled = asyncio.create_task(asyncio.sleep(0))
    cancelled.cancel()
    error = await pipeline_module._finish_cleanup_task(cancelled)
    assert isinstance(error, asyncio.CancelledError)

    async def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    failed = asyncio.create_task(fail_cleanup())
    await asyncio.sleep(0)
    error = await pipeline_module._finish_cleanup_task(failed)
    assert isinstance(error, RuntimeError)


async def test_cleanup_wait_survives_repeated_parent_cancellation() -> None:
    release = asyncio.Event()
    child = asyncio.create_task(release.wait())
    cleanup = asyncio.create_task(pipeline_module._finish_cleanup_task(child, timeout_seconds=1))
    await asyncio.sleep(0)

    cleanup.cancel()
    await asyncio.sleep(0)
    assert not cleanup.done()
    assert not child.done()
    release.set()

    assert await cleanup is None


async def test_concurrent_cleanup_error_is_attached_to_original_failure() -> None:
    original = asyncio.CancelledError("matrix cancelled")

    async def cancel_matrix() -> RunResult:
        raise original

    async def fail_when_cancelled() -> RunResult:
        try:
            await asyncio.Event().wait()
            raise AssertionError("The sibling should be cancelled before completing.")
        except asyncio.CancelledError as cancellation:
            raise RuntimeError("sibling cleanup failed") from cancellation

    with pytest.raises(asyncio.CancelledError) as raised:
        await pipeline_module._run_concurrent_matrix([cancel_matrix(), fail_when_cancelled()])

    assert raised.value is original
    assert any("sibling cleanup failed" in note for note in original.__notes__)


async def test_derivation_runtime_failure_becomes_errored_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passing_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        return TaskResult(output={"answer": 42}, status=TaskStatus.PASSED)

    def fail_derivation(
        specs: list[DeriverSpec],
        *,
        ctx: RunContext,
        observations: list[Observation],
        registry: SemanticRegistry,
    ) -> list[Observation]:
        raise RuntimeError("derivation failed")

    monkeypatch.setattr(pipeline_module, "run_python_task", passing_task)
    monkeypatch.setattr(pipeline_module, "derive_observations", fail_derivation)

    result = await run_benchmark_spec(one_run_spec(), experiment_id="exp_derivation_error")

    assert result.runs[0].status is RunStatus.ERRORED
    assert result.runs[0].task_result.output == {"answer": 42}
    assert result.runs[0].error is not None
    assert result.runs[0].error.error_type == "RuntimeError"


async def test_cooperative_runner_converts_sigterm_to_typed_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    callbacks: list[Callable[[], None]] = []
    removed: list[signal.Signals] = []

    def install_handler(
        watched_signal: signal.Signals,
        callback: Callable[[], None],
        *args: Any,
    ) -> None:
        assert watched_signal is signal.SIGTERM
        assert args == ()
        callbacks.append(callback)

    def remove_handler(watched_signal: signal.Signals) -> bool:
        removed.append(watched_signal)
        return True

    monkeypatch.setattr(loop, "add_signal_handler", install_handler)
    monkeypatch.setattr(loop, "remove_signal_handler", remove_handler)

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    execution = asyncio.create_task(_run_cooperatively(wait_forever()))
    await asyncio.sleep(0)
    callbacks[0]()

    with pytest.raises(ProcessSignalInterrupt) as raised:
        await execution
    assert raised.value.signal_number == signal.SIGTERM
    assert removed == [signal.SIGTERM]


async def test_cooperative_runner_supports_platforms_without_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()

    def unsupported_handler(
        watched_signal: signal.Signals,
        callback: Callable[[], None],
        *args: Any,
    ) -> None:
        raise NotImplementedError

    monkeypatch.setattr(loop, "add_signal_handler", unsupported_handler)

    async def answer() -> int:
        return 42

    assert await _run_cooperatively(answer()) == 42


async def test_cooperative_runner_preserves_non_signal_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()

    def install_handler(
        watched_signal: signal.Signals,
        callback: Callable[[], None],
        *args: Any,
    ) -> None:
        return None

    def remove_handler(watched_signal: signal.Signals) -> bool:
        return True

    monkeypatch.setattr(loop, "add_signal_handler", install_handler)
    monkeypatch.setattr(loop, "remove_signal_handler", remove_handler)

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    execution = asyncio.create_task(_run_cooperatively(wait_forever()))
    await asyncio.sleep(0)
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution


@pytest.mark.parametrize(
    ("failure", "extra_args", "exit_code", "message", "shows_staging"),
    [
        (
            ProcessSignalInterrupt(signal.SIGTERM),
            (),
            128 + signal.SIGTERM,
            "Benchmark interrupted by SIGTERM",
            True,
        ),
        (KeyboardInterrupt(), ("--no-record",), 130, "Benchmark interrupted", False),
    ],
)
def test_cli_reports_cooperative_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: ProcessSignalInterrupt | KeyboardInterrupt,
    extra_args: tuple[str, ...],
    exit_code: int,
    message: str,
    shows_staging: bool,
) -> None:
    spec = tmp_path / "interrupt.yaml"
    spec.write_text(
        dedent(
            """
            benchmark:
              id: interrupt
            dataset:
              cases:
                - id: case_1
            task:
              kind: python
              target: unused:run
            variants:
              - id: variant_1
            """
        ),
        encoding="utf-8",
    )

    def interrupt(
        awaitable: Coroutine[Any, Any, ExperimentResult],
    ) -> NoReturn:
        awaitable.close()
        raise failure

    monkeypatch.setattr(cli_module, "run_sync_cooperatively", interrupt)
    result = CliRunner().invoke(cli, ["run", str(spec), *extra_args])

    assert result.exit_code == exit_code
    assert message in result.output
    assert ("Staging path" in result.output) is shows_staging


async def test_cancelled_staging_finalizes_into_normal_partial_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel_task(
        target: str,
        *,
        ctx: RunContext,
        case: Case,
        search_paths: tuple[str, ...] = (),
    ) -> TaskResult:
        ctx.metric("completed_steps", 2)
        raise asyncio.CancelledError("stop")

    monkeypatch.setattr(pipeline_module, "run_python_task", cancel_task)
    output = tmp_path / "partial-record"
    recorder = FileRecorder(output)

    with pytest.raises(asyncio.CancelledError):
        await run_benchmark_spec(
            one_run_spec(),
            experiment_id="exp_partial_replay",
            recorder=recorder,
        )

    inspection = inspect_staging(recorder.staging_dir)
    assert inspection.checkpointed_run_ids == ("run_0001_0001_case_1__variant_1",)
    finalize_staging(recorder.staging_dir, output, allow_partial=True)
    replayed = replay_experiment(output)

    assert replayed.termination.status is ExperimentStatus.CANCELLED
    assert replayed.termination.partial is True
    assert replayed.termination.cross_run_derivation_complete is True
    assert replayed.termination.policies_complete is True
    assert replayed.termination.missing_run_ids == ()
    assert replayed.runs[0].status is RunStatus.CANCELLED
    assert replayed.runs[0].evaluation_status is EvaluationStatus.NOT_EVALUATED
    assert replayed.runs[0].task_result.status is TaskStatus.CANCELLED
    assert replayed.runs[0].end_reason is EndReason.CANCELLED
    assert replayed.runs[0].partial is True
    assert replayed.runs[0].task_result.output is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signals are required")
def test_cli_sigterm_commits_cancellation_evidence(tmp_path: Path) -> None:
    process, recorder, ready = start_signal_worker(tmp_path, name="sigterm")
    wait_for_ready(process, ready)

    process.send_signal(signal.SIGTERM)
    _stdout, stderr = process.communicate(timeout=15)

    assert process.returncode == 128 + signal.SIGTERM
    assert "Benchmark interrupted by SIGTERM" in stderr or process.returncode == 143
    recovered = recover_staging(recorder.with_name(f".{recorder.name}.staging"))
    assert {checkpoint.name for checkpoint in recovered.checkpoints} == {
        "committed",
        "autobench.cancelled",
    }
    assert recovered.state.termination is not None
    assert recovered.state.termination.status is ExperimentStatus.CANCELLED


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signals are required")
def test_sigkill_preserves_only_the_last_committed_checkpoint(tmp_path: Path) -> None:
    process, recorder, ready = start_signal_worker(tmp_path, name="sigkill")
    wait_for_ready(process, ready)

    process.kill()
    process.communicate(timeout=15)

    assert process.returncode == -signal.SIGKILL
    recovered = recover_staging(recorder.with_name(f".{recorder.name}.staging"))
    assert [checkpoint.name for checkpoint in recovered.checkpoints] == ["committed"]
    assert recovered.state.termination is None


def start_signal_worker(
    tmp_path: Path,
    *,
    name: str,
) -> tuple[subprocess.Popen[str], Path, Path]:
    task_module = tmp_path / f"{name}_task.py"
    ready = tmp_path / f"{name}.ready"
    task_module.write_text(
        dedent(
            """
            import asyncio
            import os
            from pathlib import Path

            async def run(ctx, case):
                ctx.metric("started", 1)
                await ctx.checkpoint("committed")
                Path(os.environ["AUTOBENCH_TEST_READY"]).write_text("ready")
                await asyncio.Event().wait()
            """
        ),
        encoding="utf-8",
    )
    spec = tmp_path / f"{name}.yaml"
    spec.write_text(
        dedent(
            f"""
            benchmark:
              id: {name}
            dataset:
              cases:
                - id: case_1
            task:
              kind: python
              target: {name}_task:run
            variants:
              - id: variant_1
            """
        ),
        encoding="utf-8",
    )
    recorder = tmp_path / f"{name}-record"
    environment = os.environ.copy()
    python_paths = [str(tmp_path), str(Path(__file__).resolve().parents[1] / "src")]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path is not None:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["AUTOBENCH_TEST_READY"] = str(ready)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "autobench",
            "run",
            str(spec),
            "--record",
            str(recorder),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process, recorder, ready


def wait_for_ready(process: subprocess.Popen[str], ready: Path) -> None:
    deadline = time.monotonic() + 10
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if ready.exists():
        return
    stdout, stderr = process.communicate(timeout=5)
    raise AssertionError(
        f"Signal worker did not reach its committed checkpoint. stdout={stdout!r} stderr={stderr!r}"
    )
