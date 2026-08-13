from __future__ import annotations as _annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Coroutine, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from autobench.data.datasets import Case
from autobench.data.variants import Variant
from autobench.errors import ErrorRecord
from autobench.evaluation.derivation import derive_observations
from autobench.evaluation.scoring import (
    ScoreRecord,
    evaluate_scoring_specs,
    has_score_errors,
    score_records_to_observations,
)
from autobench.instrumentation.manager import InstrumentationManager
from autobench.instrumentation.models import InstrumentationError, Instrumentor
from autobench.instrumentation.registry import InstrumentorStatus, resolve_instrumentors
from autobench.metrics.observations import (
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import Semantic, SemanticRegistry
from autobench.protocol import EndReason, SpanStatus
from autobench.records.storage import capture_environment
from autobench.runtime.awaitables import run_sync, settle_task
from autobench.runtime.context import ContextEvidence, RunContext
from autobench.runtime.lifecycle import RunPhase
from autobench.runtime.models import (
    BenchmarkPlan,
    EvaluationStatus,
    ExecutionCorrelation,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTermination,
    MatrixRunSpec,
    RunResult,
    RunStatus,
    merge_execution_correlation,
)
from autobench.runtime.progress import (
    ProgressErrorHandler,
    ProgressErrorPolicy,
    ProgressEventKind,
    ProgressHandler,
    _ProgressDispatcher,
)
from autobench.runtime.tasks import TaskResult, TaskStatus, run_python_task
from autobench.tracking import track

if TYPE_CHECKING:  # pragma: no cover
    from autobench.evaluation.policies import PolicyResult
    from autobench.records.staging import PartialRunSnapshot, Recorder, RecordSession
    from autobench.spec import BenchmarkSpec

CleanupResultT = TypeVar("CleanupResultT")
RecordResultT = TypeVar("RecordResultT")

_CANCELLATION_CLEANUP_TIMEOUT_SECONDS = 5.0
_CANCELLATION_GRACE_SECONDS = 0.1


class _RecordOperations:
    def __init__(self, session: RecordSession) -> None:
        self.session = session
        self.tasks: set[asyncio.Task[Any]] = set()

    def start(
        self,
        operation: Coroutine[Any, Any, RecordResultT],
    ) -> asyncio.Task[RecordResultT]:
        task = asyncio.create_task(operation)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def execute(
        self,
        operation: Coroutine[Any, Any, RecordResultT],
        *,
        description: str,
    ) -> RecordResultT:
        task = self.start(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            error = await _finish_cleanup_task(
                task,
                cancel_on_timeout=False,
                description=description,
            )
            if error is not None:
                cancellation.add_note(f"{description.lower()} did not settle: {error}")
            raise

    async def drain(self) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        while self.tasks:
            active = tuple(self.tasks)
            results = await asyncio.gather(
                *(asyncio.shield(task) for task in active),
                return_exceptions=True,
            )
            errors.extend(result for result in results if isinstance(result, BaseException))
        return tuple(errors)


def expand_matrix(
    spec: BenchmarkSpec,
    *,
    experiment_id: str,
    correlation: ExecutionCorrelation | None = None,
) -> list[MatrixRunSpec]:
    return [
        MatrixRunSpec(
            run_id=stable_run_id(
                case=case, variant=variant, case_index=case_index, variant_index=variant_index
            ),
            benchmark_id=spec.benchmark.id,
            experiment_id=experiment_id,
            case_index=case_index,
            variant_index=variant_index,
            case=case,
            variant=variant,
            correlation=correlation,
        )
        for case_index, case in enumerate(spec.dataset.cases)
        for variant_index, variant in enumerate(spec.variants)
    ]


def stable_run_id(
    *,
    case: Case,
    variant: Variant,
    case_index: int,
    variant_index: int,
) -> str:
    case_slug = _slug(case.id)
    variant_slug = _slug(variant.id)
    return f"run_{case_index + 1:04d}_{variant_index + 1:04d}_{case_slug}__{variant_slug}"


def generate_experiment_id(benchmark_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"exp_{_slug(benchmark_id)}_{timestamp}"


async def run_benchmark_spec(
    spec: BenchmarkSpec,
    *,
    experiment_id: str | None = None,
    correlation: ExecutionCorrelation | None = None,
    concurrency_limit: int | None = 1,
    instrumentors: Sequence[Instrumentor] = (),
    recorder: Recorder | None = None,
    progress_handlers: Sequence[ProgressHandler] = (),
    progress_error_policy: ProgressErrorPolicy = ProgressErrorPolicy.STRICT,
    progress_error_handler: ProgressErrorHandler | None = None,
) -> ExperimentResult:
    from autobench.spec import build_benchmark_plan

    active_experiment_id = experiment_id or generate_experiment_id(spec.benchmark.id)
    active_correlation = merge_execution_correlation(
        spec.execution.correlation,
        correlation,
    )
    plan = build_benchmark_plan(spec)
    run_specs = expand_matrix(
        spec,
        experiment_id=active_experiment_id,
        correlation=active_correlation,
    )
    environment = capture_environment()
    session: RecordSession | None = None
    recording: _RecordOperations | None = None
    staged_run_ids: list[str] = []
    started_run_ids: list[str] = []
    completed_runs: dict[str, RunResult] = {}
    cancelled_run_ids: set[str] = set()
    policy_violations: list[PolicyResult] = []
    cross_run_derivation_complete = not bool(spec.post_derive)
    policies_complete = not bool(spec.policies)
    dispatcher = _ProgressDispatcher(
        progress_handlers,
        error_policy=progress_error_policy,
        error_handler=progress_error_handler,
    )
    if recorder is not None:
        from autobench.records.staging import ExperimentStart

        session = await recorder.open(
            ExperimentStart(
                experiment_id=active_experiment_id,
                benchmark_id=spec.benchmark.id,
                plan=plan,
                runs=tuple(run_specs),
                environment=environment,
                semantic_registry=spec.semantic_registry.model_copy(deep=True),
                report_spec_data=spec.reports.model_dump(mode="json"),
                spec_snapshot=spec.model_dump(mode="json"),
                spec_hash=_spec_hash(spec),
                requires_cross_run_derivation=bool(spec.post_derive),
                requires_policies=bool(spec.policies),
                correlation=active_correlation,
            )
        )
        recording = _RecordOperations(session)

    try:
        await dispatcher.emit(
            ProgressEventKind.BENCHMARK_STARTED,
            f"Benchmark {spec.benchmark.id} started.",
            benchmark_id=spec.benchmark.id,
            experiment_id=active_experiment_id,
            case_count=plan.case_count,
            variant_count=plan.variant_count,
            planned_run_count=plan.planned_run_count,
        )
        configured, instrumentation_diagnostics = resolve_instrumentors(
            spec.instrumentation,
            reserved_ids={instrumentor.info.id for instrumentor in instrumentors},
        )
        active_instrumentors = [*configured, *instrumentors]
        instrumentor_ids = [instrumentor.info.id for instrumentor in active_instrumentors]
        duplicate_ids = sorted(
            instrumentor_id
            for instrumentor_id in set(instrumentor_ids)
            if instrumentor_ids.count(instrumentor_id) > 1
        )
        if duplicate_ids:
            raise InstrumentationError(
                f"duplicate instrumentors configured: {', '.join(duplicate_ids)}"
            )

        with InstrumentationManager() as instrumentation:
            for instrumentor in active_instrumentors:
                instrumentation.install(instrumentor)

            if concurrency_limit is None or concurrency_limit <= 1:
                runs = []
                for run_spec in run_specs:
                    run = await _run_matrix_item_observed(
                        spec,
                        run_spec,
                        dispatcher=dispatcher,
                        started_run_ids=started_run_ids,
                        completed_runs=completed_runs,
                        cancelled_run_ids=cancelled_run_ids,
                        instrumentation_diagnostics=instrumentation_diagnostics,
                        recording=recording,
                        staged_run_ids=staged_run_ids,
                    )
                    runs.append(run)
            else:
                semaphore = asyncio.Semaphore(concurrency_limit)
                runs = await _run_concurrent_matrix(
                    [
                        _run_matrix_item_limited(
                            spec,
                            run_spec,
                            semaphore,
                            dispatcher=dispatcher,
                            started_run_ids=started_run_ids,
                            completed_runs=completed_runs,
                            cancelled_run_ids=cancelled_run_ids,
                            instrumentation_diagnostics=instrumentation_diagnostics,
                            recording=recording,
                            staged_run_ids=staged_run_ids,
                        )
                        for run_spec in run_specs
                    ],
                )

        result = ExperimentResult(
            experiment_id=active_experiment_id,
            benchmark_id=spec.benchmark.id,
            plan=plan,
            runs=runs,
            environment=environment,
            termination=ExperimentTermination(
                status=ExperimentStatus.COMPLETED,
                cross_run_derivation_complete=cross_run_derivation_complete,
                policies_complete=policies_complete,
                planned_run_ids=tuple(run_spec.run_id for run_spec in run_specs),
                recorded_run_ids=tuple(run.run_id for run in runs),
            ),
            report_spec_data=spec.reports.model_dump(mode="json"),
            semantic_registry=spec.semantic_registry.model_copy(deep=True),
            spec_snapshot=spec.model_dump(mode="json"),
            spec_hash=_spec_hash(spec),
            correlation=active_correlation,
        )
        if spec.post_derive:
            from autobench.evaluation.comparison import derive_experiment_observations

            result = derive_experiment_observations(
                spec.post_derive,
                result=result,
                registry=spec.semantic_registry,
            )
            cross_run_derivation_complete = True
        if spec.policies:
            from autobench.evaluation.policies import apply_policies, evaluate_policies

            policy_violations = [
                policy
                for policy in evaluate_policies(
                    spec.policies,
                    result=result,
                    registry=spec.semantic_registry,
                )
                if not policy.passed
            ]
            result = apply_policies(
                spec.policies,
                result=result,
                registry=spec.semantic_registry,
            )
            policies_complete = True
        result = _refresh_run_statuses(result, registry=spec.semantic_registry)
        result = result.model_copy(
            update={
                "termination": result.termination.model_copy(
                    update={
                        "cross_run_derivation_complete": cross_run_derivation_complete,
                        "policies_complete": policies_complete,
                    }
                )
            }
        )
        completed_runs.update((run.run_id, run) for run in result.runs)
        if recording is not None:
            await recording.execute(
                recording.session.finish(result),
                description="Recording finalization",
            )
            await recording.execute(
                recording.session.close(),
                description="Recording close",
            )
    except BaseException as exc:
        if recording is not None:
            abort_error = await _finish_cleanup_task(
                asyncio.create_task(
                    _abort_recording(
                        recording,
                        run_specs=run_specs,
                        staged_run_ids=staged_run_ids,
                        cross_run_derivation_complete=cross_run_derivation_complete,
                        policies_complete=policies_complete,
                        failure=exc,
                    )
                ),
                cancel_on_timeout=False,
                description="Recording abort and close",
            )
            if abort_error is not None:
                exc.add_note(f"recording abort failed: {abort_error}")
        progress_error = await _finish_cleanup_task(
            asyncio.create_task(
                _emit_completion_progress(
                    dispatcher,
                    benchmark_id=spec.benchmark.id,
                    experiment_id=active_experiment_id,
                    run_specs=run_specs,
                    started_run_ids=started_run_ids,
                    completed_runs=completed_runs,
                    cancelled_run_ids=cancelled_run_ids,
                    policy_violations=policy_violations,
                    experiment_status=(
                        ExperimentStatus.CANCELLED
                        if isinstance(exc, asyncio.CancelledError)
                        else ExperimentStatus.ABORTED
                    ),
                    error=exc,
                )
            )
        )
        if progress_error is not None:
            exc.add_note(f"progress terminal delivery failed: {progress_error}")
        dispatch_error = dispatcher.error()
        if dispatch_error is not None:
            exc.add_note(str(dispatch_error))
        raise
    await _emit_completion_progress(
        dispatcher,
        benchmark_id=spec.benchmark.id,
        experiment_id=active_experiment_id,
        run_specs=run_specs,
        started_run_ids=started_run_ids,
        completed_runs=completed_runs,
        cancelled_run_ids=cancelled_run_ids,
        policy_violations=policy_violations,
        experiment_status=ExperimentStatus.COMPLETED,
    )
    dispatcher.raise_if_failed()
    return result


def run_benchmark_path(
    path: Path,
    *,
    experiment_id: str | None = None,
    correlation: ExecutionCorrelation | None = None,
    concurrency_limit: int | None = 1,
    instrumentors: Sequence[Instrumentor] = (),
    recorder: Recorder | None = None,
    progress_handlers: Sequence[ProgressHandler] = (),
    progress_error_policy: ProgressErrorPolicy = ProgressErrorPolicy.STRICT,
    progress_error_handler: ProgressErrorHandler | None = None,
) -> ExperimentResult:
    from autobench.spec import load_benchmark_spec

    spec = load_benchmark_spec(path)
    return run_sync(
        run_benchmark_spec(
            spec,
            experiment_id=experiment_id,
            correlation=correlation,
            concurrency_limit=concurrency_limit,
            instrumentors=instrumentors,
            recorder=recorder,
            progress_handlers=progress_handlers,
            progress_error_policy=progress_error_policy,
            progress_error_handler=progress_error_handler,
        )
    )


async def _run_matrix_item_limited(
    spec: BenchmarkSpec,
    run_spec: MatrixRunSpec,
    semaphore: asyncio.Semaphore,
    *,
    dispatcher: _ProgressDispatcher,
    started_run_ids: list[str],
    completed_runs: dict[str, RunResult],
    cancelled_run_ids: set[str],
    instrumentation_diagnostics: Sequence[InstrumentorStatus] = (),
    recording: _RecordOperations | None,
    staged_run_ids: list[str],
) -> RunResult:
    async with semaphore:
        return await _run_matrix_item_observed(
            spec,
            run_spec,
            dispatcher=dispatcher,
            started_run_ids=started_run_ids,
            completed_runs=completed_runs,
            cancelled_run_ids=cancelled_run_ids,
            instrumentation_diagnostics=instrumentation_diagnostics,
            recording=recording,
            staged_run_ids=staged_run_ids,
        )


async def _run_matrix_item_observed(
    spec: BenchmarkSpec,
    run_spec: MatrixRunSpec,
    *,
    dispatcher: _ProgressDispatcher,
    started_run_ids: list[str],
    completed_runs: dict[str, RunResult],
    cancelled_run_ids: set[str],
    instrumentation_diagnostics: Sequence[InstrumentorStatus],
    recording: _RecordOperations | None,
    staged_run_ids: list[str],
) -> RunResult:
    started_run_ids.append(run_spec.run_id)
    try:
        await dispatcher.emit(
            ProgressEventKind.RUN_STARTED,
            f"Run {run_spec.run_id} started.",
            benchmark_id=run_spec.benchmark_id,
            experiment_id=run_spec.experiment_id,
            run_id=run_spec.run_id,
            case_id=run_spec.case.id,
            variant_id=run_spec.variant.id,
        )
        run = await _run_matrix_item(
            spec,
            run_spec,
            instrumentation_diagnostics=instrumentation_diagnostics,
            recording=recording,
        )
    except asyncio.CancelledError:
        cancelled_run_ids.add(run_spec.run_id)
        raise
    completed_runs[run.run_id] = run
    if recording is not None:
        from autobench.records.staging import ExecutionSnapshot

        stage_task = recording.start(recording.session.stage(ExecutionSnapshot.from_result(run)))

        def retain_staged_run(completed: asyncio.Task[None]) -> None:
            if completed.cancelled() or completed.exception() is not None:
                return
            staged_run_ids.append(run.run_id)

        stage_task.add_done_callback(retain_staged_run)
        try:
            await asyncio.shield(stage_task)
        except asyncio.CancelledError as cancellation:
            stage_error = await _finish_cleanup_task(
                stage_task,
                cancel_on_timeout=False,
                description=f"Run {run.run_id} staging",
            )
            if stage_error is not None:
                cancellation.add_note(f"run staging did not settle: {stage_error}")
            raise
    return run


async def _run_matrix_item(
    spec: BenchmarkSpec,
    run_spec: MatrixRunSpec,
    *,
    instrumentation_diagnostics: Sequence[InstrumentorStatus] = (),
    recording: _RecordOperations | None = None,
) -> RunResult:
    ctx = RunContext(
        benchmark_id=run_spec.benchmark_id,
        case=run_spec.case,
        variant=run_spec.variant,
        run_id=run_spec.run_id,
        experiment_id=run_spec.experiment_id,
        capture_policy=spec.capture,
    )
    if recording is not None and recording.session.artifact_sink is not None:
        ctx.bind_artifact_sink(recording.session.artifact_sink)
    if recording is not None:

        async def persist_checkpoint(
            name: str,
            phase: RunPhase,
            task_output: Any,
            evidence: ContextEvidence,
        ) -> None:
            snapshot = _build_partial_snapshot(
                run_spec,
                name=name,
                phase=phase,
                task_output=task_output,
                evidence=evidence,
            )
            await _persist_explicit_checkpoint(recording, snapshot)

        ctx.bind_checkpoint(persist_checkpoint)
    _seed_variant_evidence(ctx)
    for status in instrumentation_diagnostics:
        ctx.diagnostic(
            "instrumentation.skipped",
            status.compatibility.status.value,
            semantic_type=Semantic.DIAGNOSTIC_EVENT,
            tags={
                "instrumentor": status.name,
                "extra": status.extra,
                "diagnostics": list(
                    status.compatibility.conflicts + status.compatibility.diagnostics
                ),
            },
            source=ObservationSource.INSTRUMENTATION,
        )

    if spec.task is None:
        error = ErrorRecord(
            error_type="TaskSkipped",
            message="No task is defined for this benchmark.",
        )
        task_result = TaskResult(status=TaskStatus.SKIPPED, error=error)
        task_result.end_reason = EndReason.DEFERRED
        ctx.set_phase(RunPhase.FINALIZING)
        ctx.finalize(status=SpanStatus.UNSET, reason=EndReason.DEFERRED)
        return _build_run_result(
            run_spec,
            task_result,
            ctx=ctx,
            error=error,
            registry=spec.semantic_registry,
        )

    if spec.task.kind != "python":
        error = ErrorRecord(
            error_type="UnsupportedTaskKind",
            message=f"Unsupported task kind: {spec.task.kind}",
        )
        ctx.error(error)
        task_result = TaskResult(
            status=TaskStatus.ERRORED,
            end_reason=EndReason.FAILED,
            error=error,
            errors=list(ctx.errors),
        )
        ctx.set_phase(RunPhase.FINALIZING)
        ctx.finalize(status=SpanStatus.ERROR, reason=EndReason.FAILED)
        return _build_run_result(
            run_spec,
            task_result,
            ctx=ctx,
            error=error,
            registry=spec.semantic_registry,
        )

    ctx.set_phase(RunPhase.EXECUTING)
    try:
        task_result = await run_python_task(
            spec.task.target,
            ctx=ctx,
            case=run_spec.case,
            search_paths=spec.task.module_search_paths,
        )
    except asyncio.CancelledError as exc:
        await _propagate_run_cancellation(
            ctx,
            run_spec,
            recording=recording,
            cancellation=exc,
        )
    except Exception as exc:
        error = ctx.error(exc)
        task_result = TaskResult(
            status=TaskStatus.ERRORED,
            end_reason=EndReason.FAILED,
            error=error,
            errors=list(ctx.errors),
            observations=list(ctx.observations),
            spans=list(ctx.spans),
            artifacts=list(ctx.artifacts),
        )
        ctx.set_phase(RunPhase.FINALIZING)
        ctx.finalize(
            status=SpanStatus.ERROR,
            reason=EndReason.FAILED,
            output=task_result.output,
        )
        return _build_run_result(
            run_spec,
            task_result,
            ctx=ctx,
            error=error,
            registry=spec.semantic_registry,
        )

    ctx.retain_task_output(task_result.output)
    ctx.set_phase(RunPhase.SCORING)
    try:
        scores = await evaluate_scoring_specs(spec.scoring, ctx=ctx, task_result=task_result)
        for observation in score_records_to_observations(scores, ctx=ctx):
            ctx.record_observation(observation)
    except asyncio.CancelledError as exc:
        await _propagate_run_cancellation(
            ctx,
            run_spec,
            recording=recording,
            cancellation=exc,
        )
    except Exception as exc:
        return _failed_post_task_run(
            spec,
            run_spec,
            ctx=ctx,
            task_result=task_result,
            error=exc,
        )

    ctx.set_phase(RunPhase.DERIVING)
    try:
        for observation in derive_observations(
            spec.derive,
            ctx=ctx,
            observations=ctx.observations,
            registry=spec.semantic_registry,
        ):
            ctx.record_observation(observation)
    except asyncio.CancelledError as exc:
        await _propagate_run_cancellation(
            ctx,
            run_spec,
            recording=recording,
            cancellation=exc,
        )
    except Exception as exc:
        return _failed_post_task_run(
            spec,
            run_spec,
            ctx=ctx,
            task_result=task_result,
            error=exc,
        )
    error = task_result.error
    if error is None and has_score_errors(scores):
        error = next(score.error for score in scores if score.error is not None)
        ctx.error(error)

    status = SpanStatus.OK
    reason = EndReason.COMPLETED
    partial = False
    if task_result.status in {TaskStatus.FAILED, TaskStatus.ERRORED} or error is not None:
        status = SpanStatus.ERROR
        reason = EndReason.FAILED
    if task_result.error is not None and task_result.error.error_type == "TimeoutError":
        reason = EndReason.TIMEOUT
        partial = True
    task_result.end_reason = reason
    task_result.partial = partial
    ctx.set_phase(RunPhase.FINALIZING)
    ctx.finalize(
        status=status,
        reason=reason,
        partial=partial,
        output=task_result.output,
    )
    task_result.errors = list(ctx.errors)
    task_result.observations = list(ctx.observations)
    task_result.spans = list(ctx.spans)
    task_result.artifacts = list(ctx.artifacts)

    return _build_run_result(
        run_spec,
        task_result,
        ctx=ctx,
        scores=scores,
        error=error,
        registry=spec.semantic_registry,
    )


def _failed_post_task_run(
    spec: BenchmarkSpec,
    run_spec: MatrixRunSpec,
    *,
    ctx: RunContext,
    task_result: TaskResult,
    error: Exception,
) -> RunResult:
    record = ctx.error(error)
    task_result.status = TaskStatus.ERRORED
    task_result.end_reason = EndReason.FAILED
    task_result.error = record
    task_result.errors = list(ctx.errors)
    task_result.observations = list(ctx.observations)
    task_result.spans = list(ctx.spans)
    task_result.artifacts = list(ctx.artifacts)
    ctx.set_phase(RunPhase.FINALIZING)
    ctx.finalize(
        status=SpanStatus.ERROR,
        reason=EndReason.FAILED,
        output=task_result.output,
    )
    return _build_run_result(
        run_spec,
        task_result,
        ctx=ctx,
        error=record,
        registry=spec.semantic_registry,
    )


def _build_partial_snapshot(
    run_spec: MatrixRunSpec,
    *,
    name: str,
    phase: RunPhase,
    task_output: Any,
    evidence: ContextEvidence,
) -> PartialRunSnapshot:
    from autobench.records.staging import PartialRunSnapshot

    return PartialRunSnapshot(
        run_id=run_spec.run_id,
        experiment_id=run_spec.experiment_id,
        benchmark_id=run_spec.benchmark_id,
        case_id=run_spec.case.id,
        variant_id=run_spec.variant.id,
        name=name,
        phase=phase,
        task_output=task_output,
        observations=evidence.observations,
        spans=evidence.spans,
        artifacts=evidence.artifacts,
        errors=evidence.errors,
        asset_versions=evidence.asset_versions,
        asset_uses=evidence.asset_uses,
        source_snapshots=evidence.source_snapshots,
        extensions=evidence.extensions,
        trace=evidence.trace,
        signal_sequence_watermark=evidence.signal_sequence_watermark,
        correlation=run_spec.correlation,
    )


async def _persist_explicit_checkpoint(
    recording: _RecordOperations,
    snapshot: PartialRunSnapshot,
) -> None:
    await recording.execute(
        recording.session.checkpoint(snapshot),
        description=f"Checkpoint {snapshot.run_id}:{snapshot.name}",
    )


async def _propagate_run_cancellation(
    ctx: RunContext,
    run_spec: MatrixRunSpec,
    *,
    recording: _RecordOperations | None,
    cancellation: asyncio.CancelledError,
) -> NoReturn:
    cancelled_phase = ctx.phase
    task_output = ctx.checkpoint_output
    ctx.error(cancellation)
    ctx.set_phase(RunPhase.FINALIZING)
    ctx.finalize(
        status=SpanStatus.ERROR,
        reason=EndReason.CANCELLED,
        partial=True,
        output=task_output,
    )
    if recording is not None:
        evidence = ctx.snapshot_evidence()
        snapshot = _build_partial_snapshot(
            run_spec,
            name="autobench.cancelled",
            phase=cancelled_phase,
            task_output=task_output,
            evidence=evidence,
        )
        cleanup_error = await _finish_cleanup_task(
            recording.start(recording.session.checkpoint(snapshot)),
            cancel_on_timeout=False,
            description=f"Cancellation checkpoint {snapshot.run_id}",
        )
        if cleanup_error is not None:
            cancellation.add_note(f"cancellation checkpoint failed: {cleanup_error}")
    raise cancellation


async def _run_concurrent_matrix(
    coroutines: list[Coroutine[Any, Any, RunResult]],
) -> list[RunResult]:
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException as exc:
        active_tasks = [task for task in tasks if not task.done()]
        for task in active_tasks:
            task.cancel()
        for task in active_tasks:
            cleanup_error = await _finish_cleanup_task(task)
            if cleanup_error is not None and not isinstance(cleanup_error, asyncio.CancelledError):
                exc.add_note(f"concurrent run cleanup failed: {cleanup_error}")
        raise


async def _abort_recording(
    recording: _RecordOperations,
    *,
    run_specs: Sequence[MatrixRunSpec],
    staged_run_ids: list[str],
    cross_run_derivation_complete: bool,
    policies_complete: bool,
    failure: BaseException,
) -> None:
    for operation_error in await recording.drain():
        failure.add_note(f"recording operation failed before abort: {operation_error}")
    planned_ids = tuple(run_spec.run_id for run_spec in run_specs)
    recorded = tuple(run_id for run_id in planned_ids if run_id in staged_run_ids)
    missing = tuple(run_id for run_id in planned_ids if run_id not in staged_run_ids)
    termination = ExperimentTermination(
        status=(
            ExperimentStatus.CANCELLED
            if isinstance(failure, asyncio.CancelledError)
            else ExperimentStatus.ABORTED
        ),
        partial=True,
        cross_run_derivation_complete=cross_run_derivation_complete,
        policies_complete=policies_complete,
        planned_run_ids=planned_ids,
        recorded_run_ids=recorded,
        missing_run_ids=missing,
        error=ErrorRecord.from_exception(failure),
    )
    abort_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        await recording.session.abort(termination)
    except BaseException as error:
        abort_error = error
    try:
        await recording.session.close()
    except BaseException as error:
        close_error = error
    if abort_error is not None:
        failure.add_note(f"recording abort failed: {abort_error}")
    if close_error is not None:
        failure.add_note(f"recording close failed: {close_error}")


async def _emit_completion_progress(
    dispatcher: _ProgressDispatcher,
    *,
    benchmark_id: str,
    experiment_id: str,
    run_specs: Sequence[MatrixRunSpec],
    started_run_ids: Sequence[str],
    completed_runs: dict[str, RunResult],
    cancelled_run_ids: set[str],
    policy_violations: Sequence[PolicyResult],
    experiment_status: ExperimentStatus,
    error: BaseException | None = None,
) -> None:
    for violation in policy_violations:
        run = completed_runs[violation.run_id]
        await dispatcher.emit(
            ProgressEventKind.POLICY_VIOLATION,
            f"Policy {violation.policy_name} failed for run {violation.run_id}.",
            benchmark_id=run.benchmark_id,
            experiment_id=run.experiment_id,
            run_id=violation.run_id,
            case_id=violation.case_id,
            variant_id=violation.variant_id,
            policy_name=violation.policy_name,
            metric=violation.metric,
            actual=violation.actual,
            reason=violation.reason,
        )

    started = set(started_run_ids)
    for run_spec in run_specs:
        if run_spec.run_id not in started:
            continue
        run = completed_runs.get(run_spec.run_id)
        if run is None:
            run_status = (
                RunStatus.CANCELLED if run_spec.run_id in cancelled_run_ids else RunStatus.ERRORED
            )
            partial = True
            end_reason = (
                EndReason.CANCELLED if run_status is RunStatus.CANCELLED else EndReason.FAILED
            )
        else:
            run_status = run.status
            partial = run.partial
            end_reason = run.end_reason
        await dispatcher.emit(
            ProgressEventKind.RUN_FINISHED,
            f"Run {run_spec.run_id} finished with status {run_status}.",
            run_status=run_status,
            benchmark_id=run_spec.benchmark_id,
            experiment_id=run_spec.experiment_id,
            run_id=run_spec.run_id,
            case_id=run_spec.case.id,
            variant_id=run_spec.variant.id,
            partial=partial,
            end_reason=end_reason.value,
        )

    await dispatcher.emit(
        ProgressEventKind.BENCHMARK_FINISHED,
        f"Benchmark finished with status {experiment_status}.",
        experiment_status=experiment_status,
        benchmark_id=benchmark_id,
        experiment_id=experiment_id,
        partial=experiment_status is not ExperimentStatus.COMPLETED,
        error_type=None if error is None else type(error).__name__,
    )


async def _finish_cleanup_task(
    task: asyncio.Task[CleanupResultT],
    *,
    timeout_seconds: float | None = None,
    cancellation_grace_seconds: float = _CANCELLATION_GRACE_SECONDS,
    cancel_on_timeout: bool = True,
    description: str = "Cleanup",
) -> BaseException | None:
    return await settle_task(
        task,
        timeout_seconds=(
            _CANCELLATION_CLEANUP_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        ),
        cancellation_grace_seconds=cancellation_grace_seconds,
        cancel_on_timeout=cancel_on_timeout,
        description=description,
    )


def _build_run_result(
    run_spec: MatrixRunSpec,
    task_result: TaskResult,
    *,
    ctx: RunContext,
    scores: list[ScoreRecord] | None = None,
    error: ErrorRecord | None,
    registry: SemanticRegistry,
) -> RunResult:
    evidence = ctx.snapshot_evidence()
    recorded_task_result = task_result.model_copy(
        update={
            "errors": list(evidence.errors),
            "observations": list(evidence.observations),
            "spans": list(evidence.spans),
            "artifacts": list(evidence.artifacts),
        },
        deep=True,
    )
    active_scores = scores or []
    evaluation_status = _evaluation_status_from_task_result(
        recorded_task_result,
        scores=active_scores,
        registry=registry,
    )
    return RunResult(
        run_id=run_spec.run_id,
        benchmark_id=run_spec.benchmark_id,
        experiment_id=run_spec.experiment_id,
        case_id=run_spec.case.id,
        variant_id=run_spec.variant.id,
        status=_run_status_from_task_result(
            recorded_task_result,
            evaluation_status=evaluation_status,
        ),
        evaluation_status=evaluation_status,
        partial=recorded_task_result.partial,
        end_reason=recorded_task_result.end_reason,
        case=run_spec.case,
        task_result=recorded_task_result,
        scores=active_scores,
        factors=list(run_spec.variant.factors),
        asset_versions=list(evidence.asset_versions),
        asset_uses=list(evidence.asset_uses),
        error=error,
        trace=evidence.trace,
        source_snapshots=evidence.source_snapshots,
        extensions=evidence.extensions,
        correlation=run_spec.correlation,
    )


def _evaluation_status_from_task_result(
    task_result: TaskResult,
    *,
    scores: list[ScoreRecord],
    registry: SemanticRegistry,
) -> EvaluationStatus:
    if task_result.status is TaskStatus.SKIPPED:
        return EvaluationStatus.SKIPPED
    if task_result.status is TaskStatus.CANCELLED:
        return EvaluationStatus.NOT_EVALUATED
    if task_result.status is TaskStatus.FAILED or task_result.status is TaskStatus.ERRORED:
        return EvaluationStatus.NOT_EVALUATED
    if has_score_errors(scores):
        return EvaluationStatus.ERRORED
    if _has_failed_outcome(task_result.observations, registry=registry):
        return EvaluationStatus.FAILED
    return EvaluationStatus.PASSED


def _run_status_from_task_result(
    task_result: TaskResult,
    *,
    evaluation_status: EvaluationStatus,
) -> RunStatus:
    if task_result.status is TaskStatus.FAILED:
        return RunStatus.FAILED
    if task_result.status is TaskStatus.SKIPPED:
        return RunStatus.SKIPPED
    if task_result.status is TaskStatus.CANCELLED:
        return RunStatus.CANCELLED
    if task_result.status is TaskStatus.ERRORED:
        return RunStatus.ERRORED
    if evaluation_status is EvaluationStatus.ERRORED:
        return RunStatus.ERRORED
    if evaluation_status is EvaluationStatus.FAILED:
        return RunStatus.FAILED
    if evaluation_status is EvaluationStatus.SKIPPED:
        return RunStatus.SKIPPED
    return RunStatus.PASSED


def _has_failed_outcome(
    observations: list[Observation],
    *,
    registry: SemanticRegistry,
) -> bool:
    for observation in observations:
        if not isinstance(observation.value, bool):
            continue
        if observation.kind is not ObservationKind.METRIC and not (
            observation.kind is ObservationKind.EVENT
            and observation.normalized_semantic_type(registry) == "policy.result"
        ):
            continue
        if observation.normalized_semantic_type(registry) == Semantic.RESULT_SUCCESS:
            if not observation.value:
                return True
            continue
        if observation.role is ObservationRole.CONSTRAINT and not observation.value:
            return True
    return False


def _seed_variant_evidence(ctx: RunContext) -> None:
    for factor in ctx.variant.factors:
        observation_value = factor.value
        observation_tags: dict[str, Any] = {}
        try:
            asset_version = ctx.attach_tracked_asset(factor.value, registry=track)
        except KeyError:
            asset_version = None
        else:
            observation_value = asset_version.version
            observation_tags["asset_id"] = asset_version.asset_id
            observation_tags["asset_version"] = asset_version.version
        ctx.factor_observation(
            factor.name,
            observation_value,
            semantic_type=factor.semantic_type,
            tags=observation_tags,
            source=ObservationSource.VARIANT,
        )


def _refresh_run_statuses(
    result: ExperimentResult,
    *,
    registry: SemanticRegistry,
) -> ExperimentResult:
    runs = [
        run.model_copy(
            update={
                "evaluation_status": _evaluation_status_from_task_result(
                    run.task_result,
                    scores=run.scores,
                    registry=registry,
                ),
                "status": _run_status_from_task_result(
                    run.task_result,
                    evaluation_status=_evaluation_status_from_task_result(
                        run.task_result,
                        scores=run.scores,
                        registry=registry,
                    ),
                ),
            }
        )
        for run in result.runs
    ]
    return result.model_copy(update={"runs": runs})


def _spec_hash(spec: BenchmarkSpec) -> str:
    payload = spec.model_dump(mode="json")
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unnamed"


__all__ = (
    "BenchmarkPlan",
    "EvaluationStatus",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentTermination",
    "MatrixRunSpec",
    "RunResult",
    "RunStatus",
    "expand_matrix",
    "generate_experiment_id",
    "run_benchmark_path",
    "run_benchmark_spec",
    "stable_run_id",
)
