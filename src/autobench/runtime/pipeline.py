from __future__ import annotations as _annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from autobench.data.datasets import Case
from autobench.data.variants import FactorValue, Variant
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
from autobench.metrics.mappings import SourceSnapshot
from autobench.metrics.observations import (
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, Semantic, SemanticRegistry
from autobench.protocol import EndReason, SpanStatus
from autobench.protocol.traces import Trace
from autobench.records.storage import EnvironmentMetadata, capture_environment
from autobench.runtime.awaitables import run_sync
from autobench.runtime.context import RunContext
from autobench.runtime.tasks import TaskResult, TaskStatus, run_python_task
from autobench.tracking import AssetUse, AssetVersion, track

if TYPE_CHECKING:  # pragma: no cover
    from autobench.spec import BenchmarkSpec


class BenchmarkPlan(BaseModel):
    benchmark_id: str
    dataset_id: str | None = None
    dataset_version: str | None = None
    dataset_hash: str | None = None
    case_ids: tuple[str, ...] = ()
    case_count: int
    variant_count: int
    planned_run_count: int
    warnings: list[str] = Field(default_factory=list)


class RunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"
    NOT_EVALUATED = "not_evaluated"


class MatrixRunSpec(BaseModel):
    run_id: str
    benchmark_id: str
    experiment_id: str
    case_index: int
    variant_index: int
    case: Case
    variant: Variant


class RunResult(BaseModel):
    run_id: str
    benchmark_id: str
    experiment_id: str
    case_id: str
    variant_id: str
    status: RunStatus
    evaluation_status: EvaluationStatus
    case: Case
    task_result: TaskResult
    scores: list[ScoreRecord] = Field(default_factory=list)
    factors: list[FactorValue] = Field(default_factory=list)
    asset_versions: list[AssetVersion] = Field(default_factory=list)
    asset_uses: list[AssetUse] = Field(default_factory=list)
    parent_run_id: str | None = None
    error: ErrorRecord | None = None
    trace: Trace | None = None
    source_snapshots: tuple[SourceSnapshot, ...] = ()


class ExperimentResult(BaseModel):
    experiment_id: str
    benchmark_id: str
    plan: BenchmarkPlan
    runs: list[RunResult]
    environment: EnvironmentMetadata
    report_spec_data: dict[str, Any] | None = None
    semantic_registry: SemanticRegistry = Field(
        default_factory=lambda: DEFAULT_SEMANTIC_REGISTRY.model_copy(deep=True)
    )
    spec_snapshot: dict[str, Any] | None = None
    spec_hash: str | None = None

    @property
    def total_count(self) -> int:
        return len(self.runs)

    @property
    def passed_count(self) -> int:
        return self.count_status(RunStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return self.count_status(RunStatus.FAILED)

    @property
    def errored_count(self) -> int:
        return self.count_status(RunStatus.ERRORED)

    @property
    def skipped_count(self) -> int:
        return self.count_status(RunStatus.SKIPPED)

    def count_status(self, status: RunStatus) -> int:
        return sum(1 for run in self.runs if run.status is status)


def expand_matrix(
    spec: BenchmarkSpec,
    *,
    experiment_id: str,
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
    concurrency_limit: int | None = 1,
    instrumentors: Sequence[Instrumentor] = (),
) -> ExperimentResult:
    from autobench.spec import build_benchmark_plan

    active_experiment_id = experiment_id or generate_experiment_id(spec.benchmark.id)
    plan = build_benchmark_plan(spec)
    run_specs = expand_matrix(spec, experiment_id=active_experiment_id)
    environment = capture_environment()

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
            runs = [
                await _run_matrix_item(
                    spec,
                    run_spec,
                    instrumentation_diagnostics=instrumentation_diagnostics,
                )
                for run_spec in run_specs
            ]
        else:
            semaphore = asyncio.Semaphore(concurrency_limit)
            runs = await asyncio.gather(
                *[
                    _run_matrix_item_limited(
                        spec,
                        run_spec,
                        semaphore,
                        instrumentation_diagnostics=instrumentation_diagnostics,
                    )
                    for run_spec in run_specs
                ]
            )

    result = ExperimentResult(
        experiment_id=active_experiment_id,
        benchmark_id=spec.benchmark.id,
        plan=plan,
        runs=runs,
        environment=environment,
        report_spec_data=spec.reports.model_dump(mode="json"),
        semantic_registry=spec.semantic_registry.model_copy(deep=True),
        spec_snapshot=spec.model_dump(mode="json"),
        spec_hash=_spec_hash(spec),
    )
    if spec.post_derive:
        from autobench.evaluation.comparison import derive_experiment_observations

        result = derive_experiment_observations(
            spec.post_derive,
            result=result,
            registry=spec.semantic_registry,
        )
    if spec.policies:
        from autobench.evaluation.policies import apply_policies

        result = apply_policies(
            spec.policies,
            result=result,
            registry=spec.semantic_registry,
        )
    return _refresh_run_statuses(result, registry=spec.semantic_registry)


def run_benchmark_path(
    path: Path,
    *,
    experiment_id: str | None = None,
    concurrency_limit: int | None = 1,
    instrumentors: Sequence[Instrumentor] = (),
) -> ExperimentResult:
    from autobench.spec import load_benchmark_spec

    spec = load_benchmark_spec(path)
    return run_sync(
        run_benchmark_spec(
            spec,
            experiment_id=experiment_id,
            concurrency_limit=concurrency_limit,
            instrumentors=instrumentors,
        )
    )


async def _run_matrix_item_limited(
    spec: BenchmarkSpec,
    run_spec: MatrixRunSpec,
    semaphore: asyncio.Semaphore,
    *,
    instrumentation_diagnostics: Sequence[InstrumentorStatus] = (),
) -> RunResult:
    async with semaphore:
        return await _run_matrix_item(
            spec,
            run_spec,
            instrumentation_diagnostics=instrumentation_diagnostics,
        )


async def _run_matrix_item(
    spec: BenchmarkSpec,
    run_spec: MatrixRunSpec,
    *,
    instrumentation_diagnostics: Sequence[InstrumentorStatus] = (),
) -> RunResult:
    ctx = RunContext(
        benchmark_id=run_spec.benchmark_id,
        case=run_spec.case,
        variant=run_spec.variant,
        run_id=run_spec.run_id,
        experiment_id=run_spec.experiment_id,
        capture_policy=spec.capture,
    )
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
        ctx.finalize(status=SpanStatus.UNSET, reason=EndReason.DEFERRED)
        return _build_run_result(
            run_spec,
            task_result,
            ctx=ctx,
            asset_versions=list(ctx.asset_versions),
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
            error=error,
            errors=list(ctx.errors),
        )
        ctx.finalize(status=SpanStatus.ERROR, reason=EndReason.FAILED)
        return _build_run_result(
            run_spec,
            task_result,
            ctx=ctx,
            asset_versions=list(ctx.asset_versions),
            error=error,
            registry=spec.semantic_registry,
        )

    try:
        task_result = await run_python_task(
            spec.task.target,
            ctx=ctx,
            case=run_spec.case,
            search_paths=spec.task.module_search_paths,
        )
    except asyncio.CancelledError as exc:
        ctx.error(exc)
        ctx.finalize(
            status=SpanStatus.ERROR,
            reason=EndReason.CANCELLED,
            partial=True,
        )
        raise
    except Exception as exc:
        error = ctx.error(exc)
        task_result = TaskResult(
            status=TaskStatus.ERRORED,
            error=error,
            errors=list(ctx.errors),
            observations=list(ctx.observations),
            spans=list(ctx.spans),
            artifacts=list(ctx.artifacts),
        )
        ctx.finalize(
            status=SpanStatus.ERROR,
            reason=EndReason.FAILED,
            output=task_result.output,
        )
        return _build_run_result(
            run_spec,
            task_result,
            ctx=ctx,
            asset_versions=list(ctx.asset_versions),
            error=error,
            registry=spec.semantic_registry,
        )

    try:
        scores = await evaluate_scoring_specs(spec.scoring, ctx=ctx, task_result=task_result)
        for observation in score_records_to_observations(scores, ctx=ctx):
            ctx.record_observation(observation)
        for observation in derive_observations(
            spec.derive,
            ctx=ctx,
            observations=ctx.observations,
            registry=spec.semantic_registry,
        ):
            ctx.record_observation(observation)
    except asyncio.CancelledError as exc:
        ctx.error(exc)
        ctx.finalize(
            status=SpanStatus.ERROR,
            reason=EndReason.CANCELLED,
            partial=True,
            output=task_result.output,
        )
        raise
    except Exception as exc:
        error = ctx.error(exc)
        task_result.status = TaskStatus.ERRORED
        task_result.error = error
        task_result.errors = list(ctx.errors)
        task_result.observations = list(ctx.observations)
        task_result.spans = list(ctx.spans)
        task_result.artifacts = list(ctx.artifacts)
        ctx.finalize(
            status=SpanStatus.ERROR,
            reason=EndReason.FAILED,
            output=task_result.output,
        )
        return _build_run_result(
            run_spec,
            task_result,
            ctx=ctx,
            asset_versions=list(ctx.asset_versions),
            error=error,
            registry=spec.semantic_registry,
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
        asset_versions=list(ctx.asset_versions),
        error=error,
        registry=spec.semantic_registry,
    )


def _build_run_result(
    run_spec: MatrixRunSpec,
    task_result: TaskResult,
    *,
    ctx: RunContext,
    scores: list[ScoreRecord] | None = None,
    asset_versions: list[AssetVersion] | None = None,
    error: ErrorRecord | None,
    registry: SemanticRegistry,
) -> RunResult:
    active_scores = scores or []
    evaluation_status = _evaluation_status_from_task_result(
        task_result,
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
            task_result,
            evaluation_status=evaluation_status,
        ),
        evaluation_status=evaluation_status,
        case=run_spec.case,
        task_result=task_result,
        scores=active_scores,
        factors=list(run_spec.variant.factors),
        asset_versions=asset_versions or [],
        asset_uses=list(ctx.asset_uses),
        error=error,
        trace=ctx.trace,
        source_snapshots=tuple(ctx.source_snapshots),
    )


def _evaluation_status_from_task_result(
    task_result: TaskResult,
    *,
    scores: list[ScoreRecord],
    registry: SemanticRegistry,
) -> EvaluationStatus:
    if task_result.status is TaskStatus.SKIPPED:
        return EvaluationStatus.SKIPPED
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
    "MatrixRunSpec",
    "RunResult",
    "RunStatus",
    "expand_matrix",
    "generate_experiment_id",
    "run_benchmark_path",
    "run_benchmark_spec",
    "stable_run_id",
)
