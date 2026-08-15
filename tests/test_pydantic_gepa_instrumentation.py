from __future__ import annotations

import builtins
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console

pytest.importorskip("pydantic_gepa")

from pydantic_gepa import Candidate, CandidateComponent, CaseResult, MetricResult
from pydantic_gepa.evaluation import Attachment
from pydantic_gepa.evaluation.traces import ComponentTrace, ErrorInfo
from pydantic_gepa.events import (
    BackendError,
    BackendProgress,
    BudgetExhausted,
    BudgetSnapshot,
    BudgetUpdated,
    CandidateAccepted,
    CandidateEvaluated,
    CandidateNormalized,
    CandidateProposed,
    CandidateRejected,
    CaseEvaluated,
    CheckpointRejected,
    CheckpointReset,
    CheckpointResumed,
    CheckpointWritten,
    ComponentsRegistered,
    DatasetDeclaration,
    EvaluationCompleted,
    EvaluationSkipped,
    EvaluationStarted,
    Event,
    FinalRescoreCompleted,
    FinalRescoreStarted,
    IterationCompleted,
    IterationStarted,
    MetricCompleted,
    MetricDeclaration,
    MetricFailed,
    MetricStarted,
    ParetoFrontUpdated,
    ReflectionCompleted,
    ReflectionStarted,
    RunCancelled,
    RunCompleted,
    RunDeclaration,
    RunFailed,
    RunStarted,
    SelectionCompleted,
    StageCompleted,
    StageFailed,
    StageStarted,
)

import autobench.instrumentation.pydantic_gepa.instrumentor as instrumentor_module
from autobench import (
    CapturePolicy,
    Case,
    Semantic,
    Variant,
    record_experiment,
    replay_experiment,
    run_benchmark_path,
    suppress_instrumentation,
)
from autobench.instrumentation import (
    AssetDiscoverySettings,
    CompatibilityStatus,
    InstrumentationManager,
    InstrumentationRuntime,
)
from autobench.instrumentation.httpx import HTTPX
from autobench.instrumentation.openai import OpenAIClient
from autobench.instrumentation.pydantic_ai import PydanticAI
from autobench.instrumentation.pydantic_gepa import PydanticGEPA
from autobench.instrumentation.pydantic_gepa.adapter import Detail, EventAdapter
from autobench.instrumentation.pydantic_gepa.assets import CandidateAssets
from autobench.instrumentation.pydantic_gepa.projection import (
    EXTENSION_KEY,
    EngineSummary,
    OptimizationExecution,
    PydanticGEPAEvidence,
)
from autobench.protocol import EndReason, SpanStatus
from autobench.protocol.traces import Trace
from autobench.reports.exporting import report_to_yaml_view
from autobench.reports.reporting import (
    OptimizationRunReport,
    build_report,
    render_markdown_report,
)
from autobench.reports.rich import _short_identifier, render_report
from autobench.runtime.context import RunContext
from autobench.runtime.instrumentation import reset_active_run_context, set_active_run_context
from autobench.tracking import AssetDefinition, AssetRepresentation, TrackingRegistry

RUN_ID = "optimization-run"
EXECUTION_ID = "optimization-execution"
ENGINE_ID = "engine-1"
PIPELINE_ID = "pipeline-1"
STEP_ID = "step-1"
EVALUATION_ID = "evaluation-1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _active(context: RunContext) -> Iterator[None]:
    token = set_active_run_context(context)
    try:
        yield
    finally:
        reset_active_run_context(token)


def _context(*, capture_policy: CapturePolicy | None = None) -> RunContext:
    return RunContext(
        benchmark_id="optimizer-benchmark",
        experiment_id="experiment-1",
        run_id="benchmark-run-1",
        case=Case(id="case-1", input="optimize"),
        variant=Variant(id="variant-1"),
        capture_policy=capture_policy,
    )


def _components() -> tuple[CandidateComponent, ...]:
    return (
        CandidateComponent(
            name="instructions",
            initial_text="seed instructions",
            kind="instructions",
            source="app.agent",
        ),
        CandidateComponent(
            name="system_prompt",
            initial_text="seed system",
            kind="system_prompt",
            asset_ref="asset:system-prompt",
        ),
        CandidateComponent(name="input", initial_text="input", kind="input_schema"),
        CandidateComponent(name="output", initial_text="output", kind="output_schema"),
        CandidateComponent(name="tool", initial_text="tool", kind="tool_schema"),
        CandidateComponent(
            name="field",
            initial_text="field",
            kind="field_description",
            source="app.Output",
        ),
        CandidateComponent(
            name="schema",
            initial_text="schema",
            kind="schema_description",
            path="app.Output",
        ),
        CandidateComponent(
            name="routing",
            initial_text="route",
            kind="custom",
            semantic_type="routing.policy.version",
        ),
    )


def _candidate(
    candidate_id: str,
    *,
    parent_id: str | None = None,
    prompt: str = "seed system",
    generation: int = 0,
) -> Candidate:
    return Candidate(
        id=candidate_id,
        parent_id=parent_id,
        generation=generation,
        values={
            "instructions": "seed instructions",
            "system_prompt": prompt,
            "input": "input",
            "output": "output",
            "tool": "tool",
            "field": "field",
            "schema": "schema",
            "routing": "route",
        },
    )


def _declaration() -> RunDeclaration:
    return RunDeclaration(
        configuration_fingerprint="configuration-fingerprint",
        composition_fingerprint="composition-fingerprint",
        objective=MetricDeclaration(
            name="accuracy",
            role="objective",
            direction="maximize",
            semantic_type=Semantic.QUALITY_CORRECTNESS,
            unit="ratio",
        ),
        datasets=DatasetDeclaration(
            train_count=2,
            validation_count=1,
            test_count=1,
            train_fingerprint="train-fingerprint",
            validation_fingerprint="validation-fingerprint",
            test_fingerprint="test-fingerprint",
        ),
        evaluation_call_limit=20,
        optimizer_cost_limit=2.5,
        checkpoint_path="checkpoints/run",
        engine_declaration={"kind": "best_of"},
    )


def _sequence(events: tuple[Event, ...]) -> tuple[Event, ...]:
    return tuple(
        event.model_copy(
            update={
                "sequence": index,
                "execution_id": EXECUTION_ID,
                "backend": "optimize_anything",
            }
        )
        for index, event in enumerate(events)
    )


def _full_events() -> tuple[Event, ...]:
    seed = _candidate("seed")
    rejected = _candidate("candidate-rejected", parent_id="seed", prompt="rejected", generation=1)
    accepted = _candidate(
        "candidate-accepted",
        parent_id="candidate-rejected",
        prompt="accepted",
        generation=2,
    )
    case_result = CaseResult[JsonValue](
        output={"answer": "accepted"},
        metrics={
            "accuracy": MetricResult(
                score=0.8,
                role="objective",
                feedback="mostly correct",
                side_info={"judge": "deterministic"},
                semantic_type=Semantic.QUALITY_CORRECTNESS,
                unit="ratio",
                direction="maximize",
            )
        },
        objectives={"objective": 0.8},
        feedback={"objective": "accepted output"},
        side_info={"routing": {"path": "primary"}},
        traces=(
            ComponentTrace(
                id="component-trace",
                component="agent",
                kind="task",
                input="input",
                output="output",
                duration_seconds=0.01,
            ),
        ),
        artifacts=(
            Attachment(
                kind="document",
                reference="artifact://evaluation",
                media_type="application/json",
                size_bytes=42,
                digest="a" * 64,
            ),
        ),
        task_error=ErrorInfo(kind="RecoverableTaskError", message="recorded failure"),
        duration_seconds=0.02,
        invocation_count=1,
        cache_hit=True,
    )
    return _sequence(
        (
            RunStarted(
                run_id=RUN_ID,
                seed=seed,
                candidate_id=seed.id,
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                declaration=_declaration(),
            ),
            ComponentsRegistered(
                run_id=RUN_ID,
                components=_components(),
                composition="best_of",
                pipeline_id=PIPELINE_ID,
            ),
            StageStarted(
                run_id=RUN_ID,
                stage_id=STEP_ID,
                stage_kind="composition",
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                step_id=STEP_ID,
            ),
            StageStarted(
                run_id=RUN_ID,
                stage_id=ENGINE_ID,
                stage_kind="engine",
                engine="custom",
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                step_id=STEP_ID,
                branch_id="branch-1",
                engine_execution_id=ENGINE_ID,
            ),
            IterationStarted(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
            ),
            ReflectionStarted(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                metadata={"proposal_id": "proposal-1"},
            ),
            ReflectionCompleted(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                metadata={"proposal_id": "proposal-1"},
            ),
            EvaluationStarted(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                candidate_id=rejected.id,
                candidate=rejected,
                evaluation_id=EVALUATION_ID,
                split="train",
                case_count=1,
            ),
            CaseEvaluated(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                candidate_id=rejected.id,
                case_id="optimizer-case-1",
                evaluation_id=EVALUATION_ID,
                split="train",
                result=case_result,
                transformed_score=0.8,
            ),
            MetricStarted(
                run_id=RUN_ID,
                engine_execution_id=ENGINE_ID,
                case_id="optimizer-case-1",
                metric="latency",
                metadata={"evaluation_id": EVALUATION_ID},
            ),
            MetricCompleted(
                run_id=RUN_ID,
                engine_execution_id=ENGINE_ID,
                case_id="optimizer-case-1",
                metric="latency",
                evaluation_id=EVALUATION_ID,
                value=0.1,
                role="diagnostic",
                semantic_type=Semantic.TIME_LATENCY,
                unit="s",
                direction="minimize",
                transformed_value=-0.1,
            ),
            MetricStarted(
                run_id=RUN_ID,
                engine_execution_id=ENGINE_ID,
                metric="broken_metric",
                metadata={"evaluation_id": EVALUATION_ID},
            ),
            MetricFailed(
                run_id=RUN_ID,
                engine_execution_id=ENGINE_ID,
                metric="broken_metric",
                metadata={"evaluation_id": EVALUATION_ID},
                error_type="JudgeError",
                message="judge unavailable",
            ),
            EvaluationCompleted(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                candidate_id=rejected.id,
                evaluation_id=EVALUATION_ID,
                split="train",
                case_count=1,
                scores=(0.8,),
            ),
            CandidateProposed(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=rejected.id,
                parent_ids=("seed",),
                candidate=rejected,
            ),
            CandidateNormalized(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=rejected.id,
                parent_ids=("seed",),
                candidate=rejected,
            ),
            CandidateEvaluated(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=rejected.id,
                score=0.4,
                scores=(0.4,),
            ),
            CandidateRejected(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=rejected.id,
                reason="below incumbent",
                score=0.4,
            ),
            CandidateProposed(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=accepted.id,
                parent_ids=("candidate-rejected",),
                candidate=accepted,
            ),
            CandidateNormalized(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=accepted.id,
                parent_ids=("candidate-rejected",),
                candidate=accepted,
            ),
            CandidateEvaluated(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=accepted.id,
                score=0.95,
                scores=(0.95,),
            ),
            CandidateAccepted(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_id=accepted.id,
                score=0.95,
            ),
            ParetoFrontUpdated(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                candidate_ids=("candidate-accepted", "candidate-rejected"),
            ),
            BackendProgress(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                name="merge_considered",
            ),
            BackendError(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                error_type="TransientError",
                message="retrying",
                will_continue=True,
            ),
            IterationCompleted(
                run_id=RUN_ID,
                engine="custom",
                engine_execution_id=ENGINE_ID,
                iteration=1,
                score=0.95,
            ),
            StageCompleted(
                run_id=RUN_ID,
                stage_id=ENGINE_ID,
                stage_kind="engine",
                engine="custom",
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                step_id=STEP_ID,
                branch_id="branch-1",
                engine_execution_id=ENGINE_ID,
                score=0.95,
                budget=BudgetSnapshot(
                    evaluation_calls=6,
                    evaluation_call_limit=10,
                    optimizer_cost=0.5,
                    optimizer_cost_limit=1.0,
                    evaluation_cost=0.25,
                    total_cost=0.75,
                ),
            ),
            SelectionCompleted(
                run_id=RUN_ID,
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                step_id=STEP_ID,
                method="best_score",
                selected_execution_id=ENGINE_ID,
                contender_execution_ids=(ENGINE_ID, "engine-2"),
                contender_scores=(0.95, 0.7),
                score=0.95,
                reason="highest validation score",
            ),
            StageCompleted(
                run_id=RUN_ID,
                stage_id=STEP_ID,
                stage_kind="composition",
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                step_id=STEP_ID,
                score=0.95,
            ),
            BudgetUpdated(
                run_id=RUN_ID,
                used=12,
                remaining=8,
                optimizer_cost=1.25,
                optimizer_cost_remaining=1.25,
                evaluation_cost=3.0,
                total_cost=4.25,
            ),
            CheckpointWritten(run_id=RUN_ID, path="checkpoints/one"),
            CheckpointResumed(run_id=RUN_ID, path="checkpoints/one"),
            CheckpointRejected(
                run_id=RUN_ID,
                path="checkpoints/stale",
                reason="fingerprint mismatch",
            ),
            CheckpointReset(run_id=RUN_ID, path="checkpoints/reset"),
            FinalRescoreStarted(run_id=RUN_ID, candidate_id=accepted.id),
            FinalRescoreCompleted(run_id=RUN_ID, candidate_id=accepted.id, score=0.97),
            BudgetExhausted(run_id=RUN_ID, used=20, resource="evaluation_calls"),
            RunCompleted(
                run_id=RUN_ID,
                candidate_id=accepted.id,
                score=0.97,
                total_metric_calls=20,
                budget=BudgetSnapshot(
                    evaluation_calls=20,
                    evaluation_call_limit=20,
                    optimizer_cost=1.25,
                    optimizer_cost_limit=2.5,
                    evaluation_cost=3.0,
                    total_cost=4.25,
                ),
            ),
        )
    )


def _capture(
    events: tuple[Event, ...],
    *,
    detail: Detail = "full",
    settings: AssetDiscoverySettings | None = None,
    capture_policy: CapturePolicy | None = None,
) -> tuple[RunContext, Trace, TrackingRegistry]:
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    instrumentor = PydanticGEPA(detail=detail)
    adapter = EventAdapter(
        runtime,
        instrumentor.info,
        CandidateAssets(
            runtime,
            instrumentor.info,
            settings or AssetDiscoverySettings(),
            target_version="0.1.0a0",
        ),
        detail=detail,
    )
    context = _context(capture_policy=capture_policy)
    with _active(context):
        for event in events:
            adapter.observe(event)
        adapter.close()
    return context, context.finalize(output="optimized"), registry


def test_full_event_contract_projects_spans_metrics_assets_lineage_and_summary() -> None:
    events = _full_events()
    context, trace, registry = _capture(events)
    integration_spans = [
        span for span in trace.spans if span.scope.instrumentor_name == "autobench.pydantic_gepa"
    ]
    operations = [span.operation for span in integration_spans]

    assert "pydantic_gepa.partial" not in operations
    assert set(operations) >= {
        "pydantic_gepa.optimization",
        "pydantic_gepa.composition_step",
        "pydantic_gepa.engine",
        "pydantic_gepa.iteration",
        "pydantic_gepa.reflection",
        "pydantic_gepa.evaluation",
        "pydantic_gepa.case",
        "pydantic_gepa.metric",
        "pydantic_gepa.candidate",
        "pydantic_gepa.final_rescore",
    }
    candidate_spans = [
        span for span in integration_spans if span.operation == "pydantic_gepa.candidate"
    ]
    accepted_span = next(span for span in candidate_spans if span.links)
    rejected_span = next(
        span for span in candidate_spans if span.span_id == accepted_span.links[0].target.span_id
    )
    assert rejected_span.status is SpanStatus.OK
    assert accepted_span.links[0].target.span_id == rejected_span.span_id
    assert any(
        error.message == "RecoverableTaskError: recorded failure" for error in context.errors
    )
    assert any(error.message == "JudgeError: judge unavailable" for error in context.errors)
    assert any(error.message == "TransientError: retrying" for error in context.errors)

    semantics = {observation.semantic_type for observation in context.observations}
    assert {
        Semantic.QUALITY_CORRECTNESS,
        Semantic.TIME_LATENCY,
        Semantic.EVALUATION_SCORE,
        Semantic.EVALUATION_LABEL,
        Semantic.EVALUATION_EXPLANATION,
        Semantic.OPTIMIZATION_EVALUATIONS_USED,
        Semantic.OPTIMIZATION_EVALUATIONS_LIMIT,
        Semantic.OPTIMIZATION_EVALUATIONS_REMAINING,
        Semantic.OPTIMIZATION_EVALUATION_COST_USED,
        Semantic.OPTIMIZATION_OPTIMIZER_COST_USED,
        Semantic.OPTIMIZATION_OPTIMIZER_COST_LIMIT,
        Semantic.OPTIMIZATION_OPTIMIZER_COST_REMAINING,
        Semantic.OPTIMIZATION_COST,
    } <= semantics
    assert not any(
        observation.semantic_type is not None and observation.semantic_type.startswith("llm.tokens")
        for observation in context.observations
    )
    evaluation_usage = [
        observation
        for observation in context.observations
        if observation.semantic_type == Semantic.OPTIMIZATION_EVALUATIONS_USED
    ]
    assert [(observation.value, observation.tags["abp.measurement_scope"]) for observation in evaluation_usage] == [
        (6, "direct"),
        (12, "direct"),
        (20, "aggregate"),
    ]

    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    execution = evidence.executions[0]
    assert execution.status == "completed"
    assert execution.objective is not None
    assert execution.objective.semantic_type == Semantic.QUALITY_CORRECTNESS
    assert execution.datasets is not None
    assert execution.datasets.model_dump() == {
        "train_count": 2,
        "validation_count": 1,
        "test_count": 1,
        "train_fingerprint": "train-fingerprint",
        "validation_fingerprint": "validation-fingerprint",
        "test_fingerprint": "test-fingerprint",
    }
    assert execution.final_candidate_id == "candidate-accepted"
    assert execution.final_score == 0.97
    assert execution.evaluations_used == 20
    assert execution.optimizer_cost_used == 1.25
    assert execution.evaluation_cost_used == 3.0
    assert execution.total_cost_used == 4.25
    assert execution.stop_reason == "budget_exhausted:evaluation_calls"
    assert execution.event_count == len(events)
    assert execution.selections[0].selected_execution_id == ENGINE_ID
    assert execution.engines[0].status == "completed"
    assert execution.engines[0].evaluations_used == 6
    assert execution.engines[0].evaluations_limit == 10
    assert execution.engines[0].optimizer_cost_used == 0.5
    assert execution.engines[0].optimizer_cost_limit == 1.0
    assert execution.engines[0].evaluation_cost_used == 0.25
    assert execution.engines[0].total_cost_used == 0.75
    candidates = {candidate.id: candidate for candidate in execution.candidates}
    assert candidates["candidate-rejected"].statuses == (
        "proposed",
        "normalized",
        "evaluated",
        "rejected",
    )
    assert candidates["candidate-accepted"].statuses == (
        "proposed",
        "normalized",
        "evaluated",
        "accepted",
        "best",
        "final",
    )
    assert set(candidates["candidate-accepted"].component_versions) == {
        component.name for component in _components()
    }
    definition = registry.resolve_locator("pydantic-gepa:app.agent#instructions")
    effective = registry.resolve_locator("pydantic-gepa:app.agent#instructions:effective")
    assert isinstance(definition, AssetDefinition)
    assert isinstance(effective, AssetDefinition)
    assert definition.representation is AssetRepresentation.DEFINITION
    assert effective.representation is AssetRepresentation.EFFECTIVE


def test_normalized_candidates_without_selection_terminal_close_as_complete() -> None:
    seed = Candidate(id="seed", values={"prompt": "seed"})
    first = Candidate(id="branch-one", parent_id="seed", values={"prompt": "one"})
    second = Candidate(id="branch-two", parent_id="seed", values={"prompt": "two"})
    context, trace, _registry = _capture(
        _sequence(
            (
                RunStarted(run_id=RUN_ID, seed=seed),
                CandidateNormalized(
                    run_id=RUN_ID,
                    candidate_id=first.id,
                    candidate=first,
                ),
                CandidateNormalized(
                    run_id=RUN_ID,
                    candidate_id=second.id,
                    candidate=second,
                ),
                RunCompleted(
                    run_id=RUN_ID,
                    candidate_id=first.id,
                    score=1.0,
                ),
            )
        )
    )

    candidate_spans = [span for span in trace.spans if span.operation == "pydantic_gepa.candidate"]
    assert len(candidate_spans) == 2
    assert all(span.status is SpanStatus.OK for span in candidate_spans)
    assert all(span.end_reason is EndReason.COMPLETED for span in candidate_spans)
    assert not any(span.partial for span in candidate_spans)
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    candidates = {candidate.id: candidate for candidate in evidence.executions[0].candidates}
    assert candidates["branch-one"].statuses == ("normalized", "final")
    assert candidates["branch-two"].statuses == ("normalized",)


@pytest.mark.parametrize(
    ("detail", "included", "excluded"),
    [
        (
            "summary",
            {"pydantic_gepa.optimization", "pydantic_gepa.engine"},
            {
                "pydantic_gepa.iteration",
                "pydantic_gepa.reflection",
                "pydantic_gepa.evaluation",
                "pydantic_gepa.case",
                "pydantic_gepa.metric",
                "pydantic_gepa.candidate",
            },
        ),
        (
            "evaluations",
            {
                "pydantic_gepa.optimization",
                "pydantic_gepa.evaluation",
                "pydantic_gepa.case",
                "pydantic_gepa.metric",
                "pydantic_gepa.candidate",
            },
            {"pydantic_gepa.iteration", "pydantic_gepa.reflection"},
        ),
        (
            "full",
            {
                "pydantic_gepa.optimization",
                "pydantic_gepa.iteration",
                "pydantic_gepa.reflection",
                "pydantic_gepa.evaluation",
                "pydantic_gepa.case",
                "pydantic_gepa.metric",
                "pydantic_gepa.candidate",
            },
            set(),
        ),
    ],
)
def test_detail_modes_control_high_cardinality_spans(
    detail: Detail,
    included: set[str],
    excluded: set[str],
) -> None:
    context, trace, _registry = _capture(_full_events(), detail=detail)
    operations = {
        span.operation
        for span in trace.spans
        if span.scope.instrumentor_name == "autobench.pydantic_gepa"
    }

    assert included <= operations
    assert operations.isdisjoint(excluded)
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    assert evidence.executions[0].final_candidate_id == "candidate-accepted"


def test_assets_reuse_unchanged_versions_and_obey_discovery_and_capture_policy() -> None:
    settings = AssetDiscoverySettings(
        representations=(AssetRepresentation.EFFECTIVE,),
        include=("prompt",),
    )
    context, _trace, registry = _capture(
        _full_events(),
        settings=settings,
        capture_policy=CapturePolicy.hashed(),
    )
    prompt_assets = [
        asset
        for asset in registry.assets.values()
        if isinstance(asset, AssetDefinition) and asset.kind == "prompt"
    ]
    prompt_versions = [
        version
        for version in registry.versions
        if version.asset_id in {asset.id for asset in prompt_assets}
    ]

    assert {asset.representation for asset in prompt_assets} == {AssetRepresentation.EFFECTIVE}
    assert len(prompt_assets) == 2
    assert len(prompt_versions) == 4
    assert all(use.representation is AssetRepresentation.EFFECTIVE for use in context.asset_uses)
    assert all(
        version.parent_version is None or version.parent_version for version in prompt_versions
    )


def test_out_of_order_duplicate_failure_and_cancellation_events_remain_bounded() -> None:
    events = _sequence(
        (
            StageCompleted(
                run_id=RUN_ID,
                stage_id="late-stage",
                stage_kind="component",
                score=0.2,
            ),
            StageCompleted(
                run_id=RUN_ID,
                stage_id="late-stage",
                stage_kind="component",
                score=0.2,
            ),
            StageStarted(
                run_id=RUN_ID,
                stage_id="failed-engine",
                stage_kind="engine",
                engine_execution_id="failed-engine",
            ),
            StageStarted(
                run_id=RUN_ID,
                stage_id="failed-engine",
                stage_kind="engine",
                engine_execution_id="failed-engine",
            ),
            StageFailed(
                run_id=RUN_ID,
                stage_id="failed-engine",
                stage_kind="engine",
                engine_execution_id="failed-engine",
                error_type="EngineError",
                message="engine failed",
            ),
            EvaluationSkipped(
                run_id=RUN_ID,
                evaluation_id="skipped",
                split="validation",
                reason="cached incumbent",
            ),
            RunCancelled(
                run_id=RUN_ID,
                error_type="CancelledError",
                message="stopped",
            ),
        )
    )
    context, trace, _registry = _capture(events)
    integration_spans = [
        span for span in trace.spans if span.scope.instrumentor_name == "autobench.pydantic_gepa"
    ]
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])

    assert evidence.executions[0].status == "cancelled"
    assert any(span.partial for span in integration_spans)
    assert any(span.end_reason is EndReason.CANCELLED for span in integration_spans)
    assert any(error.message == "EngineError: engine failed" for error in context.errors)
    assert {diagnostic.code for diagnostic in trace.diagnostics} >= {
        "keyed_span_missing_end",
        "keyed_span_duplicate_start",
    }


def test_sparse_reordered_events_preserve_partial_evidence_without_inventing_data() -> None:
    seed = Candidate(id="seed", values={})
    duplicate_metric = MetricResult(score=0.3, role="objective")
    case_result = CaseResult[JsonValue](
        output="ok",
        metrics={"duplicate": duplicate_metric},
        objectives={"duplicate": 0.3},
        duration_seconds=0.01,
        invocation_count=1,
    )
    events = _sequence(
        (
            RunStarted(run_id=RUN_ID, seed=seed),
            ComponentsRegistered(
                run_id=RUN_ID,
                components=(CandidateComponent(name="missing", initial_text="seed"),),
            ),
            StageCompleted(
                run_id=RUN_ID,
                stage_id="late-engine",
                stage_kind="engine",
                engine="custom",
            ),
            StageStarted(
                run_id=RUN_ID,
                stage_id="late-engine",
                stage_kind="engine",
                engine="custom",
            ),
            StageStarted(run_id=RUN_ID, stage_kind="rescore"),
            StageCompleted(run_id=RUN_ID, stage_kind="rescore"),
            IterationCompleted(run_id=RUN_ID, iteration=2),
            ReflectionCompleted(run_id=RUN_ID, iteration=2),
            EvaluationCompleted(
                run_id=RUN_ID,
                evaluation_id="unstarted",
                split="validation",
                case_count=1,
                scores=(),
            ),
            CaseEvaluated(
                run_id=RUN_ID,
                case_id="case-without-start",
                evaluation_id="unstarted-case",
                split="validation",
                result=case_result,
            ),
            MetricCompleted(
                run_id=RUN_ID,
                case_id="case-without-start",
                evaluation_id="unstarted-case",
                metric="duplicate",
                value=0.3,
            ),
            MetricStarted(
                run_id=RUN_ID,
                metric="non-string-evaluation-id",
                metadata={"evaluation_id": 1},
            ),
            CandidateProposed(
                run_id=RUN_ID,
                candidate_id="proposal-without-value",
            ),
            CandidateProposed(
                run_id=RUN_ID,
                candidate_id="proposal-without-value",
            ),
            CandidateNormalized(
                run_id=RUN_ID,
                candidate=Candidate(values={"prompt": "fingerprinted"}),
            ),
            CandidateEvaluated(
                run_id=RUN_ID,
                candidate_id="evaluated-without-proposal",
            ),
            CandidateEvaluated(
                run_id=RUN_ID,
                candidate_id="scored-without-proposal",
                score=0.4,
            ),
            CandidateRejected(
                run_id=RUN_ID,
                candidate_id="terminal-without-proposal",
                reason="invalid",
            ),
            BudgetUpdated(run_id=RUN_ID, used=1),
            StageStarted(
                run_id=RUN_ID,
                stage_id="left-open",
                stage_kind="component",
            ),
            RunCompleted(
                run_id=RUN_ID,
                candidate_id="terminal-without-proposal",
            ),
            BackendProgress(run_id=RUN_ID, name="late-progress"),
            RunCompleted(run_id=RUN_ID),
        )
    )

    context, trace, _registry = _capture(events)
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    execution = evidence.executions[0]

    assert execution.status == "completed"
    assert execution.final_score is None
    assert execution.evaluations_remaining is None
    assert execution.optimizer_cost_used is None
    assert execution.optimizer_cost_remaining is None
    assert execution.engines[0].execution_id == "late-engine"
    assert execution.engines[0].score is None
    candidates = {candidate.id: candidate for candidate in execution.candidates}
    assert candidates["proposal-without-value"].statuses == ("proposed",)
    assert candidates["evaluated-without-proposal"].score is None
    assert candidates["terminal-without-proposal"].statuses == ("rejected", "final")
    assert any(candidate.id.startswith("candidate:") for candidate in execution.candidates)
    assert {
        "pydantic_gepa_duplicate_start",
        "pydantic_gepa_duplicate_candidate_transition",
        "keyed_span_event_target_missing",
    } <= {diagnostic.code for diagnostic in trace.diagnostics}


def test_failure_and_unstarted_failed_stage_keep_original_error_evidence() -> None:
    context, trace, _registry = _capture(
        _sequence(
            (
                StageFailed(
                    run_id=RUN_ID,
                    stage_id="unstarted-engine",
                    stage_kind="engine",
                    engine="custom",
                    error_type="EngineError",
                    message="failed before start",
                ),
                RunFailed(
                    run_id=RUN_ID,
                    error_type="OptimizationError",
                    message="no candidate",
                ),
            )
        )
    )
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])

    assert evidence.executions[0].status == "failed"
    assert evidence.executions[0].stop_reason == "OptimizationError"
    assert evidence.executions[0].engines[0].status == "failed"
    assert any(span.status is SpanStatus.ERROR for span in trace.spans)
    assert any(error.message == "OptimizationError: no candidate" for error in context.errors)


def test_candidate_selection_respects_missing_scores_and_objective_direction() -> None:
    minimizing = _declaration().model_copy(
        update={
            "objective": MetricDeclaration(
                name="latency",
                role="objective",
                direction="minimize",
                semantic_type=Semantic.TIME_LATENCY,
                unit="s",
            ),
            "evaluation_call_limit": None,
            "optimizer_cost_limit": None,
        }
    )
    minimize_events: list[Event] = [RunStarted(run_id=RUN_ID, declaration=minimizing)]
    minimize_events.extend(
        CandidateAccepted(run_id=RUN_ID, candidate_id=candidate_id, score=score)
        for candidate_id, score in (
            ("no-score", None),
            ("none-ignored", None),
            ("first-score", 10.0),
            ("lower-score", 5.0),
            ("higher-score", 8.0),
        )
    )
    minimize_events.append(RunCompleted(run_id=RUN_ID, candidate_id="lower-score", score=5.0))
    context, _trace, _registry = _capture(_sequence(tuple(minimize_events)))
    execution = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY]).executions[0]

    assert execution.best_candidate_id == "lower-score"
    assert execution.evaluations_limit is None
    assert execution.optimizer_cost_limit is None

    maximize_events = _sequence(
        (
            RunStarted(run_id=RUN_ID, declaration=_declaration()),
            CandidateAccepted(run_id=RUN_ID, candidate_id="first", score=0.5),
            CandidateAccepted(run_id=RUN_ID, candidate_id="second", score=0.8),
            RunCompleted(run_id=RUN_ID, candidate_id="second", score=0.8),
        )
    )
    maximize_context, _trace, _registry = _capture(maximize_events)
    maximize_execution = PydanticGEPAEvidence.model_validate(
        maximize_context.extensions[EXTENSION_KEY]
    ).executions[0]
    assert maximize_execution.best_candidate_id == "second"


def test_completion_budget_snapshots_cover_cached_and_partially_reported_resources() -> None:
    unlimited_context, _trace, _registry = _capture(
        _sequence(
            (
                RunStarted(run_id=RUN_ID),
                RunCompleted(
                    run_id=RUN_ID,
                    total_metric_calls=1,
                    budget=BudgetSnapshot(
                        evaluation_calls=2,
                        optimizer_cost=0.1,
                        evaluation_cost=0.2,
                        total_cost=0.3,
                    ),
                ),
            )
        )
    )
    unlimited = PydanticGEPAEvidence.model_validate(
        unlimited_context.extensions[EXTENSION_KEY]
    ).executions[0]
    assert unlimited.evaluations_used == 2
    assert unlimited.evaluations_limit is None
    assert unlimited.evaluations_remaining is None
    assert unlimited.optimizer_cost_used == 0.1
    assert unlimited.optimizer_cost_limit is None
    assert unlimited.optimizer_cost_remaining is None
    assert unlimited.evaluation_cost_used == 0.2
    assert unlimited.total_cost_used == 0.3

    declaration = _declaration().model_copy(
        update={"evaluation_call_limit": 10, "optimizer_cost_limit": 2.0}
    )
    limited_context, _trace, _registry = _capture(
        _sequence(
            (
                RunStarted(run_id=RUN_ID, declaration=declaration),
                RunCompleted(
                    run_id=RUN_ID,
                    total_metric_calls=4,
                    budget=BudgetSnapshot(
                        evaluation_call_limit=10,
                        optimizer_cost_limit=2.0,
                    ),
                ),
            )
        )
    )
    limited = PydanticGEPAEvidence.model_validate(
        limited_context.extensions[EXTENSION_KEY]
    ).executions[0]
    assert limited.evaluations_used == 4
    assert limited.evaluations_limit == 10
    assert limited.evaluations_remaining is None
    assert limited.optimizer_cost_used is None
    assert limited.optimizer_cost_limit == 2.0
    assert limited.optimizer_cost_remaining is None


def test_evaluation_candidate_assets_bind_to_the_evaluation_span_once() -> None:
    evaluated = _candidate("evaluated-directly")
    context, trace, _registry = _capture(
        _sequence(
            (
                RunStarted(run_id=RUN_ID),
                ComponentsRegistered(run_id=RUN_ID, components=_components()),
                EvaluationStarted(
                    run_id=RUN_ID,
                    candidate_id=evaluated.id,
                    candidate=evaluated,
                    evaluation_id="direct-evaluation",
                    split="validation",
                ),
                EvaluationStarted(
                    run_id=RUN_ID,
                    candidate_id=evaluated.id,
                    candidate=evaluated,
                    evaluation_id="direct-evaluation",
                    split="validation",
                ),
                EvaluationCompleted(
                    run_id=RUN_ID,
                    candidate_id=evaluated.id,
                    evaluation_id="direct-evaluation",
                    split="validation",
                ),
                RunCompleted(run_id=RUN_ID, candidate_id=evaluated.id),
            )
        )
    )
    evaluation = next(span for span in context.spans if span.name == "pydantic_gepa.evaluation")
    evaluation_uses = [use for use in context.asset_uses if use.span_id == evaluation.id]
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])

    assert len(evaluation_uses) == len(_components())
    assert evidence.executions[0].candidates[0].statuses == ("proposed", "final")
    assert any(diagnostic.code == "keyed_span_duplicate_start" for diagnostic in trace.diagnostics)


def test_summary_mode_handles_missing_parent_seed_and_final_candidate() -> None:
    context, trace, _registry = _capture(
        (
            RunStarted(
                run_id=RUN_ID,
                execution_id=EXECUTION_ID,
                parent_execution_id="missing-parent",
                backend="plan",
                sequence=0,
            ),
            ComponentsRegistered(
                run_id=RUN_ID,
                execution_id=EXECUTION_ID,
                backend="plan",
                sequence=1,
                components=(),
            ),
            EvaluationSkipped(
                run_id=RUN_ID,
                execution_id=EXECUTION_ID,
                backend="plan",
                sequence=2,
                evaluation_id="cached",
                split="validation",
                reason="already evaluated",
            ),
            RunCompleted(
                run_id=RUN_ID,
                execution_id=EXECUTION_ID,
                backend="plan",
                sequence=3,
                candidate_id="unknown-candidate",
            ),
        ),
        detail="summary",
    )
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    root = next(span for span in trace.spans if span.operation == "pydantic_gepa.optimization")

    assert root.parent_span_id is not None
    assert evidence.executions[0].parent_execution_id == "missing-parent"
    assert evidence.executions[0].seed_candidate_id is None
    assert evidence.executions[0].final_candidate_id == "unknown-candidate"
    assert not evidence.executions[0].candidates


def test_runtime_span_desynchronization_isolated_as_conversion_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    runtime = InstrumentationRuntime(registry=TrackingRegistry())
    instrumentor = PydanticGEPA()
    adapter = EventAdapter(
        runtime,
        instrumentor.info,
        CandidateAssets(
            runtime,
            instrumentor.info,
            AssetDiscoverySettings(),
            target_version="0.1.0a0",
        ),
        detail="full",
    )
    original_start_span = runtime.start_span
    original_span_for_key = runtime.span_for_key
    monkeypatch.setattr(runtime, "start_span", lambda *_args, **_kwargs: None)
    with _active(context):
        adapter.observe(RunStarted(run_id=RUN_ID))
    monkeypatch.setattr(runtime, "start_span", original_start_span)
    monkeypatch.setattr(runtime, "span_for_key", lambda *_args, **_kwargs: None)
    with _active(context):
        adapter.observe(RunStarted(run_id=RUN_ID, declaration=_declaration()))
    monkeypatch.setattr(runtime, "span_for_key", original_span_for_key)
    with _active(context):
        adapter.close()
    trace = context.finalize(output="unchanged")

    assert "pydantic_gepa_event_conversion_failed" in {
        diagnostic.code for diagnostic in trace.diagnostics
    }


def test_nested_and_parallel_execution_ids_preserve_explicit_parentage() -> None:
    parent_started = RunStarted(
        run_id="parent",
        execution_id="parent-execution",
        backend="plan",
        sequence=0,
    )
    child_started = RunStarted(
        run_id="child",
        execution_id="child-execution",
        parent_execution_id="parent-execution",
        backend="optimize_anything",
        sequence=1,
    )
    child_completed = RunCompleted(
        run_id="child",
        execution_id="child-execution",
        parent_execution_id="parent-execution",
        backend="optimize_anything",
        sequence=4,
        score=1.0,
    )
    child_stage = StageStarted(
        run_id="child",
        execution_id="child-execution",
        parent_execution_id="parent-execution",
        backend="plan",
        sequence=2,
        stage_id="child-stage",
        stage_kind="component",
    )
    child_stage_completed = StageCompleted(
        run_id="child",
        execution_id="child-execution",
        parent_execution_id="parent-execution",
        backend="plan",
        sequence=3,
        stage_id="child-stage",
        stage_kind="component",
    )
    parent_completed = RunCompleted(
        run_id="parent",
        execution_id="parent-execution",
        backend="plan",
        sequence=5,
        score=1.0,
    )
    context, trace, _registry = _capture(
        (
            parent_started,
            child_started,
            child_stage,
            child_stage_completed,
            child_completed,
            parent_completed,
        )
    )
    roots = [span for span in trace.spans if span.operation == "pydantic_gepa.optimization"]
    assert len(roots) == 2
    child_root = next(
        span for span in roots if span.parent_span_id in {candidate.span_id for candidate in roots}
    )
    parent_root = next(span for span in roots if span.span_id == child_root.parent_span_id)
    assert child_root.parent_span_id == parent_root.span_id
    child_stage_span = next(span for span in trace.spans if span.operation == "pydantic_gepa.stage")
    assert child_stage_span.parent_span_id == parent_root.span_id
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    assert [execution.execution_id for execution in evidence.executions] == [
        "parent-execution",
        "child-execution",
    ]

    def run_one(index: int) -> tuple[str, str]:
        thread_context = _context()
        runtime = InstrumentationRuntime(registry=TrackingRegistry())
        instrumentor = PydanticGEPA()
        adapter = EventAdapter(
            runtime,
            instrumentor.info,
            CandidateAssets(
                runtime,
                instrumentor.info,
                AssetDiscoverySettings(),
                target_version="0.1.0a0",
            ),
            detail="summary",
        )
        execution_id = f"parallel-{index}"
        with _active(thread_context):
            adapter.observe(
                RunStarted(
                    run_id=f"run-{index}",
                    execution_id=execution_id,
                    backend="optimize_anything",
                )
            )
            adapter.observe(
                RunCompleted(
                    run_id=f"run-{index}",
                    execution_id=execution_id,
                    backend="optimize_anything",
                    score=float(index),
                )
            )
            adapter.close()
        thread_context.finalize(output=index)
        evidence = PydanticGEPAEvidence.model_validate(thread_context.extensions[EXTENSION_KEY])
        return str(thread_context.trace.trace_id), evidence.executions[0].execution_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(run_one, (1, 2)))
    assert len({trace_id for trace_id, _execution_id in results}) == 2
    assert {execution_id for _trace_id, execution_id in results} == {"parallel-1", "parallel-2"}


def test_parallel_backend_events_reuse_the_run_context_captured_at_start() -> None:
    context = _context()
    runtime = InstrumentationRuntime(registry=TrackingRegistry())
    instrumentor = PydanticGEPA()
    adapter = EventAdapter(
        runtime,
        instrumentor.info,
        CandidateAssets(
            runtime,
            instrumentor.info,
            AssetDiscoverySettings(),
            target_version="0.1.0a0",
        ),
        detail="full",
    )
    with _active(context):
        adapter.observe(
            RunStarted(
                run_id=RUN_ID,
                execution_id=EXECUTION_ID,
                backend="optimize_anything",
                composition="pipeline",
                pipeline_id=PIPELINE_ID,
            )
        )

    def branch(index: int) -> None:
        engine_id = f"engine-{index}"
        adapter.observe(
            StageStarted(
                run_id=RUN_ID,
                execution_id=EXECUTION_ID,
                backend="optimize_anything",
                engine=f"candidate-{index}",
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                step_id="step-0",
                branch_id=f"branch-{index}",
                engine_execution_id=engine_id,
                stage_id=engine_id,
                stage_kind="engine",
                parent_execution_id="previous-engine",
            )
        )
        adapter.observe(
            StageCompleted(
                run_id=RUN_ID,
                execution_id=EXECUTION_ID,
                backend="optimize_anything",
                engine=f"candidate-{index}",
                composition="best_of",
                pipeline_id=PIPELINE_ID,
                step_id="step-0",
                branch_id=f"branch-{index}",
                engine_execution_id=engine_id,
                stage_id=engine_id,
                stage_kind="engine",
                parent_execution_id="previous-engine",
                score=float(index),
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(branch, (0, 1)))
    adapter.observe(
        RunCompleted(
            run_id=RUN_ID,
            execution_id=EXECUTION_ID,
            backend="optimize_anything",
            score=1.0,
        )
    )
    with _active(context):
        adapter.close()
    trace = context.finalize(output="optimized")
    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    execution = evidence.executions[0]

    assert {engine.execution_id for engine in execution.engines} == {"engine-0", "engine-1"}
    assert execution.composition == "pipeline"
    assert execution.engine is None
    assert execution.parent_execution_id is None
    assert execution.step_id is None
    assert not trace.diagnostics


def test_conversion_failures_suppression_and_inactive_context_do_not_change_subject_flow() -> None:
    invalid_component = CandidateComponent(
        name="json",
        initial_text="seed",
        serialization="json_string",
    )
    events = _sequence(
        (
            RunStarted(run_id=RUN_ID, seed=Candidate(id="seed", values={"json": '"seed"'})),
            ComponentsRegistered(run_id=RUN_ID, components=(invalid_component,)),
            CandidateNormalized(
                run_id=RUN_ID,
                candidate_id="invalid",
                candidate=Candidate(id="invalid", values={"json": "not-json"}),
            ),
            RunCompleted(run_id=RUN_ID, candidate_id="invalid", score=0.0),
        )
    )
    context, trace, _registry = _capture(events)
    assert "pydantic_gepa_event_conversion_failed" in {
        diagnostic.code for diagnostic in trace.diagnostics
    }

    runtime = InstrumentationRuntime(registry=TrackingRegistry())
    instrumentor = PydanticGEPA()
    adapter = EventAdapter(
        runtime,
        instrumentor.info,
        CandidateAssets(
            runtime,
            instrumentor.info,
            AssetDiscoverySettings(),
            target_version="0.1.0a0",
        ),
        detail="full",
    )
    adapter.observe(events[0])
    suppressed_context = _context()
    with _active(suppressed_context), suppress_instrumentation("autobench.pydantic_gepa"):
        adapter.observe(events[0])
    adapter.close()
    suppressed_context.finalize(output="unchanged")
    assert EXTENSION_KEY not in suppressed_context.extensions


def test_instrumentor_compatibility_installation_and_subscription_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentor = PydanticGEPA()
    assert instrumentor.check().status is CompatibilityStatus.COMPATIBLE
    assert instrumentor.info.capabilities.asset_discovery
    assert instrumentor.info.source_convention == "pydantic-gepa"

    monkeypatch.setattr(instrumentor_module, "find_spec", lambda _name: None)
    assert PydanticGEPA().check().status is CompatibilityStatus.UNAVAILABLE
    monkeypatch.setattr(instrumentor_module, "find_spec", lambda _name: True)

    import pydantic_gepa.events as event_module

    monkeypatch.setattr(event_module, "subscribe", None)
    assert PydanticGEPA().check().status is CompatibilityStatus.UNSUPPORTED
    monkeypatch.undo()

    captured: list[Any] = []
    closed = False

    class Subscription:
        def close(self) -> None:
            nonlocal closed
            closed = True

    def subscribe(observer: Any, *, on_error: str) -> Subscription:
        captured.extend((observer, on_error))
        return Subscription()

    monkeypatch.setattr(event_module, "subscribe", subscribe)
    monkeypatch.setattr(instrumentor_module, "version", lambda _name: "0.1.0a0")
    handle = PydanticGEPA().install(InstrumentationRuntime(registry=TrackingRegistry()))
    assert captured[1] == "ignore"
    handle.close()
    assert closed

    def missing_version(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(instrumentor_module, "version", missing_version)
    assert PydanticGEPA().check().target_version is None

    original_import = builtins.__import__

    def broken_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "pydantic_gepa.events":
            raise ImportError("broken event package")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    broken = PydanticGEPA().check()
    assert broken.status is CompatibilityStatus.UNAVAILABLE
    assert "broken event package" in broken.diagnostics[0]


def test_real_subscription_handle_installs_once_through_manager() -> None:
    context = _context()
    with InstrumentationManager() as manager:
        first = manager.install(PydanticGEPA(detail="summary"))
        second = manager.install(PydanticGEPA(detail="summary"))
        assert first.info == second.info
        with _active(context):
            from pydantic_gepa.events import _dispatcher

            dispatcher = _dispatcher(
                run_id=RUN_ID,
                backend="optimize_anything",
                execution_id=EXECUTION_ID,
            )
            dispatcher.emit(RunStarted(run_id=RUN_ID))
            dispatcher.emit(RunCompleted(run_id=RUN_ID, score=1.0))
    context.finalize(output="done")

    evidence = PydanticGEPAEvidence.model_validate(context.extensions[EXTENSION_KEY])
    assert len(evidence.executions) == 1
    assert evidence.executions[0].final_score == 1.0


async def test_native_model_and_transport_evidence_remains_separate_from_optimizer_evidence() -> (
    None
):
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-optimizer",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "optimized"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            request=request,
        )

    from pydantic_gepa.events import _dispatcher

    context = _context()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    openai_client = AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=http_client,
    )
    agent = Agent(OpenAIChatModel("gpt-test", provider=OpenAIProvider(openai_client=openai_client)))
    manager = InstrumentationManager()
    for instrumentor in (PydanticGEPA(), PydanticAI(), OpenAIClient(), HTTPX()):
        manager.install(instrumentor)
    token = set_active_run_context(context)
    seed = Candidate(id="seed", values={"prompt": "seed"})
    dispatcher = _dispatcher(
        run_id=RUN_ID,
        backend="optimize_anything",
        execution_id=EXECUTION_ID,
    )
    try:
        dispatcher.emit(RunStarted(run_id=RUN_ID, seed=seed, declaration=_declaration()))
        result = await agent.run("optimize this")
        dispatcher.emit(RunCompleted(run_id=RUN_ID, candidate_id=seed.id, score=1.0))
    finally:
        reset_active_run_context(token)
        manager.close()
        await openai_client.close()
    context.finalize(output=result.output)

    assert result.output == "optimized"
    spans = {span.operation: span for span in context.trace.spans}
    assert {
        "pydantic_gepa.optimization",
        "pydantic_ai.agent.run",
        "openai.chat.completions",
        "httpx.request",
    } <= spans.keys()
    spans_by_id = {span.span_id: span for span in context.trace.spans}
    ancestor_ids: set[str] = set()
    parent_id = spans["openai.chat.completions"].parent_span_id
    while parent_id is not None:
        ancestor_ids.add(parent_id)
        parent_id = spans_by_id[parent_id].parent_span_id
    assert spans["pydantic_ai.agent.run"].span_id in ancestor_ids
    assert spans["httpx.request"].parent_span_id == spans["openai.chat.completions"].span_id
    assert spans["openai.chat.completions"].usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert spans["httpx.request"].usage == {}
    optimizer_spans = [
        span
        for span in context.trace.spans
        if span.scope.instrumentor_name == "autobench.pydantic_gepa"
    ]
    assert optimizer_spans
    assert all(not span.usage for span in optimizer_spans)


def test_optimizer_projection_replays_exports_and_renders_without_optional_runtime(
    tmp_path: Path,
) -> None:
    captured_context, _trace, _registry = _capture(_full_events())
    result = run_benchmark_path(
        PROJECT_ROOT / "examples" / "basic" / "autobench.yaml",
        experiment_id="optimizer-report",
    )
    result.runs[0].extensions[EXTENSION_KEY] = captured_context.extensions[EXTENSION_KEY]
    result.runs[1].extensions[EXTENSION_KEY] = {"schema_version": 999}
    report = build_report(result)

    assert len(report.optimizations) == 1
    assert report.optimizations[0].execution.final_score == 0.97
    assert report.optimization_warnings[0].startswith(
        f"run {result.runs[1].run_id}: invalid pydantic-gepa evidence"
    )
    rendered = StringIO()
    render_report(
        Console(file=rendered, force_terminal=False, color_system=None, width=240),
        report,
    )
    terminal = rendered.getvalue()
    assert "Pydantic-GEPA Optimizations" in terminal
    assert "Optimization Engines" in terminal
    assert "Candidate Lineage" in terminal
    assert "Optimization Component Versions" in terminal
    assert "Optimization Selections" in terminal
    assert "candidate-accepted" in terminal

    yaml_view = report_to_yaml_view(report)
    assert yaml_view["report"]["optimizations"][0]["execution"]["final_score"] == 0.97
    assert "optimization_warnings" in yaml_view["report"]
    markdown = render_markdown_report(report)
    assert "## Pydantic-GEPA Optimizations" in markdown
    assert "### Candidate Lineage" in markdown
    assert "### Optimization Evidence Warnings" in markdown

    record_path = tmp_path / "record"
    record_experiment(result, record_path)
    replayed = replay_experiment(record_path)
    assert PydanticGEPAEvidence.model_validate(
        replayed.runs[0].extensions[EXTENSION_KEY]
    ) == PydanticGEPAEvidence.model_validate(captured_context.extensions[EXTENSION_KEY])


def test_optimizer_reports_render_empty_lineage_partial_state_and_engine_resources() -> None:
    result = run_benchmark_path(
        PROJECT_ROOT / "examples" / "basic" / "autobench.yaml",
        experiment_id="optimizer-report-boundaries",
    )
    report = build_report(result)
    execution = OptimizationExecution(
        execution_id="optimization:step:engine-with-a-very-long-identifier",
        run_id="partial-optimization",
        status="partial",
        engines=(
            EngineSummary(
                execution_id="optimization:step:resource-engine",
                engine="deterministic",
                status="completed",
                evaluations_used=7,
                evaluations_limit=10,
                optimizer_cost_used=0.125,
                optimizer_cost_limit=0.5,
                evaluation_cost_used=0.25,
                total_cost_used=0.375,
            ),
            EngineSummary(
                execution_id="optimization:step:resource-without-limits",
                engine="deterministic",
                status="completed",
                evaluations_used=2,
                optimizer_cost_used=0.05,
            ),
            EngineSummary(
                execution_id="optimization:step:no-resources",
                engine="deterministic",
                status="completed",
            ),
        ),
        diagnostic_count=1,
    )
    boundary_report = report.model_copy(
        update={
            "optimizations": [
                OptimizationRunReport(
                    benchmark_run_id="benchmark-run",
                    case_id="case",
                    variant_id="variant",
                    execution=execution,
                )
            ],
            "optimization_warnings": [],
        }
    )

    rendered = StringIO()
    render_report(
        Console(file=rendered, force_terminal=False, color_system=None, width=100),
        boundary_report,
    )
    terminal = rendered.getvalue()
    assert "Optimization Engine Resources" in terminal
    assert "status=partial" in terminal
    markdown = render_markdown_report(boundary_report)
    assert "### Engine Runs" in markdown
    assert "resource-without-limits" in markdown

    no_engine_report = boundary_report.model_copy(
        update={
            "optimizations": [
                boundary_report.optimizations[0].model_copy(
                    update={"execution": execution.model_copy(update={"engines": ()})}
                )
            ]
        }
    )
    assert "### Engine Runs" not in render_markdown_report(no_engine_report)
    assert "Candidate Lineage" not in terminal
    assert "Optimization Component Versions" not in terminal
    assert "Optimization Selections" not in terminal

    empty_execution = execution.model_copy(
        update={"status": "completed", "engines": (), "diagnostic_count": 0}
    )
    empty_report = boundary_report.model_copy(
        update={
            "optimizations": [
                OptimizationRunReport(
                    benchmark_run_id="empty-run",
                    case_id="empty-case",
                    variant_id="empty-variant",
                    execution=empty_execution,
                )
            ]
        }
    )
    empty_rendered = StringIO()
    render_report(
        Console(file=empty_rendered, force_terminal=False, color_system=None, width=100),
        empty_report,
    )
    assert "Optimization Engines" not in empty_rendered.getvalue()

    markdown = render_markdown_report(boundary_report)
    assert "| partial |  | 0 |  |" in markdown
    assert "### Candidate Lineage" not in markdown

    assert _short_identifier("short") == "short"
    assert _short_identifier("prefix:middle:short-scope") == "middle:short-scope"
    assert _short_identifier("abcdefghijklmnopqrstuvwxyz") == "abcdefgh...stuvwxyz"
    assert _short_identifier("prefix:middle:a-very-long-final-scope") == ("prefix:m...al-scope")
