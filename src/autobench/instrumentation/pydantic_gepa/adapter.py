from __future__ import annotations as _annotations

from statistics import fmean
from threading import RLock
from typing import Literal

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
    EvaluationCompleted,
    EvaluationSkipped,
    EvaluationStarted,
    Event,
    FinalRescoreCompleted,
    FinalRescoreStarted,
    IterationCompleted,
    IterationStarted,
    MetricCompleted,
    MetricFailed,
    MetricStarted,
    ParetoFrontUpdated,
    ReflectionCompleted,
    ReflectionStarted,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    SelectionCompleted,
    StageCompleted,
    StageFailed,
    StageStarted,
)

from autobench.instrumentation.manager import InstrumentationRuntime
from autobench.instrumentation.models import InstrumentorInfo
from autobench.instrumentation.pydantic_gepa._evidence import Detail, _EventEvidence
from autobench.instrumentation.pydantic_gepa._state import _Candidate, _Execution
from autobench.instrumentation.pydantic_gepa.assets import CandidateAssets
from autobench.instrumentation.pydantic_gepa.projection import (
    CandidateStatus,
    DatasetSummary,
    EngineSummary,
    ObjectiveSummary,
    SelectionSummary,
)
from autobench.metrics.observations import ObservationRole
from autobench.metrics.semantics import Semantic
from autobench.protocol.context import ActiveContext, get_context, use_context
from autobench.protocol.signals import EndReason, LinkRelation, SpanStatus
from autobench.protocol.values import SerializedValue


class EventAdapter(_EventEvidence):
    """Convert ordered pydantic-gepa events into native Autobench evidence."""

    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        assets: CandidateAssets,
        *,
        detail: Detail,
    ) -> None:
        super().__init__(runtime, info, assets, detail=detail)
        self._contexts: dict[str, ActiveContext] = {}
        self._order = 0
        self._lock = RLock()

    def observe(self, event: Event) -> None:
        execution_id = event.execution_id or event.run_id
        with self._lock:
            active = self._contexts.get(execution_id)
            if active is None and event.parent_execution_id is not None:
                active = self._contexts.get(event.parent_execution_id)
            if active is None:
                active = get_context()
            if active is None or active.is_suppressed(self._info.id):
                return
            self._contexts.setdefault(execution_id, active)
            with use_context(active):
                self._observe_active(event, execution_id, active)
            if isinstance(event, RunCompleted | RunFailed | RunCancelled):
                self._contexts.pop(execution_id, None)

    def _observe_active(
        self,
        event: Event,
        execution_id: str,
        active: ActiveContext,
    ) -> None:
        state_key = (active.trace_id, execution_id)
        state = self._states.get(state_key)
        if state is None:
            state = _Execution(
                order=self._order,
                trace_id=active.trace_id,
                execution_id=execution_id,
                run_id=event.run_id,
            )
            self._order += 1
            self._states[state_key] = state
        state.event_count += 1
        self._update_identity(state, event)
        try:
            self._dispatch(state, event)
        except Exception as error:
            state.diagnostic_count += 1
            self._runtime.diagnose(
                self._info,
                "pydantic_gepa_event_conversion_failed",
                f"{event.kind}: {type(error).__name__}: {error}",
            )
        finally:
            self._write_projection(active.trace_id)

    def close(self) -> None:
        with self._lock:
            for state in self._states.values():
                for key in reversed(state.open_keys):
                    self._runtime.end_span(
                        self._info,
                        key,
                        reason=EndReason.ABANDONED,
                        partial=True,
                    )
                state.open_keys.clear()
            self._states.clear()
            self._contexts.clear()

    def _dispatch(self, state: _Execution, event: Event) -> None:
        self._ensure_root(state, event)
        if isinstance(event, RunStarted):
            self._run_started(state, event)
        elif isinstance(event, RunCompleted):
            self._run_completed(state, event)
        elif isinstance(event, RunFailed):
            self._run_failed(state, event)
        elif isinstance(event, RunCancelled):
            self._run_cancelled(state, event)
        elif isinstance(event, ComponentsRegistered):
            self._components(state, event)
        elif isinstance(event, StageStarted):
            self._stage_started(state, event)
        elif isinstance(event, StageCompleted):
            self._stage_completed(state, event)
        elif isinstance(event, StageFailed):
            self._stage_failed(state, event)
        elif isinstance(event, IterationStarted):
            self._iteration_started(state, event)
        elif isinstance(event, IterationCompleted):
            self._iteration_completed(state, event)
        elif isinstance(event, EvaluationStarted):
            self._evaluation_started(state, event)
        elif isinstance(event, EvaluationCompleted):
            self._evaluation_completed(state, event)
        elif isinstance(event, EvaluationSkipped):
            self._evaluation_skipped(state, event)
        elif isinstance(event, CaseEvaluated):
            self._case_evaluated(state, event)
        elif isinstance(event, ReflectionStarted):
            self._reflection_started(state, event)
        elif isinstance(event, ReflectionCompleted):
            self._reflection_completed(state, event)
        elif isinstance(event, CandidateProposed | CandidateNormalized):
            self._candidate_changed(state, event)
        elif isinstance(event, CandidateEvaluated):
            self._candidate_evaluated(state, event)
        elif isinstance(event, CandidateAccepted):
            self._candidate_terminal(state, event, "accepted")
        elif isinstance(event, CandidateRejected):
            self._candidate_terminal(state, event, "rejected")
        elif isinstance(event, MetricStarted):
            self._metric_started(state, event)
        elif isinstance(event, MetricCompleted):
            self._metric_completed(state, event)
        elif isinstance(event, MetricFailed):
            self._metric_failed(state, event)
        elif isinstance(event, BudgetUpdated):
            self._budget_updated(state, event)
        elif isinstance(event, BudgetExhausted):
            state.stop_reason = f"budget_exhausted:{event.resource}"
            self._event(state, event, value=event.resource)
        elif isinstance(event, SelectionCompleted):
            self._selection(state, event)
        elif isinstance(event, FinalRescoreStarted):
            self._final_rescore_started(state, event)
        elif isinstance(event, FinalRescoreCompleted):
            self._final_rescore_completed(state, event)
        elif isinstance(event, BackendError):
            self._record_error(
                self._event_span(state, event),
                event.error_type,
                event.message,
            )
            self._event(
                state,
                event,
                value={
                    "error_type": event.error_type,
                    "message": event.message,
                    "will_continue": event.will_continue,
                },
            )
        elif isinstance(event, ParetoFrontUpdated):
            if self._detail == "full":
                self._event(state, event, value=list(event.candidate_ids))
        elif isinstance(
            event,
            CheckpointWritten | CheckpointResumed | CheckpointRejected | CheckpointReset,
        ):
            self._checkpoint(state, event)
        elif isinstance(event, BackendProgress) and self._detail == "full":
            self._event(state, event, value=event.name)

    @staticmethod
    def _update_identity(state: _Execution, event: Event) -> None:
        if state.backend is None:
            state.backend = event.backend
        if not isinstance(event, RunStarted):
            return
        state.parent_execution_id = event.parent_execution_id
        state.engine = event.engine
        state.composition = event.composition
        state.pipeline_id = event.pipeline_id
        state.step_id = event.step_id
        state.branch_id = event.branch_id
        state.stage_id = event.stage_id

    def _run_started(self, state: _Execution, event: RunStarted) -> None:
        declaration = event.declaration
        if declaration is not None:
            state.objective = ObjectiveSummary(
                name=declaration.objective.name,
                role=declaration.objective.role,
                direction=declaration.objective.direction,
                semantic_type=declaration.objective.semantic_type,
                unit=declaration.objective.unit,
            )
            state.datasets = DatasetSummary.model_validate(
                declaration.datasets.model_dump(mode="json")
            )
            state.evaluations_limit = declaration.evaluation_call_limit
            state.optimizer_cost_limit = declaration.optimizer_cost_limit
            root = self._require_open_span(self._root_key(state))
            root.set_attribute(
                "configuration_fingerprint",
                declaration.configuration_fingerprint,
            )
            root.set_attribute(
                "composition_fingerprint",
                declaration.composition_fingerprint,
            )
            if declaration.evaluation_call_limit is not None:
                self._metric(
                    state,
                    "evaluation_call_limit",
                    declaration.evaluation_call_limit,
                    Semantic.OPTIMIZATION_EVALUATIONS_LIMIT,
                    unit="calls",
                )
            if declaration.optimizer_cost_limit is not None:
                self._metric(
                    state,
                    "optimizer_cost_limit",
                    declaration.optimizer_cost_limit,
                    Semantic.OPTIMIZATION_OPTIMIZER_COST_LIMIT,
                    unit="usd",
                )
        if event.seed is not None:
            candidate_id = self._candidate_id(event.seed, event.candidate_id)
            state.seed_candidate_id = candidate_id
            self._remember_candidate(state, event.seed, candidate_id, "seed", event)

    def _run_completed(self, state: _Execution, event: RunCompleted) -> None:
        state.status = "completed"
        state.final_score = event.score
        state.final_candidate_id = event.candidate_id or state.best_candidate_id
        if state.final_candidate_id is not None:
            final_candidate = state.candidates.get(state.final_candidate_id)
            if final_candidate is not None:
                final_candidate.mark("final")
                root = self._require_open_span(self._root_key(state))
                self._register_effective_assets(
                    state,
                    final_candidate,
                    span_id=root.id,
                )
        if event.total_metric_calls is not None:
            state.evaluations_used = event.total_metric_calls
        if event.budget is not None:
            budget = event.budget
            evaluation_limit_declared = state.evaluations_limit is not None
            optimizer_limit_declared = state.optimizer_cost_limit is not None
            if budget.evaluation_calls is not None:
                state.evaluations_used = budget.evaluation_calls
            state.evaluations_limit = budget.evaluation_call_limit
            state.evaluations_remaining = (
                None
                if budget.evaluation_call_limit is None or budget.evaluation_calls is None
                else max(0, budget.evaluation_call_limit - budget.evaluation_calls)
            )
            state.optimizer_cost_used = budget.optimizer_cost
            state.optimizer_cost_limit = budget.optimizer_cost_limit
            state.optimizer_cost_remaining = (
                None
                if budget.optimizer_cost_limit is None or budget.optimizer_cost is None
                else max(0.0, budget.optimizer_cost_limit - budget.optimizer_cost)
            )
            state.evaluation_cost_used = budget.evaluation_cost
            state.total_cost_used = budget.total_cost
            if not state.budget_event_seen:
                self._snapshot_metrics(
                    state,
                    event.budget,
                    include_evaluation_limit=not evaluation_limit_declared,
                    include_optimizer_limit=not optimizer_limit_declared,
                )
        if event.score is not None:
            self._metric(
                state,
                "final_score",
                event.score,
                Semantic.EVALUATION_SCORE,
                direction=self._objective_direction(state),
                role=ObservationRole.OBJECTIVE,
                tags={"score_scope": "final"},
            )
        self._close_all(state, terminal_key=self._root_key(state))

    def _run_failed(self, state: _Execution, event: RunFailed) -> None:
        state.status = "failed"
        state.stop_reason = event.error_type
        self._event(
            state,
            event,
            value={"error_type": event.error_type, "message": event.message},
        )
        self._close_all(
            state,
            terminal_key=self._root_key(state),
            error=f"{event.error_type}: {event.message}",
        )

    def _run_cancelled(self, state: _Execution, event: RunCancelled) -> None:
        state.status = "cancelled"
        state.stop_reason = event.error_type
        self._event(
            state,
            event,
            value={"error_type": event.error_type, "message": event.message},
        )
        self._close_all(
            state,
            terminal_key=self._root_key(state),
            reason=EndReason.CANCELLED,
            partial=True,
        )

    def _components(self, state: _Execution, event: ComponentsRegistered) -> None:
        span_id = self._require_open_span(self._root_key(state)).id
        for component in event.components:
            state.components[component.name] = component
            self._assets.definition(
                component,
                execution_id=state.execution_id,
                span_id=span_id,
            )
        if state.seed_candidate_id is not None:
            seed = state.candidates[state.seed_candidate_id]
            self._register_effective_assets(state, seed, span_id=span_id)

    def _stage_started(self, state: _Execution, event: StageStarted) -> None:
        key = self._stage_key(state, event)
        operation, kind = self._stage_operation(event)
        self._open(
            state,
            key,
            operation,
            parent_key=self._stage_parent(state, event),
            kind=kind,
            attributes=self._attributes(event),
        )
        if event.stage_kind == "engine":
            engine_id = event.engine_execution_id or event.stage_id or key
            state.engines[engine_id] = EngineSummary(
                execution_id=engine_id,
                engine=event.engine,
                pipeline_id=event.pipeline_id,
                step_id=event.step_id,
                branch_id=event.branch_id,
            )

    def _stage_completed(self, state: _Execution, event: StageCompleted) -> None:
        key = self._stage_key(state, event)
        if event.score is not None:
            self._metric(
                state,
                "stage_score",
                event.score,
                Semantic.EVALUATION_SCORE,
                direction=self._objective_direction(state),
                role=ObservationRole.OBJECTIVE,
                span_key=key,
                tags={"score_scope": "stage"},
            )
        if event.budget is not None:
            self._snapshot_metrics(
                state,
                event.budget,
                span_key=key,
                tags={"accounting": "aggregate_component", "resource_scope": "engine"},
            )
        self._finish_engine(state, event, status="completed", score=event.score)
        self._close(state, key, output=None if event.score is None else {"score": event.score})

    def _stage_failed(self, state: _Execution, event: StageFailed) -> None:
        self._finish_engine(state, event, status="failed", score=None)
        self._record_error(
            self._stage_key(state, event),
            event.error_type,
            event.message,
        )
        self._close(
            state,
            self._stage_key(state, event),
            attributes={"error_type": event.error_type, "message": event.message},
            status=SpanStatus.ERROR,
        )

    def _iteration_started(self, state: _Execution, event: IterationStarted) -> None:
        if self._detail != "full":
            return
        self._open(
            state,
            self._iteration_key(state, event),
            "pydantic_gepa.iteration",
            parent_key=self._engine_or_root(state, event),
            kind="workflow",
            attributes=self._attributes(event),
        )

    def _iteration_completed(self, state: _Execution, event: IterationCompleted) -> None:
        if self._detail != "full":
            return
        key = self._iteration_key(state, event)
        if event.score is not None:
            self._metric(
                state,
                "iteration_score",
                event.score,
                Semantic.EVALUATION_SCORE,
                direction=self._objective_direction(state),
                role=ObservationRole.OBJECTIVE,
                span_key=key,
                tags={"score_scope": "iteration"},
            )
        self._close(state, key, output=None if event.score is None else {"score": event.score})

    def _evaluation_started(self, state: _Execution, event: EvaluationStarted) -> None:
        if self._detail == "summary":
            return
        span = self._open(
            state,
            self._evaluation_key(state, event.evaluation_id),
            "pydantic_gepa.evaluation",
            parent_key=self._iteration_or_engine(state, event),
            kind="evaluation",
            attributes=self._attributes(event)
            | {"evaluation_id": event.evaluation_id, "split": event.split},
        )
        if event.candidate is not None and span is not None:
            candidate_id = self._candidate_id(event.candidate, event.candidate_id)
            current = state.candidates.get(candidate_id)
            candidate = self._remember_candidate(
                state,
                event.candidate,
                candidate_id,
                "proposed" if current is None else current.status,
                event,
            )
            self._register_effective_assets(state, candidate, span_id=span.id)

    def _evaluation_completed(self, state: _Execution, event: EvaluationCompleted) -> None:
        if event.scores:
            self._metric(
                state,
                "evaluation_score",
                fmean(event.scores),
                Semantic.EVALUATION_SCORE,
                direction=self._objective_direction(state),
                role=ObservationRole.OBJECTIVE,
                span_key=(
                    None
                    if self._detail == "summary"
                    else self._evaluation_key(state, event.evaluation_id)
                ),
                tags={"split": event.split, "score_scope": "evaluation"},
            )
        if self._detail != "summary":
            self._close(
                state,
                self._evaluation_key(state, event.evaluation_id),
                output={"scores": list(event.scores), "case_count": event.case_count},
            )

    def _evaluation_skipped(self, state: _Execution, event: EvaluationSkipped) -> None:
        self._event(state, event, value=event.reason)
        if self._detail != "summary":
            self._close(
                state,
                self._evaluation_key(state, event.evaluation_id),
                attributes={"skipped": True, "reason": event.reason},
                reason=EndReason.DEFERRED,
            )

    def _case_evaluated(self, state: _Execution, event: CaseEvaluated) -> None:
        if self._detail == "summary":
            return
        key = self._case_key(state, event)
        parent = self._evaluation_key(state, event.evaluation_id)
        if parent not in state.open_keys:
            parent = self._root_key(state)
        self._open(
            state,
            key,
            "pydantic_gepa.case",
            parent_key=parent,
            kind="evaluation",
            attributes=self._attributes(event)
            | {
                "evaluation_id": event.evaluation_id,
                "split": event.split,
                "cache_hit": event.result.cache_hit,
                "invocation_count": event.result.invocation_count,
                "duration_seconds": event.result.duration_seconds,
            },
        )
        for name, metric in event.result.metrics.items():
            self._case_metric(
                state,
                event,
                name,
                metric.score,
                role=metric.role,
                semantic_type=metric.semantic_type,
                unit=metric.unit,
                direction=metric.direction,
                feedback=metric.feedback,
                span_key=key,
            )
        for name, score in event.result.objectives.items():
            self._case_metric(
                state,
                event,
                name,
                score,
                role="objective",
                semantic_type=None,
                unit=None,
                direction=None,
                feedback=event.result.feedback.get(name),
                span_key=key,
            )
        if event.result.side_info:
            self._runtime.event(
                self._info,
                "evaluation.side_info",
                event.result.side_info,
                semantic_type=Semantic.EVENT_OCCURRENCE,
                span_key=key,
                tags=self._tags(event),
            )
        if event.result.traces and self._detail == "full":
            self._runtime.event(
                self._info,
                "evaluation.component_traces",
                [trace.model_dump(mode="json") for trace in event.result.traces],
                semantic_type=Semantic.EVENT_OCCURRENCE,
                span_key=key,
                tags=self._tags(event),
            )
        if event.result.artifacts:
            self._runtime.event(
                self._info,
                "evaluation.artifacts",
                [artifact.model_dump(mode="json") for artifact in event.result.artifacts],
                semantic_type=Semantic.EVENT_OCCURRENCE,
                span_key=key,
                tags=self._tags(event),
            )
        errors = tuple(
            error
            for error in (
                event.result.candidate_error,
                event.result.task_error,
                event.result.evaluator_error,
                event.result.infrastructure_error,
            )
            if error is not None
        )
        if errors:
            self._runtime.event(
                self._info,
                "evaluation.errors",
                [error.model_dump(mode="json") for error in errors],
                semantic_type=Semantic.ERROR_EXCEPTION,
                span_key=key,
                tags=self._tags(event),
            )
            current = self._require_open_span(key)
            for error in errors:
                current.record_exception(f"{error.kind}: {error.message}")
        self._close(
            state,
            key,
            output=event.result.output,
            status=SpanStatus.ERROR if errors else SpanStatus.OK,
            attributes={"error_count": len(errors)},
        )

    def _reflection_started(self, state: _Execution, event: ReflectionStarted) -> None:
        if self._detail != "full":
            return
        self._open(
            state,
            self._reflection_key(state, event),
            "pydantic_gepa.reflection",
            parent_key=self._iteration_or_engine(state, event),
            kind="reflection",
            attributes=self._attributes(event),
        )

    def _reflection_completed(self, state: _Execution, event: ReflectionCompleted) -> None:
        if self._detail == "full":
            self._close(state, self._reflection_key(state, event))

    def _candidate_changed(
        self,
        state: _Execution,
        event: CandidateProposed | CandidateNormalized,
    ) -> None:
        candidate = event.candidate
        if candidate is None:
            candidate_id = event.candidate_id or f"proposal:{event.sequence}"
        else:
            candidate_id = self._candidate_id(candidate, event.candidate_id)
        status: CandidateStatus = (
            "normalized" if isinstance(event, CandidateNormalized) else "proposed"
        )
        previous = state.candidates.get(candidate_id)
        previous_status = None if previous is None else previous.status
        candidate_state = self._remember_candidate(state, candidate, candidate_id, status, event)
        if self._detail != "summary":
            key = self._candidate_key(state, candidate_id)
            if key in state.open_keys:
                current = self._require_open_span(key)
                current.set_attribute("candidate_status", status)
                if previous_status == status:
                    self._runtime.diagnose(
                        self._info,
                        "pydantic_gepa_duplicate_candidate_transition",
                        f"candidate '{candidate_id}' repeated status '{status}'",
                    )
            else:
                self._open(
                    state,
                    key,
                    "pydantic_gepa.candidate",
                    parent_key=self._root_key(state),
                    kind="candidate",
                    attributes=self._attributes(event)
                    | {"candidate_id": candidate_id, "candidate_status": status},
                )
            current = self._require_open_span(key)
            candidate_state.span = current
            for parent_id in candidate_state.parent_ids:
                parent = state.candidates.get(parent_id)
                if parent is not None and parent.span is not None:
                    current.link_to(
                        parent.span,
                        relation=LinkRelation.RUN_LINEAGE,
                        attributes={"candidate_parent_id": parent_id},
                    )
            self._register_effective_assets(
                state,
                candidate_state,
                span_id=current.id,
            )

    def _candidate_evaluated(self, state: _Execution, event: CandidateEvaluated) -> None:
        candidate_id = event.candidate_id or f"candidate:{event.sequence}"
        candidate = state.candidates.get(candidate_id)
        if candidate is None:
            candidate = _Candidate(
                id=candidate_id,
                status="evaluated",
                statuses=["evaluated"],
            )
            state.candidates[candidate_id] = candidate
        else:
            candidate.mark("evaluated")
        candidate.score = event.score
        if event.score is not None:
            self._metric(
                state,
                "candidate_score",
                event.score,
                Semantic.EVALUATION_SCORE,
                direction=self._objective_direction(state),
                role=ObservationRole.OBJECTIVE,
                span_key=self._candidate_metric_parent(state, event, candidate_id),
                tags={"candidate_id": candidate_id, "score_scope": "candidate"},
            )

    def _candidate_terminal(
        self,
        state: _Execution,
        event: CandidateAccepted | CandidateRejected,
        status: Literal["accepted", "rejected"],
    ) -> None:
        candidate_id = event.candidate_id or f"candidate:{event.sequence}"
        candidate = state.candidates.get(candidate_id)
        if candidate is None:
            candidate = _Candidate(id=candidate_id, status=status, statuses=[status])
            state.candidates[candidate_id] = candidate
        else:
            candidate.mark(status)
        candidate.score = event.score
        if status == "accepted" and self._is_better(state, event.score, state.best_candidate_id):
            state.best_candidate_id = candidate_id
            candidate.mark("best")
        candidate_key = self._candidate_key(state, candidate_id)
        if self._detail != "summary" and candidate_key not in state.open_keys:
            candidate.span = self._open(
                state,
                candidate_key,
                "pydantic_gepa.candidate",
                parent_key=self._root_key(state),
                kind="candidate",
                attributes=self._attributes(event)
                | {"candidate_id": candidate_id, "partial_start": True},
            )
        self._runtime.event(
            self._info,
            "candidate_status",
            status,
            semantic_type=Semantic.EVALUATION_LABEL,
            span_key=(self._root_key(state) if self._detail == "summary" else candidate_key),
            tags=self._tags(event) | {"candidate_status": status},
        )
        if self._detail != "summary":
            self._close(
                state,
                candidate_key,
                output=None if event.score is None else {"score": event.score},
                attributes={
                    "candidate_status": status,
                    "reason": event.reason if isinstance(event, CandidateRejected) else None,
                },
                status=SpanStatus.OK,
            )

    def _metric_started(self, state: _Execution, event: MetricStarted) -> None:
        if self._detail == "summary":
            return
        self._open(
            state,
            self._metric_key(state, event),
            "pydantic_gepa.metric",
            parent_key=self._metric_parent(state, event),
            kind="scorer",
            attributes=self._attributes(event),
        )

    def _metric_completed(self, state: _Execution, event: MetricCompleted) -> None:
        key = self._metric_key(state, event)
        target = key if key in state.open_keys else self._metric_parent(state, event)
        identity = (event.evaluation_id or "", event.case_id, event.metric or "metric")
        if identity not in state.observed_metrics:
            state.observed_metrics.add(identity)
            self._metric(
                state,
                event.metric or "metric",
                event.value,
                event.semantic_type or Semantic.EVALUATION_SCORE,
                unit=event.unit,
                direction=self._direction(event.direction),
                role=self._role(event.role),
                span_key=None if self._detail == "summary" else target,
                tags=self._tags(event)
                | {
                    "raw_or_transformed": "raw",
                    "transformed_value": event.transformed_value,
                },
            )
        if self._detail != "summary" and key in state.open_keys:
            self._close(
                state,
                key,
                output={
                    "value": event.value,
                    "transformed_value": event.transformed_value,
                },
            )

    def _metric_failed(self, state: _Execution, event: MetricFailed) -> None:
        key = self._metric_key(state, event)
        target = key if key in state.open_keys else self._metric_parent(state, event)
        self._record_error(target, event.error_type, event.message)
        if self._detail != "summary" and key in state.open_keys:
            self._close(
                state,
                key,
                attributes={"error_type": event.error_type, "message": event.message},
                status=SpanStatus.ERROR,
            )

    def _budget_updated(self, state: _Execution, event: BudgetUpdated) -> None:
        state.budget_event_seen = True
        state.evaluations_used = event.used
        state.evaluations_remaining = event.remaining
        state.optimizer_cost_used = event.optimizer_cost
        state.optimizer_cost_remaining = event.optimizer_cost_remaining
        state.evaluation_cost_used = event.evaluation_cost
        state.total_cost_used = event.total_cost
        self._metric(
            state,
            "evaluation_calls_used",
            event.used,
            Semantic.OPTIMIZATION_EVALUATIONS_USED,
            unit="calls",
        )
        if event.remaining is not None:
            self._metric(
                state,
                "evaluation_calls_remaining",
                event.remaining,
                Semantic.OPTIMIZATION_EVALUATIONS_REMAINING,
                unit="calls",
            )
        if event.optimizer_cost is not None:
            self._metric(
                state,
                "optimizer_cost_used",
                event.optimizer_cost,
                Semantic.OPTIMIZATION_OPTIMIZER_COST_USED,
                unit="usd",
                tags={"cost_scope": "optimizer", "accounting": "aggregate_component"},
            )
        if event.optimizer_cost_remaining is not None:
            self._metric(
                state,
                "optimizer_cost_remaining",
                event.optimizer_cost_remaining,
                Semantic.OPTIMIZATION_OPTIMIZER_COST_REMAINING,
                unit="usd",
            )
        if event.evaluation_cost is not None:
            self._metric(
                state,
                "evaluation_cost_used",
                event.evaluation_cost,
                Semantic.OPTIMIZATION_EVALUATION_COST_USED,
                unit="usd",
                tags={"cost_scope": "evaluation", "accounting": "aggregate_component"},
            )
        if event.total_cost is not None:
            self._metric(
                state,
                "optimization_cost_used",
                event.total_cost,
                Semantic.OPTIMIZATION_COST,
                unit="usd",
                tags={"cost_scope": "optimization", "accounting": "aggregate"},
            )

    def _snapshot_metrics(
        self,
        state: _Execution,
        budget: BudgetSnapshot,
        *,
        span_key: str | None = None,
        tags: dict[str, SerializedValue] | None = None,
        include_evaluation_limit: bool = True,
        include_optimizer_limit: bool = True,
    ) -> None:
        metric_tags = dict(tags or {"accounting": "aggregate"})
        if budget.evaluation_calls is not None:
            self._metric(
                state,
                "evaluation_calls_used",
                budget.evaluation_calls,
                Semantic.OPTIMIZATION_EVALUATIONS_USED,
                unit="calls",
                span_key=span_key,
                tags=metric_tags,
            )
        if budget.evaluation_call_limit is not None:
            if include_evaluation_limit:
                self._metric(
                    state,
                    "evaluation_call_limit",
                    budget.evaluation_call_limit,
                    Semantic.OPTIMIZATION_EVALUATIONS_LIMIT,
                    unit="calls",
                    span_key=span_key,
                    tags=metric_tags,
                )
            if budget.evaluation_calls is not None:
                self._metric(
                    state,
                    "evaluation_calls_remaining",
                    max(0, budget.evaluation_call_limit - budget.evaluation_calls),
                    Semantic.OPTIMIZATION_EVALUATIONS_REMAINING,
                    unit="calls",
                    span_key=span_key,
                    tags=metric_tags,
                )
        if budget.optimizer_cost_limit is not None:
            if include_optimizer_limit:
                self._metric(
                    state,
                    "optimizer_cost_limit",
                    budget.optimizer_cost_limit,
                    Semantic.OPTIMIZATION_OPTIMIZER_COST_LIMIT,
                    unit="usd",
                    span_key=span_key,
                    tags=metric_tags,
                )
            if budget.optimizer_cost is not None:
                self._metric(
                    state,
                    "optimizer_cost_remaining",
                    max(0.0, budget.optimizer_cost_limit - budget.optimizer_cost),
                    Semantic.OPTIMIZATION_OPTIMIZER_COST_REMAINING,
                    unit="usd",
                    span_key=span_key,
                    tags=metric_tags,
                )
        if budget.optimizer_cost is not None:
            self._metric(
                state,
                "optimizer_cost_used",
                budget.optimizer_cost,
                Semantic.OPTIMIZATION_OPTIMIZER_COST_USED,
                unit="usd",
                span_key=span_key,
                tags=metric_tags | {"cost_scope": "optimizer", "accounting": "aggregate_component"},
            )
        if budget.evaluation_cost is not None:
            self._metric(
                state,
                "evaluation_cost_used",
                budget.evaluation_cost,
                Semantic.OPTIMIZATION_EVALUATION_COST_USED,
                unit="usd",
                span_key=span_key,
                tags=metric_tags
                | {"cost_scope": "evaluation", "accounting": "aggregate_component"},
            )
        if budget.total_cost is not None:
            self._metric(
                state,
                "optimization_cost_used",
                budget.total_cost,
                Semantic.OPTIMIZATION_COST,
                unit="usd",
                span_key=span_key,
                tags=metric_tags | {"cost_scope": "optimization", "accounting": "aggregate"},
            )

    def _selection(self, state: _Execution, event: SelectionCompleted) -> None:
        state.selections.append(
            SelectionSummary(
                method=event.method,
                selected_execution_id=event.selected_execution_id,
                contender_execution_ids=event.contender_execution_ids,
                contender_scores=event.contender_scores,
                score=event.score,
                reason=event.reason,
            )
        )
        self._event(
            state,
            event,
            value={
                "method": event.method,
                "selected_execution_id": event.selected_execution_id,
                "contender_execution_ids": list(event.contender_execution_ids),
                "contender_scores": list(event.contender_scores),
                "score": event.score,
                "reason": event.reason,
            },
        )

    def _final_rescore_started(self, state: _Execution, event: FinalRescoreStarted) -> None:
        self._open(
            state,
            self._rescore_key(state),
            "pydantic_gepa.final_rescore",
            parent_key=self._root_key(state),
            kind="evaluation",
            attributes=self._attributes(event),
        )

    def _final_rescore_completed(self, state: _Execution, event: FinalRescoreCompleted) -> None:
        state.final_score = event.score
        self._metric(
            state,
            "final_rescore",
            event.score,
            Semantic.EVALUATION_SCORE,
            direction=self._objective_direction(state),
            role=ObservationRole.OBJECTIVE,
            span_key=self._rescore_key(state),
            tags={"score_scope": "final_rescore"},
        )
        self._close(state, self._rescore_key(state), output={"score": event.score})

    def _checkpoint(
        self,
        state: _Execution,
        event: CheckpointWritten | CheckpointResumed | CheckpointRejected | CheckpointReset,
    ) -> None:
        state.checkpoint_paths.append(event.path)
        if self._detail == "full":
            value: dict[str, SerializedValue] = {"path": event.path}
            if isinstance(event, CheckpointRejected):
                value["reason"] = event.reason
            self._event(state, event, value=value)

    def _case_metric(
        self,
        state: _Execution,
        event: CaseEvaluated,
        name: str,
        value: float,
        *,
        role: str,
        semantic_type: str | None,
        unit: str | None,
        direction: str | None,
        feedback: str | None,
        span_key: str,
    ) -> None:
        identity = (event.evaluation_id, event.case_id, name)
        if identity in state.observed_metrics:
            return
        state.observed_metrics.add(identity)
        tags = self._tags(event) | {"metric_name": name, "split": event.split}
        self._metric(
            state,
            name,
            value,
            semantic_type or Semantic.EVALUATION_SCORE,
            unit=unit,
            direction=self._direction(direction),
            role=self._role(role),
            span_key=span_key,
            tags=tags,
        )
        if feedback is not None:
            self._runtime.event(
                self._info,
                f"{name}.feedback",
                feedback,
                semantic_type=Semantic.EVALUATION_EXPLANATION,
                span_key=span_key,
                tags=tags,
            )


__all__ = ("Detail", "EventAdapter")
