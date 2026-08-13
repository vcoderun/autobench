from __future__ import annotations as _annotations

from typing import Literal

from pydantic_gepa.candidates import Candidate
from pydantic_gepa.events import (
    CaseEvaluated,
    Event,
    MetricCompleted,
    RunStarted,
    StageCompleted,
    StageFailed,
    StageStarted,
)

from autobench.instrumentation.manager import CurrentSpan, InstrumentationRuntime
from autobench.instrumentation.models import InstrumentorInfo
from autobench.instrumentation.pydantic_gepa._state import _Candidate, _Execution
from autobench.instrumentation.pydantic_gepa.assets import CandidateAssets
from autobench.instrumentation.pydantic_gepa.projection import (
    EXTENSION_KEY,
    CandidateStatus,
    EngineSummary,
    PydanticGEPAEvidence,
    PydanticGEPAStatus,
)
from autobench.metrics.observations import Direction, ObservationRole
from autobench.metrics.semantics import Semantic
from autobench.protocol.ids import TraceId
from autobench.protocol.signals import EndReason, SpanStatus
from autobench.protocol.values import SerializedValue

Detail = Literal["summary", "evaluations", "full"]


class _EventEvidence:
    def __init__(
        self,
        runtime: InstrumentationRuntime,
        info: InstrumentorInfo,
        assets: CandidateAssets,
        *,
        detail: Detail,
    ) -> None:
        self._runtime = runtime
        self._info = info
        self._assets = assets
        self._detail = detail
        self._states: dict[tuple[TraceId, str], _Execution] = {}

    def _remember_candidate(
        self,
        state: _Execution,
        candidate: Candidate | None,
        candidate_id: str,
        status: CandidateStatus,
        event: Event,
    ) -> _Candidate:
        remembered = state.candidates.get(candidate_id)
        parent_ids = event.parent_ids
        if candidate is not None and candidate.parent_id is not None:
            parent_ids = tuple(dict.fromkeys((*parent_ids, candidate.parent_id)))
        if remembered is None:
            remembered = _Candidate(
                id=candidate_id,
                status=status,
                statuses=[status],
                candidate=candidate,
                parent_ids=parent_ids,
                generation=None if candidate is None else candidate.generation,
                iteration=event.iteration,
            )
            state.candidates[candidate_id] = remembered
        else:
            remembered.mark(status)
            remembered.candidate = candidate or remembered.candidate
            remembered.parent_ids = parent_ids or remembered.parent_ids
            remembered.generation = (
                remembered.generation if candidate is None else candidate.generation
            )
            remembered.iteration = event.iteration or remembered.iteration
        return remembered

    def _register_effective_assets(
        self,
        state: _Execution,
        candidate: _Candidate,
        *,
        span_id: str | None,
    ) -> None:
        if candidate.candidate is None:
            return
        for component in state.components.values():
            registered = self._assets.effective(
                component,
                candidate.candidate,
                candidate_id=candidate.id,
                execution_id=state.execution_id,
                iteration=candidate.iteration,
                status=candidate.status,
                span_id=span_id,
            )
            if registered is not None:
                candidate.component_versions[component.name] = (
                    f"{registered.version.asset_id}@{registered.version.version}"
                )

    def _ensure_root(self, state: _Execution, event: Event) -> None:
        key = self._root_key(state)
        if key in state.open_keys or key in state.closed_keys:
            return
        parent_key = None
        if event.parent_execution_id is not None:
            candidate_parent = self._root_key_for(event.parent_execution_id)
            if self._runtime.span_for_key(self._info, candidate_parent) is not None:
                parent_key = candidate_parent
        self._open(
            state,
            key,
            "pydantic_gepa.optimization",
            parent_key=parent_key,
            kind="optimization",
            attributes=self._attributes(event)
            | {"partial_start": not isinstance(event, RunStarted)},
        )

    def _open(
        self,
        state: _Execution,
        key: str,
        operation: str,
        *,
        parent_key: str | None,
        kind: str,
        attributes: dict[str, SerializedValue],
    ) -> CurrentSpan | None:
        if key in state.open_keys:
            self._runtime.start_span(
                self._info,
                key,
                operation,
                parent_key=parent_key,
                kind=kind,
                attributes=attributes,
            )
            return None
        if key in state.closed_keys:
            self._runtime.diagnose(
                self._info,
                "pydantic_gepa_duplicate_start",
                f"span key '{key}' has already completed",
            )
            return None
        span = self._runtime.start_span(
            self._info,
            key,
            operation,
            parent_key=parent_key,
            kind=kind,
            attributes=attributes,
        )
        if span is None:
            raise RuntimeError(f"could not start pydantic-gepa span '{key}'")
        state.open_keys.append(key)
        return span

    def _close(
        self,
        state: _Execution,
        key: str,
        *,
        output: SerializedValue = None,
        attributes: dict[str, SerializedValue] | None = None,
        status: SpanStatus | None = None,
        reason: EndReason | None = None,
        partial: bool | None = None,
    ) -> None:
        if key in state.closed_keys:
            self._runtime.end_span(self._info, key)
            return
        if key not in state.open_keys:
            self._open(
                state,
                key,
                "pydantic_gepa.partial",
                parent_key=self._root_key(state),
                kind="custom",
                attributes={"partial_start": True},
            )
            partial = True
        self._runtime.end_span(
            self._info,
            key,
            output=output,
            attributes=attributes,
            status=status,
            reason=reason,
            partial=partial,
        )
        state.open_keys.remove(key)
        state.closed_keys.add(key)

    def _require_open_span(self, key: str) -> CurrentSpan:
        span = self._runtime.span_for_key(self._info, key)
        if span is None:
            raise RuntimeError(f"pydantic-gepa span '{key}' is not active")
        return span

    def _close_all(
        self,
        state: _Execution,
        *,
        terminal_key: str,
        error: str | None = None,
        reason: EndReason | None = None,
        partial: bool | None = None,
    ) -> None:
        completed_normally = error is None and reason is None and partial is not True
        candidate_prefix = f"pydantic_gepa:{state.execution_id}:candidate:"
        for key in tuple(reversed(state.open_keys)):
            if key == terminal_key:
                continue
            if completed_normally and key.startswith(candidate_prefix):
                candidate_id = key.removeprefix(candidate_prefix)
                candidate = state.candidates.get(candidate_id)
                self._close(
                    state,
                    key,
                    attributes={
                        "candidate_status": None if candidate is None else candidate.status,
                    },
                    status=SpanStatus.OK,
                )
                continue
            self._close(
                state,
                key,
                reason=EndReason.ABANDONED,
                partial=True,
            )
        if terminal_key in state.open_keys:
            self._runtime.end_span(
                self._info,
                terminal_key,
                output=None if state.final_score is None else {"score": state.final_score},
                error=error,
                reason=reason,
                partial=partial,
            )
            state.open_keys.remove(terminal_key)
            state.closed_keys.add(terminal_key)

    def _metric(
        self,
        state: _Execution,
        name: str,
        value: bool | int | float,
        semantic_type: str,
        *,
        unit: str | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        span_key: str | None = None,
        tags: dict[str, SerializedValue] | None = None,
    ) -> None:
        target_key = self._root_key(state) if span_key is None else span_key
        self._runtime.metric(
            self._info,
            name,
            value,
            semantic_type=semantic_type,
            unit=unit,
            direction=direction,
            role=role,
            span_key=target_key,
            tags=dict(tags or {}),
        )

    def _event(self, state: _Execution, event: Event, *, value: SerializedValue) -> None:
        self._runtime.event(
            self._info,
            event.kind,
            value,
            semantic_type=Semantic.EVENT_OCCURRENCE,
            span_key=self._event_span(state, event),
            tags=self._tags(event),
        )

    def _record_error(
        self,
        span_key: str,
        error_type: str,
        message: str,
    ) -> None:
        current = self._runtime.span_for_key(self._info, span_key)
        if current is not None:
            current.record_exception(f"{error_type}: {message}")

    def _write_projection(self, trace_id: TraceId) -> None:
        evidence = PydanticGEPAEvidence(
            executions=tuple(
                state.summary()
                for state in sorted(
                    (state for state in self._states.values() if state.trace_id == trace_id),
                    key=lambda item: item.order,
                )
            )
        )
        self._runtime.set_extension(
            self._info,
            EXTENSION_KEY,
            evidence.model_dump(mode="json"),
        )

    def _finish_engine(
        self,
        state: _Execution,
        event: StageCompleted | StageFailed,
        *,
        status: PydanticGEPAStatus,
        score: float | None,
    ) -> None:
        if event.stage_kind != "engine":
            return
        engine_id = event.engine_execution_id or event.stage_id or self._stage_key(state, event)
        existing = state.engines.get(engine_id)
        if existing is None:
            existing = EngineSummary(
                execution_id=engine_id,
                engine=event.engine,
                pipeline_id=event.pipeline_id,
                step_id=event.step_id,
                branch_id=event.branch_id,
            )
        update: dict[str, str | int | float | None] = {"status": status, "score": score}
        if isinstance(event, StageCompleted) and event.budget is not None:
            update |= {
                "evaluations_used": event.budget.evaluation_calls,
                "evaluations_limit": event.budget.evaluation_call_limit,
                "optimizer_cost_used": event.budget.optimizer_cost,
                "optimizer_cost_limit": event.budget.optimizer_cost_limit,
                "evaluation_cost_used": event.budget.evaluation_cost,
                "total_cost_used": event.budget.total_cost,
            }
        state.engines[engine_id] = existing.model_copy(update=update)

    def _is_better(
        self,
        state: _Execution,
        score: float | None,
        current_id: str | None,
    ) -> bool:
        if score is None:
            return current_id is None
        if current_id is None:
            return True
        current = state.candidates[current_id]
        if current.score is None:
            return True
        if state.objective is not None and state.objective.direction == "minimize":
            return score < current.score
        return score > current.score

    @staticmethod
    def _candidate_id(candidate: Candidate, event_candidate_id: str | None) -> str:
        return event_candidate_id or candidate.id or f"candidate:{candidate.fingerprint()[:16]}"

    @staticmethod
    def _role(role: str) -> ObservationRole:
        return {
            "objective": ObservationRole.OBJECTIVE,
            "constraint": ObservationRole.CONSTRAINT,
            "diagnostic": ObservationRole.DIAGNOSTIC,
        }.get(role, ObservationRole.DIAGNOSTIC)

    @staticmethod
    def _direction(direction: str | None) -> Direction | None:
        if direction == "maximize":
            return Direction.MAXIMIZE
        if direction == "minimize":
            return Direction.MINIMIZE
        return None

    @staticmethod
    def _objective_direction(state: _Execution) -> Direction | None:
        return (
            None
            if state.objective is None
            else _EventEvidence._direction(state.objective.direction)
        )

    @staticmethod
    def _attributes(event: Event) -> dict[str, SerializedValue]:
        return {
            "source_event": event.kind,
            "source_sequence": event.sequence,
            "source_occurred_at": event.occurred_at.isoformat(),
            "source_monotonic_ns": event.monotonic_ns,
            "run_id": event.run_id,
            "execution_id": event.execution_id,
            "parent_execution_id": event.parent_execution_id,
            "backend": event.backend,
            "engine": event.engine,
            "composition": event.composition,
            "pipeline_id": event.pipeline_id,
            "step_id": event.step_id,
            "branch_id": event.branch_id,
            "engine_execution_id": event.engine_execution_id,
            "stage_id": event.stage_id,
            "stage_kind": event.stage_kind,
            "iteration": event.iteration,
            "candidate_id": event.candidate_id,
            "case_id": event.case_id,
            "metric": event.metric,
            "metadata": event.metadata,
        }

    @staticmethod
    def _tags(event: Event) -> dict[str, SerializedValue]:
        return {
            name: value
            for name, value in {
                "optimizer_run_id": event.run_id,
                "execution_id": event.execution_id,
                "backend": event.backend,
                "engine": event.engine,
                "composition": event.composition,
                "pipeline_id": event.pipeline_id,
                "step_id": event.step_id,
                "branch_id": event.branch_id,
                "engine_execution_id": event.engine_execution_id,
                "stage_id": event.stage_id,
                "iteration": event.iteration,
                "candidate_id": event.candidate_id,
                "case_id": event.case_id,
                "metric_name": event.metric,
            }.items()
            if value is not None
        }

    @staticmethod
    def _root_key_for(execution_id: str) -> str:
        return f"pydantic_gepa:{execution_id}:optimization"

    def _root_key(self, state: _Execution) -> str:
        return self._root_key_for(state.execution_id)

    def _stage_key(self, state: _Execution, event: Event) -> str:
        if event.stage_kind == "engine":
            return self._engine_key(state, event)
        if event.stage_kind == "rescore":
            return self._rescore_key(state)
        stage_id = event.stage_id or event.step_id or f"stage:{event.sequence}"
        return f"pydantic_gepa:{state.execution_id}:stage:{stage_id}"

    def _engine_key(self, state: _Execution, event: Event) -> str:
        engine_id = event.engine_execution_id or event.stage_id or "engine"
        return f"pydantic_gepa:{state.execution_id}:engine:{engine_id}"

    def _iteration_key(self, state: _Execution, event: Event) -> str:
        engine_id = event.engine_execution_id or "engine"
        iteration = event.iteration if event.iteration is not None else event.sequence
        return f"pydantic_gepa:{state.execution_id}:engine:{engine_id}:iteration:{iteration}"

    def _evaluation_key(self, state: _Execution, evaluation_id: str) -> str:
        return f"pydantic_gepa:{state.execution_id}:evaluation:{evaluation_id}"

    def _case_key(self, state: _Execution, event: CaseEvaluated) -> str:
        case_id = event.case_id or f"case:{event.sequence}"
        return f"{self._evaluation_key(state, event.evaluation_id)}:case:{case_id}"

    def _metric_key(self, state: _Execution, event: Event) -> str:
        evaluation_id = (
            event.evaluation_id
            if isinstance(event, MetricCompleted)
            else event.metadata.get("evaluation_id")
        )
        metric = event.metric or f"metric:{event.sequence}"
        return (
            f"pydantic_gepa:{state.execution_id}:evaluation:{evaluation_id or 'unknown'}"
            f":case:{event.case_id or 'aggregate'}:metric:{metric}"
        )

    def _reflection_key(self, state: _Execution, event: Event) -> str:
        engine_id = event.engine_execution_id or "engine"
        iteration = event.iteration if event.iteration is not None else "unknown"
        return f"pydantic_gepa:{state.execution_id}:engine:{engine_id}:reflection:{iteration}"

    def _candidate_key(self, state: _Execution, candidate_id: str) -> str:
        return f"pydantic_gepa:{state.execution_id}:candidate:{candidate_id}"

    def _rescore_key(self, state: _Execution) -> str:
        return f"pydantic_gepa:{state.execution_id}:final_rescore"

    def _stage_parent(self, state: _Execution, event: Event) -> str:
        if event.stage_kind == "engine" and event.pipeline_id is not None:
            possible = (
                f"pydantic_gepa:{state.execution_id}:stage:{event.step_id or event.pipeline_id}"
            )
            if possible in state.open_keys:
                return possible
        if event.parent_execution_id is not None:
            parent = self._root_key_for(event.parent_execution_id)
            if self._runtime.span_for_key(self._info, parent) is not None:
                return parent
        return self._root_key(state)

    def _engine_or_root(self, state: _Execution, event: Event) -> str:
        engine = self._engine_key(state, event)
        return engine if engine in state.open_keys else self._root_key(state)

    def _iteration_or_engine(self, state: _Execution, event: Event) -> str:
        iteration = self._iteration_key(state, event)
        if iteration in state.open_keys:
            return iteration
        return self._engine_or_root(state, event)

    def _metric_parent(self, state: _Execution, event: Event) -> str:
        evaluation_id = (
            event.evaluation_id
            if isinstance(event, MetricCompleted)
            else event.metadata.get("evaluation_id")
        )
        if isinstance(evaluation_id, str):
            evaluation = self._evaluation_key(state, evaluation_id)
            if evaluation in state.open_keys:
                return evaluation
        return self._root_key(state)

    def _candidate_metric_parent(
        self,
        state: _Execution,
        event: Event,
        candidate_id: str,
    ) -> str | None:
        if self._detail == "summary":
            return None
        candidate = self._candidate_key(state, candidate_id)
        if candidate in state.open_keys:
            return candidate
        return self._event_span(state, event)

    def _event_span(self, state: _Execution, event: Event) -> str:
        for key in (
            self._candidate_key(state, event.candidate_id)
            if event.candidate_id is not None
            else None,
            self._iteration_key(state, event) if event.iteration is not None else None,
            self._engine_key(state, event) if event.engine_execution_id is not None else None,
            self._root_key(state),
        ):
            if key is not None and key in state.open_keys:
                return key
        return self._root_key(state)

    @staticmethod
    def _stage_operation(event: StageStarted) -> tuple[str, str]:
        return {
            "component": ("pydantic_gepa.stage", "workflow"),
            "composition": ("pydantic_gepa.composition_step", "workflow"),
            "engine": ("pydantic_gepa.engine", "workflow"),
            "rescore": ("pydantic_gepa.final_rescore", "evaluation"),
            None: ("pydantic_gepa.stage", "workflow"),
        }[event.stage_kind]


__all__ = ()
