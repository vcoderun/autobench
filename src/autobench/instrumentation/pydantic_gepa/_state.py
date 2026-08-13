from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic_gepa.candidates import Candidate, CandidateComponent

from autobench.instrumentation.manager import CurrentSpan
from autobench.instrumentation.pydantic_gepa.projection import (
    CandidateStatus,
    CandidateSummary,
    DatasetSummary,
    EngineSummary,
    ObjectiveSummary,
    OptimizationExecution,
    PydanticGEPAStatus,
    SelectionSummary,
)
from autobench.protocol.ids import TraceId


@dataclass(slots=True)
class _Candidate:
    id: str
    status: CandidateStatus
    statuses: list[CandidateStatus] = field(default_factory=list)
    candidate: Candidate | None = None
    parent_ids: tuple[str, ...] = ()
    generation: int | None = None
    iteration: int | None = None
    score: float | None = None
    component_versions: dict[str, str] = field(default_factory=dict)
    span: CurrentSpan | None = None

    def mark(self, status: CandidateStatus) -> None:
        self.status = status
        if status not in self.statuses:
            self.statuses.append(status)

    def summary(self) -> CandidateSummary:
        return CandidateSummary(
            id=self.id,
            fingerprint=None if self.candidate is None else self.candidate.fingerprint(),
            parent_ids=self.parent_ids,
            generation=self.generation,
            iteration=self.iteration,
            status=self.status,
            statuses=tuple(self.statuses or (self.status,)),
            score=self.score,
            component_versions=dict(self.component_versions),
        )


@dataclass(slots=True)
class _Execution:
    order: int
    trace_id: TraceId
    execution_id: str
    run_id: str
    parent_execution_id: str | None = None
    backend: str | None = None
    engine: str | None = None
    composition: str | None = None
    pipeline_id: str | None = None
    step_id: str | None = None
    branch_id: str | None = None
    stage_id: str | None = None
    status: PydanticGEPAStatus = "running"
    objective: ObjectiveSummary | None = None
    datasets: DatasetSummary | None = None
    seed_candidate_id: str | None = None
    best_candidate_id: str | None = None
    final_candidate_id: str | None = None
    final_score: float | None = None
    evaluations_used: int = 0
    evaluations_limit: int | None = None
    evaluations_remaining: int | None = None
    optimizer_cost_used: float | None = None
    optimizer_cost_limit: float | None = None
    optimizer_cost_remaining: float | None = None
    evaluation_cost_used: float | None = None
    total_cost_used: float | None = None
    budget_event_seen: bool = False
    components: dict[str, CandidateComponent] = field(default_factory=dict)
    candidates: dict[str, _Candidate] = field(default_factory=dict)
    engines: dict[str, EngineSummary] = field(default_factory=dict)
    selections: list[SelectionSummary] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    event_count: int = 0
    diagnostic_count: int = 0
    open_keys: list[str] = field(default_factory=list)
    closed_keys: set[str] = field(default_factory=set)
    observed_metrics: set[tuple[str, str | None, str]] = field(default_factory=set)

    def summary(self) -> OptimizationExecution:
        return OptimizationExecution(
            execution_id=self.execution_id,
            run_id=self.run_id,
            parent_execution_id=self.parent_execution_id,
            backend=self.backend,
            engine=self.engine,
            composition=self.composition,
            pipeline_id=self.pipeline_id,
            step_id=self.step_id,
            branch_id=self.branch_id,
            stage_id=self.stage_id,
            status=self.status,
            objective=self.objective,
            datasets=self.datasets,
            seed_candidate_id=self.seed_candidate_id,
            best_candidate_id=self.best_candidate_id,
            final_candidate_id=self.final_candidate_id,
            final_score=self.final_score,
            evaluations_used=self.evaluations_used,
            evaluations_limit=self.evaluations_limit,
            evaluations_remaining=self.evaluations_remaining,
            optimizer_cost_used=self.optimizer_cost_used,
            optimizer_cost_limit=self.optimizer_cost_limit,
            optimizer_cost_remaining=self.optimizer_cost_remaining,
            evaluation_cost_used=self.evaluation_cost_used,
            total_cost_used=self.total_cost_used,
            candidates=tuple(candidate.summary() for candidate in self.candidates.values()),
            engines=tuple(self.engines.values()),
            selections=tuple(self.selections),
            checkpoint_paths=tuple(dict.fromkeys(self.checkpoint_paths)),
            stop_reason=self.stop_reason,
            event_count=self.event_count,
            diagnostic_count=self.diagnostic_count,
        )


__all__ = ()
