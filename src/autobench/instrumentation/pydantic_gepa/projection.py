from __future__ import annotations as _annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PydanticGEPAStatus = Literal[
    "running",
    "completed",
    "failed",
    "cancelled",
    "partial",
]
CandidateStatus = Literal[
    "seed",
    "proposed",
    "normalized",
    "evaluated",
    "accepted",
    "rejected",
    "best",
    "final",
]


class ObjectiveSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    role: str
    direction: str | None = None
    semantic_type: str | None = None
    unit: str | None = None


class DatasetSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    test_count: int = Field(ge=0)
    train_fingerprint: str | None = None
    validation_fingerprint: str | None = None
    test_fingerprint: str | None = None


class CandidateSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    fingerprint: str | None = None
    parent_ids: tuple[str, ...] = ()
    generation: int | None = Field(default=None, ge=0)
    iteration: int | None = Field(default=None, ge=0)
    status: CandidateStatus
    statuses: tuple[CandidateStatus, ...] = ()
    score: float | None = None
    component_versions: dict[str, str] = Field(default_factory=dict)


class EngineSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    engine: str | None = None
    pipeline_id: str | None = None
    step_id: str | None = None
    branch_id: str | None = None
    status: PydanticGEPAStatus = "running"
    score: float | None = None
    evaluations_used: int | None = Field(default=None, ge=0)
    evaluations_limit: int | None = Field(default=None, ge=0)
    optimizer_cost_used: float | None = Field(default=None, ge=0)
    optimizer_cost_limit: float | None = Field(default=None, ge=0)
    evaluation_cost_used: float | None = Field(default=None, ge=0)
    total_cost_used: float | None = Field(default=None, ge=0)


class SelectionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str
    selected_execution_id: str
    contender_execution_ids: tuple[str, ...]
    contender_scores: tuple[float, ...]
    score: float
    reason: str | None = None


class OptimizationExecution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    evaluations_used: int = Field(default=0, ge=0)
    evaluations_limit: int | None = Field(default=None, ge=0)
    evaluations_remaining: int | None = Field(default=None, ge=0)
    optimizer_cost_used: float | None = Field(default=None, ge=0)
    optimizer_cost_limit: float | None = Field(default=None, ge=0)
    optimizer_cost_remaining: float | None = Field(default=None, ge=0)
    evaluation_cost_used: float | None = Field(default=None, ge=0)
    total_cost_used: float | None = Field(default=None, ge=0)
    candidates: tuple[CandidateSummary, ...] = ()
    engines: tuple[EngineSummary, ...] = ()
    selections: tuple[SelectionSummary, ...] = ()
    checkpoint_paths: tuple[str, ...] = ()
    stop_reason: str | None = None
    event_count: int = Field(default=0, ge=0)
    diagnostic_count: int = Field(default=0, ge=0)


class PydanticGEPAEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    source_convention: Literal["pydantic-gepa"] = "pydantic-gepa"
    event_version: Literal["1"] = "1"
    executions: tuple[OptimizationExecution, ...] = ()


EXTENSION_KEY = "autobench.pydantic_gepa/v1"


__all__ = (
    "CandidateStatus",
    "CandidateSummary",
    "DatasetSummary",
    "EngineSummary",
    "EXTENSION_KEY",
    "ObjectiveSummary",
    "OptimizationExecution",
    "PydanticGEPAEvidence",
    "PydanticGEPAStatus",
    "SelectionSummary",
)
