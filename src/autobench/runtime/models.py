from __future__ import annotations as _annotations

from enum import StrEnum
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from autobench.data.datasets import Case
from autobench.data.variants import FactorValue, Variant
from autobench.errors import ErrorRecord
from autobench.evaluation.scoring import ScoreRecord
from autobench.metrics.mappings import SourceSnapshot
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
from autobench.protocol import EndReason
from autobench.protocol.traces import Trace
from autobench.records.storage import EnvironmentMetadata
from autobench.runtime.tasks import TaskResult
from autobench.tracking import AssetUse, AssetVersion

CorrelationLabel = str | int | float | bool


class ExecutionCorrelation(BaseModel):
    """Stable metadata that groups related benchmark invocations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str | None = None
    attempt: int | None = Field(default=None, ge=1)
    phase: str | None = None
    parent_experiment_id: str | None = None
    resumed_from_experiment_id: str | None = None
    labels: dict[str, CorrelationLabel] = Field(default_factory=dict)

    @field_validator(
        "group_id",
        "phase",
        "parent_experiment_id",
        "resumed_from_experiment_id",
    )
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("correlation identifiers must not be blank")
        return value

    @field_validator("labels")
    @classmethod
    def validate_labels(
        cls,
        labels: dict[str, CorrelationLabel],
    ) -> dict[str, CorrelationLabel]:
        for key, value in labels.items():
            if not key.strip():
                raise ValueError("correlation label names must not be blank")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("correlation label values must be finite")
        return labels


def merge_execution_correlation(
    base: ExecutionCorrelation | None,
    override: ExecutionCorrelation | None,
) -> ExecutionCorrelation | None:
    """Merge explicitly supplied invocation fields without erasing YAML defaults."""

    if override is None:
        return None if base is None else base.model_copy(deep=True)
    current = base or ExecutionCorrelation()
    supplied = override.model_fields_set
    labels = dict(current.labels)
    if "labels" in supplied:
        labels.update(override.labels)
    merged = ExecutionCorrelation(
        group_id=override.group_id if "group_id" in supplied else current.group_id,
        attempt=override.attempt if "attempt" in supplied else current.attempt,
        phase=override.phase if "phase" in supplied else current.phase,
        parent_experiment_id=(
            override.parent_experiment_id
            if "parent_experiment_id" in supplied
            else current.parent_experiment_id
        ),
        resumed_from_experiment_id=(
            override.resumed_from_experiment_id
            if "resumed_from_experiment_id" in supplied
            else current.resumed_from_experiment_id
        ),
        labels=labels,
    )
    if (
        merged.group_id is None
        and merged.attempt is None
        and merged.phase is None
        and merged.parent_experiment_id is None
        and merged.resumed_from_experiment_id is None
        and not merged.labels
    ):
        return None
    return merged


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
    CANCELLED = "cancelled"


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"
    NOT_EVALUATED = "not_evaluated"


class ExperimentStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


class ExperimentTermination(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ExperimentStatus = ExperimentStatus.COMPLETED
    partial: bool = False
    cross_run_derivation_complete: bool = True
    policies_complete: bool = True
    planned_run_ids: tuple[str, ...] = ()
    recorded_run_ids: tuple[str, ...] = ()
    missing_run_ids: tuple[str, ...] = ()
    error: ErrorRecord | None = None

    @model_validator(mode="after")
    def validate_run_id_sets(self) -> ExperimentTermination:
        planned = set(self.planned_run_ids)
        recorded = set(self.recorded_run_ids)
        missing = set(self.missing_run_ids)
        if len(planned) != len(self.planned_run_ids):
            raise ValueError("planned_run_ids must be unique")
        if len(recorded) != len(self.recorded_run_ids):
            raise ValueError("recorded_run_ids must be unique")
        if len(missing) != len(self.missing_run_ids):
            raise ValueError("missing_run_ids must be unique")
        if recorded & missing:
            raise ValueError("recorded_run_ids and missing_run_ids must not overlap")
        if planned and not recorded.issubset(planned):
            raise ValueError("recorded_run_ids must belong to planned_run_ids")
        if planned and not missing.issubset(planned):
            raise ValueError("missing_run_ids must belong to planned_run_ids")
        if self.status is not ExperimentStatus.COMPLETED and not self.partial:
            raise ValueError("cancelled and aborted experiments must be partial")
        if self.missing_run_ids and not self.partial:
            raise ValueError("experiments with missing runs must be partial")
        return self


class MatrixRunSpec(BaseModel):
    run_id: str
    benchmark_id: str
    experiment_id: str
    case_index: int
    variant_index: int
    case: Case
    variant: Variant
    correlation: ExecutionCorrelation | None = None


class RunResult(BaseModel):
    run_id: str
    benchmark_id: str
    experiment_id: str
    case_id: str
    variant_id: str
    status: RunStatus
    evaluation_status: EvaluationStatus
    partial: bool = False
    end_reason: EndReason = EndReason.COMPLETED
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
    correlation: ExecutionCorrelation | None = None


class ExperimentResult(BaseModel):
    experiment_id: str
    benchmark_id: str
    plan: BenchmarkPlan
    runs: list[RunResult]
    environment: EnvironmentMetadata
    termination: ExperimentTermination = Field(default_factory=ExperimentTermination)
    report_spec_data: dict[str, Any] | None = None
    semantic_registry: SemanticRegistry = Field(
        default_factory=lambda: DEFAULT_SEMANTIC_REGISTRY.model_copy(deep=True)
    )
    spec_snapshot: dict[str, Any] | None = None
    spec_hash: str | None = None
    correlation: ExecutionCorrelation | None = None

    @model_validator(mode="after")
    def validate_run_correlation(self) -> ExperimentResult:
        if any(run.correlation != self.correlation for run in self.runs):
            raise ValueError("run correlation must match the experiment correlation")
        return self

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

    @property
    def cancelled_count(self) -> int:
        return self.count_status(RunStatus.CANCELLED)

    def count_status(self, status: RunStatus) -> int:
        return sum(1 for run in self.runs if run.status is status)


__all__ = (
    "BenchmarkPlan",
    "CorrelationLabel",
    "EvaluationStatus",
    "ExecutionCorrelation",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentTermination",
    "MatrixRunSpec",
    "RunResult",
    "RunStatus",
    "merge_execution_correlation",
)
