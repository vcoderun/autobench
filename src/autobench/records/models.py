from __future__ import annotations as _annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autobench.data.datasets import Case
from autobench.data.variants import FactorValue
from autobench.errors import AutobenchError, ErrorRecord
from autobench.evaluation.extraction import ExtractionEvidence
from autobench.evaluation.scoring import ScoreRecord
from autobench.metrics.mappings import CanonicalizationResult, SourceSnapshot
from autobench.metrics.observations import Observation
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
from autobench.protocol import EndReason
from autobench.protocol.signals import PROTOCOL_VERSION
from autobench.protocol.traces import Trace
from autobench.records.artifacts import ArtifactRef
from autobench.records.storage import EnvironmentMetadata, ResolvedFileHash
from autobench.runtime.context import SpanRecord
from autobench.runtime.models import (
    BenchmarkPlan,
    EvaluationStatus,
    ExecutionCorrelation,
    ExperimentTermination,
    RunStatus,
)
from autobench.runtime.tasks import TaskStatus
from autobench.tracking import AssetUse, AssetVersion

RECORD_VERSION = 6


class RecordingError(AutobenchError):
    """Raised when an experiment cannot be recorded safely."""


class ReplayKind(StrEnum):
    EXTRACTION = "extraction"
    CANONICALIZATION = "canonicalization"


class RecordLineage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ReplayKind
    parent_run_id: str
    processor: str
    processor_version: str
    source_record_version: int
    source_protocol_version: int | None = None
    source_semantic_registry_version: int | None = None
    source_maps: tuple[str, ...] = ()


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_version: int = Field(default=RECORD_VERSION, ge=1, le=RECORD_VERSION)
    protocol_version: Literal[1] | None = None
    semantic_registry_version: int | None = Field(default=None, ge=1)
    run_id: str
    experiment_id: str
    benchmark_id: str
    case_id: str
    variant_id: str
    status: RunStatus
    evaluation_status: EvaluationStatus
    task_status: TaskStatus
    partial: bool = False
    end_reason: EndReason = EndReason.COMPLETED
    case: Case
    task_output: Any = None
    observations: tuple[Observation, ...] = ()
    scores: tuple[ScoreRecord, ...] = ()
    spans: tuple[SpanRecord, ...] = ()
    trace: Trace | None = None
    trace_artifact: ArtifactRef | None = None
    trace_extensions: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    factors: tuple[FactorValue, ...] = ()
    asset_versions: tuple[AssetVersion, ...] = ()
    asset_uses: tuple[AssetUse, ...] = ()
    parent_run_id: str | None = None
    lineage: RecordLineage | None = None
    source_snapshots: tuple[SourceSnapshot, ...] = ()
    canonicalizations: tuple[CanonicalizationResult, ...] = ()
    extractions: tuple[ExtractionEvidence, ...] = ()
    extensions: dict[str, Any] = Field(default_factory=dict)
    errors: tuple[ErrorRecord, ...] = ()
    error: ErrorRecord | None = None
    correlation: ExecutionCorrelation | None = None

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_record(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        payload = dict(raw)
        if "task_status" not in payload and "status" in payload:
            payload["task_status"] = payload["status"]
        if "evaluation_status" not in payload and "status" in payload:
            payload["evaluation_status"] = (
                EvaluationStatus.NOT_EVALUATED
                if payload["status"] == RunStatus.CANCELLED
                else payload["status"]
            )
        if "case" not in payload and "case_id" in payload:
            payload["case"] = {"id": payload["case_id"]}
        raw_status = payload.get("status")
        try:
            status = RunStatus(raw_status) if isinstance(raw_status, str) else None
        except ValueError:
            status = None
        if "partial" not in payload:
            payload["partial"] = status == RunStatus.CANCELLED
        if "end_reason" not in payload:
            if status is RunStatus.CANCELLED:
                payload["end_reason"] = EndReason.CANCELLED
            elif status in {RunStatus.FAILED, RunStatus.ERRORED}:
                payload["end_reason"] = EndReason.FAILED
            elif status is RunStatus.SKIPPED:
                payload["end_reason"] = EndReason.DEFERRED
            else:
                payload["end_reason"] = EndReason.COMPLETED
        trace = payload.get("trace")
        if payload.get("protocol_version") is None and isinstance(trace, dict):
            payload["protocol_version"] = trace.get("protocol_version", PROTOCOL_VERSION)
        if payload.get("semantic_registry_version") is None and trace is not None:
            payload["semantic_registry_version"] = DEFAULT_SEMANTIC_REGISTRY.version
        lineage = payload.get("lineage")
        if payload.get("parent_run_id") is None and isinstance(lineage, dict):
            payload["parent_run_id"] = lineage.get("parent_run_id")
        return payload


class RecordedRunPayloads(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifacts: tuple[ArtifactRef, ...] = ()
    trace: Trace | None = None
    trace_artifact: ArtifactRef | None = None


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_version: int = Field(default=RECORD_VERSION, ge=1, le=RECORD_VERSION)
    experiment_id: str
    benchmark_id: str
    plan: BenchmarkPlan
    environment: EnvironmentMetadata
    termination: ExperimentTermination = Field(default_factory=ExperimentTermination)
    semantic_registry: SemanticRegistry = Field(
        default_factory=lambda: DEFAULT_SEMANTIC_REGISTRY.model_copy(deep=True)
    )
    report_spec_data: dict[str, Any] | None = None
    spec_snapshot: dict[str, Any] | None = None
    spec_hash: str | None = None
    file_hashes: tuple[ResolvedFileHash, ...] = ()
    manifest_path: str | None = None
    run_paths: tuple[str, ...] = ()
    run_count: int
    passed_count: int
    failed_count: int
    errored_count: int
    skipped_count: int
    cancelled_count: int = 0
    correlation: ExecutionCorrelation | None = None


__all__ = (
    "ExperimentRecord",
    "RECORD_VERSION",
    "RecordingError",
    "RecordedRunPayloads",
    "RecordLineage",
    "ReplayKind",
    "RunRecord",
)
