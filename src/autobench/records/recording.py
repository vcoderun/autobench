from __future__ import annotations as _annotations

import json
from enum import StrEnum
from os.path import relpath
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autobench.data.datasets import Case, case_to_yaml_view
from autobench.data.variants import FactorValue
from autobench.errors import AutobenchError, ErrorRecord
from autobench.evaluation.extraction import ExtractionEvidence
from autobench.evaluation.scoring import ScoreRecord
from autobench.io import dump_yaml
from autobench.metrics.mappings import CanonicalizationResult, SourceSnapshot
from autobench.metrics.observations import Observation
from autobench.metrics.semantics import (
    DEFAULT_SEMANTIC_REGISTRY,
    SemanticRegistry,
    semantic_registry_payload_from_yaml_view,
    semantic_registry_to_yaml_view,
)
from autobench.protocol.signals import PROTOCOL_VERSION
from autobench.protocol.traces import Trace
from autobench.records.artifacts import ArtifactRef
from autobench.records.storage import EnvironmentMetadata, ResolvedFileHash, hash_file
from autobench.runtime.context import SpanRecord
from autobench.runtime.pipeline import (
    BenchmarkPlan,
    EvaluationStatus,
    ExperimentResult,
    RunResult,
    RunStatus,
)
from autobench.runtime.tasks import TaskStatus
from autobench.runtime.traces import (
    trace_payload_from_yaml_view,
    trace_to_yaml_view,
    trace_yaml_schema,
)
from autobench.spec import (
    BenchmarkSpec,
    benchmark_spec_payload_from_yaml_view,
    benchmark_spec_to_yaml_view,
)
from autobench.tracking import AssetVersion

RECORD_VERSION = 4
TRACE_ARTIFACT_MEDIA_TYPE = "application/vnd.autobench.abp-trace+yaml"
TRACE_INLINE_LIMIT_BYTES = 128 * 1024


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
    parent_run_id: str | None = None
    lineage: RecordLineage | None = None
    source_snapshots: tuple[SourceSnapshot, ...] = ()
    canonicalizations: tuple[CanonicalizationResult, ...] = ()
    extractions: tuple[ExtractionEvidence, ...] = ()
    extensions: dict[str, Any] = Field(default_factory=dict)
    errors: tuple[ErrorRecord, ...] = ()
    error: ErrorRecord | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_record(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        payload = dict(raw)
        if "task_status" not in payload and "status" in payload:
            payload["task_status"] = payload["status"]
        if "evaluation_status" not in payload and "status" in payload:
            payload["evaluation_status"] = payload["status"]
        if "case" not in payload and "case_id" in payload:
            payload["case"] = {"id": payload["case_id"]}
        trace = payload.get("trace")
        if payload.get("protocol_version") is None and isinstance(trace, dict):
            payload["protocol_version"] = trace.get("protocol_version", PROTOCOL_VERSION)
        if payload.get("semantic_registry_version") is None and trace is not None:
            payload["semantic_registry_version"] = DEFAULT_SEMANTIC_REGISTRY.version
        lineage = payload.get("lineage")
        if payload.get("parent_run_id") is None and isinstance(lineage, dict):
            payload["parent_run_id"] = lineage.get("parent_run_id")
        return payload


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_version: int = Field(default=RECORD_VERSION, ge=1, le=RECORD_VERSION)
    experiment_id: str
    benchmark_id: str
    plan: BenchmarkPlan
    environment: EnvironmentMetadata
    semantic_registry: SemanticRegistry = Field(
        default_factory=lambda: DEFAULT_SEMANTIC_REGISTRY.model_copy(deep=True)
    )
    report_spec_data: dict[str, Any] | None = None
    spec_snapshot: dict[str, Any] | None = None
    spec_hash: str | None = None
    file_hashes: tuple[ResolvedFileHash, ...] = ()
    run_paths: tuple[str, ...] = ()
    run_count: int
    passed_count: int
    failed_count: int
    errored_count: int
    skipped_count: int


def record_experiment(
    result: ExperimentResult,
    output_dir: Path,
    *,
    source_files: list[Path] | None = None,
    path_root: Path | None = None,
    trace_inline_limit_bytes: int = TRACE_INLINE_LIMIT_BYTES,
) -> ExperimentRecord:
    if trace_inline_limit_bytes < 1:
        raise ValueError("trace_inline_limit_bytes must be at least 1")
    if (output_dir / "experiment.yaml").exists():
        raise RecordingError(f"Experiment record already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    cases_dir = output_dir / "cases"
    _ensure_record_targets_available(
        result,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        trace_inline_limit_bytes=trace_inline_limit_bytes,
    )
    artifacts_dir.mkdir(exist_ok=True)
    cases_dir.mkdir(exist_ok=True)

    run_paths: list[str] = []
    for run in result.runs:
        run_record = run_record_from_result(
            run,
            artifacts_dir=artifacts_dir,
            root_dir=output_dir,
            semantic_registry_version=result.semantic_registry.version,
            trace_inline_limit_bytes=trace_inline_limit_bytes,
        )
        run_path = _run_record_path(output_dir, run)
        run_path.parent.mkdir(parents=True, exist_ok=True)
        dump_yaml(run_record_to_yaml_view(run_record), run_path, schema_name="run_record")
        run_paths.append(run_path.relative_to(output_dir).as_posix())

    record = ExperimentRecord(
        experiment_id=result.experiment_id,
        benchmark_id=result.benchmark_id,
        plan=result.plan,
        environment=_recorded_environment(result.environment, path_root=path_root),
        semantic_registry=result.semantic_registry,
        report_spec_data=result.report_spec_data,
        spec_snapshot=result.spec_snapshot,
        spec_hash=result.spec_hash,
        file_hashes=tuple(
            hash_file(path, relative_to=path_root)
            for path in (source_files or [])
            if path.exists() and path.is_file()
        ),
        run_paths=tuple(run_paths),
        run_count=result.total_count,
        passed_count=result.passed_count,
        failed_count=result.failed_count,
        errored_count=result.errored_count,
        skipped_count=result.skipped_count,
    )
    dump_yaml(
        experiment_record_to_yaml_view(record),
        output_dir / "experiment.yaml",
        schema_name="experiment",
    )
    dump_yaml(experiment_summary(record), output_dir / "summary.yaml", schema_name="summary")
    return record


def _recorded_environment(
    environment: EnvironmentMetadata,
    *,
    path_root: Path | None,
) -> EnvironmentMetadata:
    if path_root is None:
        return environment
    return environment.model_copy(
        update={"cwd": Path(relpath(Path(environment.cwd), path_root.resolve())).as_posix()}
    )


def run_record_from_result(
    run: RunResult,
    *,
    artifacts_dir: Path,
    root_dir: Path,
    semantic_registry_version: int = DEFAULT_SEMANTIC_REGISTRY.version,
    trace_inline_limit_bytes: int = TRACE_INLINE_LIMIT_BYTES,
) -> RunRecord:
    if trace_inline_limit_bytes < 1:
        raise ValueError("trace_inline_limit_bytes must be at least 1")
    recorded_artifacts = [
        _record_artifact(
            artifact, artifacts_dir=artifacts_dir, root_dir=root_dir, run_id=run.run_id
        )
        for artifact in run.task_result.artifacts
    ]
    errors: list[ErrorRecord] = []
    for error in [run.error, run.task_result.error, *run.task_result.errors]:
        if error is not None and error not in errors:
            errors.append(error)
    trace, trace_artifact = _record_trace(
        run.trace,
        artifacts_dir=artifacts_dir,
        root_dir=root_dir,
        run_id=run.run_id,
        inline_limit_bytes=trace_inline_limit_bytes,
    )
    return RunRecord(
        protocol_version=None if run.trace is None else run.trace.protocol_version,
        semantic_registry_version=None if run.trace is None else semantic_registry_version,
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        benchmark_id=run.benchmark_id,
        case_id=run.case_id,
        variant_id=run.variant_id,
        status=run.status,
        evaluation_status=run.evaluation_status,
        task_status=run.task_result.status,
        case=run.case,
        task_output=_to_serializable(run.task_result.output),
        observations=tuple(run.task_result.observations),
        scores=tuple(run.scores),
        spans=tuple(run.task_result.spans),
        trace=trace,
        trace_artifact=trace_artifact,
        artifacts=tuple(recorded_artifacts),
        factors=tuple(run.factors),
        asset_versions=tuple(run.asset_versions),
        parent_run_id=run.parent_run_id,
        source_snapshots=run.source_snapshots,
        errors=tuple(errors),
        error=run.error,
    )


def experiment_summary(record: ExperimentRecord) -> dict[str, Any]:
    return {
        "record": {
            "type": "summary",
            "version": record.record_version,
        },
        "summary": {
            "experiment": record.experiment_id,
            "benchmark": record.benchmark_id,
        },
        "runs": {
            "count": record.run_count,
            "passed": record.passed_count,
            "failed": record.failed_count,
            "errored": record.errored_count,
            "skipped": record.skipped_count,
        },
    }


def experiment_record_to_yaml_view(record: ExperimentRecord) -> dict[str, Any]:
    payload = _compact(
        {
            "record": {
                "type": "experiment",
                "version": record.record_version,
            },
            "experiment": {
                "id": record.experiment_id,
                "benchmark": record.benchmark_id,
            },
            "benchmark": {
                "id": record.benchmark_id,
                "dataset": _benchmark_dataset_view(record.plan),
                "cases": list(record.plan.case_ids),
                "counts": {
                    "cases": record.plan.case_count,
                    "variants": record.plan.variant_count,
                    "runs": record.plan.planned_run_count,
                },
                "warnings": list(record.plan.warnings),
                "spec": {
                    "hash": record.spec_hash,
                    "snapshot": _benchmark_spec_snapshot_view(record.spec_snapshot),
                },
            },
            "runs": {
                "count": record.run_count,
                "passed": record.passed_count,
                "failed": record.failed_count,
                "errored": record.errored_count,
                "skipped": record.skipped_count,
                "paths": list(record.run_paths),
            },
            "files": _file_hashes_view(record.file_hashes),
            "environment": _environment_yaml_view(record.environment),
            "semantic_registry": semantic_registry_to_yaml_view(record.semantic_registry)[
                "semantic_registry"
            ],
        }
    )
    if record.report_spec_data is not None:
        payload["reports"] = _to_serializable(record.report_spec_data)
    return payload


def experiment_record_payload_from_yaml_view(raw: dict[str, Any]) -> dict[str, Any]:
    record_header = raw.get("record")
    if not isinstance(record_header, dict) or record_header.get("type") != "experiment":
        return raw

    experiment = _require_mapping(raw.get("experiment"), "experiment")
    benchmark = _require_mapping(raw.get("benchmark"), "benchmark")
    runs = _require_mapping(raw.get("runs"), "runs")
    spec = benchmark.get("spec", {})
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise RecordingError("benchmark.spec must be a mapping")
    raw_environment = raw.get("environment")
    raw_semantic_registry = raw.get("semantic_registry")
    plan = benchmark.get("plan")
    if plan is None:
        plan = _benchmark_plan_payload(benchmark, runs)
    spec_snapshot = _benchmark_spec_snapshot_payload(spec.get("snapshot"))

    payload = _compact(
        {
            "record_version": record_header.get("version", RECORD_VERSION),
            "experiment_id": experiment.get("id"),
            "benchmark_id": benchmark.get("id", experiment.get("benchmark")),
            "plan": plan,
            "environment": _environment_payload(raw_environment),
            "semantic_registry": (
                semantic_registry_payload_from_yaml_view(raw_semantic_registry)
                if raw_semantic_registry is not None
                else None
            ),
            "spec_snapshot": spec_snapshot,
            "spec_hash": spec.get("hash"),
            "file_hashes": _file_hashes_payload(raw.get("files")),
            "run_paths": runs.get("paths", []),
            "run_count": runs.get("count"),
            "passed_count": runs.get("passed", 0),
            "failed_count": runs.get("failed", 0),
            "errored_count": runs.get("errored", 0),
            "skipped_count": runs.get("skipped", 0),
        }
    )
    if spec_snapshot is not None:
        payload["spec_snapshot"] = spec_snapshot
    if "reports" in raw:
        payload["report_spec_data"] = raw["reports"]
    return payload


def run_record_to_yaml_view(record: RunRecord) -> dict[str, Any]:
    output = _to_serializable(record.task_output)
    payload = {
        "record": {
            "type": "run",
            "version": record.record_version,
        },
        "protocol": _compact(
            {
                "name": "abp" if record.protocol_version is not None else None,
                "version": record.protocol_version,
                "semantic_registry": record.semantic_registry_version,
            }
        ),
        "run": _compact(
            {
                "id": record.run_id,
                "parent": record.parent_run_id,
                "experiment": record.experiment_id,
                "benchmark": record.benchmark_id,
                "case": record.case_id,
                "variant": record.variant_id,
                "status": record.status.value,
                "outcome": {
                    "evaluation": record.evaluation_status.value,
                    "task": record.task_status.value,
                },
            }
        ),
        "case": case_to_yaml_view(record.case),
        "variant": _variant_view(record),
        "scores": _scores_view(record.scores),
        "metrics": _observations_view(record.observations),
        "trace": _trace_view(record),
        "spans": _spans_view(record.spans),
        "artifacts": _artifacts_view(record.artifacts),
        "assets": _asset_versions_view(record.asset_versions),
        "errors": _errors_view(record),
        "canonicalization": _compact(
            {
                "source_snapshots": [
                    snapshot.model_dump(mode="json") for snapshot in record.source_snapshots
                ],
                "results": [result.model_dump(mode="json") for result in record.canonicalizations],
            }
        ),
        "extraction": _compact(
            {
                "results": [result.model_dump(mode="json") for result in record.extractions],
            }
        ),
        "lineage": None if record.lineage is None else record.lineage.model_dump(mode="json"),
        "extensions": _to_serializable(record.extensions),
    }
    compacted = _compact(payload)
    if record.task_output is not None:
        compacted["output"] = output
    return compacted


def _benchmark_dataset_view(plan: BenchmarkPlan) -> dict[str, Any]:
    return _compact(
        {
            "id": plan.dataset_id,
            "version": plan.dataset_version,
            "hash": plan.dataset_hash,
        }
    )


def _benchmark_spec_snapshot_view(snapshot: dict[str, Any] | None) -> Any:
    if snapshot is None:
        return None
    try:
        normalized_snapshot = benchmark_spec_payload_from_yaml_view(snapshot)
        return benchmark_spec_to_yaml_view(BenchmarkSpec.model_validate(normalized_snapshot))
    except Exception:
        return _to_serializable(snapshot)


def _benchmark_spec_snapshot_payload(snapshot: Any) -> Any:
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        return snapshot
    try:
        return benchmark_spec_payload_from_yaml_view(snapshot)
    except Exception:
        return snapshot


def _benchmark_plan_payload(
    raw_benchmark: dict[str, Any],
    raw_runs: dict[str, Any],
) -> dict[str, Any]:
    raw_dataset = raw_benchmark.get("dataset")
    dataset = _require_mapping(raw_dataset, "benchmark.dataset") if raw_dataset is not None else {}
    raw_counts = raw_benchmark.get("counts")
    counts = _require_mapping(raw_counts, "benchmark.counts") if raw_counts is not None else {}
    raw_cases = raw_benchmark.get("cases", [])
    if raw_cases is None:
        raw_cases = []
    if not isinstance(raw_cases, list):
        raise RecordingError("benchmark.cases must be a list")
    raw_warnings = raw_benchmark.get("warnings", [])
    if raw_warnings is None:
        raw_warnings = []
    if not isinstance(raw_warnings, list):
        raise RecordingError("benchmark.warnings must be a list")
    return _compact(
        {
            "benchmark_id": raw_benchmark.get("id"),
            "dataset_id": dataset.get("id"),
            "dataset_version": dataset.get("version"),
            "dataset_hash": dataset.get("hash"),
            "case_ids": raw_cases,
            "case_count": counts.get("cases", len(raw_cases)),
            "variant_count": counts.get("variants"),
            "planned_run_count": counts.get("runs", raw_runs.get("count")),
            "warnings": raw_warnings,
        }
    )


def _environment_yaml_view(environment: EnvironmentMetadata) -> dict[str, Any]:
    return {
        "python": environment.python_version,
        "platform": environment.platform,
        "cwd": environment.cwd,
    }


def _environment_payload(raw_environment: Any) -> dict[str, Any] | None:
    if raw_environment is None:
        return None
    environment = _require_mapping(raw_environment, "environment")
    return _compact(
        {
            "python_version": environment.get("python", environment.get("python_version")),
            "platform": environment.get("platform"),
            "cwd": environment.get("cwd"),
        }
    )


def _file_hashes_view(file_hashes: tuple[ResolvedFileHash, ...]) -> dict[str, str]:
    return {file_hash.path: file_hash.sha256 for file_hash in file_hashes}


def _file_hashes_payload(raw_files: Any) -> list[dict[str, Any]]:
    if raw_files is None:
        return []
    if isinstance(raw_files, list):
        return [dict(file_hash) for file_hash in raw_files if isinstance(file_hash, dict)]
    if not isinstance(raw_files, dict):
        raise RecordingError("files must be a mapping or list")
    return [
        {"path": path, "sha256": sha256}
        for path, sha256 in raw_files.items()
        if isinstance(path, str) and isinstance(sha256, str)
    ]


def run_record_payload_from_yaml_view(raw: dict[str, Any]) -> dict[str, Any]:
    record_header = raw.get("record")
    if not isinstance(record_header, dict) or record_header.get("type") != "run":
        return raw

    run = _require_mapping(raw.get("run"), "run")
    outcome = run.get("outcome", {})
    if outcome is None:
        outcome = {}
    if not isinstance(outcome, dict):
        raise RecordingError("run.outcome must be a mapping")

    protocol = raw.get("protocol", {})
    if protocol is None:
        protocol = {}
    if not isinstance(protocol, dict):
        raise RecordingError("protocol must be a mapping")
    if protocol.get("name") not in (None, "abp"):
        raise RecordingError("protocol.name must be 'abp'")
    trace, trace_artifact, trace_extensions = _trace_payload(raw.get("trace"), protocol)
    canonicalization = raw.get("canonicalization", {})
    if canonicalization is None:
        canonicalization = {}
    if not isinstance(canonicalization, dict):
        raise RecordingError("canonicalization must be a mapping")
    extraction = raw.get("extraction", {})
    if extraction is None:
        extraction = {}
    if not isinstance(extraction, dict):
        raise RecordingError("extraction must be a mapping")

    payload: dict[str, Any] = {
        "record_version": record_header.get("version", RECORD_VERSION),
        "protocol_version": protocol.get("version"),
        "semantic_registry_version": protocol.get("semantic_registry"),
        "run_id": run.get("id"),
        "experiment_id": run.get("experiment"),
        "benchmark_id": run.get("benchmark"),
        "case_id": run.get("case"),
        "variant_id": run.get("variant"),
        "status": run.get("status"),
        "evaluation_status": outcome.get("evaluation", run.get("status")),
        "task_status": outcome.get("task", run.get("status")),
        "case": raw.get("case", {"id": run.get("case")}),
        "task_output": raw.get("output"),
        "scores": _scores_payload(raw.get("scores")),
        "observations": _observations_payload(raw.get("metrics")),
        "spans": _spans_payload(raw.get("spans")),
        "trace": trace,
        "trace_artifact": trace_artifact,
        "trace_extensions": trace_extensions,
        "artifacts": _artifacts_payload(raw.get("artifacts")),
        "factors": _factors_payload(raw.get("variant")),
        "asset_versions": _asset_versions_payload(raw.get("assets")),
        "parent_run_id": run.get("parent"),
        "lineage": raw.get("lineage"),
        "source_snapshots": canonicalization.get("source_snapshots", []),
        "canonicalizations": canonicalization.get("results", []),
        "extractions": extraction.get("results", []),
        "extensions": _record_extensions(raw),
    }
    errors = _errors_payload(raw.get("errors"))
    if errors:
        payload["errors"] = errors
        payload["error"] = errors[0]
    elif raw.get("error") is not None:
        payload["error"] = raw["error"]
        payload["errors"] = [raw["error"]]
    return _compact(payload)


def _variant_view(record: RunRecord) -> dict[str, Any]:
    return _compact(
        {
            "id": record.variant_id,
            "factors": _factor_mapping(record.factors),
        }
    )


def _factor_mapping(factors: tuple[FactorValue, ...]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for factor in factors:
        if factor.semantic_type is None and not factor.optimize:
            mapped[factor.name] = _to_serializable(factor.value)
            continue
        mapped[factor.name] = _compact(
            {
                "value": _to_serializable(factor.value),
                "semantic": factor.semantic_type,
                "optimize": factor.optimize or None,
            }
        )
    return mapped


def _scores_view(scores: tuple[ScoreRecord, ...]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for score in scores:
        mapped[_unique_key(mapped, score.name)] = _compact(
            {
                "value": _to_serializable(score.value),
                "passed": score.value if isinstance(score.value, bool) else None,
                "semantic": score.semantic_type,
                "unit": score.unit,
                "goal": score.direction.value if score.direction is not None else None,
                "role": score.role.value if score.role is not None else None,
                "optional": score.optional or None,
                "actual": _to_serializable(score.actual_value),
                "expected": _to_serializable(score.expected_value),
                "span": score.span_id,
                "error": _model_dump(score.error) if score.error is not None else None,
                "tags": _to_serializable(score.tags),
            }
        )
    return mapped


def _observations_view(observations: tuple[Observation, ...]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {
        "factors": {},
        "measurements": {},
        "diagnostics": {},
        "events": {},
    }
    for sequence, observation in enumerate(observations, start=1):
        group = _observation_group(observation)
        groups[group][_unique_key(groups[group], observation.name)] = _compact(
            {
                "sequence": sequence,
                "id": observation.id,
                "name": observation.name,
                "kind": observation.kind.value,
                "value": _to_serializable(observation.value),
                "semantic": observation.semantic_type,
                "unit": observation.unit,
                "goal": observation.direction.value if observation.direction is not None else None,
                "role": observation.role.value if observation.role is not None else None,
                "span": observation.span_id,
                "source": str(observation.source) if observation.source is not None else None,
                "tags": _to_serializable(observation.tags),
                "case": observation.case_id,
                "variant": observation.variant_id,
            }
        )
    return {group: entries for group, entries in groups.items() if entries}


def _observation_group(observation: Observation) -> str:
    if observation.kind.value == "factor":
        return "factors"
    if observation.kind.value == "metric":
        if observation.role is not None and observation.role.value == "diagnostic":
            return "diagnostics"
        return "measurements"
    return "events"


def _spans_view(spans: tuple[SpanRecord, ...]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for span in spans:
        mapped[span.id] = _compact(
            {
                "name": span.name,
                "kind": str(span.kind),
                "parent": span.parent_id,
                "started_at": span.model_dump(mode="json")["started_at"],
                "ended_at": span.model_dump(mode="json")["ended_at"],
                "duration": span.duration_seconds,
                "input": _to_serializable(span.input),
                "output": _to_serializable(span.output),
                "attributes": _to_serializable(span.attributes),
                "usage": _to_serializable(span.usage),
                "observations": span.observations,
                "artifacts": span.artifacts,
                "error": _model_dump(span.error) if span.error is not None else None,
                "tags": _to_serializable(span.tags),
            }
        )
    return mapped


def _trace_view(record: RunRecord) -> dict[str, Any]:
    if record.trace_artifact is not None:
        tags = record.trace_artifact.tags
        return _compact(
            {
                "id": tags.get("trace_id"),
                "partial": tags.get("partial"),
                "spans": tags.get("span_count"),
                "signals": tags.get("signal_count"),
                "artifact": {
                    "id": record.trace_artifact.id,
                    "name": record.trace_artifact.name,
                    "media": record.trace_artifact.media_type,
                    "path": record.trace_artifact.value,
                },
                "extensions": _to_serializable(record.trace_extensions),
            }
        )
    if record.trace is None:
        return {}
    return trace_to_yaml_view(record.trace, extensions=record.trace_extensions)["trace"]


def _trace_payload(
    raw_trace: Any,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    if raw_trace is None:
        return None, None, {}
    trace = _require_mapping(raw_trace, "trace")
    raw_artifact = trace.get("artifact")
    if raw_artifact is not None:
        artifact = _require_mapping(raw_artifact, "trace.artifact")
        extensions = trace.get("extensions", {})
        if not isinstance(extensions, dict):
            raise RecordingError("trace.extensions must be a mapping")
        return (
            None,
            _compact(
                {
                    "id": artifact.get("id", "abp_trace"),
                    "name": artifact.get("name", "ABP trace"),
                    "media_type": artifact.get("media", artifact.get("media_type")),
                    "value": artifact.get("path", artifact.get("value")),
                    "tags": {
                        "trace_id": trace.get("id"),
                        "partial": trace.get("partial", False),
                        "span_count": trace.get("spans", 0),
                        "signal_count": trace.get("signals", 0),
                    },
                }
            ),
            dict(extensions),
        )
    payload, extensions = trace_payload_from_yaml_view(trace)
    payload.setdefault("protocol", protocol.get("name", "abp"))
    payload.setdefault("protocol_version", protocol.get("version", PROTOCOL_VERSION))
    return payload, None, extensions


def _record_extensions(raw: dict[str, Any]) -> dict[str, Any]:
    extensions = raw.get("extensions", {})
    if not isinstance(extensions, dict):
        raise RecordingError("extensions must be a mapping")
    known = {
        "record",
        "protocol",
        "run",
        "case",
        "variant",
        "scores",
        "metrics",
        "trace",
        "spans",
        "artifacts",
        "assets",
        "errors",
        "error",
        "canonicalization",
        "extraction",
        "lineage",
        "extensions",
        "output",
    }
    preserved = dict(extensions)
    preserved.update({key: value for key, value in raw.items() if key not in known})
    return preserved


def _artifacts_view(artifacts: tuple[ArtifactRef, ...]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for artifact in artifacts:
        mapped[_unique_key(mapped, artifact.name)] = _compact(
            {
                "id": artifact.id,
                "name": artifact.name,
                "media": artifact.media_type,
                "path": artifact.value if isinstance(artifact.value, str) else None,
                "value": None
                if isinstance(artifact.value, str)
                else _to_serializable(artifact.value),
                "span": artifact.span_id,
                "tags": _to_serializable(artifact.tags),
            }
        )
    return mapped


def _asset_versions_view(asset_versions: tuple[AssetVersion, ...]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for asset_version in asset_versions:
        mapped[_unique_key(mapped, asset_version.asset_id)] = _compact(
            {
                "asset_id": asset_version.asset_id,
                "version": asset_version.version,
                "content_hash": asset_version.content_hash,
                "source_hash": asset_version.source_hash,
                "source_path": asset_version.source_path,
                "git_commit": asset_version.git_commit,
                "parent": asset_version.parent_version,
                "metadata": _to_serializable(asset_version.metadata),
            }
        )
    return mapped


def _errors_view(record: RunRecord) -> dict[str, Any]:
    if record.error is None and not record.errors:
        return {}
    all_errors = [_model_dump(error) for error in record.errors]
    return _compact(
        {
            "primary": _model_dump(record.error) if record.error is not None else None,
            "all": all_errors,
        }
    )


def _scores_payload(raw_scores: Any) -> list[dict[str, Any]]:
    if raw_scores is None:
        return []
    if isinstance(raw_scores, list):
        return [dict(score) for score in raw_scores if isinstance(score, dict)]
    if not isinstance(raw_scores, dict):
        raise RecordingError("scores must be a mapping or list")
    scores: list[dict[str, Any]] = []
    for name, raw_score in raw_scores.items():
        score = _require_mapping(raw_score, f"scores.{name}")
        scores.append(
            _compact(
                {
                    "name": score.get("name", name),
                    "semantic_type": score.get("semantic", score.get("semantic_type")),
                    "value": score.get("value"),
                    "unit": score.get("unit"),
                    "direction": score.get("goal", score.get("direction")),
                    "role": score.get("role"),
                    "optional": score.get("optional", False),
                    "actual_value": score.get("actual", score.get("actual_value")),
                    "expected_value": score.get("expected", score.get("expected_value")),
                    "span_id": score.get("span", score.get("span_id")),
                    "error": score.get("error"),
                    "tags": score.get("tags", {}),
                }
            )
        )
    return scores


def _observations_payload(raw_metrics: Any) -> list[dict[str, Any]]:
    if raw_metrics is None:
        return []
    if isinstance(raw_metrics, list):
        return [dict(metric) for metric in raw_metrics if isinstance(metric, dict)]
    if not isinstance(raw_metrics, dict):
        raise RecordingError("metrics must be a mapping or list")

    observations: list[tuple[int, bool, dict[str, Any]]] = []
    for group, default_kind in (
        ("factors", "factor"),
        ("measurements", "metric"),
        ("diagnostics", "metric"),
        ("events", "event"),
    ):
        raw_group = raw_metrics.get(group)
        if raw_group is None:
            continue
        if not isinstance(raw_group, dict):
            raise RecordingError(f"metrics.{group} must be a mapping")
        for name, raw_observation in raw_group.items():
            observation = _require_mapping(raw_observation, f"metrics.{group}.{name}")
            sequence = observation.get("sequence")
            if sequence is not None and (
                not isinstance(sequence, int) or isinstance(sequence, bool)
            ):
                raise RecordingError(f"metrics.{group}.{name}.sequence must be an integer")
            observations.append(
                (
                    0 if sequence is None else sequence,
                    sequence is not None,
                    _compact(
                        {
                            "id": observation.get("id", name),
                            "name": observation.get("name", name),
                            "kind": observation.get("kind", default_kind),
                            "semantic_type": observation.get(
                                "semantic", observation.get("semantic_type")
                            ),
                            "value": observation.get("value"),
                            "unit": observation.get("unit"),
                            "direction": observation.get("goal", observation.get("direction")),
                            "role": observation.get("role"),
                            "span_id": observation.get("span", observation.get("span_id")),
                            "source": observation.get("source"),
                            "tags": observation.get("tags", {}),
                            "case_id": observation.get("case", observation.get("case_id")),
                            "variant_id": observation.get("variant", observation.get("variant_id")),
                        }
                    ),
                )
            )
    if all(has_sequence for _, has_sequence, _ in observations):
        observations.sort(key=lambda item: item[0])
    return [observation for _, _, observation in observations]


def _spans_payload(raw_spans: Any) -> list[dict[str, Any]]:
    if raw_spans is None:
        return []
    if isinstance(raw_spans, list):
        return [dict(span) for span in raw_spans if isinstance(span, dict)]
    if not isinstance(raw_spans, dict):
        raise RecordingError("spans must be a mapping or list")
    spans: list[dict[str, Any]] = []
    for span_id, raw_span in raw_spans.items():
        span = _require_mapping(raw_span, f"spans.{span_id}")
        spans.append(
            _compact(
                {
                    "id": span.get("id", span_id),
                    "name": span.get("name", span_id),
                    "kind": span.get("kind", "custom"),
                    "parent_id": span.get("parent", span.get("parent_id")),
                    "started_at": span.get("started_at"),
                    "ended_at": span.get("ended_at"),
                    "duration_seconds": span.get("duration", span.get("duration_seconds")),
                    "input": span.get("input"),
                    "output": span.get("output"),
                    "attributes": span.get("attributes", {}),
                    "usage": span.get("usage", {}),
                    "observations": span.get("observations", []),
                    "artifacts": span.get("artifacts", []),
                    "error": span.get("error"),
                    "tags": span.get("tags", {}),
                }
            )
        )
    return spans


def _artifacts_payload(raw_artifacts: Any) -> list[dict[str, Any]]:
    if raw_artifacts is None:
        return []
    if isinstance(raw_artifacts, list):
        return [dict(artifact) for artifact in raw_artifacts if isinstance(artifact, dict)]
    if not isinstance(raw_artifacts, dict):
        raise RecordingError("artifacts must be a mapping or list")
    artifacts: list[dict[str, Any]] = []
    for name, raw_artifact in raw_artifacts.items():
        artifact = _require_mapping(raw_artifact, f"artifacts.{name}")
        artifacts.append(
            _compact(
                {
                    "id": artifact.get("id", name),
                    "name": artifact.get("name", name),
                    "media_type": artifact.get("media", artifact.get("media_type")),
                    "value": artifact.get("path", artifact.get("value")),
                    "span_id": artifact.get("span", artifact.get("span_id")),
                    "tags": artifact.get("tags", {}),
                }
            )
        )
    return artifacts


def _factors_payload(raw_variant: Any) -> list[dict[str, Any]]:
    if raw_variant is None:
        return []
    variant = _require_mapping(raw_variant, "variant")
    raw_factors = variant.get("factors", {})
    if isinstance(raw_factors, list):
        return [dict(factor) for factor in raw_factors if isinstance(factor, dict)]
    if not isinstance(raw_factors, dict):
        raise RecordingError("variant.factors must be a mapping or list")
    factors: list[dict[str, Any]] = []
    for name, raw_factor in raw_factors.items():
        if isinstance(raw_factor, dict):
            factors.append(
                {
                    "name": name,
                    "value": raw_factor.get("value"),
                    "semantic_type": raw_factor.get("semantic", raw_factor.get("semantic_type")),
                    "optimize": raw_factor.get("optimize", False),
                }
            )
        else:
            factors.append({"name": name, "value": raw_factor})
    return factors


def _asset_versions_payload(raw_assets: Any) -> list[dict[str, Any]]:
    if raw_assets is None:
        return []
    if isinstance(raw_assets, list):
        return [dict(asset) for asset in raw_assets if isinstance(asset, dict)]
    if not isinstance(raw_assets, dict):
        raise RecordingError("assets must be a mapping or list")
    versions: list[dict[str, Any]] = []
    for asset_id, raw_asset in raw_assets.items():
        asset = _require_mapping(raw_asset, f"assets.{asset_id}")
        versions.append(
            _compact(
                {
                    "asset_id": asset.get("asset_id", asset_id),
                    "version": asset.get("version"),
                    "content_hash": asset.get("content_hash"),
                    "source_hash": asset.get("source_hash"),
                    "source_path": asset.get("source_path"),
                    "git_commit": asset.get("git_commit"),
                    "parent_version": asset.get("parent", asset.get("parent_version")),
                    "metadata": asset.get("metadata", {}),
                }
            )
        )
    return versions


def _errors_payload(raw_errors: Any) -> list[dict[str, Any]]:
    if raw_errors is None:
        return []
    if isinstance(raw_errors, list):
        return [dict(error) for error in raw_errors if isinstance(error, dict)]
    if not isinstance(raw_errors, dict):
        raise RecordingError("errors must be a mapping or list")
    raw_all = raw_errors.get("all")
    if isinstance(raw_all, list):
        return [dict(error) for error in raw_all if isinstance(error, dict)]
    primary = raw_errors.get("primary")
    if isinstance(primary, dict):
        return [dict(primary)]
    return []


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecordingError(f"{label} must be a mapping")
    return dict(value)


def _unique_key(mapped: dict[str, Any], key: str) -> str:
    candidate = key
    index = 2
    while candidate in mapped:
        candidate = f"{key}_{index}"
        index += 1
    return candidate


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: compacted
            for key, item in value.items()
            if (compacted := _compact(item)) not in (None, {}, [], ())
        }
    if isinstance(value, list):
        return [compacted for item in value if (compacted := _compact(item)) is not None]
    return value


def _record_trace(
    trace: Trace | None,
    *,
    artifacts_dir: Path,
    root_dir: Path,
    run_id: str,
    inline_limit_bytes: int,
) -> tuple[Trace | None, ArtifactRef | None]:
    if trace is None or _trace_size(trace) <= inline_limit_bytes:
        return trace, None

    path = _trace_artifact_path(artifacts_dir, run_id=run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RecordingError(f"Trace artifact already exists: {path}")
    dump_yaml(
        trace_to_yaml_view(trace),
        path,
        schema_name="trace",
        schema=trace_yaml_schema(),
    )
    return (
        None,
        ArtifactRef(
            id="abp_trace",
            name="ABP trace",
            media_type=TRACE_ARTIFACT_MEDIA_TYPE,
            value=path.relative_to(root_dir).as_posix(),
            tags={
                "trace_id": trace.trace_id,
                "partial": trace.partial,
                "span_count": len(trace.spans),
                "signal_count": len(trace.signals),
            },
        ),
    )


def _trace_size(trace: Trace) -> int:
    return len(
        json.dumps(
            trace.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _record_artifact(
    artifact: ArtifactRef,
    *,
    artifacts_dir: Path,
    root_dir: Path,
    run_id: str,
) -> ArtifactRef:
    run_artifacts_dir = artifacts_dir / _path_part(run_id)
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _artifact_record_path(artifacts_dir, run_id=run_id, artifact=artifact)
    if artifact_path.exists():
        raise RecordingError(f"Artifact already exists: {artifact_path}")
    payload_path = _artifact_payload_path(artifacts_dir, run_id=run_id, artifact=artifact)
    if payload_path.exists():
        raise RecordingError(f"Artifact already exists: {payload_path}")

    payload = _compact(
        {
            "record": {
                "type": "artifact",
                "version": 1,
            },
            "artifact": {
                "id": artifact.id,
                "name": artifact.name,
                "media_type": artifact.media_type,
                "span_id": artifact.span_id,
                "tags": _to_serializable(artifact.tags),
                "payload": payload_path.relative_to(root_dir).as_posix(),
            },
        }
    )
    if isinstance(artifact.value, str):
        payload_path.write_text(artifact.value, encoding="utf-8")
    else:
        dump_yaml(
            _compact(
                {
                    "record": {
                        "type": "artifact_payload",
                        "version": 1,
                    },
                    "artifact": {
                        "id": artifact.id,
                        "name": artifact.name,
                        "media_type": artifact.media_type,
                    },
                    "payload": _to_serializable(artifact.value),
                }
            ),
            payload_path,
            schema_name="artifact_payload",
        )
    dump_yaml(payload, artifact_path, schema_name="artifact")
    return artifact.model_copy(
        update={"value": payload_path.relative_to(root_dir).as_posix()},
    )


def _ensure_record_targets_available(
    result: ExperimentResult,
    *,
    output_dir: Path,
    artifacts_dir: Path,
    trace_inline_limit_bytes: int,
) -> None:
    reserved_paths = [output_dir / "experiment.yaml", output_dir / "summary.yaml"]
    for run in result.runs:
        reserved_paths.append(_run_record_path(output_dir, run))
        reserved_paths.extend(
            _artifact_record_path(artifacts_dir, run_id=run.run_id, artifact=artifact)
            for artifact in run.task_result.artifacts
        )
        if run.trace is not None and _trace_size(run.trace) > trace_inline_limit_bytes:
            reserved_paths.append(_trace_artifact_path(artifacts_dir, run_id=run.run_id))
        reserved_paths.extend(
            _artifact_payload_path(artifacts_dir, run_id=run.run_id, artifact=artifact)
            for artifact in run.task_result.artifacts
        )

    existing_paths = [path for path in reserved_paths if path.exists()]
    if existing_paths:
        raise RecordingError(f"Record target already exists: {existing_paths[0]}")


def _artifact_record_path(artifacts_dir: Path, *, run_id: str, artifact: ArtifactRef) -> Path:
    return artifacts_dir / _path_part(run_id) / f"{_path_part(artifact.id)}.meta.yaml"


def _artifact_payload_path(artifacts_dir: Path, *, run_id: str, artifact: ArtifactRef) -> Path:
    base = artifacts_dir / _path_part(run_id) / _path_part(artifact.id)
    if isinstance(artifact.value, str):
        if artifact.media_type == "text/markdown":
            return base.with_suffix(".md")
        if artifact.media_type is not None and artifact.media_type.startswith("text/"):
            return base.with_suffix(".txt")
    return base.with_suffix(".yaml")


def _trace_artifact_path(artifacts_dir: Path, *, run_id: str) -> Path:
    return artifacts_dir / _path_part(run_id) / "trace.yaml"


def _run_record_path(root_dir: Path, run: RunResult) -> Path:
    return root_dir / "cases" / _path_part(run.case_id) / _path_part(run.variant_id) / "run.yaml"


def _path_part(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value)
    return normalized.strip("_") or "unnamed"


def _model_dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


__all__ = (
    "ExperimentRecord",
    "RECORD_VERSION",
    "TRACE_ARTIFACT_MEDIA_TYPE",
    "TRACE_INLINE_LIMIT_BYTES",
    "RecordLineage",
    "RecordingError",
    "ReplayKind",
    "RunRecord",
    "experiment_record_payload_from_yaml_view",
    "experiment_record_to_yaml_view",
    "experiment_summary",
    "record_experiment",
    "run_record_payload_from_yaml_view",
    "run_record_from_result",
    "run_record_to_yaml_view",
)
