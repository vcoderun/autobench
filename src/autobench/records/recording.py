from __future__ import annotations as _annotations

import json
from collections.abc import Mapping
from os.path import relpath
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autobench.errors import ErrorRecord
from autobench.io import dump_yaml
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY
from autobench.protocol.traces import Trace
from autobench.records.artifacts import ArtifactRef, ArtifactSource
from autobench.records.files import (
    LogicalRecordTarget,
    RecordDurability,
    RecordFileKind,
    atomic_write_text,
    build_manifest,
    create_temporary_record_directory,
    hash_and_size,
    publish_record_directory,
    remove_temporary_record_directory,
    validate_logical_targets,
    validate_manifest,
)
from autobench.records.models import (
    RECORD_VERSION,
    ExperimentRecord,
    RecordedRunPayloads,
    RecordingError,
    RecordLineage,
    ReplayKind,
    RunRecord,
)
from autobench.records.storage import EnvironmentMetadata, hash_file
from autobench.records.views import (
    experiment_record_payload_from_yaml_view,
    experiment_record_to_yaml_view,
    experiment_summary,
    manifest_to_yaml_view,
    run_record_payload_from_yaml_view,
    run_record_to_yaml_view,
)
from autobench.runtime.models import ExperimentResult, ExperimentTermination, RunResult
from autobench.runtime.traces import trace_to_yaml_view, trace_yaml_schema
from autobench.tracking import TrackingRegistry, track

TRACE_ARTIFACT_MEDIA_TYPE = "application/vnd.autobench.abp-trace+yaml"
TRACE_INLINE_LIMIT_BYTES = 128 * 1024


def record_experiment(
    result: ExperimentResult,
    output_dir: Path,
    *,
    source_files: list[Path] | None = None,
    path_root: Path | None = None,
    trace_inline_limit_bytes: int = TRACE_INLINE_LIMIT_BYTES,
    asset_registry: TrackingRegistry = track,
    durability: RecordDurability = "atomic",
) -> ExperimentRecord:
    if trace_inline_limit_bytes < 1:
        raise ValueError("trace_inline_limit_bytes must be at least 1")
    if output_dir.is_symlink() or (
        output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))
    ):
        raise RecordingError(f"Record target already exists: {output_dir}")

    targets = validate_logical_targets(
        _logical_record_targets(result, trace_inline_limit_bytes=trace_inline_limit_bytes)
    )
    staging = create_temporary_record_directory(output_dir)
    try:
        artifacts_dir = staging / "artifacts"
        referenced_asset_ids = {
            version.asset_id for run in result.runs for version in run.asset_versions
        }
        persisted_asset_ids = {
            asset_id for asset_id in referenced_asset_ids if asset_registry.has_asset(asset_id)
        }
        if persisted_asset_ids:
            asset_registry.write_assets(
                staging / "assets",
                asset_ids=persisted_asset_ids,
                content_path=artifacts_dir / "asset-content.sqlite3",
                root_dir=staging,
            )

        run_paths: list[str] = []
        for run in result.runs:
            run_record = run_record_from_result(
                run,
                artifacts_dir=artifacts_dir,
                root_dir=staging,
                semantic_registry_version=result.semantic_registry.version,
                trace_inline_limit_bytes=trace_inline_limit_bytes,
                durability=durability,
            )
            run_path = case_run_record_path(staging, run)
            _write_yaml(
                run_record_to_yaml_view(run_record),
                run_path,
                schema_name="run_record",
                durability=durability,
            )
            run_paths.append(run_path.relative_to(staging).as_posix())

        termination = _recorded_termination(result)
        record = ExperimentRecord(
            experiment_id=result.experiment_id,
            benchmark_id=result.benchmark_id,
            plan=result.plan,
            environment=_recorded_environment(result.environment, path_root=path_root),
            termination=termination,
            semantic_registry=result.semantic_registry,
            report_spec_data=result.report_spec_data,
            spec_snapshot=result.spec_snapshot,
            spec_hash=result.spec_hash,
            file_hashes=tuple(
                hash_file(path, relative_to=path_root)
                for path in (source_files or [])
                if path.exists() and path.is_file()
            ),
            manifest_path="manifest.yaml",
            run_paths=tuple(run_paths),
            run_count=result.total_count,
            passed_count=result.passed_count,
            failed_count=result.failed_count,
            errored_count=result.errored_count,
            skipped_count=result.skipped_count,
            cancelled_count=result.cancelled_count,
            correlation=result.correlation,
        )
        _write_yaml(
            experiment_record_to_yaml_view(record),
            staging / "experiment.yaml",
            schema_name="experiment",
            durability=durability,
        )
        _write_yaml(
            experiment_summary(record),
            staging / "summary.yaml",
            schema_name="summary",
            durability=durability,
        )
        manifest = build_manifest(staging, experiment_id=result.experiment_id, targets=targets)
        _write_yaml(
            manifest_to_yaml_view(manifest),
            staging / "manifest.yaml",
            schema_name="manifest",
            durability=durability,
        )
        validate_manifest(staging, manifest)
        publish_record_directory(staging, output_dir, durability=durability)
        return record
    finally:
        remove_temporary_record_directory(staging)


def run_record_from_result(
    run: RunResult,
    *,
    artifacts_dir: Path,
    root_dir: Path,
    semantic_registry_version: int = DEFAULT_SEMANTIC_REGISTRY.version,
    trace_inline_limit_bytes: int = TRACE_INLINE_LIMIT_BYTES,
    durability: RecordDurability = "atomic",
    recorded_payloads: RecordedRunPayloads | None = None,
    prepared_artifacts: Mapping[str, Path] | None = None,
) -> RunRecord:
    if trace_inline_limit_bytes < 1:
        raise ValueError("trace_inline_limit_bytes must be at least 1")
    recorded_artifacts = (
        list(recorded_payloads.artifacts)
        if recorded_payloads is not None
        else [
            record_artifact(
                artifact,
                artifacts_dir=artifacts_dir,
                root_dir=root_dir,
                run_id=run.run_id,
                durability=durability,
                prepared_path=(
                    None if prepared_artifacts is None else prepared_artifacts.get(artifact.id)
                ),
            )
            for artifact in run.task_result.artifacts
        ]
    )
    errors: list[ErrorRecord] = []
    for error in [run.error, run.task_result.error, *run.task_result.errors]:
        if error is not None and error not in errors:
            errors.append(error)
    if recorded_payloads is None:
        trace, trace_artifact = _record_trace(
            run.trace,
            artifacts_dir=artifacts_dir,
            root_dir=root_dir,
            run_id=run.run_id,
            inline_limit_bytes=trace_inline_limit_bytes,
            durability=durability,
        )
    else:
        trace = recorded_payloads.trace
        trace_artifact = recorded_payloads.trace_artifact
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
        partial=run.partial,
        end_reason=run.end_reason,
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
        asset_uses=tuple(run.asset_uses),
        parent_run_id=run.parent_run_id,
        source_snapshots=run.source_snapshots,
        errors=tuple(errors),
        error=run.error,
        extensions=run.extensions,
        correlation=run.correlation,
    )


def _recorded_termination(result: ExperimentResult) -> ExperimentTermination:
    if result.termination.planned_run_ids or result.termination.recorded_run_ids:
        return result.termination
    run_ids = tuple(run.run_id for run in result.runs)
    return result.termination.model_copy(
        update={"planned_run_ids": run_ids, "recorded_run_ids": run_ids}
    )


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


def _record_trace(
    trace: Trace | None,
    *,
    artifacts_dir: Path,
    root_dir: Path,
    run_id: str,
    inline_limit_bytes: int,
    durability: RecordDurability,
) -> tuple[Trace | None, ArtifactRef | None]:
    if trace is None or _trace_size(trace) <= inline_limit_bytes:
        return trace, None
    path = trace_artifact_path(artifacts_dir, run_id=run_id)
    if path.exists():
        raise RecordingError(f"Trace artifact already exists: {path}")
    _write_yaml(
        trace_to_yaml_view(trace),
        path,
        schema_name="trace",
        schema=trace_yaml_schema(),
        durability=durability,
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


def record_artifact(
    artifact: ArtifactRef,
    *,
    artifacts_dir: Path,
    root_dir: Path,
    run_id: str,
    durability: RecordDurability,
    prepared_path: Path | None = None,
) -> ArtifactRef:
    artifact_path = artifact_record_path(artifacts_dir, run_id=run_id, artifact=artifact)
    payload_path = artifact_payload_path(artifacts_dir, run_id=run_id, artifact=artifact)
    if prepared_path is not None and artifact_path.is_file() and payload_path.is_file():
        payload_hash, payload_byte_count = hash_and_size(payload_path)
        if artifact.sha256 != payload_hash or artifact.byte_count != payload_byte_count:
            raise RecordingError(f"Prepared artifact changed after capture: {artifact.id}")
        return artifact.model_copy(
            update={
                "value": payload_path.relative_to(root_dir).as_posix(),
                "sha256": payload_hash,
                "byte_count": payload_byte_count,
            }
        )
    if artifact_path.exists() or (payload_path.exists() and prepared_path is None):
        raise RecordingError(
            f"Artifact already exists: {artifact_path if artifact_path.exists() else payload_path}"
        )
    payload = _compact(
        {
            "record": {"type": "artifact", "version": 1},
            "artifact": {
                "id": artifact.id,
                "name": artifact.name,
                "media_type": artifact.media_type,
                "span_id": artifact.span_id,
                "tags": _to_serializable(artifact.tags),
                "payload": payload_path.relative_to(root_dir).as_posix(),
                "source": artifact.source.value,
                "state": artifact.state.value,
                "sha256": artifact.sha256,
                "byte_count": artifact.byte_count,
                "filename": artifact.filename,
                "symlink_followed": artifact.symlink_followed,
            },
        }
    )
    if prepared_path is not None:
        if prepared_path != payload_path or not payload_path.is_file():
            raise RecordingError(f"Prepared artifact payload is unavailable: {artifact.id}")
    elif isinstance(artifact.value, str):
        atomic_write_text(payload_path, artifact.value, durability=durability)
    else:
        _write_yaml(
            _compact(
                {
                    "record": {"type": "artifact_payload", "version": 1},
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
            durability=durability,
        )
    payload_hash, payload_byte_count = hash_and_size(payload_path)
    if artifact.sha256 is not None and artifact.sha256 != payload_hash:
        raise RecordingError(f"Prepared artifact hash mismatch: {artifact.id}")
    if artifact.byte_count is not None and artifact.byte_count != payload_byte_count:
        raise RecordingError(f"Prepared artifact byte count mismatch: {artifact.id}")
    payload["artifact"]["sha256"] = payload_hash
    payload["artifact"]["byte_count"] = payload_byte_count
    _write_yaml(payload, artifact_path, schema_name="artifact", durability=durability)
    return artifact.model_copy(
        update={
            "value": payload_path.relative_to(root_dir).as_posix(),
            "sha256": payload_hash,
            "byte_count": payload_byte_count,
        }
    )


def _logical_record_targets(
    result: ExperimentResult,
    *,
    trace_inline_limit_bytes: int,
) -> tuple[LogicalRecordTarget, ...]:
    targets = [
        LogicalRecordTarget(
            path="experiment.yaml", kind=RecordFileKind.EXPERIMENT, identity=result.experiment_id
        ),
        LogicalRecordTarget(
            path="summary.yaml", kind=RecordFileKind.SUMMARY, identity=result.experiment_id
        ),
        LogicalRecordTarget(
            path="manifest.yaml", kind=RecordFileKind.OTHER, identity=result.experiment_id
        ),
    ]
    for run in result.runs:
        run_path = case_run_record_path(Path(), run).as_posix()
        targets.append(
            LogicalRecordTarget(path=run_path, kind=RecordFileKind.RUN, identity=run.run_id)
        )
        for artifact in run.task_result.artifacts:
            targets.extend(
                (
                    LogicalRecordTarget(
                        path=artifact_record_path(
                            Path("artifacts"), run_id=run.run_id, artifact=artifact
                        ).as_posix(),
                        kind=RecordFileKind.ARTIFACT,
                        identity=f"{run.run_id}:{artifact.id}:metadata",
                    ),
                    LogicalRecordTarget(
                        path=artifact_payload_path(
                            Path("artifacts"), run_id=run.run_id, artifact=artifact
                        ).as_posix(),
                        kind=RecordFileKind.ARTIFACT,
                        identity=f"{run.run_id}:{artifact.id}:payload",
                    ),
                )
            )
        if run.trace is not None and _trace_size(run.trace) > trace_inline_limit_bytes:
            targets.append(
                LogicalRecordTarget(
                    path=trace_artifact_path(Path("artifacts"), run_id=run.run_id).as_posix(),
                    kind=RecordFileKind.TRACE,
                    identity=run.run_id,
                )
            )
    return tuple(targets)


def artifact_record_path(artifacts_dir: Path, *, run_id: str, artifact: ArtifactRef) -> Path:
    return artifacts_dir / path_component(run_id) / f"{path_component(artifact.id)}.meta.yaml"


def artifact_payload_path(artifacts_dir: Path, *, run_id: str, artifact: ArtifactRef) -> Path:
    base = artifacts_dir / path_component(run_id) / path_component(artifact.id)
    if artifact.source is not ArtifactSource.VALUE:
        if artifact.filename is not None and Path(artifact.filename).suffix:
            return base.with_suffix(Path(artifact.filename).suffix)
        if artifact.media_type is not None and artifact.media_type.startswith("text/"):
            return base.with_suffix(".txt")
        return base.with_suffix(".bin")
    if isinstance(artifact.value, str):
        if artifact.media_type == "text/markdown":
            return base.with_suffix(".md")
        if artifact.media_type is not None and artifact.media_type.startswith("text/"):
            return base.with_suffix(".txt")
    return base.with_suffix(".yaml")


def trace_artifact_path(artifacts_dir: Path, *, run_id: str) -> Path:
    return artifacts_dir / path_component(run_id) / "trace.yaml"


def case_run_record_path(root_dir: Path, run: RunResult) -> Path:
    return (
        root_dir
        / "cases"
        / path_component(run.case_id)
        / path_component(run.variant_id)
        / "run.yaml"
    )


def path_component(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value)
    return normalized.strip("_") or "unnamed"


def _write_yaml(
    value: Any,
    path: Path,
    *,
    schema_name: str,
    durability: RecordDurability,
    schema: dict[str, Any] | None = None,
) -> None:
    atomic_write_text(
        path,
        dump_yaml(value, schema_name=schema_name, schema=schema),
        durability=durability,
    )


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


def _to_serializable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(
            mode="json",
            fallback=lambda item: f"<{type(item).__qualname__}>",
        )
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
    "RecordDurability",
    "RecordLineage",
    "RecordedRunPayloads",
    "RecordingError",
    "ReplayKind",
    "RunRecord",
    "artifact_payload_path",
    "artifact_record_path",
    "case_run_record_path",
    "experiment_record_payload_from_yaml_view",
    "experiment_record_to_yaml_view",
    "experiment_summary",
    "path_component",
    "record_experiment",
    "run_record_payload_from_yaml_view",
    "run_record_from_result",
    "run_record_to_yaml_view",
    "trace_artifact_path",
)
