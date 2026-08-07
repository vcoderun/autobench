from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import AsyncIterable, AsyncIterator, Collection, Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from autobench.errors import ErrorRecord
from autobench.io import dump_yaml, load_yaml
from autobench.metrics.mappings import SourceSnapshot
from autobench.metrics.observations import Observation
from autobench.metrics.semantics import SemanticRegistry
from autobench.protocol import EndReason
from autobench.protocol.traces import Trace
from autobench.records.artifacts import (
    ArtifactOverflow,
    ArtifactRef,
    ArtifactSource,
    ArtifactState,
    ArtifactTransferError,
    SymlinkPolicy,
)
from autobench.records.files import (
    LogicalRecordTarget,
    ManifestEntry,
    RecordDurability,
    RecordFileKind,
    atomic_write_text,
    build_manifest,
    create_temporary_record_directory,
    hash_and_size,
    normalize_logical_path,
    publish_record_directory,
    remove_temporary_record_directory,
    sync_directory,
    validate_logical_targets,
    validate_manifest,
)
from autobench.records.models import (
    ExperimentRecord,
    RecordedRunPayloads,
    RecordingError,
    RunRecord,
)
from autobench.records.recording import (
    TRACE_INLINE_LIMIT_BYTES,
    artifact_payload_path,
    artifact_record_path,
    experiment_record_to_yaml_view,
    experiment_summary,
    path_component,
    record_artifact,
    run_record_from_result,
    run_record_to_yaml_view,
    trace_artifact_path,
)
from autobench.records.storage import EnvironmentMetadata, ResolvedFileHash, hash_file
from autobench.records.views import (
    manifest_to_yaml_view,
    run_record_payload_from_yaml_view,
)
from autobench.runtime.context import SpanRecord
from autobench.runtime.lifecycle import RunPhase
from autobench.runtime.models import (
    BenchmarkPlan,
    EvaluationStatus,
    ExecutionCorrelation,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTermination,
    MatrixRunSpec,
    RunResult,
    RunStatus,
)
from autobench.runtime.tasks import TaskStatus
from autobench.tracking import AssetUse, AssetVersion, TrackingRegistry, track

STAGING_VERSION = 1
STAGING_STATE_PATH = "staging.yaml"
STAGING_MANIFEST_PATH = "staging-manifest.yaml"


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class _AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


class StagingStatus(StrEnum):
    ACTIVE = "active"
    FINALIZING = "finalizing"
    ABORTED = "aborted"


class StagingHealth(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    CORRUPT = "corrupt"
    CONFLICTING = "conflicting"


class ExperimentStart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = STAGING_VERSION
    experiment_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    plan: BenchmarkPlan
    runs: tuple[MatrixRunSpec, ...]
    environment: EnvironmentMetadata
    semantic_registry: SemanticRegistry
    report_spec_data: dict[str, Any] | None = None
    spec_snapshot: dict[str, Any] | None = None
    spec_hash: str | None = None
    file_hashes: tuple[ResolvedFileHash, ...] = ()
    requires_cross_run_derivation: bool = False
    requires_policies: bool = False
    correlation: ExecutionCorrelation | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> ExperimentStart:
        run_ids = tuple(run.run_id for run in self.runs)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("planned run ids must be unique")
        if self.plan.planned_run_count != len(run_ids):
            raise ValueError("plan count must match the planned runs")
        if any(run.experiment_id != self.experiment_id for run in self.runs):
            raise ValueError("planned runs must belong to the experiment")
        if any(run.benchmark_id != self.benchmark_id for run in self.runs):
            raise ValueError("planned runs must belong to the benchmark")
        if any(run.correlation != self.correlation for run in self.runs):
            raise ValueError("planned run correlation must match the experiment correlation")
        return self


class ExecutionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run: RunResult
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    signal_sequence_watermark: int = Field(default=0, ge=0)

    @classmethod
    def from_result(cls, run: RunResult) -> ExecutionSnapshot:
        copied = run.model_copy(deep=True)
        watermark = (
            max(
                (signal.sequence for signal in copied.trace.signals),
                default=0,
            )
            if copied.trace is not None
            else 0
        )
        return cls(run=copied, signal_sequence_watermark=watermark)


class PartialRunSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    phase: RunPhase
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_status: TaskStatus = TaskStatus.CANCELLED
    end_reason: EndReason = EndReason.CANCELLED
    task_output: Any = None
    observations: tuple[Observation, ...] = ()
    spans: tuple[SpanRecord, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()
    asset_versions: tuple[AssetVersion, ...] = ()
    asset_uses: tuple[AssetUse, ...] = ()
    source_snapshots: tuple[SourceSnapshot, ...] = ()
    trace: Trace | None = None
    signal_sequence_watermark: int = Field(default=0, ge=0)
    correlation: ExecutionCorrelation | None = None


class StagedRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    record_path: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    signal_sequence_watermark: int = Field(ge=0)
    files: tuple[ManifestEntry, ...]


class StagedCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    signal_sequence_watermark: int = Field(ge=0)
    file: ManifestEntry


class StagingManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = STAGING_VERSION
    experiment_id: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    runs: tuple[StagedRun, ...] = ()
    checkpoints: tuple[StagedCheckpoint, ...] = ()
    payloads: tuple[ManifestEntry, ...] = ()

    @model_validator(mode="after")
    def validate_identities(self) -> StagingManifest:
        run_ids = tuple(run.run_id for run in self.runs)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("staging manifest contains duplicate run ids")
        checkpoint_keys = tuple((item.run_id, item.name) for item in self.checkpoints)
        if len(checkpoint_keys) != len(set(checkpoint_keys)):
            raise ValueError("staging manifest contains duplicate checkpoints")
        paths = [entry.path for run in self.runs for entry in run.files]
        paths.extend(checkpoint.file.path for checkpoint in self.checkpoints)
        paths.extend(entry.path for entry in self.payloads)
        if len(paths) != len(set(paths)):
            raise ValueError("staging manifest contains conflicting file paths")
        return self


class StagingState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = STAGING_VERSION
    experiment_id: str = Field(min_length=1)
    status: StagingStatus = StagingStatus.ACTIVE
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    termination: ExperimentTermination | None = None


class StagingInspection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    experiment_id: str
    health: StagingHealth
    recoverable: bool
    status: StagingStatus
    planned_run_ids: tuple[str, ...]
    complete_run_ids: tuple[str, ...] = ()
    checkpointed_run_ids: tuple[str, ...] = ()
    missing_run_ids: tuple[str, ...] = ()
    corrupt_run_ids: tuple[str, ...] = ()
    conflicting_run_ids: tuple[str, ...] = ()
    orphaned_files: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class RecoveredStaging(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: ExperimentStart
    state: StagingState
    manifest: StagingManifest
    inspection: StagingInspection
    runs: tuple[RunRecord, ...]
    checkpoints: tuple[PartialRunSnapshot, ...] = ()


class Recorder(Protocol):
    async def open(self, start: ExperimentStart) -> RecordSession: ...


class RecordSession(Protocol):
    async def stage(self, snapshot: ExecutionSnapshot) -> None: ...

    async def checkpoint(self, snapshot: PartialRunSnapshot) -> None: ...

    async def finish(self, result: ExperimentResult) -> ExperimentRecord: ...

    async def abort(self, termination: ExperimentTermination) -> None: ...

    async def close(self) -> None: ...


class FileRecorder:
    def __init__(
        self,
        output_dir: Path,
        *,
        source_files: Collection[Path] = (),
        path_root: Path | None = None,
        durability: RecordDurability = "atomic",
        trace_inline_limit_bytes: int = TRACE_INLINE_LIMIT_BYTES,
        asset_registry: TrackingRegistry = track,
    ) -> None:
        if trace_inline_limit_bytes < 1:
            raise ValueError("trace_inline_limit_bytes must be at least 1")
        self.output_dir = output_dir
        self.staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
        self.source_files = tuple(source_files)
        self.path_root = path_root
        self.durability: RecordDurability = durability
        self.trace_inline_limit_bytes = trace_inline_limit_bytes
        self.asset_registry = asset_registry

    async def open(self, start: ExperimentStart) -> FileRecordSession:
        return await asyncio.to_thread(self.open_sync, start)

    def open_sync(self, start: ExperimentStart) -> FileRecordSession:
        if self.output_dir.is_symlink() or (
            self.output_dir.exists()
            and (not self.output_dir.is_dir() or any(self.output_dir.iterdir()))
        ):
            raise RecordingError(f"Record target already exists: {self.output_dir}")
        if self.staging_dir.exists() or self.staging_dir.is_symlink():
            raise RecordingError(
                f"Staging target already exists; inspect or recover it first: {self.staging_dir}"
            )
        active_start = start.model_copy(
            update={
                "file_hashes": source_file_hashes(
                    self.source_files,
                    path_root=self.path_root,
                )
            }
        )
        validate_logical_targets(
            tuple(
                LogicalRecordTarget(
                    path=(
                        f"cases/{path_component(run.case.id)}/"
                        f"{path_component(run.variant.id)}/run.yaml"
                    ),
                    kind=RecordFileKind.RUN,
                    identity=run.run_id,
                )
                for run in active_start.runs
            )
        )
        self.staging_dir.mkdir(parents=True)
        try:
            state = StagingState(experiment_id=active_start.experiment_id)
            manifest = StagingManifest(experiment_id=active_start.experiment_id)
            atomic_write_text(
                self.staging_dir / STAGING_STATE_PATH,
                dump_yaml(
                    experiment_start_to_yaml_view(active_start, state),
                    schema_name="staging",
                ),
                durability=self.durability,
            )
            atomic_write_text(
                self.staging_dir / STAGING_MANIFEST_PATH,
                dump_yaml(staging_manifest_to_yaml_view(manifest), schema_name="staging_manifest"),
                durability=self.durability,
            )
        except BaseException:
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            raise
        return FileRecordSession(
            recorder=self,
            start=active_start,
            state=state,
            manifest=manifest,
        )


class FileRecordSession:
    def __init__(
        self,
        *,
        recorder: FileRecorder,
        start: ExperimentStart,
        state: StagingState,
        manifest: StagingManifest,
    ) -> None:
        self.recorder = recorder
        self.start = start
        self.state = state
        self.manifest = manifest
        self.state_lock = asyncio.Lock()
        self.run_locks = {run.run_id: asyncio.Lock() for run in start.runs}
        self.prepared_artifacts: dict[tuple[str, str], tuple[ArtifactRef, Path]] = {}
        self.prepared_artifacts_lock = RLock()
        self.closed = False
        self.finished = False

    def prepare_file(
        self,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Path,
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        symlinks: SymlinkPolicy,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef:
        self.require_open()
        self._require_run(run_id)
        self._require_unprepared(run_id, artifact_id)
        followed = source.is_symlink()
        if followed and symlinks is SymlinkPolicy.REJECT:
            raise RecordingError(
                f"Artifact source is a symlink and symlinks are rejected: {source.name}"
            )
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise RecordingError(
                f"Artifact source does not exist or cannot be read: {source.name}"
            ) from exc
        if resolved.is_dir():
            raise RecordingError("Artifact sources must be files; directories are not supported.")
        if not resolved.is_file():
            raise RecordingError(f"Artifact source is not a regular file: {source.name}")
        active_filename = self._artifact_filename(filename or source.name)
        with resolved.open("rb") as stream:
            return self._prepare_sync_chunks(
                run_id=run_id,
                artifact_id=artifact_id,
                name=name,
                source=iter(lambda: stream.read(1024 * 1024), b""),
                media_type=media_type,
                max_bytes=max_bytes,
                overflow=overflow,
                filename=active_filename,
                span_id=span_id,
                tags=tags,
                artifact_source=ArtifactSource.FILE,
                symlink_followed=followed,
                close_source=False,
            )

    def prepare_stream(
        self,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Iterable[bytes],
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef:
        self.require_open()
        self._require_run(run_id)
        self._require_unprepared(run_id, artifact_id)
        return self._prepare_sync_chunks(
            run_id=run_id,
            artifact_id=artifact_id,
            name=name,
            source=source,
            media_type=media_type,
            max_bytes=max_bytes,
            overflow=overflow,
            filename=None if filename is None else self._artifact_filename(filename),
            span_id=span_id,
            tags=tags,
            artifact_source=ArtifactSource.STREAM,
            symlink_followed=False,
            close_source=True,
        )

    async def prepare_stream_async(
        self,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: AsyncIterable[bytes],
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
    ) -> ArtifactRef:
        self.require_open()
        self._require_run(run_id)
        self._require_unprepared(run_id, artifact_id)
        active_filename = None if filename is None else self._artifact_filename(filename)
        prototype = ArtifactRef(
            id=artifact_id,
            name=name,
            media_type=media_type,
            span_id=span_id,
            tags=tags,
            source=ArtifactSource.STREAM,
            filename=active_filename,
        )
        target = artifact_payload_path(
            self.recorder.staging_dir / "artifacts",
            run_id=run_id,
            artifact=prototype,
        )
        temporary, stream = self._open_prepared_target(target)
        digest = hashlib.sha256()
        byte_count = 0
        state = ArtifactState.COMPLETE
        failure: BaseException | None = None
        iterator: AsyncIterator[bytes] = aiter(source)
        try:
            async for chunk in iterator:
                chunk = self._artifact_chunk(chunk)
                remaining = max_bytes - byte_count
                if len(chunk) <= remaining:
                    stream.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                    continue
                if remaining:
                    prefix = chunk[:remaining]
                    stream.write(prefix)
                    digest.update(prefix)
                    byte_count += len(prefix)
                state = (
                    ArtifactState.TRUNCATED
                    if overflow is ArtifactOverflow.TRUNCATE
                    else ArtifactState.PARTIAL
                )
                if overflow is ArtifactOverflow.FAIL:
                    failure = ArtifactTransferError(
                        f"Artifact {name!r} exceeded max_bytes={max_bytes}.",
                        prototype,
                    )
                break
        except BaseException as exc:
            state = ArtifactState.PARTIAL
            failure = exc
        finally:
            try:
                if isinstance(iterator, _AsyncClosable):
                    await iterator.aclose()
                if source is not iterator and isinstance(source, _AsyncClosable):
                    await source.aclose()
            except BaseException as close_error:
                state = ArtifactState.PARTIAL
                if failure is None:
                    failure = close_error
            self._commit_prepared_stream(stream, temporary, target)
        artifact = prototype.model_copy(
            update={
                "state": state,
                "sha256": digest.hexdigest(),
                "byte_count": byte_count,
            }
        )
        self._retain_prepared(run_id, artifact, target)
        if failure is not None:
            if isinstance(failure, ArtifactTransferError):
                raise ArtifactTransferError(str(failure), artifact) from failure.__cause__
            raise failure
        return artifact

    def prepared_artifact(self, *, run_id: str, artifact_id: str) -> ArtifactRef | None:
        with self.prepared_artifacts_lock:
            prepared = self.prepared_artifacts.get((run_id, artifact_id))
            return None if prepared is None else prepared[0]

    def _prepare_sync_chunks(
        self,
        *,
        run_id: str,
        artifact_id: str,
        name: str,
        source: Iterable[bytes],
        media_type: str | None,
        max_bytes: int,
        overflow: ArtifactOverflow,
        filename: str | None,
        span_id: str | None,
        tags: dict[str, Any],
        artifact_source: ArtifactSource,
        symlink_followed: bool,
        close_source: bool,
    ) -> ArtifactRef:
        prototype = ArtifactRef(
            id=artifact_id,
            name=name,
            media_type=media_type,
            span_id=span_id,
            tags=tags,
            source=artifact_source,
            filename=filename,
            symlink_followed=symlink_followed,
        )
        target = artifact_payload_path(
            self.recorder.staging_dir / "artifacts",
            run_id=run_id,
            artifact=prototype,
        )
        temporary, stream = self._open_prepared_target(target)
        digest = hashlib.sha256()
        byte_count = 0
        state = ArtifactState.COMPLETE
        failure: BaseException | None = None
        iterator: Iterator[bytes] = iter(source)
        try:
            for chunk in iterator:
                chunk = self._artifact_chunk(chunk)
                remaining = max_bytes - byte_count
                if len(chunk) <= remaining:
                    stream.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                    continue
                if remaining:
                    prefix = chunk[:remaining]
                    stream.write(prefix)
                    digest.update(prefix)
                    byte_count += len(prefix)
                state = (
                    ArtifactState.TRUNCATED
                    if overflow is ArtifactOverflow.TRUNCATE
                    else ArtifactState.PARTIAL
                )
                if overflow is ArtifactOverflow.FAIL:
                    failure = ArtifactTransferError(
                        f"Artifact {name!r} exceeded max_bytes={max_bytes}.",
                        prototype,
                    )
                break
        except BaseException as exc:
            state = ArtifactState.PARTIAL
            failure = exc
        finally:
            try:
                if close_source and isinstance(iterator, _Closable):
                    iterator.close()
                if close_source and source is not iterator and isinstance(source, _Closable):
                    source.close()
            except BaseException as close_error:
                state = ArtifactState.PARTIAL
                if failure is None:
                    failure = close_error
            self._commit_prepared_stream(stream, temporary, target)
        artifact = prototype.model_copy(
            update={
                "state": state,
                "sha256": digest.hexdigest(),
                "byte_count": byte_count,
            }
        )
        self._retain_prepared(run_id, artifact, target)
        if failure is not None:
            if isinstance(failure, ArtifactTransferError):
                raise ArtifactTransferError(str(failure), artifact) from failure.__cause__
            raise failure
        return artifact

    def _open_prepared_target(self, target: Path) -> tuple[Path, BinaryIO]:
        if target.exists() or target.is_symlink():
            raise RecordingError(f"Prepared artifact already exists: {target.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        return Path(temporary_name), os.fdopen(descriptor, "wb")

    def _commit_prepared_stream(self, stream: BinaryIO, temporary: Path, target: Path) -> None:
        try:
            stream.flush()
            if self.recorder.durability == "synced":
                os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, target)
            if self.recorder.durability == "synced":
                sync_directory(target.parent)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise

    def _retain_prepared(self, run_id: str, artifact: ArtifactRef, path: Path) -> None:
        with self.prepared_artifacts_lock:
            self.prepared_artifacts[(run_id, artifact.id)] = (artifact, path)

    def _require_unprepared(self, run_id: str, artifact_id: str) -> None:
        with self.prepared_artifacts_lock:
            if (run_id, artifact_id) in self.prepared_artifacts:
                raise RecordingError(f"Prepared artifact already exists: {run_id}:{artifact_id}")

    def _require_run(self, run_id: str) -> None:
        if run_id not in self.run_locks:
            raise RecordingError(f"Run is not part of the experiment plan: {run_id}")

    @staticmethod
    def _artifact_filename(filename: str) -> str:
        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise RecordingError(f"Artifact filename must not contain path traversal: {filename!r}")
        return filename

    @staticmethod
    def _artifact_chunk(chunk: bytes) -> bytes:
        if not isinstance(chunk, bytes):
            raise TypeError(f"Artifact streams must yield bytes, received {type(chunk).__name__}.")
        return chunk

    async def stage(self, snapshot: ExecutionSnapshot) -> None:
        self.require_open()
        run = snapshot.run
        try:
            run_lock = self.run_locks[run.run_id]
        except KeyError as exc:
            raise RecordingError(f"Run is not part of the experiment plan: {run.run_id}") from exc
        run_spec = next(item for item in self.start.runs if item.run_id == run.run_id)
        if (
            run.experiment_id != self.start.experiment_id
            or run.benchmark_id != self.start.benchmark_id
            or run.case_id != run_spec.case.id
            or run.variant_id != run_spec.variant.id
            or run.correlation != self.start.correlation
        ):
            raise RecordingError(f"Run identity does not match its plan entry: {run.run_id}")
        snapshot_hash = execution_snapshot_hash(snapshot)
        async with run_lock:
            async with self.state_lock:
                existing = next(
                    (item for item in self.manifest.runs if item.run_id == run.run_id),
                    None,
                )
                if existing is None:
                    pass
                else:
                    if existing.snapshot_hash == snapshot_hash:
                        return
                    raise RecordingError(f"Run {run.run_id} was staged with different content.")
            staged = await asyncio.to_thread(
                self.write_execution_snapshot,
                snapshot,
                snapshot_hash,
            )
            async with self.state_lock:
                ordered = {item.run_id: item for item in self.manifest.runs}
                ordered[staged.run_id] = staged
                plan_order = {
                    run_spec.run_id: index for index, run_spec in enumerate(self.start.runs)
                }
                revision = self.manifest.revision + 1
                staged_paths = {entry.path for entry in staged.files}
                manifest = self.manifest.model_copy(
                    update={
                        "revision": revision,
                        "runs": tuple(
                            sorted(ordered.values(), key=lambda item: plan_order[item.run_id])
                        ),
                        "payloads": tuple(
                            entry
                            for entry in self.manifest.payloads
                            if entry.path not in staged_paths
                        ),
                    }
                )
                state = self.state.model_copy(
                    update={"revision": revision, "updated_at": datetime.now(UTC)}
                )
                await asyncio.to_thread(self.write_state_and_manifest, state, manifest)
                self.state = state
                self.manifest = manifest

    async def checkpoint(self, snapshot: PartialRunSnapshot) -> None:
        self.require_open()
        if snapshot.run_id not in self.run_locks:
            raise RecordingError(f"Run is not part of the experiment plan: {snapshot.run_id}")
        run_spec = next(item for item in self.start.runs if item.run_id == snapshot.run_id)
        if (
            snapshot.experiment_id != self.start.experiment_id
            or snapshot.benchmark_id != self.start.benchmark_id
            or snapshot.case_id != run_spec.case.id
            or snapshot.variant_id != run_spec.variant.id
            or snapshot.correlation != self.start.correlation
        ):
            raise RecordingError(
                f"Checkpoint identity does not match its plan entry: {snapshot.run_id}"
            )
        async with self.run_locks[snapshot.run_id]:
            active_snapshot, payload_entries = await asyncio.to_thread(
                self._materialize_checkpoint_artifacts,
                snapshot,
            )
            snapshot_hash = partial_snapshot_hash(active_snapshot)
            async with self.state_lock:
                previous = next(
                    (
                        item
                        for item in self.manifest.checkpoints
                        if item.run_id == snapshot.run_id and item.name == snapshot.name
                    ),
                    None,
                )
                if previous is None:
                    pass
                else:
                    if previous.snapshot_hash == snapshot_hash:
                        return
                    raise RecordingError(
                        f"Checkpoint {snapshot.run_id}:{snapshot.name} has conflicting content."
                    )
            path = (
                self.recorder.staging_dir
                / "checkpoints"
                / path_component(snapshot.run_id)
                / f"{path_component(snapshot.name)}.yaml"
            )
            relative_path = path.relative_to(self.recorder.staging_dir).as_posix()
            await asyncio.to_thread(
                atomic_write_text,
                path,
                dump_yaml(partial_snapshot_to_yaml_view(active_snapshot), schema_name="checkpoint"),
                durability=self.recorder.durability,
            )
            entry = manifest_entry(
                self.recorder.staging_dir,
                relative_path,
                kind=RecordFileKind.RUN,
                identity=f"{snapshot.run_id}:{snapshot.name}",
            )
            checkpoint = StagedCheckpoint(
                run_id=snapshot.run_id,
                name=snapshot.name,
                path=relative_path,
                snapshot_hash=snapshot_hash,
                captured_at=active_snapshot.captured_at,
                signal_sequence_watermark=active_snapshot.signal_sequence_watermark,
                file=entry,
            )
            async with self.state_lock:
                checkpoints = {(item.run_id, item.name): item for item in self.manifest.checkpoints}
                checkpoints[(snapshot.run_id, snapshot.name)] = checkpoint
                revision = self.manifest.revision + 1
                committed_payloads = {item.path: item for item in self.manifest.payloads}
                committed_payloads.update({item.path: item for item in payload_entries})
                manifest = self.manifest.model_copy(
                    update={
                        "revision": revision,
                        "checkpoints": tuple(
                            sorted(
                                checkpoints.values(),
                                key=lambda item: (item.run_id, item.name),
                            )
                        ),
                        "payloads": tuple(
                            committed_payloads[path] for path in sorted(committed_payloads)
                        ),
                    }
                )
                state = self.state.model_copy(
                    update={"revision": revision, "updated_at": datetime.now(UTC)}
                )
                await asyncio.to_thread(self.write_state_and_manifest, state, manifest)
                self.state = state
                self.manifest = manifest

    def _materialize_checkpoint_artifacts(
        self,
        snapshot: PartialRunSnapshot,
    ) -> tuple[PartialRunSnapshot, tuple[ManifestEntry, ...]]:
        artifacts: list[ArtifactRef] = []
        entries: dict[str, ManifestEntry] = {}
        artifacts_dir = self.recorder.staging_dir / "artifacts"
        for artifact in snapshot.artifacts:
            if artifact.source is ArtifactSource.VALUE:
                artifacts.append(artifact)
                continue
            with self.prepared_artifacts_lock:
                prepared = self.prepared_artifacts.get((snapshot.run_id, artifact.id))
            if prepared is None:
                raise RecordingError(
                    f"Prepared artifact is unavailable for checkpoint: {snapshot.run_id}:{artifact.id}"
                )
            prepared_ref, prepared_path = prepared
            payload_path = artifact_payload_path(
                artifacts_dir,
                run_id=snapshot.run_id,
                artifact=prepared_ref,
            )
            metadata_path = artifact_record_path(
                artifacts_dir,
                run_id=snapshot.run_id,
                artifact=prepared_ref,
            )
            if metadata_path.exists():
                digest, byte_count = hash_and_size(payload_path)
                if digest != prepared_ref.sha256 or byte_count != prepared_ref.byte_count:
                    raise RecordingError(
                        f"Prepared artifact changed after capture: {snapshot.run_id}:{artifact.id}"
                    )
                recorded = prepared_ref.model_copy(
                    update={"value": payload_path.relative_to(self.recorder.staging_dir).as_posix()}
                )
            else:
                recorded = record_artifact(
                    prepared_ref,
                    artifacts_dir=artifacts_dir,
                    root_dir=self.recorder.staging_dir,
                    run_id=snapshot.run_id,
                    durability=self.recorder.durability,
                    prepared_path=prepared_path,
                )
            artifacts.append(recorded)
            for path in (metadata_path, payload_path):
                relative_path = path.relative_to(self.recorder.staging_dir).as_posix()
                entries[relative_path] = manifest_entry(
                    self.recorder.staging_dir,
                    relative_path,
                    kind=RecordFileKind.ARTIFACT,
                    identity=f"{snapshot.run_id}:{artifact.id}:{path.name}",
                )
        return snapshot.model_copy(update={"artifacts": tuple(artifacts)}), tuple(
            entries[path] for path in sorted(entries)
        )

    async def finish(self, result: ExperimentResult) -> ExperimentRecord:
        self.require_open()
        if result.experiment_id != self.start.experiment_id:
            raise RecordingError("Experiment result does not belong to the recording session.")
        staged_ids = {item.run_id for item in self.manifest.runs}
        planned_ids = tuple(run.run_id for run in self.start.runs)
        missing = tuple(run_id for run_id in planned_ids if run_id not in staged_ids)
        if missing:
            raise RecordingError(f"Cannot finish with missing staged runs: {list(missing)}")
        finalizing = self.state.model_copy(
            update={
                "status": StagingStatus.FINALIZING,
                "updated_at": datetime.now(UTC),
                "termination": result.termination,
            }
        )
        async with self.state_lock:
            await asyncio.to_thread(self.write_state_and_manifest, finalizing, self.manifest)
            self.state = finalizing
        record = await asyncio.to_thread(self.publish_result, result)
        self.finished = True
        await asyncio.to_thread(shutil.rmtree, self.recorder.staging_dir, True)
        return record

    async def abort(self, termination: ExperimentTermination) -> None:
        if self.closed or self.finished:
            return
        async with self.state_lock:
            state = self.state.model_copy(
                update={
                    "status": StagingStatus.ABORTED,
                    "updated_at": datetime.now(UTC),
                    "termination": termination,
                }
            )
            await asyncio.to_thread(self.write_state_and_manifest, state, self.manifest)
            self.state = state

    async def close(self) -> None:
        self.closed = True

    def require_open(self) -> None:
        if self.closed:
            raise RecordingError("Record session is closed.")
        if self.finished:
            raise RecordingError("Record session is already finished.")

    def write_execution_snapshot(
        self,
        snapshot: ExecutionSnapshot,
        snapshot_hash: str,
    ) -> StagedRun:
        run = snapshot.run
        staging = self.recorder.staging_dir
        run_path = (
            staging
            / "cases"
            / path_component(run.case_id)
            / path_component(run.variant_id)
            / "run.yaml"
        )
        artifacts_dir = staging / "artifacts"
        asset_dir = staging / "assets" / path_component(run.run_id)
        targets = [
            LogicalRecordTarget(
                path=run_path.relative_to(staging).as_posix(),
                kind=RecordFileKind.RUN,
                identity=run.run_id,
            )
        ]
        external_trace_path: Path | None = None
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
        if run.trace is not None and trace_size(run.trace) > self.recorder.trace_inline_limit_bytes:
            external_trace_path = trace_artifact_path(Path("artifacts"), run_id=run.run_id)
            targets.append(
                LogicalRecordTarget(
                    path=external_trace_path.as_posix(),
                    kind=RecordFileKind.TRACE,
                    identity=run.run_id,
                )
            )
        validate_logical_targets(tuple(targets))
        if run_path.exists():
            raise RecordingError(f"Uncommitted staging data already exists for run {run.run_id}.")
        with self.prepared_artifacts_lock:
            prepared_artifacts = {
                artifact_id: path
                for (prepared_run_id, artifact_id), (_, path) in self.prepared_artifacts.items()
                if prepared_run_id == run.run_id
            }
        record = run_record_from_result(
            run,
            artifacts_dir=artifacts_dir,
            root_dir=staging,
            semantic_registry_version=self.start.semantic_registry.version,
            trace_inline_limit_bytes=self.recorder.trace_inline_limit_bytes,
            durability=self.recorder.durability,
            prepared_artifacts=prepared_artifacts,
        )
        referenced_asset_ids = {version.asset_id for version in run.asset_versions}
        persisted_asset_ids = {
            asset_id
            for asset_id in referenced_asset_ids
            if self.recorder.asset_registry.has_asset(asset_id)
        }
        if persisted_asset_ids:
            self.recorder.asset_registry.write_assets(
                asset_dir,
                asset_ids=persisted_asset_ids,
                content_path=asset_dir / "content.sqlite3",
                root_dir=staging,
            )
        atomic_write_text(
            run_path,
            dump_yaml(run_record_to_yaml_view(record), schema_name="run_record"),
            durability=self.recorder.durability,
        )
        owned_paths = [run_path]
        artifact_root = artifacts_dir / path_component(run.run_id)
        if artifact_root.exists():
            owned_paths.extend(path for path in artifact_root.rglob("*") if path.is_file())
        if asset_dir.exists():
            owned_paths.extend(path for path in asset_dir.rglob("*") if path.is_file())
        files: list[ManifestEntry] = []
        for path in sorted(owned_paths):
            relative_path = path.relative_to(staging)
            if path == run_path:
                kind = RecordFileKind.RUN
            elif path.is_relative_to(asset_dir):
                kind = RecordFileKind.ASSET
            elif external_trace_path is not None and relative_path == external_trace_path:
                kind = RecordFileKind.TRACE
            else:
                kind = RecordFileKind.ARTIFACT
            files.append(
                manifest_entry(
                    staging,
                    relative_path.as_posix(),
                    kind=kind,
                    identity=run.run_id if path == run_path else f"{run.run_id}:{path.name}",
                )
            )
        return StagedRun(
            run_id=run.run_id,
            case_id=run.case_id,
            variant_id=run.variant_id,
            record_path=run_path.relative_to(staging).as_posix(),
            snapshot_hash=snapshot_hash,
            captured_at=snapshot.captured_at,
            signal_sequence_watermark=snapshot.signal_sequence_watermark,
            files=tuple(files),
        )

    def write_state_and_manifest(
        self,
        state: StagingState,
        manifest: StagingManifest,
    ) -> None:
        atomic_write_text(
            self.recorder.staging_dir / STAGING_STATE_PATH,
            dump_yaml(
                experiment_start_to_yaml_view(self.start, state),
                schema_name="staging",
            ),
            durability=self.recorder.durability,
        )
        atomic_write_text(
            self.recorder.staging_dir / STAGING_MANIFEST_PATH,
            dump_yaml(staging_manifest_to_yaml_view(manifest), schema_name="staging_manifest"),
            durability=self.recorder.durability,
        )

    def publish_result(self, result: ExperimentResult) -> ExperimentRecord:
        recovered = recover_staging(self.recorder.staging_dir)
        finalizing = create_temporary_record_directory(self.recorder.output_dir)
        try:
            copy_committed_files(recovered, finalizing)
            staged_by_id = {run.run_id: run for run in recovered.manifest.runs}
            run_paths: list[str] = []
            for run in result.runs:
                staged = staged_by_id[run.run_id]
                path = finalizing / staged.record_path
                existing = RunRecord.model_validate(
                    run_record_payload_from_yaml_view(load_yaml(path))
                )
                updated = run_record_from_result(
                    run,
                    artifacts_dir=finalizing / "artifacts",
                    root_dir=finalizing,
                    semantic_registry_version=result.semantic_registry.version,
                    trace_inline_limit_bytes=self.recorder.trace_inline_limit_bytes,
                    durability=self.recorder.durability,
                    recorded_payloads=RecordedRunPayloads(
                        artifacts=existing.artifacts,
                        trace=existing.trace,
                        trace_artifact=existing.trace_artifact,
                    ),
                )
                atomic_write_text(
                    path,
                    dump_yaml(run_record_to_yaml_view(updated), schema_name="run_record"),
                    durability=self.recorder.durability,
                )
                run_paths.append(staged.record_path)
            record = experiment_record_from_result(
                result,
                run_paths=tuple(run_paths),
                file_hashes=self.start.file_hashes,
            )
            write_final_metadata(
                finalizing,
                record,
                durability=self.recorder.durability,
            )
            publish_record_directory(
                finalizing,
                self.recorder.output_dir,
                durability=self.recorder.durability,
            )
            return record
        finally:
            remove_temporary_record_directory(finalizing)


def inspect_staging(path: Path) -> StagingInspection:
    start, state = load_staging_state(path)
    planned_ids = tuple(run.run_id for run in start.runs)
    try:
        manifest = load_staging_manifest(path)
    except (RecordingError, ValidationError) as exc:
        return StagingInspection(
            path=path,
            experiment_id=start.experiment_id,
            health=StagingHealth.CONFLICTING,
            recoverable=False,
            status=state.status,
            planned_run_ids=planned_ids,
            missing_run_ids=planned_ids,
            diagnostics=(str(exc),),
        )
    corrupt: list[str] = []
    conflicting: list[str] = []
    diagnostics: list[str] = []
    complete: list[str] = []
    plan_by_id = {run.run_id: run for run in start.runs}
    if manifest.experiment_id != start.experiment_id:
        diagnostics.append(
            "staging manifest experiment does not match staging state: "
            f"{manifest.experiment_id!r} != {start.experiment_id!r}"
        )
        conflicting.extend(planned_ids)
    if manifest.revision != state.revision:
        diagnostics.append(
            "staging state and manifest revisions differ; only manifest-committed evidence "
            f"will be recovered ({state.revision} != {manifest.revision})"
        )
    expected_paths = {STAGING_STATE_PATH, STAGING_MANIFEST_PATH}
    for staged in manifest.runs:
        run_spec = plan_by_id.get(staged.run_id)
        if run_spec is None:
            conflicting.append(staged.run_id)
            diagnostics.append(f"staged run is not in the experiment plan: {staged.run_id}")
        elif staged.case_id != run_spec.case.id or staged.variant_id != run_spec.variant.id:
            conflicting.append(staged.run_id)
            diagnostics.append(f"staged run identity does not match its plan: {staged.run_id}")
        if staged.record_path not in {entry.path for entry in staged.files}:
            conflicting.append(staged.run_id)
            diagnostics.append(
                f"staged run record is not committed by its file list: {staged.run_id}"
            )
        invalid = validate_staged_entries(path, staged.files, diagnostics=diagnostics)
        if invalid:
            corrupt.append(staged.run_id)
        else:
            try:
                record = RunRecord.model_validate(
                    run_record_payload_from_yaml_view(load_yaml(path / staged.record_path))
                )
            except (OSError, RecordingError, ValidationError) as exc:
                corrupt.append(staged.run_id)
                diagnostics.append(f"invalid staged run record {staged.run_id}: {exc}")
            else:
                if (
                    record.run_id != staged.run_id
                    or record.experiment_id != start.experiment_id
                    or record.benchmark_id != start.benchmark_id
                    or record.case_id != staged.case_id
                    or record.variant_id != staged.variant_id
                ):
                    conflicting.append(staged.run_id)
                    diagnostics.append(
                        f"staged run payload identity does not match its manifest: {staged.run_id}"
                    )
                else:
                    complete.append(staged.run_id)
        expected_paths.update(entry.path for entry in staged.files)
    checkpointed: list[str] = []
    for checkpoint in manifest.checkpoints:
        run_spec = plan_by_id.get(checkpoint.run_id)
        if run_spec is None:
            conflicting.append(checkpoint.run_id)
            diagnostics.append(f"checkpoint run is not in the experiment plan: {checkpoint.run_id}")
        if checkpoint.path != checkpoint.file.path:
            conflicting.append(checkpoint.run_id)
            diagnostics.append(
                f"checkpoint path is not committed by its file entry: {checkpoint.run_id}:{checkpoint.name}"
            )
        if validate_staged_entries(path, (checkpoint.file,), diagnostics=diagnostics):
            corrupt.append(checkpoint.run_id)
        else:
            try:
                snapshot = partial_snapshot_from_yaml_view(load_yaml(path / checkpoint.path))
            except (OSError, RecordingError, ValidationError) as exc:
                corrupt.append(checkpoint.run_id)
                diagnostics.append(
                    f"invalid staged checkpoint {checkpoint.run_id}:{checkpoint.name}: {exc}"
                )
            else:
                if (
                    snapshot.run_id != checkpoint.run_id
                    or snapshot.name != checkpoint.name
                    or snapshot.experiment_id != start.experiment_id
                    or snapshot.benchmark_id != start.benchmark_id
                    or (
                        run_spec is not None
                        and (
                            snapshot.case_id != run_spec.case.id
                            or snapshot.variant_id != run_spec.variant.id
                        )
                    )
                ):
                    conflicting.append(checkpoint.run_id)
                    diagnostics.append(
                        "staged checkpoint payload identity does not match its manifest: "
                        f"{checkpoint.run_id}:{checkpoint.name}"
                    )
                else:
                    checkpointed.append(checkpoint.run_id)
        expected_paths.add(checkpoint.file.path)
    if validate_staged_entries(path, manifest.payloads, diagnostics=diagnostics):
        corrupt.extend(
            entry.identity.partition(":")[0]
            for entry in manifest.payloads
            if entry.identity.partition(":")[0] in plan_by_id
        )
    expected_paths.update(entry.path for entry in manifest.payloads)
    actual_paths = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and not item.name.endswith(".tmp")
    }
    orphaned = tuple(sorted(actual_paths - expected_paths))
    missing = tuple(run_id for run_id in planned_ids if run_id not in complete)
    fatal_conflict = bool(conflicting)
    if corrupt:
        health = StagingHealth.CORRUPT
    elif fatal_conflict or orphaned or manifest.revision != state.revision:
        health = StagingHealth.CONFLICTING
        if orphaned:
            diagnostics.append("staging contains uncommitted files; recovery will ignore them")
    elif not missing:
        health = StagingHealth.COMPLETE
    elif complete or checkpointed:
        health = StagingHealth.PARTIAL
    else:
        health = StagingHealth.MISSING
    return StagingInspection(
        path=path,
        experiment_id=start.experiment_id,
        health=health,
        recoverable=not corrupt and not fatal_conflict,
        status=state.status,
        planned_run_ids=planned_ids,
        complete_run_ids=tuple(run_id for run_id in planned_ids if run_id in complete),
        checkpointed_run_ids=tuple(dict.fromkeys(checkpointed)),
        missing_run_ids=missing,
        corrupt_run_ids=tuple(dict.fromkeys(corrupt)),
        conflicting_run_ids=tuple(dict.fromkeys(conflicting)),
        orphaned_files=orphaned,
        diagnostics=tuple(diagnostics),
    )


def recover_staging(path: Path) -> RecoveredStaging:
    inspection = inspect_staging(path)
    if not inspection.recoverable:
        raise RecordingError(
            f"Staging evidence is {inspection.health.value}: {list(inspection.diagnostics)}"
        )
    start, state = load_staging_state(path)
    manifest = load_staging_manifest(path)
    runs = tuple(
        RunRecord.model_validate(
            run_record_payload_from_yaml_view(load_yaml(path / staged.record_path))
        )
        for staged in manifest.runs
    )
    checkpoints = tuple(
        partial_snapshot_from_yaml_view(load_yaml(path / checkpoint.path))
        for checkpoint in manifest.checkpoints
    )
    return RecoveredStaging(
        start=start,
        state=state,
        manifest=manifest,
        inspection=inspection,
        runs=runs,
        checkpoints=checkpoints,
    )


def finalize_staging(
    path: Path,
    output_dir: Path,
    *,
    allow_partial: bool = False,
    durability: RecordDurability = "atomic",
) -> ExperimentRecord:
    recovered = recover_staging(path)
    complete_by_id = {run.run_id: run for run in recovered.runs}
    latest_checkpoints: dict[str, PartialRunSnapshot] = {}
    for checkpoint in recovered.checkpoints:
        current = latest_checkpoints.get(checkpoint.run_id)
        if current is None or checkpoint.captured_at > current.captured_at:
            latest_checkpoints[checkpoint.run_id] = checkpoint
    run_records: dict[str, RunRecord] = dict(complete_by_id)
    run_paths = {
        run.run_id: staged.record_path
        for run, staged in zip(
            recovered.runs,
            recovered.manifest.runs,
            strict=True,
        )
    }
    for run_spec in recovered.start.runs:
        checkpoint = latest_checkpoints.get(run_spec.run_id)
        if run_spec.run_id in run_records or checkpoint is None:
            continue
        record = checkpoint_run_record(
            checkpoint,
            run_spec=run_spec,
            semantic_registry_version=recovered.start.semantic_registry.version,
        )
        run_records[run_spec.run_id] = record
        run_paths[run_spec.run_id] = (
            f"cases/{path_component(run_spec.case.id)}/"
            f"{path_component(run_spec.variant.id)}/run.yaml"
        )
    planned_ids = tuple(run.run_id for run in recovered.start.runs)
    recorded_ids = tuple(run_id for run_id in planned_ids if run_id in run_records)
    missing = tuple(run_id for run_id in planned_ids if run_id not in run_records)
    post_processing_incomplete = (
        len(complete_by_id) != len(recovered.start.runs)
        or recovered.start.requires_cross_run_derivation
        or recovered.start.requires_policies
    )
    partial = bool(missing or post_processing_incomplete)
    if partial and not allow_partial:
        raise RecordingError(
            "Staging is incomplete; pass allow_partial=True to publish explicit partial evidence."
        )
    partial_status = ExperimentStatus.ABORTED
    if (
        recovered.state.termination is not None
        and recovered.state.termination.status is ExperimentStatus.CANCELLED
    ):
        partial_status = ExperimentStatus.CANCELLED
    termination = ExperimentTermination(
        status=partial_status if partial else ExperimentStatus.COMPLETED,
        partial=partial,
        cross_run_derivation_complete=not recovered.start.requires_cross_run_derivation,
        policies_complete=not recovered.start.requires_policies,
        planned_run_ids=planned_ids,
        recorded_run_ids=recorded_ids,
        missing_run_ids=missing,
        error=recovered.state.termination.error
        if recovered.state.termination is not None
        else None,
    )
    ordered_records = tuple(run_records[run_id] for run_id in recorded_ids)
    ordered_paths = tuple(run_paths[run_id] for run_id in recorded_ids)
    record = experiment_record_from_recovery(
        recovered,
        termination=termination,
        runs=ordered_records,
        run_paths=ordered_paths,
    )
    finalizing = create_temporary_record_directory(output_dir)
    try:
        copy_committed_files(recovered, finalizing)
        for run_record, run_path in zip(ordered_records, ordered_paths, strict=True):
            target = finalizing / run_path
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                target,
                dump_yaml(run_record_to_yaml_view(run_record), schema_name="run_record"),
                durability=durability,
            )
        write_final_metadata(finalizing, record, durability=durability)
        publish_record_directory(finalizing, output_dir, durability=durability)
        return record
    finally:
        remove_temporary_record_directory(finalizing)


def archive_staging(
    path: Path,
    destination: Path,
    *,
    durability: RecordDurability = "atomic",
) -> Path:
    load_staging_state(path)
    temporary = create_temporary_record_directory(destination)
    try:
        shutil.copytree(path, temporary, dirs_exist_ok=True)
        publish_record_directory(temporary, destination, durability=durability)
    finally:
        remove_temporary_record_directory(temporary)
    return destination


def discard_staging(path: Path) -> None:
    load_staging_state(path)
    shutil.rmtree(path)


def experiment_start_to_yaml_view(
    start: ExperimentStart,
    state: StagingState,
) -> dict[str, Any]:
    return {
        "staging": {
            "type": "experiment",
            "version": start.version,
            "status": state.status.value,
            "revision": state.revision,
            "updated_at": state.updated_at,
        },
        "experiment": {
            "id": start.experiment_id,
            "benchmark": start.benchmark_id,
            "started_at": start.started_at,
            "correlation": (
                None
                if start.correlation is None
                else start.correlation.model_dump(mode="json", exclude_none=True)
            ),
            "termination": (
                None if state.termination is None else state.termination.model_dump(mode="json")
            ),
        },
        "plan": start.plan.model_dump(mode="json"),
        "runs": [run.model_dump(mode="json") for run in start.runs],
        "post_processing": {
            "cross_run_derivation": start.requires_cross_run_derivation,
            "policies": start.requires_policies,
        },
        "environment": start.environment.model_dump(mode="json"),
        "semantic_registry": start.semantic_registry.model_dump(mode="json"),
        "report": start.report_spec_data,
        "benchmark_spec": start.spec_snapshot,
        "spec_hash": start.spec_hash,
        "source_files": [file_hash.model_dump(mode="json") for file_hash in start.file_hashes],
    }


def staging_manifest_to_yaml_view(manifest: StagingManifest) -> dict[str, Any]:
    return {
        "staging": {
            "type": "manifest",
            "version": manifest.version,
            "revision": manifest.revision,
        },
        "experiment": {"id": manifest.experiment_id},
        "runs": [run.model_dump(mode="json") for run in manifest.runs],
        "checkpoints": [item.model_dump(mode="json") for item in manifest.checkpoints],
        "payloads": [item.model_dump(mode="json") for item in manifest.payloads],
    }


def partial_snapshot_to_yaml_view(snapshot: PartialRunSnapshot) -> dict[str, Any]:
    return {
        "checkpoint": {
            "version": STAGING_VERSION,
            "name": snapshot.name,
            "phase": snapshot.phase.value,
            "captured_at": snapshot.captured_at,
            "signal_sequence_watermark": snapshot.signal_sequence_watermark,
        },
        "run": {
            "id": snapshot.run_id,
            "experiment": snapshot.experiment_id,
            "benchmark": snapshot.benchmark_id,
            "case": snapshot.case_id,
            "variant": snapshot.variant_id,
            "task_status": snapshot.task_status.value,
            "end_reason": snapshot.end_reason.value,
            "correlation": (
                None
                if snapshot.correlation is None
                else snapshot.correlation.model_dump(mode="json", exclude_none=True)
            ),
        },
        "evidence": {
            "output": snapshot.task_output,
            "observations": [item.model_dump(mode="json") for item in snapshot.observations],
            "spans": [item.model_dump(mode="json") for item in snapshot.spans],
            "artifacts": [item.model_dump(mode="json") for item in snapshot.artifacts],
            "errors": [item.model_dump(mode="json") for item in snapshot.errors],
            "asset_versions": [item.model_dump(mode="json") for item in snapshot.asset_versions],
            "asset_uses": [item.model_dump(mode="json") for item in snapshot.asset_uses],
            "source_snapshots": [
                item.model_dump(mode="json") for item in snapshot.source_snapshots
            ],
            "trace": None if snapshot.trace is None else snapshot.trace.model_dump(mode="json"),
        },
    }


def partial_snapshot_from_yaml_view(value: Any) -> PartialRunSnapshot:
    if not isinstance(value, dict):
        raise RecordingError("Checkpoint YAML must be a mapping.")
    checkpoint = value.get("checkpoint")
    run = value.get("run")
    evidence = value.get("evidence")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(run, dict)
        or not isinstance(evidence, dict)
    ):
        raise RecordingError("Checkpoint YAML is missing checkpoint, run, or evidence sections.")
    return PartialRunSnapshot.model_validate(
        {
            "run_id": run.get("id"),
            "experiment_id": run.get("experiment"),
            "benchmark_id": run.get("benchmark"),
            "case_id": run.get("case"),
            "variant_id": run.get("variant"),
            "name": checkpoint.get("name"),
            "phase": checkpoint.get("phase"),
            "captured_at": checkpoint.get("captured_at"),
            "signal_sequence_watermark": checkpoint.get("signal_sequence_watermark", 0),
            "task_status": run.get("task_status"),
            "end_reason": run.get("end_reason"),
            "task_output": evidence.get("output"),
            "observations": evidence.get("observations", []),
            "spans": evidence.get("spans", []),
            "artifacts": evidence.get("artifacts", []),
            "errors": evidence.get("errors", []),
            "asset_versions": evidence.get("asset_versions", []),
            "asset_uses": evidence.get("asset_uses", []),
            "source_snapshots": evidence.get("source_snapshots", []),
            "trace": evidence.get("trace"),
            "correlation": run.get("correlation"),
        }
    )


def load_staging_state(path: Path) -> tuple[ExperimentStart, StagingState]:
    state_path = path / STAGING_STATE_PATH
    if path.is_symlink() or not path.is_dir() or not state_path.is_file():
        raise RecordingError(f"Not an Autobench staging directory: {path}")
    value = load_yaml(state_path)
    if not isinstance(value, dict):
        raise RecordingError("Staging state YAML must be a mapping.")
    staging = value.get("staging")
    experiment = value.get("experiment")
    post_processing = value.get("post_processing")
    if not isinstance(staging, dict) or not isinstance(experiment, dict):
        raise RecordingError("Staging state is missing staging or experiment sections.")
    if not isinstance(post_processing, dict):
        post_processing = {}
    start = ExperimentStart.model_validate(
        {
            "version": staging.get("version"),
            "experiment_id": experiment.get("id"),
            "benchmark_id": experiment.get("benchmark"),
            "started_at": experiment.get("started_at"),
            "plan": value.get("plan"),
            "runs": value.get("runs", []),
            "environment": value.get("environment"),
            "semantic_registry": value.get("semantic_registry"),
            "report_spec_data": value.get("report"),
            "spec_snapshot": value.get("benchmark_spec"),
            "spec_hash": value.get("spec_hash"),
            "file_hashes": value.get("source_files", []),
            "requires_cross_run_derivation": post_processing.get("cross_run_derivation", False),
            "requires_policies": post_processing.get("policies", False),
            "correlation": experiment.get("correlation"),
        }
    )
    state = StagingState.model_validate(
        {
            "version": staging.get("version"),
            "experiment_id": experiment.get("id"),
            "status": staging.get("status"),
            "revision": staging.get("revision", 0),
            "updated_at": staging.get("updated_at"),
            "termination": experiment.get("termination"),
        }
    )
    return start, state


def load_staging_manifest(path: Path) -> StagingManifest:
    value = load_yaml(path / STAGING_MANIFEST_PATH)
    if not isinstance(value, dict):
        raise RecordingError("Staging manifest YAML must be a mapping.")
    staging = value.get("staging")
    experiment = value.get("experiment")
    if not isinstance(staging, dict) or not isinstance(experiment, dict):
        raise RecordingError("Staging manifest is missing staging or experiment sections.")
    return StagingManifest.model_validate(
        {
            "version": staging.get("version"),
            "experiment_id": experiment.get("id"),
            "revision": staging.get("revision", 0),
            "runs": value.get("runs", []),
            "checkpoints": value.get("checkpoints", []),
            "payloads": value.get("payloads", []),
        }
    )


def execution_snapshot_hash(snapshot: ExecutionSnapshot) -> str:
    return hashlib.sha256(
        json.dumps(
            snapshot.run.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def partial_snapshot_hash(snapshot: PartialRunSnapshot) -> str:
    return hashlib.sha256(
        json.dumps(
            snapshot.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def trace_size(trace: Trace) -> int:
    return len(
        json.dumps(
            trace.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def manifest_entry(
    root: Path,
    relative_path: str,
    *,
    kind: RecordFileKind,
    identity: str,
) -> ManifestEntry:
    normalized = normalize_logical_path(relative_path)
    path = root / normalized
    digest, byte_count = hash_and_size(path)
    return ManifestEntry(
        path=normalized,
        sha256=digest,
        byte_count=byte_count,
        kind=kind,
        identity=identity,
    )


def validate_staged_entries(
    root: Path,
    entries: tuple[ManifestEntry, ...],
    *,
    diagnostics: list[str],
) -> bool:
    invalid = False
    resolved_root = root.resolve()
    for entry in entries:
        path = root / entry.path
        if not path.exists() or not path.is_file():
            diagnostics.append(f"missing staged file: {entry.path}")
            invalid = True
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            diagnostics.append(f"staged file escapes its root: {entry.path}")
            invalid = True
            continue
        digest, byte_count = hash_and_size(path)
        if byte_count != entry.byte_count:
            diagnostics.append(f"staged byte count mismatch: {entry.path}")
            invalid = True
        if digest != entry.sha256:
            diagnostics.append(f"staged hash mismatch: {entry.path}")
            invalid = True
    return invalid


def copy_committed_files(recovered: RecoveredStaging, destination: Path) -> None:
    entries = [entry for run in recovered.manifest.runs for entry in run.files]
    entries.extend(recovered.manifest.payloads)
    for entry in entries:
        source = recovered.inspection.path / entry.path
        target = destination / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def experiment_record_from_result(
    result: ExperimentResult,
    *,
    run_paths: tuple[str, ...],
    file_hashes: tuple[ResolvedFileHash, ...],
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=result.experiment_id,
        benchmark_id=result.benchmark_id,
        plan=result.plan,
        environment=result.environment,
        termination=result.termination,
        semantic_registry=result.semantic_registry,
        report_spec_data=result.report_spec_data,
        spec_snapshot=result.spec_snapshot,
        spec_hash=result.spec_hash,
        file_hashes=file_hashes,
        manifest_path="manifest.yaml",
        run_paths=run_paths,
        run_count=result.total_count,
        passed_count=result.passed_count,
        failed_count=result.failed_count,
        errored_count=result.errored_count,
        skipped_count=result.skipped_count,
        cancelled_count=result.cancelled_count,
        correlation=result.correlation,
    )


def experiment_record_from_recovery(
    recovered: RecoveredStaging,
    *,
    termination: ExperimentTermination,
    runs: tuple[RunRecord, ...],
    run_paths: tuple[str, ...],
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=recovered.start.experiment_id,
        benchmark_id=recovered.start.benchmark_id,
        plan=recovered.start.plan,
        environment=recovered.start.environment,
        termination=termination,
        semantic_registry=recovered.start.semantic_registry,
        report_spec_data=recovered.start.report_spec_data,
        spec_snapshot=recovered.start.spec_snapshot,
        spec_hash=recovered.start.spec_hash,
        file_hashes=recovered.start.file_hashes,
        manifest_path="manifest.yaml",
        run_paths=run_paths,
        run_count=len(runs),
        passed_count=sum(run.status is RunStatus.PASSED for run in runs),
        failed_count=sum(run.status is RunStatus.FAILED for run in runs),
        errored_count=sum(run.status is RunStatus.ERRORED for run in runs),
        skipped_count=sum(run.status is RunStatus.SKIPPED for run in runs),
        cancelled_count=sum(run.status is RunStatus.CANCELLED for run in runs),
        correlation=recovered.start.correlation,
    )


def checkpoint_run_record(
    snapshot: PartialRunSnapshot,
    *,
    run_spec: MatrixRunSpec,
    semantic_registry_version: int,
) -> RunRecord:
    errors = snapshot.errors
    return RunRecord(
        protocol_version=None if snapshot.trace is None else snapshot.trace.protocol_version,
        semantic_registry_version=semantic_registry_version,
        run_id=snapshot.run_id,
        experiment_id=snapshot.experiment_id,
        benchmark_id=snapshot.benchmark_id,
        case_id=snapshot.case_id,
        variant_id=snapshot.variant_id,
        status=RunStatus.CANCELLED,
        evaluation_status=EvaluationStatus.NOT_EVALUATED,
        task_status=snapshot.task_status,
        partial=True,
        end_reason=snapshot.end_reason,
        case=run_spec.case,
        task_output=snapshot.task_output,
        observations=snapshot.observations,
        spans=snapshot.spans,
        trace=snapshot.trace,
        artifacts=snapshot.artifacts,
        factors=tuple(run_spec.variant.factors),
        asset_versions=snapshot.asset_versions,
        asset_uses=snapshot.asset_uses,
        source_snapshots=snapshot.source_snapshots,
        errors=errors,
        error=errors[0] if errors else None,
        correlation=snapshot.correlation,
    )


def write_final_metadata(
    root: Path,
    record: ExperimentRecord,
    *,
    durability: RecordDurability,
) -> None:
    atomic_write_text(
        root / "experiment.yaml",
        dump_yaml(experiment_record_to_yaml_view(record), schema_name="experiment"),
        durability=durability,
    )
    atomic_write_text(
        root / "summary.yaml",
        dump_yaml(experiment_summary(record), schema_name="summary"),
        durability=durability,
    )
    targets = {
        target.path: target
        for target in (
            LogicalRecordTarget(
                path="experiment.yaml",
                kind=RecordFileKind.EXPERIMENT,
                identity=record.experiment_id,
            ),
            LogicalRecordTarget(
                path="summary.yaml",
                kind=RecordFileKind.SUMMARY,
                identity=record.experiment_id,
            ),
            *(
                LogicalRecordTarget(
                    path=path,
                    kind=RecordFileKind.RUN,
                    identity=path,
                )
                for path in record.run_paths
            ),
        )
    }
    manifest = build_manifest(root, experiment_id=record.experiment_id, targets=targets)
    atomic_write_text(
        root / "manifest.yaml",
        dump_yaml(manifest_to_yaml_view(manifest), schema_name="manifest"),
        durability=durability,
    )
    validate_manifest(root, manifest)


def source_file_hashes(
    source_files: Collection[Path],
    *,
    path_root: Path | None,
) -> tuple[ResolvedFileHash, ...]:
    return tuple(
        hash_file(path, relative_to=path_root)
        for path in source_files
        if path.exists() and path.is_file()
    )


__all__ = (
    "ExecutionSnapshot",
    "ExperimentStart",
    "FileRecordSession",
    "FileRecorder",
    "PartialRunSnapshot",
    "RecordSession",
    "Recorder",
    "RecoveredStaging",
    "RunPhase",
    "STAGING_MANIFEST_PATH",
    "STAGING_STATE_PATH",
    "STAGING_VERSION",
    "StagedCheckpoint",
    "StagedRun",
    "StagingHealth",
    "StagingInspection",
    "StagingManifest",
    "StagingState",
    "StagingStatus",
    "archive_staging",
    "discard_staging",
    "finalize_staging",
    "inspect_staging",
    "recover_staging",
    "source_file_hashes",
)
