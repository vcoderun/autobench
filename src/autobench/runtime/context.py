from __future__ import annotations as _annotations

import asyncio
import json
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from contextvars import Token
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any
from weakref import WeakValueDictionary

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from autobench._version import __version__
from autobench.data.datasets import Case
from autobench.data.variants import Variant
from autobench.errors import ErrorRecord
from autobench.evaluation.measurement import Measurement
from autobench.metrics.mappings import SourceSnapshot
from autobench.metrics.observations import (
    Direction,
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import Semantic, SemanticType
from autobench.protocol.capture import CapturePolicy, CaptureResult, CaptureSession
from autobench.protocol.collector import Emitter, LocalCollector
from autobench.protocol.context import ActiveContext, attach_context, get_context, reset_context
from autobench.protocol.ids import SpanId
from autobench.protocol.signals import (
    AbstractionLayer,
    CaptureLevel,
    CaptureMechanism,
    EndReason,
    ExecutionRef,
    InstrumentationScope,
    KnownSpanKind,
    SpanStatus,
)
from autobench.protocol.traces import Trace
from autobench.protocol.values import EvidenceRef, ReferenceKind, ReferenceStore, SerializedValue
from autobench.records.artifacts import (
    ArtifactOverflow,
    ArtifactRef,
    ArtifactSink,
    ArtifactSinkRequiredError,
    ArtifactSource,
    ArtifactState,
    SymlinkPolicy,
)
from autobench.runtime.awaitables import settle_task
from autobench.runtime.lifecycle import RunPhase
from autobench.tracking import (
    AssetCandidate,
    AssetProvenance,
    AssetRepresentation,
    AssetSensitivity,
    AssetUse,
    AssetVersion,
    RegisteredAsset,
    TrackedAsset,
    TrackingRegistry,
    canonical_asset_hash,
    track,
)

_RUN_CONTEXTS: WeakValueDictionary[str, RunContext] = WeakValueDictionary()
_ARTIFACT_TRANSFER_SETTLE_SECONDS = 5.0


class DurationMetricSpec(BaseModel):
    name: str = "duration"
    semantic_type: SemanticType = Semantic.TIME_LATENCY
    unit: str = "s"
    direction: Direction | None = Direction.MINIMIZE
    role: ObservationRole | None = ObservationRole.DIAGNOSTIC
    tags: dict[str, Any] = Field(default_factory=dict)


class SpanKind(StrEnum):
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    PARSER = "parser"
    WORKFLOW = "workflow"
    OPTIMIZATION = "optimization"
    CANDIDATE = "candidate"
    EVALUATION = "evaluation"
    REFLECTION = "reflection"
    CUSTOM = "custom"


class SpanRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    kind: SpanKind | str = SpanKind.CUSTOM
    parent_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    input: Any = None
    output: Any = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    error: ErrorRecord | None = None
    tags: dict[str, Any] = Field(default_factory=dict)


class CheckResult(BaseModel):
    name: str
    passed: bool
    observation: Observation
    reason: str | None = None

    def skip(self, message: str) -> dict[str, Any]:
        return {
            "skipped": True,
            "check": self.name,
            "passed": self.passed,
            "reason": message,
        }


class MeasurementRecord(BaseModel):
    metrics: tuple[Observation, ...]
    samples_artifact: ArtifactRef | None = None


class ContextEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    observations: tuple[Observation, ...]
    spans: tuple[SpanRecord, ...]
    artifacts: tuple[ArtifactRef, ...]
    errors: tuple[ErrorRecord, ...]
    asset_versions: tuple[AssetVersion, ...]
    asset_uses: tuple[AssetUse, ...]
    source_snapshots: tuple[SourceSnapshot, ...]
    extensions: dict[str, JsonValue] = Field(default_factory=dict)
    trace: Trace
    signal_sequence_watermark: int


CheckpointHandler = Callable[
    [str, RunPhase, Any, ContextEvidence],
    Awaitable[None],
]


class RunContext:
    def __init__(
        self,
        *,
        benchmark_id: str,
        case: Case,
        variant: Variant,
        run_id: str = "run_1",
        experiment_id: str = "experiment_1",
        capture_policy: CapturePolicy | None = None,
    ) -> None:
        self.benchmark_id = benchmark_id
        self.case = case
        self.variant = variant
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.observations: list[Observation] = []
        self.spans: list[SpanRecord] = []
        self.artifacts: list[ArtifactRef] = []
        self.errors: list[ErrorRecord] = []
        self.asset_versions: list[AssetVersion] = []
        self.asset_uses: list[AssetUse] = []
        self.source_snapshots: list[SourceSnapshot] = []
        self.extensions: dict[str, JsonValue] = {}
        self._evidence_lock = RLock()
        self._collector = LocalCollector()
        self._capture = CaptureSession(capture_policy)
        self._execution = ExecutionRef(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            run_id=run_id,
            case_id=case.id,
            variant_id=variant.id,
        )
        self._emitter = Emitter(
            self._collector,
            InstrumentationScope(
                instrumentor_name="autobench.manual",
                instrumentor_version=__version__,
                package_name="autobench",
                package_version=__version__,
                mechanism=CaptureMechanism.MANUAL,
                layer=AbstractionLayer.APPLICATION,
            ),
            execution=self._execution,
        )
        root = self._emitter.start_span(
            "benchmark.run",
            kind=KnownSpanKind.TASK,
            attributes={
                "benchmark_id": benchmark_id,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "case_id": case.id,
                "variant_id": variant.id,
            },
            capture=CaptureLevel.METADATA,
        )
        self._root_span_id = root.span_id
        self._legacy_to_abp: dict[str, SpanId] = {}
        self._abp_to_legacy: dict[SpanId, str] = {}
        self._span_emitters: dict[str, Emitter] = {}
        self._span_started_monotonic: dict[str, int] = {}
        self._ended_spans: set[str] = set()
        self._span_error_refs: dict[str, list[EvidenceRef]] = {}
        self._error_refs: list[EvidenceRef] = []
        self._trace: Trace | None = None
        self._observation_index = 0
        self._span_index = 0
        self._artifact_index = 0
        self._asset_version_keys: set[tuple[str, str]] = set()
        self._asset_use_keys: set[tuple[str, str, str, str | None]] = set()
        self._phase = RunPhase.RESOLVING
        self._checkpoint_output: Any = None
        self._checkpoint_handler: CheckpointHandler | None = None
        self._artifact_sink: ArtifactSink | None = None
        _RUN_CONTEXTS[self._emitter.trace_id] = self

    @property
    def trace(self) -> Trace:
        with self._evidence_lock:
            if self._trace is not None:
                return self._trace
            return self._collector.snapshot(self._emitter.trace_id)

    @property
    def finalized(self) -> bool:
        return self._trace is not None

    @property
    def phase(self) -> RunPhase:
        with self._evidence_lock:
            return self._phase

    @property
    def checkpoint_output(self) -> Any:
        with self._evidence_lock:
            return self._checkpoint_output

    @property
    def reference_store(self) -> ReferenceStore:
        return self._capture.store

    @property
    def capture_policy(self) -> CapturePolicy:
        return self._capture.policy

    @property
    def active_context(self) -> ActiveContext:
        return ActiveContext(
            collector=self._collector,
            trace_id=self._emitter.trace_id,
            current_span_id=self._root_span_id,
            execution=self._execution,
            capture_policy=self._capture.policy,
        )

    @property
    def active_span_id(self) -> str | None:
        """Return the legacy span identity selected by the active ABP context."""

        active = get_context()
        if (
            active is None
            or active.collector is not self._collector
            or active.trace_id != self._emitter.trace_id
            or active.current_span_id is None
        ):
            return None
        with self._evidence_lock:
            return self._abp_to_legacy.get(active.current_span_id)

    def factor(self, name: str) -> Any:
        for factor in self.variant.factors:
            if factor.name == name:
                return factor.value
        raise KeyError(f"Unknown variant factor: {name}")

    def retain_source_snapshot(self, snapshot: SourceSnapshot) -> SourceSnapshot:
        with self._evidence_lock:
            self.source_snapshots.append(snapshot)
        return snapshot

    def set_extension(self, name: str, value: JsonValue) -> None:
        """Store one integration-owned, JSON-safe run projection."""

        if not name.strip():
            raise ValueError("Extension names must not be empty.")
        with self._evidence_lock:
            self.extensions[name] = deepcopy(value)

    def snapshot_evidence(self) -> ContextEvidence:
        with self._evidence_lock:
            trace = self.trace
            return ContextEvidence(
                observations=tuple(item.model_copy(deep=True) for item in self.observations),
                spans=tuple(item.model_copy(deep=True) for item in self.spans),
                artifacts=tuple(item.model_copy(deep=True) for item in self.artifacts),
                errors=tuple(item.model_copy(deep=True) for item in self.errors),
                asset_versions=tuple(item.model_copy(deep=True) for item in self.asset_versions),
                asset_uses=tuple(item.model_copy(deep=True) for item in self.asset_uses),
                source_snapshots=tuple(
                    item.model_copy(deep=True) for item in self.source_snapshots
                ),
                extensions=deepcopy(self.extensions),
                trace=trace.model_copy(deep=True),
                signal_sequence_watermark=max(
                    (signal.sequence for signal in trace.signals),
                    default=0,
                ),
            )

    def bind_checkpoint(self, handler: CheckpointHandler) -> None:
        with self._evidence_lock:
            if self._checkpoint_handler is not None:
                raise RuntimeError("A checkpoint handler is already bound to this run context.")
            self._checkpoint_handler = handler

    def bind_artifact_sink(self, sink: ArtifactSink) -> None:
        with self._evidence_lock:
            if self._artifact_sink is not None:
                raise RuntimeError("An artifact sink is already bound to this run context.")
            self._artifact_sink = sink

    def set_phase(self, phase: RunPhase) -> None:
        with self._evidence_lock:
            self._phase = phase

    def retain_task_output(self, output: Any) -> None:
        with self._evidence_lock:
            self._checkpoint_output = output

    async def checkpoint(self, name: str) -> None:
        active_name = name.strip()
        if not active_name:
            raise ValueError("Checkpoint names must not be empty.")
        if active_name.startswith("autobench."):
            raise ValueError("Checkpoint names beginning with 'autobench.' are reserved.")
        with self._evidence_lock:
            handler = self._checkpoint_handler
            if handler is None:
                raise RuntimeError(
                    "Checkpoints require durable recording; run the benchmark with a recorder."
                )
            phase = self._phase
            output = self._checkpoint_output
            evidence = self.snapshot_evidence()
        await handler(active_name, phase, output, evidence)

    def span(
        self,
        name: str,
        *,
        kind: SpanKind | str = SpanKind.CUSTOM,
        input: Any = None,
        attributes: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        duration_metric: DurationMetricSpec | dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        instrumentation_scope: InstrumentationScope | None = None,
        parent_span_id: str | None = None,
    ) -> Span:
        metric_spec = None
        if duration_metric is not None:
            metric_spec = (
                duration_metric
                if isinstance(duration_metric, DurationMetricSpec)
                else DurationMetricSpec.model_validate(duration_metric)
            )
        return Span(
            context=self,
            name=name,
            kind=kind,
            input=input,
            attributes=attributes or {},
            usage=usage or {},
            duration_metric=metric_spec,
            tags=tags or {},
            instrumentation_scope=instrumentation_scope,
            parent_span_id=parent_span_id,
        )

    def metric(
        self,
        name: str,
        value: Any,
        *,
        semantic_type: SemanticType | None = None,
        unit: str | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        source: ObservationSource = ObservationSource.TASK_OBSERVATION,
    ) -> Observation:
        return self._append_observation(
            name=name,
            kind=ObservationKind.METRIC,
            value=value,
            semantic_type=semantic_type,
            unit=unit,
            direction=direction,
            role=role,
            span_id=span_id,
            tags=tags,
            source=source,
        )

    def factor_observation(
        self,
        name: str,
        value: Any,
        *,
        semantic_type: SemanticType | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        source: ObservationSource = ObservationSource.TASK_OBSERVATION,
    ) -> Observation:
        return self._append_observation(
            name=name,
            kind=ObservationKind.FACTOR,
            value=value,
            semantic_type=semantic_type,
            span_id=span_id,
            tags=tags,
            source=source,
        )

    def event(
        self,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        source: ObservationSource = ObservationSource.TASK_OBSERVATION,
    ) -> Observation:
        return self._append_observation(
            name=name,
            kind=ObservationKind.EVENT,
            value=value,
            semantic_type=semantic_type,
            span_id=span_id,
            tags=tags,
            source=source,
        )

    def diagnostic(
        self,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        source: ObservationSource = ObservationSource.TASK_OBSERVATION,
    ) -> Observation:
        return self._append_observation(
            name=name,
            kind=ObservationKind.EVENT,
            value=value,
            semantic_type=semantic_type,
            role=ObservationRole.DIAGNOSTIC,
            span_id=span_id,
            tags=tags,
            source=source,
        )

    def outcome(
        self,
        success: bool,
        *,
        name: str = "success",
        semantic_type: SemanticType = Semantic.RESULT_SUCCESS,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self.metric(
            name,
            success,
            semantic_type=semantic_type,
            role=ObservationRole.OBJECTIVE,
            span_id=span_id,
            tags=tags,
        )

    def skip_reason(
        self,
        reason: str,
        *,
        name: str = "skip_reason",
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self.diagnostic(name, reason, span_id=span_id, tags=tags)

    def check(
        self,
        name: str,
        passed: bool,
        *,
        reason: str | None = None,
        semantic_type: SemanticType = Semantic.QUALITY_CORRECTNESS,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> CheckResult:
        observation_tags = dict(tags or {})
        if reason is not None:
            observation_tags["reason"] = reason
        observation = self.metric(
            name,
            passed,
            semantic_type=semantic_type,
            role=ObservationRole.CONSTRAINT,
            span_id=span_id,
            tags=observation_tags,
        )
        return CheckResult(name=name, passed=passed, observation=observation, reason=reason)

    def metrics(
        self,
        namespace: str,
        values: dict[str, Any],
        *,
        semantic_types: dict[str, SemanticType] | None = None,
        units: dict[str, str] | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> list[Observation]:
        observations: list[Observation] = []
        for key, value in values.items():
            observations.append(
                self.metric(
                    f"{namespace}.{key}",
                    value,
                    semantic_type=None if semantic_types is None else semantic_types.get(key),
                    unit=None if units is None else units.get(key),
                    direction=direction,
                    role=role,
                    span_id=span_id,
                    tags=tags,
                )
            )
        return observations

    def record_measurement(
        self,
        name: str,
        measurement: Measurement,
        *,
        semantic_type: SemanticType = Semantic.TIME_LATENCY,
        unit: str = "ms",
        direction: Direction | None = Direction.MINIMIZE,
        role: ObservationRole | None = ObservationRole.DIAGNOSTIC,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        include_samples_artifact: bool = True,
    ) -> MeasurementRecord:
        values = {
            "median_ms": measurement.median_ms,
            "p95_ms": measurement.p95_ms,
            "mean_ms": measurement.mean_ms,
            "min_ms": measurement.min_ms,
            "max_ms": measurement.max_ms,
            "stddev_ms": measurement.standard_deviation_ms,
            "noise_pct": measurement.range_noise_pct,
            "repetitions": measurement.repetition_count,
            "timed_out": measurement.timed_out,
        }
        measurement_semantics = {
            "median_ms": semantic_type,
            "p95_ms": f"{semantic_type}.p95",
            "mean_ms": f"{semantic_type}.mean",
            "min_ms": f"{semantic_type}.min",
            "max_ms": f"{semantic_type}.max",
        }
        metrics = tuple(
            self.metrics(
                name,
                values,
                semantic_types=measurement_semantics,
                units={
                    "median_ms": unit,
                    "p95_ms": unit,
                    "mean_ms": unit,
                    "min_ms": unit,
                    "max_ms": unit,
                    "stddev_ms": unit,
                    "noise_pct": "%",
                },
                direction=direction,
                role=role,
                span_id=span_id,
                tags=tags,
            )
        )
        samples_artifact = None
        if include_samples_artifact:
            samples_artifact = self.artifact(
                f"{name}.samples_ms",
                measurement.samples_ms,
                media_type="application/x.autobench.samples+yaml",
                span_id=span_id,
                tags=tags,
            )
        return MeasurementRecord(metrics=metrics, samples_artifact=samples_artifact)

    def error(
        self,
        error: BaseException | ErrorRecord | str,
        *,
        span_id: str | None = None,
    ) -> ErrorRecord:
        with self._evidence_lock:
            if isinstance(error, ErrorRecord):
                record = error.model_copy(update={"span_id": error.span_id or span_id})
            elif isinstance(error, BaseException):
                record = ErrorRecord.from_exception(error, span_id=span_id)
            else:
                record = ErrorRecord(error_type="Error", message=error, span_id=span_id)

            self.errors.append(record)
            error_reference = EvidenceRef(
                kind=ReferenceKind.ERROR,
                id=f"error_{len(self.errors)}",
                media_type="application/x.autobench.error+json",
            )
            self._error_refs.append(error_reference)
            abp_span_id = self._abp_span_id(span_id)
            emitter = self._emitter_for_legacy_span(span_id)
            captured = self._capture_value(
                record.model_dump(mode="json"),
                semantic_type=Semantic.ERROR_EXCEPTION,
                path=f"errors.{error_reference.id}",
                span_id=abp_span_id,
                level=CaptureLevel.REDACTED,
            )
            emitter.event(
                abp_span_id,
                record.error_type,
                Semantic.ERROR_EXCEPTION,
                body=None if captured.reference is not None else captured.value,
                reference=captured.reference,
                attributes={"error_id": error_reference.id},
            )
            emitter.reference(
                error_reference,
                span_id=abp_span_id,
                semantic_type=Semantic.ERROR_EXCEPTION,
                name=record.error_type,
            )
            if span_id is not None:
                span_record = self._span_by_id(span_id)
                if span_record is not None:
                    span_record.error = record
                    self._span_error_refs.setdefault(span_id, []).append(error_reference)
            return record

    def artifact(
        self,
        name: str,
        value: Any,
        *,
        media_type: str | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        with self._evidence_lock:
            artifact = ArtifactRef(
                id=self._next_artifact_id(),
                name=name,
                value=value,
                media_type=media_type,
                span_id=span_id,
                tags=tags or {},
            )
            self.artifacts.append(artifact)
            if span_id is not None:
                span_record = self._span_by_id(span_id)
                if span_record is not None:
                    span_record.artifacts.append(artifact.id)
            abp_span_id = self._abp_span_id(span_id)
            emitter = self._emitter_for_legacy_span(span_id)
            captured = self._capture_value(
                value,
                semantic_type=Semantic.ARTIFACT_CONTENT,
                path=f"artifacts.{name}",
                span_id=abp_span_id,
                level=CaptureLevel.FULL,
                media_type=media_type,
            )
            reference = captured.reference
            if reference is None and not captured.omitted:
                if isinstance(captured.value, str):
                    content = captured.value.encode()
                    active_media_type = media_type or "text/plain"
                else:
                    content = json.dumps(
                        captured.value,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                    active_media_type = media_type or "application/json"
                reference = self._capture.store.add_artifact(content, media_type=active_media_type)
            if reference is not None:
                emitter.reference(
                    reference,
                    span_id=abp_span_id,
                    semantic_type=Semantic.ARTIFACT_CONTENT,
                    name=name,
                    attributes={"artifact_id": artifact.id},
                )
            self._append_observation(
                name=name,
                kind=ObservationKind.ARTIFACT,
                value=artifact.id,
                span_id=span_id,
                tags=tags,
            )
            return artifact

    def artifact_file(
        self,
        name: str,
        source: Path,
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        symlinks: SymlinkPolicy = SymlinkPolicy.REJECT,
        filename: str | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        sink = self._require_artifact_sink()
        active_max_bytes = self._artifact_max_bytes(max_bytes)
        with self._evidence_lock:
            artifact_id = self._next_artifact_id()
        active_tags = dict(tags or {})
        try:
            artifact = sink.prepare_file(
                run_id=self.run_id,
                artifact_id=artifact_id,
                name=name,
                source=source,
                media_type=media_type,
                max_bytes=active_max_bytes,
                overflow=overflow,
                symlinks=symlinks,
                filename=filename,
                span_id=span_id,
                tags=active_tags,
            )
        except BaseException:
            self._retain_interrupted_artifact(sink, artifact_id, span_id=span_id)
            raise
        return self._retain_prepared_artifact(artifact)

    async def artifact_file_async(
        self,
        name: str,
        source: Path,
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        symlinks: SymlinkPolicy = SymlinkPolicy.REJECT,
        filename: str | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        sink = self._require_artifact_sink()
        active_max_bytes = self._artifact_max_bytes(max_bytes)
        with self._evidence_lock:
            artifact_id = self._next_artifact_id()
        active_tags = dict(tags or {})
        transfer = asyncio.create_task(
            sink.prepare_file_async(
                run_id=self.run_id,
                artifact_id=artifact_id,
                name=name,
                source=source,
                media_type=media_type,
                max_bytes=active_max_bytes,
                overflow=overflow,
                symlinks=symlinks,
                filename=filename,
                span_id=span_id,
                tags=active_tags,
            )
        )
        try:
            artifact = await asyncio.shield(transfer)
        except asyncio.CancelledError as cancellation:
            transfer_error = await settle_task(
                transfer,
                timeout_seconds=_ARTIFACT_TRANSFER_SETTLE_SECONDS,
                cancel_on_timeout=False,
                description=f"Artifact {artifact_id} transfer",
            )
            retained = self._retain_interrupted_artifact(
                sink,
                artifact_id,
                span_id=span_id,
            )
            if not retained:
                self._retain_prepared_artifact(
                    ArtifactRef(
                        id=artifact_id,
                        name=name,
                        media_type=media_type,
                        span_id=span_id,
                        tags=active_tags,
                        source=ArtifactSource.FILE,
                        state=ArtifactState.PARTIAL,
                        filename=filename or source.name,
                    )
                )
            if transfer_error is not None:
                cancellation.add_note(f"artifact transfer did not settle: {transfer_error}")
            raise
        except BaseException:
            self._retain_interrupted_artifact(sink, artifact_id, span_id=span_id)
            raise
        return self._retain_prepared_artifact(artifact)

    def artifact_stream(
        self,
        name: str,
        source: Iterable[bytes],
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        filename: str | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        sink = self._require_artifact_sink()
        active_max_bytes = self._artifact_max_bytes(max_bytes)
        with self._evidence_lock:
            artifact_id = self._next_artifact_id()
        try:
            artifact = sink.prepare_stream(
                run_id=self.run_id,
                artifact_id=artifact_id,
                name=name,
                source=source,
                media_type=media_type,
                max_bytes=active_max_bytes,
                overflow=overflow,
                filename=filename,
                span_id=span_id,
                tags=dict(tags or {}),
            )
        except BaseException:
            self._retain_interrupted_artifact(sink, artifact_id, span_id=span_id)
            raise
        return self._retain_prepared_artifact(artifact)

    async def artifact_stream_async(
        self,
        name: str,
        source: AsyncIterable[bytes],
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        filename: str | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        sink = self._require_artifact_sink()
        active_max_bytes = self._artifact_max_bytes(max_bytes)
        with self._evidence_lock:
            artifact_id = self._next_artifact_id()
        try:
            artifact = await sink.prepare_stream_async(
                run_id=self.run_id,
                artifact_id=artifact_id,
                name=name,
                source=source,
                media_type=media_type,
                max_bytes=active_max_bytes,
                overflow=overflow,
                filename=filename,
                span_id=span_id,
                tags=dict(tags or {}),
            )
        except BaseException:
            self._retain_interrupted_artifact(sink, artifact_id, span_id=span_id)
            raise
        return self._retain_prepared_artifact(artifact)

    def _require_artifact_sink(self) -> ArtifactSink:
        with self._evidence_lock:
            if self._artifact_sink is None:
                raise ArtifactSinkRequiredError(
                    "File and stream artifacts require an active durable recorder."
                )
            return self._artifact_sink

    def _artifact_max_bytes(self, max_bytes: int | None) -> int:
        active = self.capture_policy.max_artifact_bytes if max_bytes is None else max_bytes
        if active < 1:
            raise ValueError("max_bytes must be at least 1")
        return active

    def _retain_interrupted_artifact(
        self,
        sink: ArtifactSink,
        artifact_id: str,
        *,
        span_id: str | None,
    ) -> bool:
        artifact = sink.prepared_artifact(run_id=self.run_id, artifact_id=artifact_id)
        if artifact is None:
            return False
        self._retain_prepared_artifact(
            artifact.model_copy(update={"span_id": artifact.span_id or span_id})
        )
        return True

    def _retain_prepared_artifact(self, artifact: ArtifactRef) -> ArtifactRef:
        with self._evidence_lock:
            self.artifacts.append(artifact)
            if artifact.span_id is not None:
                span_record = self._span_by_id(artifact.span_id)
                if span_record is not None:
                    span_record.artifacts.append(artifact.id)
            abp_span_id = self._abp_span_id(artifact.span_id)
            self._emitter_for_legacy_span(artifact.span_id).reference(
                EvidenceRef(
                    kind=ReferenceKind.ARTIFACT,
                    id=artifact.id,
                    media_type=artifact.media_type,
                ),
                span_id=abp_span_id,
                semantic_type=Semantic.ARTIFACT_CONTENT,
                name=artifact.name,
                attributes={
                    "artifact_id": artifact.id,
                    "artifact_state": artifact.state.value,
                    "artifact_sha256": artifact.sha256,
                    "artifact_byte_count": artifact.byte_count,
                },
            )
            self._append_observation(
                name=artifact.name,
                kind=ObservationKind.ARTIFACT,
                value=artifact.id,
                span_id=artifact.span_id,
                tags=artifact.tags,
            )
            return artifact

    def attach_tracked_asset(
        self,
        target: Any,
        *,
        registry: TrackingRegistry | None = None,
        span_id: str | None = None,
    ) -> AssetVersion:
        active_registry = registry or track
        asset_version = active_registry.asset_version_of(target)
        asset = active_registry.asset_of(target)
        with self._evidence_lock:
            self._attach_asset_reference(asset, asset_version, span_id=span_id)
            source_locator = asset.id
            use_key = (asset.id, asset_version.version, source_locator, span_id)
            if use_key not in self._asset_use_keys:
                self.asset_uses.append(
                    AssetUse(
                        asset_id=asset.id,
                        version=asset_version.version,
                        representation=AssetRepresentation.DEFINITION,
                        source_locator=source_locator,
                        span_id=span_id,
                        provenance=AssetProvenance(
                            system="autobench",
                            key=asset.id,
                            instrumentor="autobench.tracking",
                        ),
                    )
                )
                self._asset_use_keys.add(use_key)
        return asset_version

    def attach_discovered_asset(self, registered: RegisteredAsset) -> AssetUse:
        use = registered.use
        with self._evidence_lock:
            self._attach_asset_reference(registered.asset, registered.version, span_id=use.span_id)
            use_key = (use.asset_id, use.version, use.source_locator, use.span_id)
            if use_key not in self._asset_use_keys:
                self.asset_uses.append(use)
                self._asset_use_keys.add(use_key)
        return use

    def prepare_discovered_asset(
        self,
        candidate: AssetCandidate,
        *,
        span_id: str | None,
    ) -> AssetCandidate:
        fingerprint = canonical_asset_hash(candidate.canonical_content)
        level = self.capture_policy.level_for_asset(
            candidate.semantic_type,
        )
        if candidate.sensitivity is AssetSensitivity.SENSITIVE and level is CaptureLevel.METADATA:
            level = CaptureLevel.HASH
        elif candidate.sensitivity is AssetSensitivity.PUBLIC and level is CaptureLevel.METADATA:
            level = CaptureLevel.FULL
        captured = self._capture_value(
            candidate.canonical_content,
            semantic_type=candidate.semantic_type,
            path=f"assets.{candidate.source_locator}",
            span_id=self._abp_span_id(span_id),
            level=level,
        )
        content: SerializedValue
        if captured.omitted:
            content = {"omitted": True, "sha256": fingerprint}
        elif captured.reference is not None:
            content = captured.reference.model_dump(mode="json")
        else:
            content = captured.value
        return candidate.model_copy(
            update={
                "canonical_content": content,
                "content_fingerprint": fingerprint,
            }
        )

    def _attach_asset_reference(
        self,
        asset: TrackedAsset,
        asset_version: AssetVersion,
        *,
        span_id: str | None,
    ) -> None:
        asset_key = (asset_version.asset_id, asset_version.version)
        if asset_key not in self._asset_version_keys:
            self.asset_versions.append(asset_version)
            self._asset_version_keys.add(asset_key)
            reference_kind = ReferenceKind.ASSET
            if asset.kind == "prompt":
                reference_kind = ReferenceKind.PROMPT
            elif asset.kind == "tool":
                reference_kind = ReferenceKind.TOOL
            elif asset.kind in {"pydantic_model", "dataclass", "typed_class", "type"}:
                reference_kind = ReferenceKind.OUTPUT_SCHEMA
            abp_span_id = self._abp_span_id(span_id)
            emitter = self._emitter_for_legacy_span(span_id)
            captured = self._capture_value(
                asset_version,
                semantic_type=asset.semantic_type,
                path=f"assets.{asset.id}",
                span_id=abp_span_id,
                asset_version=asset_version,
                reference_kind=reference_kind,
            )
            if captured.reference is not None:
                emitter.reference(
                    captured.reference,
                    span_id=abp_span_id,
                    semantic_type=asset.semantic_type,
                    name=asset.name,
                    attributes={"kind": asset.kind},
                )

    def _append_observation(
        self,
        *,
        name: str,
        kind: ObservationKind,
        value: Any,
        semantic_type: SemanticType | None = None,
        unit: str | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        source: ObservationSource = ObservationSource.TASK_OBSERVATION,
    ) -> Observation:
        observation = Observation(
            id=self._next_observation_id(),
            name=name,
            kind=kind,
            semantic_type=semantic_type,
            value=value,
            unit=unit,
            direction=direction,
            role=role,
            span_id=span_id,
            source=source,
            tags=tags or {},
            case_id=self.case.id,
            variant_id=self.variant.id,
        )
        return self.record_observation(observation)

    def record_observation(self, observation: Observation) -> Observation:
        with self._evidence_lock:
            self.observations.append(observation)
            if observation.span_id is not None:
                span_record = self._span_by_id(observation.span_id)
                if span_record is not None:
                    span_record.observations.append(observation.id)
            self._emit_observation(observation)
            return observation

    def _start_span(
        self,
        name: str,
        *,
        kind: SpanKind | str = SpanKind.CUSTOM,
        input: Any = None,
        attributes: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        tags: dict[str, Any],
        instrumentation_scope: InstrumentationScope | None = None,
        parent_span_id: str | None = None,
    ) -> tuple[SpanRecord, int]:
        with self._evidence_lock:
            if self.finalized:
                raise RuntimeError("RunContext is finalized.")
            if parent_span_id is not None:
                parent_record = self._span_by_id(parent_span_id)
                if parent_record is None:
                    raise ValueError(f"Unknown parent span: {parent_span_id}")
                if parent_record.ended_at is not None:
                    raise ValueError(f"Parent span has ended: {parent_span_id}")
                parent_abp_id = self._legacy_to_abp[parent_span_id]
            else:
                active = get_context()
                parent_abp_id = self._root_span_id
                if (
                    active is not None
                    and active.collector is self._collector
                    and active.trace_id == self._emitter.trace_id
                    and active.current_span_id is not None
                ):
                    parent_abp_id = active.current_span_id
            parent_id = self._abp_to_legacy.get(parent_abp_id)
            captured_attributes = self._capture_mapping(
                attributes or {},
                path=f"spans.{name}.attributes",
                span_id=parent_abp_id,
            )
            captured_tags = self._capture_mapping(
                tags,
                path=f"spans.{name}.tags",
                span_id=parent_abp_id,
            )
            if captured_tags:
                captured_attributes["tags"] = captured_tags
            emitter = self._emitter
            if instrumentation_scope is not None:
                emitter = Emitter(
                    self._collector,
                    instrumentation_scope,
                    trace_id=self._emitter.trace_id,
                    execution=self._execution,
                )
            start = emitter.start_span(
                name,
                parent_span_id=parent_abp_id,
                kind=str(kind),
                attributes=captured_attributes,
                capture=self._capture.policy.default_level,
            )
            span_record = SpanRecord(
                id=self._next_span_id(),
                name=name,
                kind=kind,
                parent_id=parent_id,
                started_at=start.emitted_at,
                input=input,
                attributes=attributes or {},
                usage=usage or {},
                tags=tags,
            )
            self.spans.append(span_record)
            self._legacy_to_abp[span_record.id] = start.span_id
            self._abp_to_legacy[start.span_id] = span_record.id
            self._span_emitters[span_record.id] = emitter
            self._span_started_monotonic[span_record.id] = start.monotonic_ns
            if input is not None:
                captured_input = self._capture_value(
                    input,
                    semantic_type=Semantic.OPERATION_INPUT,
                    path=f"spans.{name}.input",
                    span_id=start.span_id,
                )
                emitter.event(
                    start.span_id,
                    "input",
                    Semantic.OPERATION_INPUT,
                    body=None if captured_input.reference is not None else captured_input.value,
                    reference=captured_input.reference,
                )
            return span_record, start.monotonic_ns

    def _finish_span(
        self,
        span_record: SpanRecord,
        *,
        started_at: int,
        duration_metric: DurationMetricSpec | None,
        error: BaseException | None = None,
        status: SpanStatus | None = None,
        reason: EndReason | None = None,
        partial: bool | None = None,
    ) -> None:
        with self._evidence_lock:
            if span_record.id in self._ended_spans:
                return
            emitter = self._span_emitters[span_record.id]
            abp_span_id = self._legacy_to_abp[span_record.id]
            error_refs = tuple(self._span_error_refs.get(span_record.id, ()))
            span_status = SpanStatus.OK if status is None else status
            end_reason = EndReason.COMPLETED if reason is None else reason
            is_partial = False if partial is None else partial
            if error is not None or error_refs:
                span_status = SpanStatus.ERROR
                if reason is None or reason is EndReason.COMPLETED:
                    end_reason = EndReason.FAILED
            elif span_status is SpanStatus.ERROR and reason is None:
                end_reason = EndReason.FAILED
            if isinstance(error, asyncio.CancelledError):
                end_reason = EndReason.CANCELLED
                is_partial = True
            elif isinstance(error, TimeoutError):
                end_reason = EndReason.TIMEOUT
                is_partial = True
            captured_output = self._capture_value(
                span_record.output,
                semantic_type=Semantic.OPERATION_OUTPUT,
                path=f"spans.{span_record.name}.output",
                span_id=abp_span_id,
            )
            end = emitter.end_span(
                span_id=abp_span_id,
                attributes=self._capture_mapping(
                    span_record.attributes,
                    path=f"spans.{span_record.name}.attributes",
                    span_id=abp_span_id,
                ),
                output=None if captured_output.reference is not None else captured_output.value,
                output_reference=captured_output.reference,
                status=span_status,
                reason=end_reason,
                errors=error_refs,
                partial=is_partial,
                usage=self._capture_mapping(
                    span_record.usage,
                    path=f"spans.{span_record.name}.usage",
                    span_id=abp_span_id,
                ),
            )
            self._ended_spans.add(span_record.id)
            duration_seconds = max(0, end.monotonic_ns - started_at) / 1_000_000_000
            span_record.ended_at = end.emitted_at
            span_record.duration_seconds = duration_seconds

            if duration_metric is not None:
                self.metric(
                    duration_metric.name,
                    duration_seconds,
                    semantic_type=duration_metric.semantic_type,
                    unit=duration_metric.unit,
                    direction=duration_metric.direction,
                    role=duration_metric.role,
                    span_id=span_record.id,
                    tags=duration_metric.tags,
                )

    def finalize(
        self,
        *,
        status: SpanStatus = SpanStatus.OK,
        reason: EndReason = EndReason.COMPLETED,
        partial: bool = False,
        output: Any = None,
    ) -> Trace:
        with self._evidence_lock:
            if self._trace is not None:
                return self._trace
            for span_record in self.spans:
                if span_record.id in self._ended_spans:
                    continue
                self._finish_span(
                    span_record,
                    started_at=self._span_started_monotonic[span_record.id],
                    duration_metric=None,
                    reason=EndReason.ABANDONED,
                    partial=True,
                )
            if self._error_refs and status is SpanStatus.OK:
                status = SpanStatus.ERROR
                reason = EndReason.FAILED
            captured_output = self._capture_value(
                output,
                semantic_type=Semantic.OPERATION_OUTPUT,
                path="benchmark.run.output",
                span_id=self._root_span_id,
            )
            self._emitter.end_span(
                self._root_span_id,
                output=None if captured_output.reference is not None else captured_output.value,
                output_reference=captured_output.reference,
                status=status,
                reason=reason,
                errors=tuple(self._error_refs),
                partial=partial,
            )
            self._trace = self._collector.finish(
                self._emitter.trace_id,
                error=status is SpanStatus.ERROR,
            )
            return self._trace

    def _emit_observation(self, observation: Observation) -> None:
        emitter = self._emitter_for_legacy_span(observation.span_id)
        abp_span_id = self._abp_span_id(observation.span_id)
        semantic_type = observation.semantic_type
        if semantic_type is None:
            if observation.kind is ObservationKind.FACTOR:
                semantic_type = Semantic.FACTOR_VALUE
            elif observation.role is ObservationRole.DIAGNOSTIC:
                semantic_type = Semantic.DIAGNOSTIC_EVENT
            else:
                semantic_type = Semantic.EVENT_OCCURRENCE
        captured = self._capture_value(
            observation.value,
            semantic_type=semantic_type,
            path=f"observations.{observation.name}",
            span_id=abp_span_id,
        )
        attributes: dict[str, SerializedValue] = {
            "observation_id": observation.id,
            "kind": observation.kind.value,
            "source": ("unspecified" if observation.source is None else str(observation.source)),
        }
        if observation.span_id is not None:
            attributes["legacy_span_id"] = observation.span_id
        if observation.tags:
            attributes["tags"] = self._capture_mapping(
                observation.tags,
                path=f"observations.{observation.name}.tags",
                span_id=abp_span_id,
            )
        if (
            observation.kind is ObservationKind.METRIC
            and not captured.omitted
            and isinstance(captured.value, (bool, int, float))
        ):
            emitter.measurement(
                abp_span_id,
                observation.name,
                semantic_type,
                captured.value,
                unit=observation.unit,
                direction=observation.direction,
                role=observation.role,
                attributes=attributes,
            )
            return
        if captured.omitted:
            attributes["capture_omitted"] = True
        emitter.event(
            abp_span_id,
            observation.name,
            semantic_type,
            body=None if captured.reference is not None else captured.value,
            reference=captured.reference,
            attributes=attributes,
        )

    def _capture_value(
        self,
        value: Any,
        *,
        semantic_type: SemanticType | None,
        path: str,
        span_id: SpanId,
        level: CaptureLevel | None = None,
        asset_version: AssetVersion | None = None,
        reference_kind: ReferenceKind | None = None,
        media_type: str | None = None,
    ) -> CaptureResult:
        captured = self._capture.capture(
            value,
            semantic_type=semantic_type,
            path=path,
            level=level,
            asset_version=asset_version,
            reference_kind=reference_kind,
            media_type=media_type,
        )
        for diagnostic in captured.diagnostics:
            self._emitter_for_abp_span(span_id).diagnostic(
                diagnostic.code,
                diagnostic.message,
                severity=diagnostic.severity,
                span_id=span_id,
                path=diagnostic.path,
                semantic_type=diagnostic.semantic_type,
                details=diagnostic.details,
            )
        return captured

    def _capture_mapping(
        self,
        values: dict[str, Any],
        *,
        path: str,
        span_id: SpanId,
    ) -> dict[str, SerializedValue]:
        captured_values: dict[str, SerializedValue] = {}
        for name, value in values.items():
            captured = self._capture_value(
                value,
                semantic_type=name,
                path=f"{path}.{name}",
                span_id=span_id,
            )
            if captured.omitted:
                continue
            if captured.reference is None:
                captured_values[name] = captured.value
            else:
                captured_values[name] = captured.reference.model_dump(mode="json")
        return captured_values

    def _abp_span_id(self, legacy_span_id: str | None) -> SpanId:
        if legacy_span_id is not None:
            mapped = self._legacy_to_abp.get(legacy_span_id)
            if mapped is not None:
                return mapped
            return self._root_span_id
        active = get_context()
        if (
            active is not None
            and active.collector is self._collector
            and active.trace_id == self._emitter.trace_id
            and active.current_span_id is not None
        ):
            return active.current_span_id
        return self._root_span_id

    def _span_by_id(self, span_id: str) -> SpanRecord | None:
        for span in self.spans:
            if span.id == span_id:
                return span
        return None

    def _emitter_for_legacy_span(self, span_id: str | None) -> Emitter:
        if span_id is None:
            return self._emitter
        return self._span_emitters.get(span_id, self._emitter)

    def _emitter_for_abp_span(self, span_id: SpanId) -> Emitter:
        legacy_span_id = self._abp_to_legacy.get(span_id)
        return self._emitter_for_legacy_span(legacy_span_id)

    def _next_observation_id(self) -> str:
        self._observation_index += 1
        return f"obs_{self._observation_index}"

    def _next_span_id(self) -> str:
        self._span_index += 1
        return f"span_{self._span_index}"

    def _next_artifact_id(self) -> str:
        self._artifact_index += 1
        return f"artifact_{self._artifact_index}"


class Span(AbstractContextManager["Span"]):
    def __init__(
        self,
        *,
        context: RunContext,
        name: str,
        kind: SpanKind | str,
        input: Any,
        attributes: dict[str, Any],
        usage: dict[str, Any],
        duration_metric: DurationMetricSpec | None,
        tags: dict[str, Any],
        instrumentation_scope: InstrumentationScope | None,
        parent_span_id: str | None,
    ) -> None:
        self._context = context
        self._name = name
        self._kind = kind
        self._input = input
        self._attributes = attributes
        self._usage = usage
        self._duration_metric = duration_metric
        self._tags = tags
        self._instrumentation_scope = instrumentation_scope
        self._parent_span_id = parent_span_id
        self._record: SpanRecord | None = None
        self._started_at: int | None = None
        self._active_token: Token[ActiveContext | None] | None = None

    @property
    def id(self) -> str:
        if self._record is None:
            raise RuntimeError("Span has not started.")
        return self._record.id

    @property
    def record(self) -> SpanRecord:
        if self._record is None:
            raise RuntimeError("Span has not started.")
        return self._record

    def __enter__(self) -> Span:
        self.start()
        self.resume()
        return self

    def start(self) -> Span:
        """Start this span without changing the active context stack."""

        if self._record is not None:
            raise RuntimeError("Span has already started.")
        self._record, self._started_at = self._context._start_span(
            self._name,
            kind=self._kind,
            input=self._input,
            attributes=self._attributes,
            usage=self._usage,
            tags=self._tags,
            instrumentation_scope=self._instrumentation_scope,
            parent_span_id=self._parent_span_id,
        )
        return self

    def resume(self) -> None:
        if self._record is None:
            raise RuntimeError("Span has not started.")
        if self._active_token is not None:
            return
        active = get_context()
        abp_span_id = self._context._legacy_to_abp[self._record.id]
        if (
            active is not None
            and active.collector is self._context._collector
            and active.trace_id == self._context._emitter.trace_id
        ):
            protocol_context = active.with_span(abp_span_id)
        else:
            protocol_context = self._context.active_context.with_span(abp_span_id)
        self._active_token = attach_context(protocol_context)

    def suspend(self) -> None:
        if self._active_token is None:
            return
        reset_context(self._active_token)
        self._active_token = None

    def finish(
        self,
        *,
        error: BaseException | None = None,
        status: SpanStatus | None = None,
        reason: EndReason | None = None,
        partial: bool | None = None,
    ) -> None:
        if error is not None:
            self._context.error(error, span_id=self.id)
        try:
            if self._started_at is not None:
                self._context._finish_span(
                    self.record,
                    started_at=self._started_at,
                    duration_metric=self._duration_metric,
                    error=error,
                    status=status,
                    reason=reason,
                    partial=partial,
                )
        finally:
            self.suspend()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.finish(error=exc_value)
        return None

    def __iter__(self) -> Iterator[SpanRecord]:
        yield self.record

    def set_output(self, value: Any) -> None:
        self.record.output = value

    def set_attribute(self, name: str, value: Any) -> None:
        self.record.attributes[name] = value

    def set_usage(self, name: str, value: Any) -> None:
        self.record.usage[name] = value

    def metric(
        self,
        name: str,
        value: Any,
        *,
        semantic_type: SemanticType | None = None,
        unit: str | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._context.metric(
            name,
            value,
            semantic_type=semantic_type,
            unit=unit,
            direction=direction,
            role=role,
            span_id=self.id,
            tags=tags,
        )

    def artifact_file(
        self,
        name: str,
        source: Path,
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        symlinks: SymlinkPolicy = SymlinkPolicy.REJECT,
        filename: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return self._context.artifact_file(
            name,
            source,
            media_type=media_type,
            max_bytes=max_bytes,
            overflow=overflow,
            symlinks=symlinks,
            filename=filename,
            span_id=self.id,
            tags=tags,
        )

    async def artifact_file_async(
        self,
        name: str,
        source: Path,
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        symlinks: SymlinkPolicy = SymlinkPolicy.REJECT,
        filename: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return await self._context.artifact_file_async(
            name,
            source,
            media_type=media_type,
            max_bytes=max_bytes,
            overflow=overflow,
            symlinks=symlinks,
            filename=filename,
            span_id=self.id,
            tags=tags,
        )

    def artifact_stream(
        self,
        name: str,
        source: Iterable[bytes],
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        filename: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return self._context.artifact_stream(
            name,
            source,
            media_type=media_type,
            max_bytes=max_bytes,
            overflow=overflow,
            filename=filename,
            span_id=self.id,
            tags=tags,
        )

    async def artifact_stream_async(
        self,
        name: str,
        source: AsyncIterable[bytes],
        *,
        media_type: str | None = None,
        max_bytes: int | None = None,
        overflow: ArtifactOverflow = ArtifactOverflow.FAIL,
        filename: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return await self._context.artifact_stream_async(
            name,
            source,
            media_type=media_type,
            max_bytes=max_bytes,
            overflow=overflow,
            filename=filename,
            span_id=self.id,
            tags=tags,
        )

    def factor(
        self,
        name: str,
        value: Any,
        *,
        semantic_type: SemanticType | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._context.factor_observation(
            name,
            value,
            semantic_type=semantic_type,
            span_id=self.id,
            tags=tags,
        )

    def event(
        self,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._context.event(
            name,
            value,
            semantic_type=semantic_type,
            span_id=self.id,
            tags=tags,
        )

    def diagnostic(
        self,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._context.diagnostic(
            name,
            value,
            semantic_type=semantic_type,
            span_id=self.id,
            tags=tags,
        )

    def outcome(
        self,
        success: bool,
        *,
        name: str = "success",
        semantic_type: SemanticType = Semantic.RESULT_SUCCESS,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._context.outcome(
            success,
            name=name,
            semantic_type=semantic_type,
            span_id=self.id,
            tags=tags,
        )

    def skip_reason(
        self,
        reason: str,
        *,
        name: str = "skip_reason",
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._context.skip_reason(reason, name=name, span_id=self.id, tags=tags)

    def check(
        self,
        name: str,
        passed: bool,
        *,
        reason: str | None = None,
        semantic_type: SemanticType = Semantic.QUALITY_CORRECTNESS,
        tags: dict[str, Any] | None = None,
    ) -> CheckResult:
        return self._context.check(
            name,
            passed,
            reason=reason,
            semantic_type=semantic_type,
            span_id=self.id,
            tags=tags,
        )

    def metrics(
        self,
        namespace: str,
        values: dict[str, Any],
        *,
        semantic_types: dict[str, SemanticType] | None = None,
        units: dict[str, str] | None = None,
        direction: Direction | None = None,
        role: ObservationRole | None = None,
        tags: dict[str, Any] | None = None,
    ) -> list[Observation]:
        return self._context.metrics(
            namespace,
            values,
            semantic_types=semantic_types,
            units=units,
            direction=direction,
            role=role,
            span_id=self.id,
            tags=tags,
        )

    def record_measurement(
        self,
        name: str,
        measurement: Measurement,
        *,
        semantic_type: SemanticType = Semantic.TIME_LATENCY,
        unit: str = "ms",
        direction: Direction | None = Direction.MINIMIZE,
        role: ObservationRole | None = ObservationRole.DIAGNOSTIC,
        tags: dict[str, Any] | None = None,
        include_samples_artifact: bool = True,
    ) -> MeasurementRecord:
        return self._context.record_measurement(
            name,
            measurement,
            semantic_type=semantic_type,
            unit=unit,
            direction=direction,
            role=role,
            span_id=self.id,
            tags=tags,
            include_samples_artifact=include_samples_artifact,
        )

    def error(self, error: BaseException | ErrorRecord | str) -> ErrorRecord:
        return self._context.error(error, span_id=self.id)

    def artifact(
        self,
        name: str,
        value: Any,
        *,
        media_type: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return self._context.artifact(
            name,
            value,
            media_type=media_type,
            span_id=self.id,
            tags=tags,
        )


def active_run_context() -> RunContext | None:
    active = get_context()
    if active is None:
        return None
    return _RUN_CONTEXTS.get(active.trace_id)


__all__ = (
    "CheckResult",
    "ContextEvidence",
    "DurationMetricSpec",
    "MeasurementRecord",
    "RunContext",
    "Span",
    "SpanKind",
    "SpanRecord",
)
