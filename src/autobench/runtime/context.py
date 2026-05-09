from __future__ import annotations as _annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from time import perf_counter
from types import TracebackType
from typing import Any

from pydantic import BaseModel, Field

from autobench.data.datasets import Case
from autobench.data.variants import Variant
from autobench.errors import ErrorRecord
from autobench.evaluation.measurement import Measurement
from autobench.metrics.observations import (
    Direction,
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import Semantic, SemanticType
from autobench.records.artifacts import ArtifactRef
from autobench.tracking import AssetVersion, TrackingRegistry, track


class DurationMetricSpec(BaseModel):
    name: str = "duration"
    semantic_type: SemanticType = Semantic.TIME_LATENCY
    unit: str = "s"
    direction: Direction | None = Direction.MINIMIZE
    role: ObservationRole | None = ObservationRole.DIAGNOSTIC
    tags: dict[str, Any] = Field(default_factory=dict)


class SpanRecord(BaseModel):
    id: str
    name: str
    parent_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float | None = None
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


class RunContext:
    def __init__(
        self,
        *,
        benchmark_id: str,
        case: Case,
        variant: Variant,
        run_id: str = "run_1",
        experiment_id: str = "experiment_1",
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
        self._span_stack: list[str] = []
        self._observation_index = 0
        self._span_index = 0
        self._artifact_index = 0
        self._asset_version_keys: set[tuple[str, str]] = set()

    def factor(self, name: str) -> Any:
        for factor in self.variant.factors:
            if factor.name == name:
                return factor.value
        raise KeyError(f"Unknown variant factor: {name}")

    def span(
        self,
        name: str,
        *,
        duration_metric: DurationMetricSpec | dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
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
            duration_metric=metric_spec,
            tags=tags or {},
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
        )

    def factor_observation(
        self,
        name: str,
        value: Any,
        *,
        semantic_type: SemanticType | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._append_observation(
            name=name,
            kind=ObservationKind.FACTOR,
            value=value,
            semantic_type=semantic_type,
            span_id=span_id,
            tags=tags,
        )

    def event(
        self,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._append_observation(
            name=name,
            kind=ObservationKind.EVENT,
            value=value,
            semantic_type=semantic_type,
            span_id=span_id,
            tags=tags,
        )

    def diagnostic(
        self,
        name: str,
        value: Any = True,
        *,
        semantic_type: SemanticType | None = None,
        span_id: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation:
        return self._append_observation(
            name=name,
            kind=ObservationKind.EVENT,
            value=value,
            semantic_type=semantic_type,
            role=ObservationRole.DIAGNOSTIC,
            span_id=span_id,
            tags=tags,
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
        metrics = tuple(
            self.metrics(
                name,
                values,
                semantic_types={
                    "median_ms": semantic_type,
                    "p95_ms": semantic_type,
                    "mean_ms": semantic_type,
                    "min_ms": semantic_type,
                    "max_ms": semantic_type,
                },
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
        if isinstance(error, ErrorRecord):
            record = error.model_copy(update={"span_id": error.span_id or span_id})
        elif isinstance(error, BaseException):
            record = ErrorRecord.from_exception(error, span_id=span_id)
        else:
            record = ErrorRecord(error_type="Error", message=error, span_id=span_id)

        self.errors.append(record)
        if span_id is not None:
            span_record = self._span_by_id(span_id)
            if span_record is not None:
                span_record.error = record
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
        self._append_observation(
            name=name,
            kind=ObservationKind.ARTIFACT,
            value=artifact.id,
            span_id=span_id,
            tags=tags,
        )
        return artifact

    def attach_tracked_asset(
        self,
        target: Any,
        *,
        registry: TrackingRegistry | None = None,
    ) -> AssetVersion:
        active_registry = registry or track
        asset_version = active_registry.asset_version_of(target)
        asset_key = (asset_version.asset_id, asset_version.version)
        if asset_key not in self._asset_version_keys:
            self.asset_versions.append(asset_version)
            self._asset_version_keys.add(asset_key)
        return asset_version

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
            source=ObservationSource.TASK_OBSERVATION,
            tags=tags or {},
            case_id=self.case.id,
            variant_id=self.variant.id,
        )
        self.observations.append(observation)
        if span_id is not None:
            span_record = self._span_by_id(span_id)
            if span_record is not None:
                span_record.observations.append(observation.id)
        return observation

    def _start_span(self, name: str, *, tags: dict[str, Any]) -> tuple[SpanRecord, float]:
        parent_id = self._span_stack[-1] if self._span_stack else None
        span_record = SpanRecord(
            id=self._next_span_id(),
            name=name,
            parent_id=parent_id,
            started_at=datetime.now(UTC),
            tags=tags,
        )
        self.spans.append(span_record)
        self._span_stack.append(span_record.id)
        return span_record, perf_counter()

    def _finish_span(
        self,
        span_record: SpanRecord,
        *,
        started_at: float,
        duration_metric: DurationMetricSpec | None,
    ) -> None:
        duration_seconds = perf_counter() - started_at
        span_record.ended_at = datetime.now(UTC)
        span_record.duration_seconds = duration_seconds
        if self._span_stack and self._span_stack[-1] == span_record.id:
            self._span_stack.pop()
        elif span_record.id in self._span_stack:
            self._span_stack.remove(span_record.id)

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

    def _span_by_id(self, span_id: str) -> SpanRecord | None:
        for span in self.spans:
            if span.id == span_id:
                return span
        return None

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
        duration_metric: DurationMetricSpec | None,
        tags: dict[str, Any],
    ) -> None:
        self._context = context
        self._name = name
        self._duration_metric = duration_metric
        self._tags = tags
        self._record: SpanRecord | None = None
        self._started_at: float | None = None

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
        self._record, self._started_at = self._context._start_span(
            self._name,
            tags=self._tags,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if exc_value is not None:
            self._context.error(exc_value, span_id=self.id)
        if self._started_at is not None:
            self._context._finish_span(
                self.record,
                started_at=self._started_at,
                duration_metric=self._duration_metric,
            )
        return None

    def __iter__(self) -> Iterator[SpanRecord]:
        yield self.record

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


__all__ = (
    "CheckResult",
    "DurationMetricSpec",
    "MeasurementRecord",
    "RunContext",
    "Span",
    "SpanRecord",
)
