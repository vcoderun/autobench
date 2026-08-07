from __future__ import annotations as _annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autobench.errors import AutobenchError
from autobench.records.models import ExperimentRecord, RunRecord
from autobench.records.replay import load_experiment_record, load_run_record

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

OTLPResourceValue = str | bool | int | float


class OTLPExportError(AutobenchError):
    """Raised when immutable Autobench evidence cannot be exported through OTLP."""


class OTLPSettings(BaseModel):
    """Vendor-neutral HTTP/protobuf exporter settings kept outside benchmark specs."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    endpoint: str | None = Field(default=None, min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=10, gt=0)
    certificate_file: Path | None = None
    service_name: str = Field(default="autobench", min_length=1)
    service_namespace: str | None = Field(default=None, min_length=1)
    resource_attributes: dict[str, OTLPResourceValue] = Field(default_factory=dict)
    include_captured_content: bool = False

    @field_validator("endpoint", "service_name", "service_namespace")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("OTLP endpoint and service identifiers must not be blank")
        return value

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() for key in headers):
            raise ValueError("OTLP header names must not be blank")
        return headers

    @field_validator("resource_attributes")
    @classmethod
    def validate_resource_attributes(
        cls,
        attributes: dict[str, OTLPResourceValue],
    ) -> dict[str, OTLPResourceValue]:
        for key in attributes:
            if not key.strip():
                raise ValueError("OTLP resource attribute names must not be blank")
        return attributes


class OTLPExportResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    benchmark_id: str
    record_version: int
    run_count: int = Field(ge=0)
    trace_count: int = Field(ge=0)
    abp_span_count: int = Field(ge=0)
    exported_span_count: int = Field(ge=0)
    partial_run_count: int = Field(ge=0)
    partial_trace_count: int = Field(ge=0)
    endpoint: str


def export_record_otlp(
    record_dir: Path,
    *,
    settings: OTLPSettings | None = None,
    exporter: SpanExporter | None = None,
) -> OTLPExportResult:
    """Load one immutable record directory and export its evidence through OTLP."""

    experiment = load_experiment_record(record_dir)
    runs = tuple(
        load_run_record(record_dir / run_path, root_dir=record_dir)
        for run_path in experiment.run_paths
    )
    return export_otlp(
        experiment,
        runs,
        settings=settings,
        exporter=exporter,
    )


def export_otlp(
    experiment: ExperimentRecord,
    runs: Sequence[RunRecord],
    *,
    settings: OTLPSettings | None = None,
    exporter: SpanExporter | None = None,
) -> OTLPExportResult:
    """Export validated immutable record models without mutating their ABP evidence."""

    if len(runs) != experiment.run_count:
        raise OTLPExportError(
            f"Experiment declares {experiment.run_count} runs but {len(runs)} were supplied."
        )
    run_ids: set[str] = set()
    for run in runs:
        if run.run_id in run_ids:
            raise OTLPExportError(f"Duplicate run supplied for OTLP export: {run.run_id}")
        run_ids.add(run.run_id)
        if run.experiment_id != experiment.experiment_id:
            raise OTLPExportError(f"Run {run.run_id!r} belongs to another experiment.")
        if run.benchmark_id != experiment.benchmark_id:
            raise OTLPExportError(f"Run {run.run_id!r} belongs to another benchmark.")
        if run.correlation != experiment.correlation:
            raise OTLPExportError(f"Run {run.run_id!r} has inconsistent execution correlation.")

    try:
        from autobench.exporters._otel import export_records
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("opentelemetry"):
            raise
        raise OTLPExportError(
            "OTLP export requires the 'autobench[otlp]' optional dependency."
        ) from exc
    try:
        return export_records(
            experiment,
            tuple(runs),
            OTLPSettings() if settings is None else settings,
            exporter,
        )
    except OTLPExportError:
        raise
    except Exception as exc:
        raise OTLPExportError(f"Could not map Autobench evidence to OTLP spans: {exc}") from exc


__all__ = (
    "OTLPExportError",
    "OTLPExportResult",
    "OTLPResourceValue",
    "OTLPSettings",
    "export_otlp",
    "export_record_otlp",
)
