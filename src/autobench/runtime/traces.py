from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, Field

from autobench.errors import ErrorRecord
from autobench.metrics.observations import (
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import Semantic
from autobench.protocol.traces import Trace
from autobench.records.artifacts import ArtifactRef
from autobench.runtime.context import RunContext, SpanKind, SpanRecord

TRACE_RECORD_VERSION = 1


class TraceEnvelope(BaseModel):
    trace_id: str
    name: str
    input: Any = None
    output: Any = None
    spans: tuple[SpanRecord, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    errors: tuple[ErrorRecord, ...] = ()
    raw_artifact: ArtifactRef | None = None


def attach_trace(ctx: RunContext, trace: TraceEnvelope) -> list[Observation]:
    existing_span_ids = {span.id for span in ctx.spans}
    for span in trace.spans:
        if span.id not in existing_span_ids:
            ctx.spans.append(span)
            existing_span_ids.add(span.id)

    if trace.raw_artifact is not None and trace.raw_artifact.id not in {
        artifact.id for artifact in ctx.artifacts
    }:
        ctx.artifacts.append(trace.raw_artifact)

    for error in trace.errors:
        ctx.errors.append(error)

    observations = trace_to_observations(
        trace,
        case_id=ctx.case.id,
        variant_id=ctx.variant.id,
        id_prefix=f"trace_{len(ctx.observations) + 1}",
    )
    ctx.observations.extend(observations)
    return observations


def trace_to_observations(
    trace: TraceEnvelope,
    *,
    case_id: str | None = None,
    variant_id: str | None = None,
    id_prefix: str = "trace",
) -> list[Observation]:
    observations: list[Observation] = []
    for span in trace.spans:
        observations.extend(
            _span_usage_observations(
                span,
                trace_id=trace.trace_id,
                case_id=case_id,
                variant_id=variant_id,
                id_prefix=f"{id_prefix}_{len(observations) + 1}",
            )
        )
        if span.error is not None:
            observations.append(
                _trace_event(
                    f"{id_prefix}_{len(observations) + 1}",
                    name="span_error",
                    value=span.error.message,
                    trace_id=trace.trace_id,
                    span=span,
                    case_id=case_id,
                    variant_id=variant_id,
                )
            )
    for error in trace.errors:
        observations.append(
            Observation(
                id=f"{id_prefix}_{len(observations) + 1}",
                name="trace_error",
                kind=ObservationKind.EVENT,
                value=error.message,
                role=ObservationRole.DIAGNOSTIC,
                source=ObservationSource.IMPORTED,
                tags={"trace_id": trace.trace_id, "error_type": error.error_type},
                case_id=case_id,
                variant_id=variant_id,
            )
        )
    return observations


def _span_usage_observations(
    span: SpanRecord,
    *,
    trace_id: str,
    case_id: str | None,
    variant_id: str | None,
    id_prefix: str,
) -> list[Observation]:
    observations: list[Observation] = []
    if span.kind == SpanKind.LLM or str(span.kind) == SpanKind.LLM.value:
        semantic_keys = {
            "input_tokens": Semantic.LLM_TOKENS_INPUT,
            "prompt_tokens": Semantic.LLM_TOKENS_INPUT,
            "output_tokens": Semantic.LLM_TOKENS_OUTPUT,
            "completion_tokens": Semantic.LLM_TOKENS_OUTPUT,
            "total_tokens": Semantic.LLM_TOKENS_TOTAL,
            "requests": "llm.requests",
        }
        for usage_key, semantic_type in semantic_keys.items():
            if usage_key not in span.usage:
                continue
            observations.append(
                Observation(
                    id=f"{id_prefix}_{len(observations) + 1}",
                    name=f"{span.name}.{usage_key}",
                    kind=ObservationKind.METRIC,
                    semantic_type=semantic_type,
                    value=span.usage[usage_key],
                    role=ObservationRole.DIAGNOSTIC,
                    span_id=span.id,
                    source=ObservationSource.IMPORTED,
                    tags={"trace_id": trace_id, "span_kind": str(span.kind)},
                    case_id=case_id,
                    variant_id=variant_id,
                )
            )
        for attribute_key, semantic_type in (
            ("model", Semantic.LLM_MODEL_NAME),
            ("model_name", Semantic.LLM_MODEL_NAME),
            ("provider", Semantic.LLM_PROVIDER),
        ):
            if attribute_key not in span.attributes:
                continue
            observations.append(
                Observation(
                    id=f"{id_prefix}_{len(observations) + 1}",
                    name=f"{span.name}.{attribute_key}",
                    kind=ObservationKind.FACTOR,
                    semantic_type=semantic_type,
                    value=span.attributes[attribute_key],
                    span_id=span.id,
                    source=ObservationSource.IMPORTED,
                    tags={"trace_id": trace_id, "span_kind": str(span.kind)},
                    case_id=case_id,
                    variant_id=variant_id,
                )
            )
    if span.duration_seconds is not None:
        observations.append(
            Observation(
                id=f"{id_prefix}_{len(observations) + 1}",
                name=f"{span.name}.duration",
                kind=ObservationKind.METRIC,
                semantic_type=Semantic.TIME_LATENCY,
                value=span.duration_seconds,
                unit="s",
                role=ObservationRole.DIAGNOSTIC,
                span_id=span.id,
                source=ObservationSource.IMPORTED,
                tags={"trace_id": trace_id, "span_kind": str(span.kind)},
                case_id=case_id,
                variant_id=variant_id,
            )
        )
    return observations


def _trace_event(
    observation_id: str,
    *,
    name: str,
    value: Any,
    trace_id: str,
    span: SpanRecord,
    case_id: str | None,
    variant_id: str | None,
) -> Observation:
    return Observation(
        id=observation_id,
        name=name,
        kind=ObservationKind.EVENT,
        value=value,
        role=ObservationRole.DIAGNOSTIC,
        span_id=span.id,
        source=ObservationSource.IMPORTED,
        tags={"trace_id": trace_id, "span_kind": str(span.kind)},
        case_id=case_id,
        variant_id=variant_id,
    )


def trace_to_yaml_view(
    trace: Trace,
    *,
    extensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = trace.model_dump(mode="json")
    if extensions:
        payload["extensions"] = extensions
    return {
        "record": {"type": "trace", "version": TRACE_RECORD_VERSION},
        "trace": payload,
    }


def trace_payload_from_yaml_view(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    record = raw.get("record")
    payload = (
        raw.get("trace") if isinstance(record, dict) and record.get("type") == "trace" else raw
    )
    if not isinstance(payload, dict):
        raise ValueError("trace must be a mapping")

    trace_payload = dict(payload)
    extensions = trace_payload.pop("extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError("trace.extensions must be a mapping")
    extensions = dict(extensions)
    known_fields = set(Trace.model_fields)
    for key in tuple(trace_payload):
        if key not in known_fields:
            extensions[key] = trace_payload.pop(key)
    return trace_payload, extensions


def trace_yaml_schema() -> dict[str, Any]:
    trace_schema = Trace.model_json_schema(ref_template="#/$defs/{model}")
    definitions = trace_schema.pop("$defs", {})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Autobench ABP Trace YAML",
        "type": "object",
        "required": ["record", "trace"],
        "properties": {
            "record": {
                "type": "object",
                "required": ["type", "version"],
                "properties": {
                    "type": {"const": "trace"},
                    "version": {"const": TRACE_RECORD_VERSION},
                },
                "additionalProperties": False,
            },
            "trace": trace_schema,
        },
        "$defs": definitions,
        "additionalProperties": False,
    }


__all__ = (
    "TRACE_RECORD_VERSION",
    "TraceEnvelope",
    "attach_trace",
    "trace_payload_from_yaml_view",
    "trace_to_observations",
    "trace_to_yaml_view",
    "trace_yaml_schema",
)
