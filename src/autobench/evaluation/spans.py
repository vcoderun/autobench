from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, Field

from autobench.metrics.observations import Observation
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
from autobench.runtime.context import SpanRecord


class SpanSelector(BaseModel):
    kind: str | None = None
    name: str | None = None
    tag: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None
    semantic_type: str | None = None


def select_spans(
    selector: SpanSelector | None,
    *,
    spans: list[SpanRecord],
    observations: list[Observation],
    registry: SemanticRegistry | None = None,
) -> list[SpanRecord]:
    if selector is None:
        return list(spans)

    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    selected: list[SpanRecord] = []
    for span in spans:
        if selector.kind is not None and str(span.kind) != selector.kind:
            continue
        if selector.name is not None and span.name != selector.name:
            continue
        if selector.path is not None and _span_path(span, spans=spans) != selector.path:
            continue
        if selector.tag and not _contains_tag_values(span.tags, selector.tag):
            continue
        if selector.semantic_type is not None and not _span_has_semantic(
            span,
            observations=observations,
            semantic_type=selector.semantic_type,
            registry=active_registry,
        ):
            continue
        selected.append(span)
    return selected


def _span_path(span: SpanRecord, *, spans: list[SpanRecord]) -> str:
    spans_by_id = {candidate.id: candidate for candidate in spans}
    names = [span.name]
    parent_id = span.parent_id
    while parent_id is not None and parent_id in spans_by_id:
        parent = spans_by_id[parent_id]
        names.append(parent.name)
        parent_id = parent.parent_id
    return ".".join(reversed(names))


def _contains_tag_values(tags: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(name in tags and tags[name] == value for name, value in expected.items())


def _span_has_semantic(
    span: SpanRecord,
    *,
    observations: list[Observation],
    semantic_type: str,
    registry: SemanticRegistry,
) -> bool:
    for observation in observations:
        if observation.span_id != span.id:
            continue
        if registry.is_a(observation.semantic_type, semantic_type):
            return True
    return False


__all__ = (
    "SpanSelector",
    "select_spans",
)
