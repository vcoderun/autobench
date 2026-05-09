from __future__ import annotations as _annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry, SemanticType


class ObservationKind(StrEnum):
    METRIC = "metric"
    FACTOR = "factor"
    ARTIFACT = "artifact"
    EVENT = "event"


class Direction(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"
    NONE = "none"


class ObservationRole(StrEnum):
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    DIAGNOSTIC = "diagnostic"
    METADATA = "metadata"


class ObservationSource(StrEnum):
    SCORE = "score"
    DERIVED = "derived"
    TASK_OBSERVATION = "task_observation"
    INSTRUMENTATION = "instrumentation"
    VARIANT = "variant"
    IMPORTED = "imported"


class Observation(BaseModel):
    id: str
    name: str
    kind: ObservationKind
    semantic_type: SemanticType | None = None
    value: Any
    unit: str | None = None
    direction: Direction | None = None
    role: ObservationRole | None = None
    span_id: str | None = None
    source: ObservationSource | str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    case_id: str | None = None
    variant_id: str | None = None

    @model_validator(mode="after")
    def _validate_kind_rules(self) -> Observation:
        if self.kind is ObservationKind.FACTOR and self.direction is not None:
            raise ValueError("factor observations cannot declare direction")
        if (
            self.kind in {ObservationKind.ARTIFACT, ObservationKind.EVENT}
            and self.direction is not None
        ):
            raise ValueError("artifact and event observations cannot declare direction")
        return self

    def normalized_semantic_type(self, registry: SemanticRegistry | None = None) -> str | None:
        active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
        return active_registry.normalize(self.semantic_type)


def filter_observations(
    observations: list[Observation],
    *,
    name: str | None = None,
    kind: ObservationKind | None = None,
    role: ObservationRole | None = None,
    source: ObservationSource | str | None = None,
    semantic_type: str | None = None,
    parent_semantic_type: str | None = None,
    span_id: str | None = None,
    registry: SemanticRegistry | None = None,
) -> list[Observation]:
    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    filtered: list[Observation] = []

    for observation in observations:
        if name is not None and observation.name != name:
            continue
        if kind is not None and observation.kind is not kind:
            continue
        if role is not None and observation.role is not role:
            continue
        if source is not None and observation.source != source:
            continue
        if span_id is not None and observation.span_id != span_id:
            continue
        if semantic_type is not None and observation.normalized_semantic_type(
            active_registry
        ) != active_registry.normalize(semantic_type):
            continue
        if parent_semantic_type is not None and not active_registry.is_a(
            observation.semantic_type,
            parent_semantic_type,
        ):
            continue
        filtered.append(observation)

    return filtered


__all__ = (
    "Direction",
    "Observation",
    "ObservationKind",
    "ObservationRole",
    "ObservationSource",
    "filter_observations",
)
