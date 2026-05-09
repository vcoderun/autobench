from __future__ import annotations as _annotations

from typing import Final

from pydantic import BaseModel, Field

from autobench.metrics.observations import Observation, ObservationSource
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry

SOURCE_PRIORITY: Final[dict[str | None, int]] = {
    ObservationSource.SCORE.value: 0,
    ObservationSource.DERIVED.value: 1,
    ObservationSource.TASK_OBSERVATION.value: 2,
    ObservationSource.INSTRUMENTATION.value: 3,
    ObservationSource.VARIANT.value: 4,
    ObservationSource.IMPORTED.value: 5,
    None: 6,
}


class ProjectionKey(BaseModel):
    semantic_type: str | None
    name: str
    role: str | None
    case_id: str | None
    variant_id: str | None


class ProjectedObservation(BaseModel):
    key: ProjectionKey
    observation: Observation
    candidates: list[Observation] = Field(default_factory=list)
    ambiguous: bool = False


def source_priority(source: ObservationSource | str | None) -> int:
    normalized = source.value if isinstance(source, ObservationSource) else source
    return SOURCE_PRIORITY.get(normalized, SOURCE_PRIORITY[None])


def observation_projection_key(
    observation: Observation,
    *,
    registry: SemanticRegistry | None = None,
) -> ProjectionKey:
    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    role = observation.role.value if observation.role is not None else None
    return ProjectionKey(
        semantic_type=observation.normalized_semantic_type(active_registry),
        name=observation.name,
        role=role,
        case_id=observation.case_id,
        variant_id=observation.variant_id,
    )


def project_observations(
    observations: list[Observation],
    *,
    registry: SemanticRegistry | None = None,
) -> list[ProjectedObservation]:
    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    grouped: dict[
        tuple[str | None, str, str | None, str | None, str | None], list[Observation]
    ] = {}
    order: list[tuple[str | None, str, str | None, str | None, str | None]] = []

    for observation in observations:
        key = observation_projection_key(observation, registry=active_registry)
        key_tuple = (
            key.semantic_type,
            key.name,
            key.role,
            key.case_id,
            key.variant_id,
        )
        if key_tuple not in grouped:
            grouped[key_tuple] = []
            order.append(key_tuple)
        grouped[key_tuple].append(observation)

    projected: list[ProjectedObservation] = []
    for key_tuple in order:
        candidates = grouped[key_tuple]
        ordered_candidates = sorted(
            enumerate(candidates),
            key=lambda item: (source_priority(item[1].source), item[0]),
        )
        best_priority = source_priority(ordered_candidates[0][1].source)
        best_candidates = [
            observation
            for _, observation in ordered_candidates
            if source_priority(observation.source) == best_priority
        ]
        projected.append(
            ProjectedObservation(
                key=ProjectionKey(
                    semantic_type=key_tuple[0],
                    name=key_tuple[1],
                    role=key_tuple[2],
                    case_id=key_tuple[3],
                    variant_id=key_tuple[4],
                ),
                observation=ordered_candidates[0][1],
                candidates=candidates,
                ambiguous=len(best_candidates) > 1,
            )
        )

    return projected


__all__ = (
    "ProjectedObservation",
    "ProjectionKey",
    "SOURCE_PRIORITY",
    "observation_projection_key",
    "project_observations",
    "source_priority",
)
