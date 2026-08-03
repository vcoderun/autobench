from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autobench.metrics.observations import Observation, ObservationKind, ObservationSource
from autobench.metrics.projection import observation_priority, project_observations
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry


class ObservationQuery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    observations: list[Observation] = Field(default_factory=list)
    registry: SemanticRegistry = Field(
        default_factory=lambda: DEFAULT_SEMANTIC_REGISTRY.model_copy(deep=True)
    )

    def all(
        self,
        *,
        projected: bool = False,
    ) -> list[Observation]:
        if not projected:
            return list(self.observations)
        return [
            item.observation
            for item in project_observations(self.observations, registry=self.registry)
        ]

    def exact(
        self,
        semantic_type: str,
        *,
        kind: ObservationKind | tuple[ObservationKind, ...] | None = None,
        source: ObservationSource | str | None = None,
        projected: bool = True,
    ) -> list[Observation]:
        normalized = self.registry.normalize(semantic_type)
        return [
            observation
            for observation in self._iter(projected=projected)
            if observation.normalized_semantic_type(self.registry) == normalized
            and _kind_matches(observation, kind)
            and _source_matches(observation, source)
        ]

    def related(
        self,
        semantic_type: str,
        *,
        kind: ObservationKind | tuple[ObservationKind, ...] | None = None,
        source: ObservationSource | str | None = None,
        projected: bool = True,
    ) -> list[Observation]:
        return [
            observation
            for observation in self._iter(projected=projected)
            if self.registry.is_a(observation.semantic_type, semantic_type)
            and _kind_matches(observation, kind)
            and _source_matches(observation, source)
        ]

    def first_exact(
        self,
        semantic_type: str,
        *,
        kind: ObservationKind | tuple[ObservationKind, ...] | None = None,
        source: ObservationSource | str | None = None,
        projected: bool = True,
    ) -> Observation | None:
        matches = self.exact(
            semantic_type,
            kind=kind,
            source=source,
            projected=projected,
        )
        return _preferred(matches)

    def first_related(
        self,
        semantic_type: str,
        *,
        kind: ObservationKind | tuple[ObservationKind, ...] | None = None,
        source: ObservationSource | str | None = None,
        projected: bool = True,
    ) -> Observation | None:
        matches = self.related(
            semantic_type,
            kind=kind,
            source=source,
            projected=projected,
        )
        return _preferred(matches)

    def values(
        self,
        semantic_type: str,
        *,
        related: bool = False,
        kind: ObservationKind | tuple[ObservationKind, ...] | None = None,
        source: ObservationSource | str | None = None,
        projected: bool = True,
    ) -> list[Any]:
        selector = self.related if related else self.exact
        return [
            observation.value
            for observation in selector(
                semantic_type,
                kind=kind,
                source=source,
                projected=projected,
            )
        ]

    def _iter(self, *, projected: bool) -> list[Observation]:
        return self.all(projected=projected)


def _kind_matches(
    observation: Observation,
    kind: ObservationKind | tuple[ObservationKind, ...] | None,
) -> bool:
    if kind is None:
        return True
    if isinstance(kind, tuple):
        return observation.kind in kind
    return observation.kind is kind


def _source_matches(
    observation: Observation,
    source: ObservationSource | str | None,
) -> bool:
    if source is None:
        return True
    return observation.source == source


def _preferred(observations: list[Observation]) -> Observation | None:
    if not observations:
        return None
    return min(
        enumerate(observations),
        key=lambda item: (*observation_priority(item[1]), item[0]),
    )[1]


__all__ = ("ObservationQuery",)
