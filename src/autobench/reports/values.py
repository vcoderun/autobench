from __future__ import annotations as _annotations

from collections.abc import Sequence
from math import prod
from statistics import median, pstdev
from typing import Any

from autobench.metrics.observations import Observation, ObservationKind
from autobench.metrics.projection import observation_priority
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry
from autobench.reports.models import AggregationFn
from autobench.runtime.models import RunResult


def metric_value(
    run: RunResult,
    semantic_type: str,
    *,
    registry: SemanticRegistry | None = None,
) -> Any | None:
    observation = metric_observation(run, semantic_type, registry=registry)
    return observation.value if observation is not None else None


def metric_observation(
    run: RunResult,
    semantic_type: str,
    *,
    registry: SemanticRegistry | None = None,
) -> Observation | None:
    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    normalized = active_registry.normalize(semantic_type)
    candidates = [
        observation
        for observation in run.task_result.observations
        if observation.kind is ObservationKind.METRIC
        and observation.normalized_semantic_type(active_registry) == normalized
    ]
    candidates.extend(
        score.to_observation(
            observation_id=f"{run.run_id}_report_score_{index}",
            case_id=run.case_id,
            variant_id=run.variant_id,
        )
        for index, score in enumerate(run.scores)
        if active_registry.normalize(score.semantic_type) == normalized
    )
    if not candidates:
        return None
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (*observation_priority(item[1]), item[0]),
    )
    return ordered[0][1]


def metric_metadata(
    runs: Sequence[RunResult],
    semantic_type: str,
    *,
    registry: SemanticRegistry,
) -> tuple[str | None, str | None, str | None]:
    observations = [
        observation
        for run in runs
        if (observation := metric_observation(run, semantic_type, registry=registry)) is not None
    ]
    directions = {item.direction.value for item in observations if item.direction is not None}
    units = {item.unit for item in observations if item.unit is not None}
    roles = {item.role.value for item in observations if item.role is not None}
    semantic_info = registry.types.get(registry.normalize(semantic_type) or semantic_type)
    if not units and semantic_info is not None and semantic_info.unit is not None:
        units.add(semantic_info.unit)
    return (
        next(iter(directions)) if len(directions) == 1 else None,
        next(iter(units)) if len(units) == 1 else None,
        next(iter(roles)) if len(roles) == 1 else None,
    )


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def aggregate_values(values: list[Any], fn: AggregationFn) -> Any | None:
    if not values:
        return None
    if fn == "count":
        return len(values)
    if fn == "ratio_true":
        return sum(1 for value in values if bool(value)) / len(values)

    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if not numeric_values:
        return None
    if fn == "mean":
        return sum(numeric_values) / len(numeric_values)
    if fn == "sum":
        return sum(numeric_values)
    if fn == "min":
        return min(numeric_values)
    if fn == "max":
        return max(numeric_values)
    if fn == "median":
        return median(numeric_values)
    if fn == "p95":
        return _percentile(numeric_values, 95.0)
    if fn == "stddev":
        return pstdev(numeric_values)
    if fn == "geomean":
        if any(value <= 0.0 for value in numeric_values):
            return None
        return prod(numeric_values) ** (1.0 / len(numeric_values))
    raise ValueError(f"Unsupported aggregation: {fn}")


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * weight
    )


__all__ = (
    "aggregate_values",
    "metric_metadata",
    "metric_observation",
    "metric_value",
    "numeric_value",
)
