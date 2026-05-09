from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, Field

from autobench.metrics.semantics import SemanticType


class FactorValue(BaseModel):
    name: str = Field(min_length=1)
    value: Any
    semantic_type: SemanticType | None = None
    optimize: bool = False


class Variant(BaseModel):
    id: str = Field(min_length=1)
    label: str | None = None
    factors: list[FactorValue] = Field(default_factory=list)


def normalize_variant_factors(
    raw_factors: list[FactorValue] | list[dict[str, Any]] | dict[str, Any] | None,
) -> list[FactorValue]:
    if raw_factors is None:
        return []

    if isinstance(raw_factors, list):
        normalized: list[FactorValue] = []
        for item in raw_factors:
            if isinstance(item, FactorValue):
                normalized.append(item)
            else:
                normalized.append(FactorValue.model_validate(item))
        return normalized

    normalized = []
    for name, raw_value in raw_factors.items():
        if isinstance(raw_value, dict):
            payload = dict(raw_value)
            payload.setdefault("name", name)
        else:
            payload = {"name": name, "value": raw_value}
        normalized.append(FactorValue.model_validate(payload))
    return normalized


__all__ = (
    "FactorValue",
    "Variant",
    "normalize_variant_factors",
)
