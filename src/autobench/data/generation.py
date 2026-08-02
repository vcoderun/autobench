from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, Field

from autobench.data.datasets import Case
from autobench.data.ingestion import ReviewStatus


class GeneratedCaseBatch(BaseModel):
    generator_asset_version: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    cases: tuple[Case, ...] = ()


class CaseGeneratorInput(BaseModel):
    seed_cases: tuple[Case, ...] = ()
    prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def mark_generated_case(
    case: Case,
    *,
    generator_asset_version: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    review_status: ReviewStatus = ReviewStatus.CANDIDATE,
) -> Case:
    metadata = dict(case.metadata)
    metadata["source"] = "synthetic"
    metadata["review_status"] = review_status.value
    if generator_asset_version is not None:
        metadata["generator_asset_version"] = generator_asset_version
    if model_provider is not None:
        metadata["model_provider"] = model_provider
    if model_name is not None:
        metadata["model_name"] = model_name
    return case.model_copy(update={"metadata": metadata})


def generated_batch_from_cases(
    cases: list[Case],
    *,
    generator_asset_version: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> GeneratedCaseBatch:
    marked_cases = tuple(
        mark_generated_case(
            case,
            generator_asset_version=generator_asset_version,
            model_provider=model_provider,
            model_name=model_name,
        )
        for case in cases
    )
    return GeneratedCaseBatch(
        generator_asset_version=generator_asset_version,
        model_provider=model_provider,
        model_name=model_name,
        cases=marked_cases,
    )


__all__ = (
    "CaseGeneratorInput",
    "GeneratedCaseBatch",
    "generated_batch_from_cases",
    "mark_generated_case",
)
