from __future__ import annotations

from autobench import (
    Case,
    CaseGeneratorInput,
    GeneratedCaseBatch,
    GeneratedCaseReview,
    GenerationCost,
    GenerationDeterminism,
    GenerationUsage,
    ReviewStatus,
)


def generate_routing_cases(request: CaseGeneratorInput) -> GeneratedCaseBatch:
    """Build deterministic routing cases from the reviewed seed request."""

    route = str(request.settings.get("route", "billing"))
    cases = tuple(
        Case(
            id=f"generated-{index}",
            input={"message": f"Refund request {index}"},
            expected={"route": route},
        )
        for index in range(1, 3)
    )
    return GeneratedCaseBatch(
        generator_asset_version=request.prompt_asset_version,
        determinism=GenerationDeterminism.GUARANTEED,
        usage=GenerationUsage(requests=1),
        cost=GenerationCost(amount=0),
        reviews=(GeneratedCaseReview(case_id="generated-1", status=ReviewStatus.ACCEPTED),),
        cases=cases,
    )
