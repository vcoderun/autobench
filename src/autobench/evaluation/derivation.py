from __future__ import annotations as _annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, Field

from autobench.evaluation.pricing import ModelPricing, PricingTable, load_pricing_table
from autobench.metrics.observations import (
    Direction,
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.query import ObservationQuery
from autobench.metrics.semantics import (
    DEFAULT_SEMANTIC_REGISTRY,
    Semantic,
    SemanticRegistry,
    SemanticType,
)
from autobench.runtime.context import RunContext


class DerivedMetricOutput(BaseModel):
    name: str = Field(min_length=1)
    semantic_type: SemanticType
    unit: str | None = None
    direction: Direction | None = None
    role: ObservationRole | None = None


class TokenCostInputs(BaseModel):
    input_tokens: SemanticType = Semantic.LLM_TOKENS_INPUT
    output_tokens: SemanticType = Semantic.LLM_TOKENS_OUTPUT
    provider: SemanticType = Semantic.LLM_PROVIDER
    model: SemanticType = Semantic.LLM_MODEL_NAME


class TokenCostDeriverSpec(BaseModel):
    kind: Literal["token_cost"] = "token_cost"
    output: DerivedMetricOutput = Field(
        default_factory=lambda: DerivedMetricOutput(
            name="cost",
            semantic_type=Semantic.MONEY_COST,
            unit="usd",
            direction=Direction.MINIMIZE,
            role=ObservationRole.CONSTRAINT,
        )
    )
    inputs: TokenCostInputs = Field(default_factory=TokenCostInputs)
    pricing: str = Field(min_length=1)


DeriverSpec: TypeAlias = Annotated[TokenCostDeriverSpec, Field(discriminator="kind")]


class Deriver(Protocol):  # pragma: no cover
    def derive(
        self,
        *,
        ctx: RunContext,
        observations: list[Observation],
        registry: SemanticRegistry,
    ) -> list[Observation]: ...


class TokenCostDeriver:
    def __init__(self, spec: TokenCostDeriverSpec) -> None:
        self.spec = spec

    def derive(
        self,
        *,
        ctx: RunContext,
        observations: list[Observation],
        registry: SemanticRegistry,
    ) -> list[Observation]:
        query = ObservationQuery(observations=observations, registry=registry)
        metric_kinds = (ObservationKind.METRIC, ObservationKind.FACTOR)

        input_tokens = query.first_related(
            self.spec.inputs.input_tokens,
            kind=metric_kinds,
        )
        output_tokens = query.first_related(
            self.spec.inputs.output_tokens,
            kind=metric_kinds,
        )
        provider = query.first_related(
            self.spec.inputs.provider,
            kind=metric_kinds,
        )
        model = query.first_related(
            self.spec.inputs.model,
            kind=metric_kinds,
        )

        if input_tokens is None or output_tokens is None or provider is None or model is None:
            missing = [
                name
                for name, value in (
                    ("input_tokens", input_tokens),
                    ("output_tokens", output_tokens),
                    ("provider", provider),
                    ("model", model),
                )
                if value is None
            ]
            return [
                _diagnostic_observation(
                    ctx=ctx,
                    name="token_cost_missing_inputs",
                    message="Missing inputs required for token cost derivation.",
                    tags={"missing": missing},
                )
            ]

        pricing = load_pricing_table(Path(self.spec.pricing))
        resolved_pricing = pricing.resolve_model_pricing(
            provider=str(provider.value),
            model=str(model.value),
        )
        if resolved_pricing is None:
            return [
                _diagnostic_observation(
                    ctx=ctx,
                    name="token_cost_unknown_pricing",
                    message="No pricing entry found for model/provider.",
                    tags={
                        "provider": str(provider.value),
                        "model": str(model.value),
                    },
                )
            ]
        resolved_model_id, model_pricing = resolved_pricing
        input_rate = model_pricing.input_rate_for_tokens(float(input_tokens.value))
        output_rate = model_pricing.output_rate_for_tokens(float(output_tokens.value))
        if input_rate is None or output_rate is None:
            return [
                _diagnostic_observation(
                    ctx=ctx,
                    name="token_cost_missing_rates",
                    message="Pricing entry did not define input/output rates.",
                    tags={
                        "provider": str(provider.value),
                        "model": str(model.value),
                        "model_id": resolved_model_id,
                    },
                )
            ]

        cost = (float(input_tokens.value) / 1_000_000.0) * input_rate + (
            float(output_tokens.value) / 1_000_000.0
        ) * output_rate
        return [
            Observation(
                id=ctx._next_observation_id(),
                name=self.spec.output.name,
                kind=ObservationKind.METRIC,
                semantic_type=self.spec.output.semantic_type,
                value=cost,
                unit=self.spec.output.unit,
                direction=self.spec.output.direction,
                role=self.spec.output.role,
                source=ObservationSource.DERIVED,
                tags={
                    "provider": str(provider.value),
                    "model": str(model.value),
                    "model_id": resolved_model_id,
                    "pricing_path": self.spec.pricing,
                },
                case_id=ctx.case.id,
                variant_id=ctx.variant.id,
            )
        ]


def derive_observations(
    derive: list[DeriverSpec],
    *,
    ctx: RunContext,
    observations: list[Observation],
    registry: SemanticRegistry | None = None,
) -> list[Observation]:
    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    derived: list[Observation] = []
    for spec in derive:
        deriver = build_deriver(spec)
        result = deriver.derive(
            ctx=ctx,
            observations=[*observations, *derived],
            registry=active_registry,
        )
        derived.extend(result)
    return derived


def build_deriver(spec: DeriverSpec) -> Deriver:
    if isinstance(spec, TokenCostDeriverSpec):
        return TokenCostDeriver(spec)
    raise TypeError(f"Unsupported deriver spec: {type(spec).__name__}")


def _diagnostic_observation(
    *,
    ctx: RunContext,
    name: str,
    message: str,
    tags: dict[str, Any],
) -> Observation:
    return Observation(
        id=ctx._next_observation_id(),
        name=name,
        kind=ObservationKind.EVENT,
        semantic_type=None,
        value=message,
        role=ObservationRole.DIAGNOSTIC,
        source=ObservationSource.DERIVED,
        tags=tags,
        case_id=ctx.case.id,
        variant_id=ctx.variant.id,
    )


__all__ = (
    "Deriver",
    "DeriverSpec",
    "DerivedMetricOutput",
    "ModelPricing",
    "PricingTable",
    "TokenCostDeriver",
    "TokenCostDeriverSpec",
    "TokenCostInputs",
    "build_deriver",
    "derive_observations",
    "load_pricing_table",
)
