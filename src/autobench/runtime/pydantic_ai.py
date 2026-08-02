from __future__ import annotations as _annotations

from pydantic import BaseModel

from autobench.metrics.observations import Direction, Observation, ObservationRole
from autobench.metrics.semantics import Semantic
from autobench.runtime.context import RunContext


class PydanticAIUsage(BaseModel):
    requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    model_name: str | None = None
    provider: str | None = None


def record_pydantic_ai_usage(
    ctx: RunContext,
    usage: PydanticAIUsage,
    *,
    span_id: str | None = None,
) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    metric_values = {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }
    semantic_types = {
        "input_tokens": Semantic.LLM_TOKENS_INPUT,
        "output_tokens": Semantic.LLM_TOKENS_OUTPUT,
        "total_tokens": Semantic.LLM_TOKENS_TOTAL,
    }
    for name, value in metric_values.items():
        if value is None:
            continue
        observations.append(
            ctx.metric(
                f"pydantic_ai.{name}",
                value,
                semantic_type=semantic_types.get(name),
                direction=Direction.MINIMIZE if name == "requests" else None,
                role=ObservationRole.DIAGNOSTIC,
                span_id=span_id,
            )
        )
    if usage.model_name is not None:
        observations.append(
            ctx.factor_observation(
                "pydantic_ai.model",
                usage.model_name,
                semantic_type=Semantic.LLM_MODEL_NAME,
                span_id=span_id,
            )
        )
    if usage.provider is not None:
        observations.append(
            ctx.factor_observation(
                "pydantic_ai.provider",
                usage.provider,
                semantic_type=Semantic.LLM_PROVIDER,
                span_id=span_id,
            )
        )
    return tuple(observations)


__all__ = (
    "PydanticAIUsage",
    "record_pydantic_ai_usage",
)
