from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai import RunContext as AgentRunContext
from pydantic_ai.exceptions import ModelRetry
from rich.console import Console
from rich.table import Table

from autobench import (
    Benchmark,
    Case,
    Direction,
    ExactScorer,
    HTTPXInstrumentation,
    ObservationRole,
    OpenAIInstrumentation,
    RunContext,
    Semantic,
    track,
)
from autobench.instrumentation.pydantic_ai import PydanticAI


@dataclass
class SupportDeps:
    lookup_attempts: int = 0


@track.type
class OrderSnapshot(BaseModel):
    order_id: str = Field(description="Canonical order identifier.")
    status: Literal["processing", "shipped", "delayed"]
    estimated_delivery: str


@track.type
class SupportAnswer(BaseModel):
    answer: str = Field(description="A concise answer grounded in the order lookup result.")
    order_id: str
    status: Literal["processing", "shipped", "delayed"]


@track.tool
def lookup_order(ctx: AgentRunContext[SupportDeps], order_id: str) -> OrderSnapshot:
    """Return the current status and delivery estimate for an order."""
    ctx.deps.lookup_attempts += 1
    if ctx.deps.lookup_attempts == 1:
        raise ModelRetry("The order service was temporarily unavailable; retry the lookup.")
    return OrderSnapshot(
        order_id=order_id,
        status="delayed",
        estimated_delivery="2026-08-06",
    )


SYSTEM_PROMPT = track.prompt(
    name="order_support",
    text=(
        "You are an order support agent. Always call lookup_order before answering. "
        "Do not invent status or delivery information."
    ),
)


async def run(ctx: RunContext, case: Case) -> SupportAnswer:
    model = ctx.factor("model")
    if not isinstance(model, str):
        raise TypeError("The model factor must be a string Pydantic AI model identifier.")
    agent = Agent[SupportDeps, SupportAnswer](
        model,
        name="order_support",
        deps_type=SupportDeps,
        output_type=SupportAnswer,
        instructions=str(SYSTEM_PROMPT),
        tools=[lookup_order],
        retries=2,
    )
    async with agent.run_stream(str(case.input), deps=SupportDeps()) as stream:
        return await stream.get_output()


def main() -> None:
    model = os.environ.get("PYDANTIC_AI_MODEL")
    if model is None:
        raise SystemExit(
            "Set PYDANTIC_AI_MODEL to a configured provider model, for example "
            "openrouter:openai/gpt-5.6-luna."
        )

    benchmark = (
        Benchmark("pydantic-ai-order-support")
        .description("Evaluate a tool-using structured-output support agent.")
        .dataset(
            [
                Case(
                    id="delayed-order",
                    input="Where is order ORD-1042? Use the order service before answering.",
                    expected={"status": "delayed"},
                )
            ]
        )
        .variants(
            [
                {
                    "id": "configured_model",
                    "factors": [
                        {
                            "name": "model",
                            "value": model,
                            "semantic_type": Semantic.LLM_MODEL_REQUESTED,
                            "optimize": True,
                        }
                    ],
                }
            ]
        )
        .task("__main__:run")
        .scoring(
            [
                ExactScorer(
                    name="order_status",
                    semantic_type=Semantic.QUALITY_CORRECTNESS,
                    actual="output.status",
                    expected="case.expected.status",
                    direction=Direction.MAXIMIZE,
                    role=ObservationRole.OBJECTIVE,
                )
            ]
        )
        .instrument(PydanticAI(assets=[SYSTEM_PROMPT]))
    )
    if model.startswith("openai:"):
        benchmark.instrument(OpenAIInstrumentation(), HTTPXInstrumentation())

    experiment = benchmark.run()

    run_result = experiment.runs[0]
    table = Table(title="Pydantic AI Evidence", show_lines=True)
    table.add_column("Operation")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Usage")
    if run_result.trace is not None:
        for span in run_result.trace.spans:
            table.add_row(
                span.operation,
                span.kind,
                span.status,
                ", ".join(f"{key}={value}" for key, value in span.usage.items()) or "-",
            )
    Console().print(table)
    Console().print(run_result.task_result.output)


if __name__ == "__main__":
    main()
