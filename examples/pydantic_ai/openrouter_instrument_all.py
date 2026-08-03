from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai import RunContext as AgentRunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from rich.console import Console
from rich.pretty import pretty_repr
from rich.table import Table

from autobench import (
    Benchmark,
    Case,
    Direction,
    ExactScorer,
    ObservationRole,
    RunContext,
    RunResult,
    Semantic,
    record_experiment,
    track,
)
from autobench.instrumentation.pydantic_ai import PydanticAI


@dataclass(frozen=True)
class Catalog:
    prices: dict[str, int]


@track.type
class ShoppingAnswer(BaseModel):
    product: str = Field(description="The exact product returned by the catalog tool.")
    price_usd: int = Field(description="The integer catalog price in US dollars.")
    recommendation: Literal["buy", "skip"]
    reason: str = Field(description="A short reason grounded in the catalog result.")


@track.tool
def lookup_price(ctx: AgentRunContext[Catalog], product: str) -> int:
    """Look up the current catalog price for a product."""
    price = ctx.deps.prices.get(product.lower())
    if price is None:
        raise ValueError(f"Unknown catalog product: {product}")
    return price


INSTRUCTIONS = track.prompt(
    name="shopping_advisor",
    text=(
        "You are a shopping advisor. Always call lookup_price before answering. "
        "Recommend buy only when the exact catalog price is at most the user's budget."
    ),
)


async def run(ctx: RunContext, case: Case) -> ShoppingAnswer:
    requested_model = ctx.factor("model")
    if not isinstance(requested_model, str):
        raise TypeError("The model factor must contain an OpenRouter model slug.")

    model_name = requested_model.removeprefix("openrouter:")
    model = OpenAIChatModel(model_name, provider=OpenRouterProvider())
    agent = Agent[Catalog, ShoppingAnswer](
        model,
        name="shopping_advisor",
        deps_type=Catalog,
        output_type=ShoppingAnswer,
        instructions=str(INSTRUCTIONS),
        tools=[lookup_price],
    )
    async with agent.run_stream(
        str(case.input),
        deps=Catalog(prices={"mechanical keyboard": 89}),
    ) as stream:
        return await stream.get_output()


def print_evidence(run_result: RunResult, *, record_dir: Path) -> None:
    console = Console()
    trace = run_result.trace

    summary = Table(title="Autobench Run", show_lines=True)
    summary.add_column("Run")
    summary.add_column("Status")
    summary.add_column("Spans", justify="right")
    summary.add_column("Observations", justify="right")
    summary.add_column("Record")
    summary.add_row(
        run_result.run_id,
        run_result.status,
        str(len(trace.spans) if trace is not None else 0),
        str(len(run_result.task_result.observations)),
        str(record_dir),
    )
    console.print(summary)

    if trace is not None:
        spans = Table(title="ABP Trace: Pydantic AI -> OpenAI -> HTTPX", show_lines=True)
        spans.add_column("Operation")
        spans.add_column("Layer")
        spans.add_column("Kind")
        spans.add_column("Status")
        spans.add_column("Duration ms", justify="right")
        spans.add_column("Usage")
        spans.add_column("Attributes")
        for span in trace.spans:
            duration = "-" if span.duration_ns is None else f"{span.duration_ns / 1_000_000:.2f}"
            spans.add_row(
                span.operation,
                span.scope.layer,
                span.kind,
                span.status,
                duration,
                json.dumps(span.usage, sort_keys=True),
                json.dumps(span.attributes, sort_keys=True),
            )
        console.print(spans)

    observations = Table(title="Semantic Observations", show_lines=True)
    observations.add_column("Name")
    observations.add_column("Semantic type")
    observations.add_column("Value")
    observations.add_column("Source")
    for observation in run_result.task_result.observations:
        observations.add_row(
            observation.name,
            observation.semantic_type,
            pretty_repr(observation.value, max_width=72, max_length=20, max_depth=4),
            observation.source,
        )
    console.print(observations)

    if trace is not None and trace.diagnostics:
        diagnostics = Table(title="Trace Diagnostics", show_lines=True)
        diagnostics.add_column("Severity")
        diagnostics.add_column("Code")
        diagnostics.add_column("Message")
        for diagnostic in trace.diagnostics:
            diagnostics.add_row(diagnostic.severity, diagnostic.code, diagnostic.message)
        console.print(diagnostics)

    console.print(run_result.task_result.output)


@click.command()
@click.option(
    "--model",
    envvar="OPENROUTER_MODEL",
    required=True,
    help="Pydantic AI model ID or OpenRouter slug, for example openrouter:openai/gpt-5.6-luna.",
)
@click.option(
    "--record",
    "record_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="New directory in which Autobench writes replayable YAML evidence.",
)
def main(model: str, record_dir: Path) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise click.UsageError("Set OPENROUTER_API_KEY before running this live example.")

    benchmark = (
        Benchmark("openrouter-shopping-agent")
        .description("Collect layered ABP evidence from a real Pydantic AI OpenRouter call.")
        .dataset(
            [
                Case(
                    id="keyboard-budget",
                    input=(
                        "Should I buy the mechanical keyboard if my budget is $100? "
                        "Use the catalog before deciding."
                    ),
                    expected={"product": "mechanical keyboard", "recommendation": "buy"},
                )
            ]
        )
        .variants(
            [
                {
                    "id": "openrouter_model",
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
                    name="recommendation",
                    semantic_type=Semantic.QUALITY_CORRECTNESS,
                    actual="output.recommendation",
                    expected="case.expected.recommendation",
                    direction=Direction.MAXIMIZE,
                    role=ObservationRole.OBJECTIVE,
                )
            ]
        )
        .instrument(PydanticAI(assets=[INSTRUCTIONS]))
        .instrument_all()
    )
    experiment = benchmark.run()
    record_experiment(
        experiment,
        record_dir,
        source_files=[Path(__file__)],
        path_root=Path.cwd(),
    )
    print_evidence(experiment.runs[0], record_dir=record_dir)


if __name__ == "__main__":
    main()
