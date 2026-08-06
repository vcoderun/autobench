from __future__ import annotations

from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from rich.console import Console
from rich.table import Table

from autobench import (
    Benchmark,
    Case,
    Direction,
    ExactScorer,
    ObservationRole,
    RunContext,
    Semantic,
    record_experiment,
)


class RouteDecision(BaseModel):
    route: Literal["standard", "priority"]
    reason: str = Field(description="Why the selected route fits the request.")


class RetrievalCapability(AbstractCapability[None]):
    id = "retrieval"

    def get_instructions(self) -> str:
        return "Ground routing decisions in the current support policy."


def lookup_policy(topic: str) -> str:
    """Return the active support policy for a topic."""

    return f"{topic}: enterprise outages use priority support"


AGENT = Agent[None, RouteDecision](
    TestModel(
        custom_output_args={
            "route": "priority",
            "reason": "The enterprise outage policy requires priority support.",
        }
    ),
    name="support-router",
    deps_type=type(None),
    output_type=RouteDecision,
    instructions="Route each support request using the policy tool.",
    tools=[Tool[None](lookup_policy)],
    capabilities=[RetrievalCapability()],
)


async def run(ctx: RunContext, case: Case) -> RouteDecision:
    del ctx
    return (await AGENT.run(str(case.input))).output


def build_benchmark() -> Benchmark:
    return (
        Benchmark("automatic-pydantic-assets")
        .description("Discover Pydantic AI behavioral assets without tracking decorators.")
        .dataset(
            [
                Case(
                    id="enterprise-outage",
                    input="An enterprise customer reports a production outage.",
                    expected={"route": "priority"},
                )
            ]
        )
        .variants([{"id": "baseline"}])
        .task("__main__:run")
        .scoring(
            [
                ExactScorer(
                    name="route",
                    semantic_type=Semantic.QUALITY_CORRECTNESS,
                    actual="output.route",
                    expected="case.expected.route",
                    direction=Direction.MAXIMIZE,
                    role=ObservationRole.OBJECTIVE,
                )
            ]
        )
        .instrument_all(
            exclude={"openai", "openai_agents", "httpx"},
            assets={
                "include": [
                    "agent",
                    "capability",
                    "output_schema",
                    "prompt",
                    "tool",
                    "toolset",
                ]
            },
        )
    )


def render_assets(record_dir: Path, benchmark: Benchmark) -> None:
    experiment = benchmark.run()
    record_experiment(
        experiment,
        record_dir,
        source_files=[Path(__file__)],
        path_root=Path.cwd(),
    )

    table = Table(title="Automatically Discovered Pydantic AI Assets", show_lines=True)
    table.add_column("Asset")
    table.add_column("Version")
    table.add_column("Representation")
    table.add_column("Scope")
    table.add_column("Source")
    for use in experiment.runs[0].asset_uses:
        table.add_row(
            use.asset_id,
            use.version,
            use.representation,
            use.scope or "-",
            use.source_locator,
        )
    Console().print(table)


@click.command()
@click.option(
    "--record",
    "record_dir",
    type=click.Path(path_type=Path),
    required=True,
)
def main(record_dir: Path) -> None:
    render_assets(record_dir, build_benchmark())


if __name__ == "__main__":
    main()
