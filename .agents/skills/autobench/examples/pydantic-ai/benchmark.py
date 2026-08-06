from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel

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
        .description("Discover Pydantic AI behavior without tracking decorators.")
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


def main(record_dir: Path) -> None:
    experiment = build_benchmark().run()
    record_experiment(
        experiment,
        record_dir,
        source_files=[Path(__file__)],
        path_root=Path.cwd(),
    )
    print(record_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.record)
