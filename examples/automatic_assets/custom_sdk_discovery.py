from __future__ import annotations

from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from autobench import (
    Benchmark,
    CapturePolicy,
    Case,
    Direction,
    ExactScorer,
    InstrumentAssetSpec,
    ObservationRole,
    RunContext,
    Semantic,
    SpanKind,
    instrument_method,
    record_experiment,
)


class RoutingOutput(BaseModel):
    queue: Literal["billing", "technical"]
    confidence: float = Field(ge=0, le=1)


class WorkflowClient:
    def execute(
        self,
        *,
        instructions: str,
        tools: list[dict[str, str]],
        output_type: type[RoutingOutput],
        message: str,
    ) -> RoutingOutput:
        del instructions, tools, output_type, message
        return RoutingOutput(queue="technical", confidence=0.98)


CLIENT = WorkflowClient()


def run(ctx: RunContext, case: Case) -> RoutingOutput:
    del ctx
    return CLIENT.execute(
        instructions="Route infrastructure failures to technical support.",
        tools=[
            {
                "name": "lookup_incident",
                "description": "Look up an incident by its public identifier.",
            }
        ],
        output_type=RoutingOutput,
        message=str(case.input),
    )


def build_benchmark() -> Benchmark:
    return (
        Benchmark("custom-sdk-assets")
        .description("Add behavioral lineage to an arbitrary SDK method.")
        .capture(CapturePolicy.full())
        .dataset(
            [
                Case(
                    id="database-outage",
                    input="The production database is unavailable.",
                    expected={"queue": "technical"},
                )
            ]
        )
        .variants([{"id": "baseline"}])
        .task("__main__:run")
        .scoring(
            [
                ExactScorer(
                    name="queue",
                    semantic_type=Semantic.QUALITY_CORRECTNESS,
                    actual="output.queue",
                    expected="case.expected.queue",
                    direction=Direction.MAXIMIZE,
                    role=ObservationRole.OBJECTIVE,
                )
            ]
        )
    )


def render_assets(record_dir: Path, benchmark: Benchmark) -> None:
    handle = instrument_method(
        WorkflowClient,
        "execute",
        span="workflow_client.execute",
        span_kind=SpanKind.WORKFLOW,
        operation_family="workflow_client.execute",
        assets=[
            InstrumentAssetSpec(
                kind="prompt",
                local_id="instructions",
                value_path="kwargs.instructions",
            ),
            InstrumentAssetSpec(
                kind="tool",
                local_id="tools",
                value_path="kwargs.tools",
                many=True,
            ),
            InstrumentAssetSpec(
                kind="output_schema",
                local_id="output",
                value_path="kwargs.output_type",
            ),
        ],
    )
    try:
        experiment = benchmark.run()
    finally:
        handle.close()
    record_experiment(
        experiment,
        record_dir,
        source_files=[Path(__file__)],
        path_root=Path.cwd(),
    )

    table = Table(title="Custom SDK Asset Lineage", show_lines=True)
    table.add_column("Asset")
    table.add_column("Version")
    table.add_column("Representation")
    table.add_column("Span")
    for use in experiment.runs[0].asset_uses:
        table.add_row(
            use.asset_id,
            use.version,
            use.representation,
            use.span_id or "-",
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
