from __future__ import annotations as _annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import click
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_gepa import (
    CandidateContext,
    Component,
    DataSplit,
    DerivedValueInjection,
    Example,
    Optimization,
)
from pydantic_gepa.experimental.optimize_anything import (
    Engine,
    EngineResult,
    EvaluationServer,
    OptimizationTask,
    OptimizeAnythingConfig,
    Single,
)
from pydantic_gepa.values import SerializableValue

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

MODEL = "openrouter:openai/gpt-5.6-luna"
TARGET_PROMPT = "Return refund_request for requests asking for a refund."


class RoutingAnswer(BaseModel):
    label: str


@dataclass(frozen=True, slots=True)
class LiveDeps:
    source: str = "autobench-example"


@dataclass(slots=True)
class FixedResult:
    best_candidate: str | dict[str, str]
    best_score: float
    total_evals: int = 0
    eval_log: list[dict[str, SerializableValue]] = field(default_factory=list)
    metadata: dict[str, SerializableValue] = field(default_factory=dict)


@dataclass(slots=True)
class LiveCandidate:
    name: str = "live-candidate"

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        del task
        score, evidence = server.evaluate_examples(TARGET_PROMPT, split="val")
        return FixedResult(
            best_candidate=TARGET_PROMPT,
            best_score=score,
            total_evals=1,
            metadata={"validation_evidence": evidence},
        )

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result, output_dir


def run(_ctx: RunContext, case: Case) -> RoutingAnswer:
    active_prompt = CandidateContext[str]("live.prompt")
    component = Component(
        name="prompt",
        initial_text="Classify the request.",
        kind="system_prompt",
        source="examples.pydantic_gepa.live_agent",
        path="instructions",
    )

    def run_agent(message: str) -> RoutingAnswer:
        agent = Agent[LiveDeps, RoutingAnswer](
            MODEL,
            deps_type=LiveDeps,
            output_type=RoutingAnswer,
            instructions=active_prompt.require(),
        )
        return agent.run_sync(message, deps=LiveDeps()).output

    optimization = Optimization.from_examples(
        data=DataSplit.from_sets(
            train=(
                Example(
                    name="train-refund",
                    inputs=str(case.input),
                    expected_output=RoutingAnswer(label="refund_request"),
                ),
            ),
            validation=(
                Example(
                    name="validation-refund",
                    inputs=str(case.input),
                    expected_output=RoutingAnswer(label="refund_request"),
                ),
            ),
        ),
        task=run_agent,
        score=lambda score: float(
            score.expected_output is not None and score.output.label == score.expected_output.label
        ),
        components=(component,),
        injections=(
            DerivedValueInjection(
                component="prompt",
                context=active_prompt,
                required_components=("prompt",),
                derive_value=lambda candidate: component.decode(candidate["prompt"]),
            ),
        ),
        backend="optimize_anything",
        optimization_objective="Improve refund routing instructions.",
    )
    result = optimization.optimize(
        config=OptimizeAnythingConfig(
            composition=Single(
                engine=Engine.custom(
                    LiveCandidate(),
                    candidate_mode="text",
                    max_evals=2,
                )
            ),
            component="prompt",
        )
    )
    return run_agent(component.decode(result.best_candidate.values["prompt"]))


@click.command()
@click.option(
    "--record",
    "record_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="New directory for replayable Autobench evidence.",
)
def main(record_dir: Path) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise click.UsageError("Set OPENROUTER_API_KEY before running this live example.")
    benchmark = (
        Benchmark("pydantic-gepa-live-agent")
        .description("Capture pydantic-gepa, Pydantic AI, OpenAI, and HTTPX in one trace.")
        .dataset(
            [
                Case(
                    id="refund-request",
                    input="I need my money back for order ORD-1042.",
                    expected={"label": "refund_request"},
                )
            ]
        )
        .task("__main__:run")
        .scoring(
            [
                ExactScorer(
                    name="routing",
                    semantic_type=Semantic.QUALITY_CORRECTNESS,
                    actual="output.label",
                    expected="case.expected.label",
                    direction=Direction.MAXIMIZE,
                    role=ObservationRole.OBJECTIVE,
                )
            ]
        )
        .instrument_all()
    )
    record_experiment(benchmark.run(), record_dir)


if __name__ == "__main__":
    main()
