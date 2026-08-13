from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel
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

from autobench import Case, RunContext


class MultiComponentInput(BaseModel):
    prompt: str
    tool_schema: str
    output_schema: str


class MultiComponentOutcome(BaseModel):
    prompt: str
    tool_schema: str
    output_schema: str
    score: float


@dataclass(slots=True)
class FixedResult:
    best_candidate: str | dict[str, str]
    best_score: float
    total_evals: int = 0
    eval_log: list[dict[str, SerializableValue]] = field(default_factory=list)
    metadata: dict[str, SerializableValue] = field(default_factory=dict)


@dataclass(slots=True)
class ComponentEngine:
    name: str
    candidate: dict[str, str]

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        del task
        score, evidence = server.evaluate_examples(self.candidate, split="val")
        return FixedResult(
            best_candidate=self.candidate,
            best_score=score,
            total_evals=1,
            metadata={"validation_evidence": evidence},
        )

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result, output_dir


def run(_ctx: RunContext, case: Case) -> MultiComponentOutcome:
    inputs = MultiComponentInput.model_validate(case.input)
    contexts = {
        "prompt": CandidateContext[str]("components.prompt"),
        "tool_schema": CandidateContext[str]("components.tool_schema"),
        "output_schema": CandidateContext[str]("components.output_schema"),
    }
    components = (
        Component(
            name="prompt",
            initial_text="Answer without guidance.",
            kind="system_prompt",
            source="examples.pydantic_gepa.multi_component",
            path="system_prompt",
        ),
        Component(
            name="tool_schema",
            initial_text="search(query)",
            kind="tool_schema",
            source="examples.pydantic_gepa.multi_component",
            path="tools.search",
        ),
        Component(
            name="output_schema",
            initial_text="answer: string",
            kind="output_schema",
            source="examples.pydantic_gepa.multi_component",
            path="output.answer",
        ),
    )
    by_name = {component.name: component for component in components}
    target = inputs.model_dump()
    optimization = Optimization.from_examples(
        data=DataSplit.from_sets(
            train=(Example(name="train", inputs="train", expected_output=target),),
            validation=(Example(name="validation", inputs="validation", expected_output=target),),
        ),
        task=lambda _sample: {name: context.require() for name, context in contexts.items()},
        score=lambda score: float(score.output == score.expected_output),
        components=components,
        injections=tuple(
            DerivedValueInjection(
                component=name,
                context=context,
                required_components=(name,),
                derive_value=lambda candidate, component=by_name[name]: component.decode(
                    candidate[component.name]
                ),
            )
            for name, context in contexts.items()
        ),
        backend="optimize_anything",
        optimization_objective="Improve prompt, tool schema, and output schema together.",
    )
    encoded_target = {name: by_name[name].encode(value) for name, value in target.items()}
    result = optimization.optimize(
        config=OptimizeAnythingConfig(
            composition=Single(
                engine=Engine.custom(
                    ComponentEngine("component-editor", encoded_target),
                    candidate_mode="components",
                    max_evals=3,
                )
            )
        )
    )
    values = {name: by_name[name].decode(result.best_candidate.values[name]) for name in by_name}
    return MultiComponentOutcome(score=result.best_score, **values)


__all__ = ("run",)
