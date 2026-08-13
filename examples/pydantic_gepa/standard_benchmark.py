from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel
from pydantic_gepa import (
    CandidateContext,
    Component,
    DataSplit,
    DerivedValueInjection,
    Example,
    GEPAConfig,
    Optimization,
)
from pydantic_gepa.configuration import BudgetConfig, ReflectionConfig
from pydantic_gepa.values import SerializableValue

from autobench import Case, RunContext


class StandardInput(BaseModel):
    seed: str
    target: str


class StandardOutcome(BaseModel):
    candidate: str
    score: float
    evaluation_calls: int


def run(_ctx: RunContext, case: Case) -> StandardOutcome:
    inputs = StandardInput.model_validate(case.input)
    active_prompt = CandidateContext[str]("standard.prompt")
    component = Component(
        name="prompt",
        initial_text=inputs.seed,
        kind="system_prompt",
        source="examples.pydantic_gepa.standard",
        path="system_prompt",
    )

    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, SerializableValue]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        del reflective_dataset
        if "prompt" not in components_to_update:
            return candidate
        return {**candidate, "prompt": component.encode(inputs.target)}

    optimization = Optimization.from_examples(
        data=DataSplit.from_sets(
            train=(Example(name="train", inputs="train", expected_output=inputs.target),),
            validation=(
                Example(name="validation", inputs="validation", expected_output=inputs.target),
            ),
        ),
        task=lambda _sample: active_prompt.require(),
        score=lambda score: float(score.output == score.expected_output),
        components=(component,),
        injections=(
            DerivedValueInjection(
                component="prompt",
                context=active_prompt,
                required_components=("prompt",),
                derive_value=lambda candidate: component.decode(candidate["prompt"]),
            ),
        ),
    )
    result = optimization.optimize(
        config=GEPAConfig(
            reflection=ReflectionConfig(proposer=propose, minibatch_size=1),
            budget=BudgetConfig(max_metric_calls=6),
        )
    )
    return StandardOutcome(
        candidate=component.decode(result.best_candidate.values["prompt"]),
        score=result.best_score,
        evaluation_calls=result.total_metric_calls or 0,
    )


__all__ = ("run",)
