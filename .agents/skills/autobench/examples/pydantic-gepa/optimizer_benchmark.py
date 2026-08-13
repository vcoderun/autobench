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
    BestOf,
    Engine,
    EngineResult,
    EvaluationServer,
    OptimizationTask,
    OptimizeAnythingConfig,
    Pipeline,
    Single,
)
from pydantic_gepa.values import SerializableValue

from autobench import Case, RunContext


class OptimizationInput(BaseModel):
    seed: str
    weak_candidate: str
    strong_candidate: str
    continuation: str


class OptimizationOutcome(BaseModel):
    candidate: str
    score: float
    validation_score: float
    evaluation_calls: int
    engine_runs: int


@dataclass(slots=True)
class FixedResult:
    best_candidate: str | dict[str, str]
    best_score: float
    total_evals: int = 0
    eval_log: list[dict[str, SerializableValue]] = field(default_factory=list)
    metadata: dict[str, SerializableValue] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluatedCandidate:
    """A deterministic optimizer engine that uses the real evaluation server."""

    name: str
    candidate: str

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


def run(_ctx: RunContext, case: Case) -> OptimizationOutcome:
    inputs = OptimizationInput.model_validate(case.input)
    expected = str(case.expected["candidate"])
    active_prompt = CandidateContext[str]("optimizer.prompt")
    component = Component(
        name="prompt",
        initial_text=inputs.seed,
        kind="system_prompt",
        source="examples.pydantic_gepa.agent",
        path="system_prompt",
    )
    optimization = Optimization.from_examples(
        data=DataSplit.from_sets(
            train=(Example(name="train", inputs="train", expected_output=expected),),
            validation=(Example(name="validation", inputs="validation", expected_output=expected),),
            test=(Example(name="test", inputs="test", expected_output=expected),),
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
        backend="optimize_anything",
        optimization_objective="Return the expected prompt exactly.",
        background="The candidate is a versioned system-prompt component.",
    )
    result = optimization.optimize(
        config=OptimizeAnythingConfig(
            composition=Pipeline(
                steps=(
                    BestOf(
                        engines=(
                            _engine("weak", inputs.weak_candidate),
                            _engine("strong", inputs.strong_candidate),
                        )
                    ),
                    Single(engine=_engine("continuation", inputs.continuation)),
                )
            ),
            component="prompt",
        )
    )
    composition = result.composition
    if composition is None:
        raise RuntimeError("Optimize Anything did not return composition evidence.")
    return OptimizationOutcome(
        candidate=component.decode(result.best_candidate.values["prompt"]),
        score=result.best_score,
        validation_score=result.scores.validation or 0.0,
        evaluation_calls=result.budget.evaluation_calls or 0,
        engine_runs=len(composition.engine_runs),
    )


def _engine(name: str, candidate: str) -> Engine:
    return Engine.custom(
        EvaluatedCandidate(name=name, candidate=candidate),
        candidate_mode="text",
        max_evals=4,
    )


__all__ = ("run",)
