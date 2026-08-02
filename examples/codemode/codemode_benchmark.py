from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, TypeAdapter

from autobench import Case, Direction, ObservationRole, RunContext, Semantic

TEXT_ADAPTER = TypeAdapter(str)
YAML_MAPPING_ADAPTER = TypeAdapter(dict[str, Any])


class ScenarioInput(BaseModel):
    name: str
    description: str
    code: str


class CodeModeOutput(BaseModel):
    success: bool
    coverage: float
    generated_case_count: int
    refinement_rounds: int


async def run(ctx: RunContext, case: Case) -> CodeModeOutput:
    try:
        from vowel.codemode import CodeModeGenerator
        from vowel.runner import Function, RunEvals
    except ImportError as exc:
        raise RuntimeError(
            "The CodeMode example requires the project that provides vowel.codemode."
        ) from exc

    scenario = ScenarioInput.model_validate(case.input)
    spec_model = TEXT_ADAPTER.validate_python(ctx.factor("spec_model"))
    exploration_model = TEXT_ADAPTER.validate_python(ctx.factor("exploration_model"))
    function = Function(
        name=scenario.name,
        description=scenario.description,
        code=scenario.code,
    )
    function_impl = function.impl
    generator = CodeModeGenerator(
        spec_model=spec_model,
        exploration_model=exploration_model,
    )

    with ctx.span(
        "generate_eval_spec",
        kind="agent",
        input={"function": scenario.name},
        attributes={
            "spec_model": spec_model,
            "exploration_model": exploration_model,
        },
        duration_metric={
            "name": "generation_latency",
            "semantic_type": Semantic.TIME_LATENCY,
            "unit": "s",
            "direction": Direction.MINIMIZE,
        },
    ) as span:
        result = await generator.generate(
            function,
            run_id=ctx.run_id,
            run_evals=True,
            max_refinement_rounds=2,
            min_coverage=1.0,
            inject_durations=False,
        )
        summary = (
            RunEvals.from_source(result.yaml_spec)
            .with_functions({scenario.name: function_impl})
            .ignore_duration()
            .run()
        )
        generated_case_count = _generated_case_count(result.yaml_spec)
        output = CodeModeOutput(
            success=summary.all_passed,
            coverage=summary.coverage,
            generated_case_count=generated_case_count,
            refinement_rounds=result.refinement_rounds,
        )
        span.metric(
            "coverage",
            output.coverage,
            semantic_type=Semantic.COVERAGE_RATIO,
            direction=Direction.MAXIMIZE,
            role=ObservationRole.OBJECTIVE,
        )
        span.metric(
            "generated_case_count",
            generated_case_count,
            semantic_type="evaluation.case.count",
            unit="cases",
            role=ObservationRole.DIAGNOSTIC,
        )
        span.artifact("generated_eval_spec", result.yaml_spec, media_type="application/yaml")
        span.artifact(
            "exploration_results",
            repr(result.exploration_results),
            media_type="text/plain",
        )
        span.set_output(output.model_dump())
    return output


def _generated_case_count(spec_source: str) -> int:
    raw: Any = yaml.safe_load(spec_source)
    if not isinstance(raw, dict) or not raw:
        return 0
    root = YAML_MAPPING_ADAPTER.validate_python(raw)
    benchmark_body = next(iter(root.values()))
    if not isinstance(benchmark_body, dict):
        return 0
    dataset = YAML_MAPPING_ADAPTER.validate_python(benchmark_body).get("dataset")
    return len(dataset) if isinstance(dataset, list) else 0


__all__ = ("run",)
