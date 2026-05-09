from __future__ import annotations as _annotations

import importlib
from dataclasses import dataclass
from inspect import isawaitable
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from autobench.errors import ErrorRecord, TaskResolutionError
from autobench.metrics.observations import (
    Direction,
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import SemanticType
from autobench.runtime.context import RunContext
from autobench.runtime.tasks import TaskResult


class ScoreRecord(BaseModel):
    name: str
    semantic_type: SemanticType
    value: Any | None = None
    unit: str | None = None
    direction: Direction | None = None
    role: ObservationRole | None = None
    optional: bool = False
    actual_value: Any | None = None
    expected_value: Any | None = None
    error: ErrorRecord | None = None
    tags: dict[str, Any] = Field(default_factory=dict)

    def to_observation(
        self,
        *,
        observation_id: str,
        case_id: str,
        variant_id: str,
    ) -> Observation:
        return Observation(
            id=observation_id,
            name=self.name,
            kind=ObservationKind.METRIC,
            semantic_type=self.semantic_type,
            value=self.value,
            unit=self.unit,
            direction=self.direction,
            role=self.role,
            source=ObservationSource.SCORE,
            tags=self.tags,
            case_id=case_id,
            variant_id=variant_id,
        )


class ScoringSpecBase(BaseModel):
    name: str = Field(min_length=1)
    semantic_type: SemanticType
    unit: str | None = None
    direction: Direction | None = None
    role: ObservationRole | None = None
    optional: bool = False


class OutputMetricScorer(ScoringSpecBase):
    kind: Literal["output"] = "output"
    path: str = Field(min_length=1)


class PassFailScorer(ScoringSpecBase):
    kind: Literal["pass_fail"] = "pass_fail"
    path: str = Field(min_length=1)


class ExactScorer(ScoringSpecBase):
    kind: Literal["exact"] = "exact"
    actual: str = Field(min_length=1)
    expected: str = Field(min_length=1)


class SchemaScorer(ScoringSpecBase):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["schema"] = "schema"
    path: str = "output"
    schema_definition: dict[str, Any] = Field(default_factory=dict, alias="schema")


class PythonScorer(ScoringSpecBase):
    kind: Literal["python"] = "python"
    target: str = Field(min_length=1)
    module_search_paths: tuple[str, ...] = Field(default_factory=tuple, exclude=True)


ScoringSpec: TypeAlias = Annotated[
    OutputMetricScorer | PassFailScorer | ExactScorer | SchemaScorer | PythonScorer,
    Field(discriminator="kind"),
]


@dataclass(slots=True)
class ScoringCall:
    ctx: RunContext
    task_result: TaskResult

    @property
    def output(self) -> Any:
        return self.task_result.output

    @property
    def case(self) -> Any:
        return self.ctx.case

    @property
    def variant(self) -> Any:
        return self.ctx.variant

    @property
    def observations(self) -> list[Observation]:
        return self.task_result.observations


async def evaluate_scoring_specs(
    scoring: list[ScoringSpec],
    *,
    ctx: RunContext,
    task_result: TaskResult,
) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    call = ScoringCall(ctx=ctx, task_result=task_result)
    subjects = {
        "output": task_result.output,
        "case": ctx.case,
        "variant": ctx.variant,
    }

    for spec in scoring:
        try:
            record = await _evaluate_scoring_spec(spec, call=call, subjects=subjects)
        except Exception as exc:
            record = ScoreRecord(
                name=spec.name,
                semantic_type=spec.semantic_type,
                unit=spec.unit,
                direction=spec.direction,
                role=spec.role,
                optional=spec.optional,
                error=ErrorRecord.from_exception(exc),
            )
        records.append(record)

    return records


def score_records_to_observations(
    records: list[ScoreRecord],
    *,
    ctx: RunContext,
) -> list[Observation]:
    observations: list[Observation] = []
    for record in records:
        if record.error is not None:
            continue
        observations.append(
            record.to_observation(
                observation_id=ctx._next_observation_id(),
                case_id=ctx.case.id,
                variant_id=ctx.variant.id,
            )
        )
    return observations


def has_score_errors(records: list[ScoreRecord]) -> bool:
    return any(record.error is not None and not record.optional for record in records)


async def _evaluate_scoring_spec(
    spec: ScoringSpec,
    *,
    call: ScoringCall,
    subjects: dict[str, Any],
) -> ScoreRecord:
    if isinstance(spec, OutputMetricScorer):
        value = resolve_dotted_path(subjects, spec.path)
        return _build_score_record(spec, value=value)

    if isinstance(spec, PassFailScorer):
        actual_value = resolve_dotted_path(subjects, spec.path)
        return _build_score_record(spec, value=bool(actual_value), actual_value=actual_value)

    if isinstance(spec, ExactScorer):
        actual_value = resolve_dotted_path(subjects, spec.actual)
        expected_value = resolve_dotted_path(subjects, spec.expected)
        return _build_score_record(
            spec,
            value=1.0 if actual_value == expected_value else 0.0,
            actual_value=actual_value,
            expected_value=expected_value,
        )

    if isinstance(spec, SchemaScorer):
        actual_value = resolve_dotted_path(subjects, spec.path)
        return _build_score_record(
            spec,
            value=validate_schema_value(actual_value, spec.schema_definition),
            actual_value=actual_value,
        )

    scorer = resolve_python_scorer(spec.target, search_paths=spec.module_search_paths)
    result = scorer(call)
    if isawaitable(result):
        result = await result
    if isinstance(result, ScoreRecord):
        return result
    return _build_score_record(spec, value=result)


def resolve_dotted_path(subjects: dict[str, Any], path: str) -> Any:
    current: Any = subjects
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Path segment '{part}' not found in mapping.")
            current = current[part]
        else:
            if not hasattr(current, part):
                raise KeyError(
                    f"Path segment '{part}' not found on object '{type(current).__name__}'."
                )
            current = getattr(current, part)
        if callable(current):
            current = current()
    return current


def validate_schema_value(value: Any, schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type is not None and schema_type != "object":
        raise ValueError("Only schema.type='object' is supported in v0.")
    if not isinstance(value, dict):
        return False
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ValueError("schema.required must be a list when provided.")
    return all(key in value for key in required)


def resolve_python_scorer(target: str, *, search_paths: tuple[str, ...] = ()) -> Any:
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise TaskResolutionError("Python scorer targets must use 'module:function' format.")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        try:
            from autobench.runtime.tasks import _temporary_sys_path

            with _temporary_sys_path(search_paths):
                module = importlib.import_module(module_name)
        except Exception as fallback_exc:
            raise TaskResolutionError(
                f"Could not import scorer module '{module_name}'."
            ) from fallback_exc
    try:
        scorer = getattr(module, attribute_name)
    except AttributeError as exc:
        raise TaskResolutionError(
            f"Python scorer target '{target}' does not define '{attribute_name}'."
        ) from exc
    if not callable(scorer):
        raise TaskResolutionError(f"Python scorer target '{target}' is not callable.")
    return scorer


def _build_score_record(
    spec: ScoringSpecBase,
    *,
    value: Any,
    actual_value: Any | None = None,
    expected_value: Any | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        name=spec.name,
        semantic_type=spec.semantic_type,
        value=value,
        unit=spec.unit,
        direction=spec.direction,
        role=spec.role,
        optional=spec.optional,
        actual_value=actual_value,
        expected_value=expected_value,
    )


__all__ = (
    "ExactScorer",
    "OutputMetricScorer",
    "PassFailScorer",
    "PythonScorer",
    "SchemaScorer",
    "ScoreRecord",
    "ScoringCall",
    "ScoringSpec",
    "evaluate_scoring_specs",
    "has_score_errors",
    "resolve_dotted_path",
    "resolve_python_scorer",
    "score_records_to_observations",
    "validate_schema_value",
)
