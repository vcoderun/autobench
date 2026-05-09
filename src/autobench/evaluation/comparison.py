from __future__ import annotations as _annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from autobench.evaluation.derivation import DerivedMetricOutput
from autobench.metrics.observations import (
    Direction,
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry, SemanticType
from autobench.reports.reporting import metric_observation
from autobench.runtime.pipeline import ExperimentResult, RunResult

PairedBaselineFormula = Literal[
    "baseline_over_candidate",
    "candidate_over_baseline",
    "candidate_minus_baseline",
    "baseline_minus_candidate",
    "percent_change_from_baseline",
]
PostDerivationMissingPolicy = Literal["skip", "diagnostic"]
RunMatchKeyKind = Literal["case_id", "factor"]
ComparisonVerdict = Literal["improved", "regressed", "unchanged", "inconclusive"]


class RunMatchKey(BaseModel):
    kind: RunMatchKeyKind = "case_id"
    name: str = ""

    @model_validator(mode="after")
    def _validate_name(self) -> RunMatchKey:
        if self.kind == "factor" and not self.name:
            raise ValueError("factor match keys require name")
        if self.kind == "case_id" and self.name:
            raise ValueError("case_id match keys cannot declare name")
        return self


class RelativeThreshold(BaseModel):
    kind: Literal["relative_noise"] = "relative_noise"
    pct: float = Field(ge=0.0)


class ComparisonVerdictSpec(BaseModel):
    output: DerivedMetricOutput
    threshold: RelativeThreshold = Field(default_factory=lambda: RelativeThreshold(pct=0.0))


class PairedBaselineDeriverSpec(BaseModel):
    kind: Literal["paired_baseline"] = "paired_baseline"
    baseline_variant: str = Field(min_length=1)
    match_on: tuple[RunMatchKey, ...] = Field(default_factory=lambda: (RunMatchKey(),))
    metric: SemanticType
    output: DerivedMetricOutput
    formula: PairedBaselineFormula = "baseline_over_candidate"
    threshold: RelativeThreshold | None = None
    verdict: ComparisonVerdictSpec | None = None
    include_baseline: bool = False
    missing: PostDerivationMissingPolicy = "diagnostic"
    zero_division: PostDerivationMissingPolicy = "diagnostic"
    diagnostics_name: str = "paired_baseline_unavailable"


PostDeriverSpec: TypeAlias = Annotated[PairedBaselineDeriverSpec, Field(discriminator="kind")]


def derive_experiment_observations(
    post_derive: list[PostDeriverSpec],
    *,
    result: ExperimentResult,
    registry: SemanticRegistry | None = None,
) -> ExperimentResult:
    if not post_derive:
        return result

    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    runs = result.runs
    for spec in post_derive:
        runs = _apply_paired_baseline_deriver(spec, runs=runs, registry=active_registry)
    return result.model_copy(update={"runs": runs})


def _apply_paired_baseline_deriver(
    spec: PairedBaselineDeriverSpec,
    *,
    runs: list[RunResult],
    registry: SemanticRegistry,
) -> list[RunResult]:
    baseline_by_key = {
        key: run
        for run in runs
        if run.variant_id == spec.baseline_variant
        if (key := _run_match_key(run, spec.match_on)) is not None
    }
    updated_runs: list[RunResult] = []
    for run in runs:
        if run.variant_id == spec.baseline_variant and not spec.include_baseline:
            updated_runs.append(run)
            continue
        key = _run_match_key(run, spec.match_on)
        if key is None:
            derived = _diagnostics_or_empty(
                spec,
                run=run,
                reason="missing_match_key",
                tags={"baseline_variant": spec.baseline_variant},
            )
            updated_runs.append(_append_observations(run, derived) if derived else run)
            continue
        derived = _derive_for_run(
            spec,
            run=run,
            baseline=baseline_by_key.get(key),
            registry=registry,
        )
        updated_runs.append(_append_observations(run, derived) if derived else run)
    return updated_runs


def _derive_for_run(
    spec: PairedBaselineDeriverSpec,
    *,
    run: RunResult,
    baseline: RunResult | None,
    registry: SemanticRegistry,
) -> list[Observation]:
    if baseline is None:
        return _diagnostics_or_empty(
            spec,
            run=run,
            reason="missing_baseline_run",
            tags={"baseline_variant": spec.baseline_variant},
        )

    baseline_observation = metric_observation(baseline, spec.metric, registry=registry)
    candidate_observation = metric_observation(run, spec.metric, registry=registry)
    if baseline_observation is None or candidate_observation is None:
        return _diagnostics_or_empty(
            spec,
            run=run,
            reason="missing_metric",
            tags={
                "baseline_variant": spec.baseline_variant,
                "baseline_run_id": baseline.run_id,
                "metric": spec.metric,
                "missing_baseline_metric": baseline_observation is None,
                "missing_candidate_metric": candidate_observation is None,
            },
        )

    baseline_value = _numeric_value(baseline_observation.value)
    candidate_value = _numeric_value(candidate_observation.value)
    if baseline_value is None or candidate_value is None:
        return _diagnostics_or_empty(
            spec,
            run=run,
            reason="non_numeric_metric",
            tags={
                "baseline_variant": spec.baseline_variant,
                "baseline_run_id": baseline.run_id,
                "metric": spec.metric,
            },
        )

    value = _calculate_formula(spec.formula, baseline=baseline_value, candidate=candidate_value)
    if value is None:
        return _zero_division_diagnostics_or_empty(
            spec,
            run=run,
            baseline_run_id=baseline.run_id,
        )

    tags: dict[str, Any] = {
        "baseline_variant": spec.baseline_variant,
        "baseline_run_id": baseline.run_id,
        "formula": spec.formula,
        "source_metric": spec.metric,
    }
    if spec.threshold is not None:
        tags["relative_delta_pct"] = _relative_delta_pct(
            baseline=baseline_value,
            candidate=candidate_value,
        )
        tags["significant"] = _is_significant(
            spec.threshold,
            baseline=baseline_value,
            candidate=candidate_value,
        )

    observations = [
        Observation(
            id=_next_post_observation_id(run, 1),
            name=spec.output.name,
            kind=ObservationKind.METRIC,
            semantic_type=spec.output.semantic_type,
            value=value,
            unit=spec.output.unit,
            direction=spec.output.direction,
            role=spec.output.role,
            source=ObservationSource.DERIVED,
            tags=tags,
            case_id=run.case_id,
            variant_id=run.variant_id,
        )
    ]
    if spec.verdict is not None:
        observations.append(
            _verdict_observation(
                spec,
                run=run,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                direction=candidate_observation.direction or baseline_observation.direction,
            )
        )
    return observations


def _append_observations(run: RunResult, observations: list[Observation]) -> RunResult:
    task_result = run.task_result.model_copy(
        update={"observations": [*run.task_result.observations, *observations]}
    )
    return run.model_copy(update={"task_result": task_result})


def _diagnostics_or_empty(
    spec: PairedBaselineDeriverSpec,
    *,
    run: RunResult,
    reason: str,
    tags: dict[str, Any],
) -> list[Observation]:
    if spec.missing == "skip":
        return []
    return [_diagnostic(spec, run=run, reason=reason, tags=tags)]


def _zero_division_diagnostics_or_empty(
    spec: PairedBaselineDeriverSpec,
    *,
    run: RunResult,
    baseline_run_id: str,
) -> list[Observation]:
    if spec.zero_division == "skip":
        return []
    return [
        _diagnostic(
            spec,
            run=run,
            reason="zero_division",
            tags={
                "baseline_variant": spec.baseline_variant,
                "baseline_run_id": baseline_run_id,
                "metric": spec.metric,
                "formula": spec.formula,
            },
        )
    ]


def _diagnostic(
    spec: PairedBaselineDeriverSpec,
    *,
    run: RunResult,
    reason: str,
    tags: dict[str, Any],
) -> Observation:
    return Observation(
        id=_next_post_observation_id(run, 1),
        name=spec.diagnostics_name,
        kind=ObservationKind.EVENT,
        semantic_type=None,
        value=reason,
        role=ObservationRole.DIAGNOSTIC,
        source=ObservationSource.DERIVED,
        tags=tags,
        case_id=run.case_id,
        variant_id=run.variant_id,
    )


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def classify_metric_comparison(
    *,
    baseline: float,
    candidate: float,
    direction: Direction | None,
    threshold_pct: float = 0.0,
) -> ComparisonVerdict:
    if _relative_delta_pct(baseline=baseline, candidate=candidate) <= threshold_pct:
        return "unchanged"
    if direction is Direction.MAXIMIZE:
        return "improved" if candidate > baseline else "regressed"
    if direction is Direction.MINIMIZE:
        return "improved" if candidate < baseline else "regressed"
    return "inconclusive"


def _verdict_observation(
    spec: PairedBaselineDeriverSpec,
    *,
    run: RunResult,
    baseline_value: float,
    candidate_value: float,
    direction: Direction | None,
) -> Observation:
    if spec.verdict is None:
        raise ValueError("verdict spec is required")  # pragma: no cover
    verdict = classify_metric_comparison(
        baseline=baseline_value,
        candidate=candidate_value,
        direction=direction,
        threshold_pct=spec.verdict.threshold.pct,
    )
    return Observation(
        id=_next_post_observation_id(run, 2),
        name=spec.verdict.output.name,
        kind=ObservationKind.EVENT,
        semantic_type=spec.verdict.output.semantic_type,
        value=verdict,
        role=spec.verdict.output.role,
        source=ObservationSource.DERIVED,
        tags={
            "baseline_variant": spec.baseline_variant,
            "source_metric": spec.metric,
            "threshold_pct": spec.verdict.threshold.pct,
        },
        case_id=run.case_id,
        variant_id=run.variant_id,
    )


def _relative_delta_pct(*, baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return abs((candidate - baseline) / baseline) * 100.0


def _is_significant(
    threshold: RelativeThreshold,
    *,
    baseline: float,
    candidate: float,
) -> bool:
    return _relative_delta_pct(baseline=baseline, candidate=candidate) > threshold.pct


def _run_match_key(run: RunResult, match_on: tuple[RunMatchKey, ...]) -> tuple[str, ...] | None:
    values: list[str] = []
    for key in match_on:
        if key.kind == "case_id":
            values.append(run.case_id)
            continue
        factor_found, factor_value = _factor_value(run, key.name)
        if not factor_found:
            return None
        values.append(str(factor_value))
    return tuple(values)


def _factor_value(run: RunResult, name: str) -> tuple[bool, Any]:
    for factor in run.factors:
        if factor.name == name:
            return True, factor.value
    return False, None


def _calculate_formula(
    formula: PairedBaselineFormula,
    *,
    baseline: float,
    candidate: float,
) -> float | None:
    if formula == "baseline_over_candidate":
        return _divide(baseline, candidate)
    if formula == "candidate_over_baseline":
        return _divide(candidate, baseline)
    if formula == "candidate_minus_baseline":
        return candidate - baseline
    if formula == "baseline_minus_candidate":
        return baseline - candidate
    if formula == "percent_change_from_baseline":
        ratio = _divide(candidate - baseline, baseline)
        return None if ratio is None else ratio * 100.0
    raise ValueError(f"Unsupported paired baseline formula: {formula}")  # pragma: no cover


def _divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _next_post_observation_id(run: RunResult, offset: int) -> str:
    return f"{run.run_id}_post_obs_{len(run.task_result.observations) + offset}"


__all__ = (
    "ComparisonVerdict",
    "ComparisonVerdictSpec",
    "PairedBaselineDeriverSpec",
    "PairedBaselineFormula",
    "PostDerivationMissingPolicy",
    "PostDeriverSpec",
    "RelativeThreshold",
    "RunMatchKey",
    "RunMatchKeyKind",
    "classify_metric_comparison",
    "derive_experiment_observations",
)
