from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from autobench.metrics.observations import (
    Observation,
    ObservationKind,
    ObservationRole,
    ObservationSource,
)
from autobench.metrics.semantics import DEFAULT_SEMANTIC_REGISTRY, SemanticRegistry, SemanticType
from autobench.reports.reporting import metric_value
from autobench.runtime.pipeline import ExperimentResult, RunResult


class BetweenRequirement(BaseModel):
    min: float
    max: float
    inclusive: bool = True

    @model_validator(mode="after")
    def _validate_bounds(self) -> BetweenRequirement:
        if self.min > self.max:
            raise ValueError("between min cannot be greater than max")
        return self


class PolicySpec(BaseModel):
    name: str = Field(min_length=1)
    metric: SemanticType
    must_equal: Any = None
    must_not_equal: Any = None
    must_greater: float | None = None
    must_greater_equal: float | None = None
    must_less: float | None = None
    must_less_equal: float | None = None
    must_in: tuple[Any, ...] | None = None
    must_not_in: tuple[Any, ...] | None = None
    must_between: BetweenRequirement | None = None

    @model_validator(mode="after")
    def _validate_single_requirement(self) -> PolicySpec:
        configured = [
            self.must_equal is not None,
            self.must_not_equal is not None,
            self.must_greater is not None,
            self.must_greater_equal is not None,
            self.must_less is not None,
            self.must_less_equal is not None,
            self.must_in is not None,
            self.must_not_in is not None,
            self.must_between is not None,
        ]
        if sum(configured) != 1:
            raise ValueError("policy must declare exactly one requirement")
        return self


class PolicyResult(BaseModel):
    policy_name: str
    run_id: str
    case_id: str
    variant_id: str
    metric: SemanticType
    passed: bool
    actual: Any = None
    reason: str | None = None


def apply_policies(
    policies: list[PolicySpec],
    *,
    result: ExperimentResult,
    registry: SemanticRegistry | None = None,
) -> ExperimentResult:
    if not policies:
        return result

    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    updated_runs = [
        _append_policy_observations(
            run,
            evaluate_run_policies(policies, run=run, registry=active_registry),
        )
        for run in result.runs
    ]
    return result.model_copy(update={"runs": updated_runs})


def evaluate_policies(
    policies: list[PolicySpec],
    *,
    result: ExperimentResult,
    registry: SemanticRegistry | None = None,
) -> list[PolicyResult]:
    active_registry = registry or DEFAULT_SEMANTIC_REGISTRY
    return [
        policy_result
        for run in result.runs
        for policy_result in evaluate_run_policies(policies, run=run, registry=active_registry)
    ]


def evaluate_run_policies(
    policies: list[PolicySpec],
    *,
    run: RunResult,
    registry: SemanticRegistry | None = None,
) -> list[PolicyResult]:
    return [_evaluate_policy(policy, run=run, registry=registry) for policy in policies]


def _evaluate_policy(
    policy: PolicySpec,
    *,
    run: RunResult,
    registry: SemanticRegistry | None,
) -> PolicyResult:
    actual = metric_value(run, policy.metric, registry=registry)
    if actual is None:
        return _policy_result(policy, run=run, passed=False, actual=None, reason="missing_metric")

    passed = _passes(policy, actual)
    return _policy_result(
        policy,
        run=run,
        passed=passed,
        actual=actual,
        reason=None if passed else "requirement_failed",
    )


def _passes(policy: PolicySpec, actual: Any) -> bool:
    if policy.must_equal is not None:
        return actual == policy.must_equal
    if policy.must_not_equal is not None:
        return actual != policy.must_not_equal
    if policy.must_in is not None:
        return actual in policy.must_in
    if policy.must_not_in is not None:
        return actual not in policy.must_not_in

    numeric = (
        None if isinstance(actual, bool) or not isinstance(actual, int | float) else float(actual)
    )
    if numeric is None:
        return False
    if policy.must_greater is not None:
        return numeric > policy.must_greater
    if policy.must_greater_equal is not None:
        return numeric >= policy.must_greater_equal
    if policy.must_less is not None:
        return numeric < policy.must_less
    if policy.must_less_equal is not None:
        return numeric <= policy.must_less_equal
    if policy.must_between is not None:
        requirement = policy.must_between
        if requirement.inclusive:
            return requirement.min <= numeric <= requirement.max
        return requirement.min < numeric < requirement.max
    raise ValueError("policy must declare exactly one requirement")  # pragma: no cover


def _policy_result(
    policy: PolicySpec,
    *,
    run: RunResult,
    passed: bool,
    actual: Any,
    reason: str | None,
) -> PolicyResult:
    return PolicyResult(
        policy_name=policy.name,
        run_id=run.run_id,
        case_id=run.case_id,
        variant_id=run.variant_id,
        metric=policy.metric,
        passed=passed,
        actual=actual,
        reason=reason,
    )


def _append_policy_observations(run: RunResult, results: list[PolicyResult]) -> RunResult:
    observations = [
        Observation(
            id=f"{run.run_id}_policy_{index + 1}",
            name=result.policy_name,
            kind=ObservationKind.EVENT,
            semantic_type="policy.result",
            value=result.passed,
            role=ObservationRole.CONSTRAINT,
            source=ObservationSource.DERIVED,
            tags={
                "metric": result.metric,
                "actual": result.actual,
                "reason": result.reason,
            },
            case_id=run.case_id,
            variant_id=run.variant_id,
        )
        for index, result in enumerate(results)
    ]
    task_result = run.task_result.model_copy(
        update={"observations": [*run.task_result.observations, *observations]}
    )
    return run.model_copy(update={"task_result": task_result})


__all__ = (
    "BetweenRequirement",
    "PolicyResult",
    "PolicySpec",
    "apply_policies",
    "evaluate_policies",
    "evaluate_run_policies",
)
