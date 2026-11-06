from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose
from typing import Any

from autobench.metrics.observations import Direction
from autobench.metrics.semantics import SemanticRegistry
from autobench.reports.inference import exact_sign_test, paired_success_difference
from autobench.reports.models import (
    DEFAULT_LEADERBOARD_METRICS,
    ComparisonOutcome,
    ComparisonReport,
    MetricAggregation,
    MetricComparisonReport,
    PairedBinaryTable,
)
from autobench.reports.values import aggregate_values, metric_metadata, metric_value, numeric_value
from autobench.runtime.models import ExperimentResult, RunResult


def compare_variants(
    result: ExperimentResult,
    *,
    baseline: str,
    candidate: str,
    metrics: tuple[MetricAggregation, ...] = DEFAULT_LEADERBOARD_METRICS,
    registry: SemanticRegistry | None = None,
    case_tags: tuple[str, ...] = (),
    name: str | None = None,
) -> ComparisonReport:
    """Compare two variants over the cases they share.

    Args:
        result: Experiment evidence to compare.
        baseline: Baseline variant id.
        candidate: Candidate variant id.
        metrics: Metrics to aggregate and pair by case.
        registry: Semantic registry used to resolve metric values.
        case_tags: Restrict both variants to cases carrying every listed tag.
        name: Optional comparison label, required to tell apart comparisons of the same pair.

    Returns:
        Aggregate deltas, case-paired outcomes, and exact paired tests for every metric.
    """
    selected_runs = [run for run in result.runs if has_case_tags(run, case_tags)]
    return compare_runs(
        baseline=baseline,
        candidate=candidate,
        baseline_runs=[run for run in selected_runs if run.variant_id == baseline],
        candidate_runs=[run for run in selected_runs if run.variant_id == candidate],
        metrics=metrics,
        registry=registry or result.semantic_registry,
        case_tags=case_tags,
        name=name,
    )


def compare_runs(
    *,
    baseline: str,
    candidate: str,
    baseline_runs: Sequence[RunResult],
    candidate_runs: Sequence[RunResult],
    registry: SemanticRegistry,
    metrics: tuple[MetricAggregation, ...] = DEFAULT_LEADERBOARD_METRICS,
    case_tags: tuple[str, ...] = (),
    name: str | None = None,
) -> ComparisonReport:
    """Compare two labelled groups of runs over the cases they share.

    The groups are taken as given: callers select the runs, for example by variant or by the
    asset version they used, and pass the labels that identify each side in the report.

    Args:
        baseline: Label of the baseline group.
        candidate: Label of the candidate group.
        baseline_runs: Runs of the baseline group.
        candidate_runs: Runs of the candidate group.
        registry: Semantic registry used to resolve metric values.
        metrics: Metrics to aggregate and pair by case.
        case_tags: Case tags the caller selected the runs by, recorded on the report.
        name: Optional comparison label.

    Returns:
        Aggregate deltas, case-paired outcomes, and exact paired tests for every metric.
    """
    factor_deltas = _factor_deltas(baseline_runs, candidate_runs)
    metric_deltas: dict[str, dict[str, Any]] = {}
    metric_results: list[MetricComparisonReport] = []
    baseline_by_case = {run.case_id: run for run in baseline_runs}
    candidate_by_case = {run.case_id: run for run in candidate_runs}
    paired_case_ids = tuple(sorted(set(baseline_by_case) & set(candidate_by_case)))
    all_case_ids = set(baseline_by_case) | set(candidate_by_case)

    for metric in metrics:
        baseline_samples = [
            value
            for run in baseline_runs
            if (value := metric_value(run, metric.semantic_type, registry=registry)) is not None
        ]
        candidate_samples = [
            value
            for run in candidate_runs
            if (value := metric_value(run, metric.semantic_type, registry=registry)) is not None
        ]
        baseline_value = aggregate_values(
            baseline_samples,
            metric.fn,
        )
        candidate_value = aggregate_values(
            candidate_samples,
            metric.fn,
        )
        delta = _numeric_delta(baseline_value, candidate_value)
        metric_deltas[metric.name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
        direction, unit, _ = metric_metadata(
            [*baseline_runs, *candidate_runs],
            metric.semantic_type,
            registry=registry,
        )
        tally = _paired_outcomes(
            paired_case_ids,
            baseline_by_case=baseline_by_case,
            candidate_by_case=candidate_by_case,
            semantic_type=metric.semantic_type,
            direction=direction,
            registry=registry,
        )
        metric_results.append(
            MetricComparisonReport(
                name=metric.name,
                semantic_type=registry.normalize(metric.semantic_type) or metric.semantic_type,
                aggregation=metric.fn,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=delta,
                relative_delta=_relative_delta(baseline_value, delta),
                direction=direction,
                unit=unit,
                outcome=_comparison_outcome(delta, direction),
                baseline_count=len(baseline_samples),
                candidate_count=len(candidate_samples),
                paired_count=tally.paired_count,
                missing_pair_count=len(paired_case_ids) - tally.paired_count,
                wins=tally.wins,
                ties=tally.ties,
                losses=tally.losses,
                sign_test=(
                    exact_sign_test(tally.wins, tally.losses)
                    if tally.paired_count and _is_directional(direction)
                    else None
                ),
                binary_table=tally.binary_table,
                success_difference=(
                    None
                    if tally.binary_table is None
                    else paired_success_difference(tally.binary_table)
                ),
            )
        )

    return ComparisonReport(
        name=name,
        baseline=baseline,
        candidate=candidate,
        case_tags=case_tags,
        run_count=min(len(baseline_runs), len(candidate_runs)),
        factor_deltas=factor_deltas,
        metric_deltas=metric_deltas,
        confounded=len(factor_deltas) > 1,
        baseline_factors=_factor_map(baseline_runs),
        candidate_factors=_factor_map(candidate_runs),
        paired_count=len(paired_case_ids),
        missing_pair_count=len(all_case_ids) - len(paired_case_ids),
        metric_results=tuple(metric_results),
    )


def _paired_outcomes(
    case_ids: Sequence[str],
    *,
    baseline_by_case: dict[str, RunResult],
    candidate_by_case: dict[str, RunResult],
    semantic_type: str,
    direction: str | None,
    registry: SemanticRegistry,
) -> _PairedTally:
    wins = 0
    ties = 0
    losses = 0
    pairs: list[tuple[bool | float, bool | float]] = []
    for case_id in case_ids:
        baseline = _paired_value(
            metric_value(baseline_by_case[case_id], semantic_type, registry=registry)
        )
        candidate = _paired_value(
            metric_value(candidate_by_case[case_id], semantic_type, registry=registry)
        )
        if baseline is None or candidate is None:
            continue
        pairs.append((baseline, candidate))
        outcome = _comparison_outcome(float(candidate) - float(baseline), direction)
        if outcome == "improved":
            wins += 1
        elif outcome == "regressed":
            losses += 1
        elif outcome == "unchanged":
            ties += 1
    return _PairedTally(
        wins=wins,
        ties=ties,
        losses=losses,
        paired_count=len(pairs),
        binary_table=_binary_table(pairs),
    )


@dataclass(frozen=True, slots=True)
class _PairedTally:
    wins: int
    ties: int
    losses: int
    paired_count: int
    binary_table: PairedBinaryTable | None


def _paired_value(value: Any) -> bool | float | None:
    # Booleans stay out of numeric aggregates, but a pass/fail pair still orders cleanly.
    if isinstance(value, bool):
        return value
    return numeric_value(value)


def _binary_table(pairs: Sequence[tuple[bool | float, bool | float]]) -> PairedBinaryTable | None:
    if not pairs or not all(
        isinstance(baseline, bool) and isinstance(candidate, bool) for baseline, candidate in pairs
    ):
        return None
    return PairedBinaryTable(
        both=sum(1 for baseline, candidate in pairs if baseline and candidate),
        candidate_only=sum(1 for baseline, candidate in pairs if candidate and not baseline),
        baseline_only=sum(1 for baseline, candidate in pairs if baseline and not candidate),
        neither=sum(1 for baseline, candidate in pairs if not baseline and not candidate),
    )


def has_case_tags(run: RunResult, case_tags: tuple[str, ...]) -> bool:
    """Return whether the run's case carries every listed tag.

    Args:
        run: Run to check.
        case_tags: Required case tags; an empty tuple selects every run.

    Returns:
        ``True`` when no listed tag is missing from the case.
    """
    return set(case_tags).issubset(run.case.tags)


def _numeric_delta(baseline: Any, candidate: Any) -> float | None:
    baseline_number = numeric_value(baseline)
    candidate_number = numeric_value(candidate)
    if baseline_number is None or candidate_number is None:
        return None
    return candidate_number - baseline_number


def _relative_delta(baseline: Any, delta: float | None) -> float | None:
    baseline_number = numeric_value(baseline)
    if baseline_number is None or delta is None or isclose(baseline_number, 0.0, abs_tol=1e-12):
        return None
    return delta / abs(baseline_number)


def _is_directional(direction: str | None) -> bool:
    # Without a direction a pair has no winner, so a sign test would report no evidence.
    return direction in {Direction.MAXIMIZE.value, Direction.MINIMIZE.value}


def _comparison_outcome(delta: float | None, direction: str | None) -> ComparisonOutcome:
    if delta is None or not _is_directional(direction):
        return "indeterminate"
    if isclose(delta, 0.0, abs_tol=1e-12):
        return "unchanged"
    if direction == Direction.MAXIMIZE.value:
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _factor_deltas(
    baseline_runs: Sequence[RunResult],
    candidate_runs: Sequence[RunResult],
) -> dict[str, dict[str, Any]]:
    baseline_factors = _factor_map(baseline_runs)
    candidate_factors = _factor_map(candidate_runs)
    deltas: dict[str, dict[str, Any]] = {}
    for name in sorted(set(baseline_factors) | set(candidate_factors)):
        baseline_value = baseline_factors.get(name)
        candidate_value = candidate_factors.get(name)
        if baseline_value != candidate_value:
            deltas[name] = {"baseline": baseline_value, "candidate": candidate_value}
    return deltas


def _factor_map(runs: Sequence[RunResult]) -> dict[str, Any]:
    if not runs:
        return {}
    return {factor.name: factor.value for factor in runs[0].factors}


__all__ = (
    "compare_runs",
    "compare_variants",
    "has_case_tags",
)
