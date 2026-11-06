from __future__ import annotations as _annotations

from autobench.metrics.semantics import SemanticRegistry
from autobench.reports.comparison import compare_runs, has_case_tags
from autobench.reports.models import (
    DEFAULT_LEADERBOARD_METRICS,
    AssetTrajectoryReport,
    AssetTrajectoryStep,
    MetricAggregation,
)
from autobench.runtime.models import ExperimentResult, RunResult


def build_asset_trajectory(
    result: ExperimentResult,
    *,
    asset_id: str,
    metrics: tuple[MetricAggregation, ...] = DEFAULT_LEADERBOARD_METRICS,
    registry: SemanticRegistry | None = None,
    case_tags: tuple[str, ...] = (),
) -> AssetTrajectoryReport:
    """Compare every recorded version of an asset with its parent on the cases both ran.

    A run belongs to a version when it used exactly one version of the asset. A run that used
    several versions of it cannot be attributed to either and is counted as ambiguous. Versions
    are ordered so that a parent precedes its children; a version whose parent has no runs in
    the experiment is listed without a comparison.

    Args:
        result: Experiment evidence to analyze.
        asset_id: Tracked asset whose versions form the trajectory.
        metrics: Metrics to aggregate and pair by case.
        registry: Semantic registry used to resolve metric values.
        case_tags: Restrict the trajectory to cases carrying every listed tag.

    Returns:
        One step per version, paired against its parent when the parent was evaluated.
    """
    active_registry = registry or result.semantic_registry
    runs_by_version: dict[str, list[RunResult]] = {}
    parents: dict[str, str | None] = {}
    ambiguous_run_count = 0
    for run in result.runs:
        if not has_case_tags(run, case_tags):
            continue
        versions = {item.version: item for item in run.asset_versions if item.asset_id == asset_id}
        if len(versions) > 1:
            ambiguous_run_count += 1
            continue
        for version in versions.values():
            parents.setdefault(version.version, version.parent_version)
            runs_by_version.setdefault(version.version, []).append(run)

    steps: list[AssetTrajectoryStep] = []
    for version in _lineage_order(parents):
        parent = parents[version]
        parent_runs = None if parent is None else runs_by_version.get(parent)
        steps.append(
            AssetTrajectoryStep(
                version=version,
                parent_version=parent,
                run_count=len(runs_by_version[version]),
                comparison=(
                    None
                    if parent is None or parent_runs is None
                    else compare_runs(
                        baseline=parent,
                        candidate=version,
                        baseline_runs=parent_runs,
                        candidate_runs=runs_by_version[version],
                        metrics=metrics,
                        registry=active_registry,
                        case_tags=case_tags,
                    )
                ),
            )
        )
    return AssetTrajectoryReport(
        asset_id=asset_id,
        case_tags=case_tags,
        ambiguous_run_count=ambiguous_run_count,
        steps=tuple(steps),
    )


def _lineage_order(parents: dict[str, str | None]) -> tuple[str, ...]:
    # Walk each version up to its earliest observed ancestor, then emit that chain top-down, so
    # parents come first while unrelated branches keep the order the runs first used them in.
    ordered: list[str] = []
    placed: set[str] = set()
    for version in parents:
        chain: list[str] = []
        current: str | None = version
        while current is not None and current in parents and current not in placed:
            if current in chain:
                break
            chain.append(current)
            current = parents[current]
        ordered.extend(reversed(chain))
        placed.update(chain)
    return tuple(ordered)


__all__ = ("build_asset_trajectory",)
