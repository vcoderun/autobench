from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass

from autobench.data.datasets import Case
from autobench.metrics.semantics import SemanticRegistry
from autobench.reports.comparison import compare_runs, has_case_tags
from autobench.reports.inference import holm_adjusted
from autobench.reports.models import (
    DEFAULT_LEADERBOARD_METRICS,
    AssetTrajectoryReport,
    AssetTrajectoryStep,
    ComparisonReport,
    MetricAggregation,
    TrackedAssetSummary,
)
from autobench.runtime.models import ExperimentResult, RunResult

ROOT_VERSION = "root"
LEAF_VERSION = "leaf"


def build_asset_trajectory(
    results: ExperimentResult | Sequence[ExperimentResult],
    *,
    asset_id: str,
    metrics: tuple[MetricAggregation, ...] = DEFAULT_LEADERBOARD_METRICS,
    registry: SemanticRegistry | None = None,
    case_tags: tuple[str, ...] = (),
    variants: tuple[str, ...] = (),
) -> AssetTrajectoryReport:
    """Compare every recorded version of an asset with its parent on the cases both ran.

    A run belongs to a version when it used exactly one version of the asset. A run that used
    several versions of it cannot be attributed to either and is counted as ambiguous. Versions
    are ordered so that a parent precedes its children; a version whose parent has no runs is
    listed without a comparison. Each step's sign-test p-values are also Holm-adjusted across
    the tested steps of the trajectory, metric by metric.

    Records of one benchmark can be pooled, for example one record per generation of a
    self-modifying agent; a case must then mean the same thing in every record.

    Args:
        results: Experiment evidence to analyze, from one record or several.
        asset_id: Tracked asset whose versions form the trajectory.
        metrics: Metrics to aggregate and pair by case.
        registry: Semantic registry used to resolve metric values; defaults to the registry of
            the first record.
        case_tags: Restrict the trajectory to cases carrying every listed tag.
        variants: Restrict the trajectory to runs of the listed variants.

    Returns:
        One step per version, paired against its parent when the parent was evaluated.

    Raises:
        ValueError: If no record is given, or the pooled records disagree on the benchmark or on
            the content of a case.
    """
    pooled = _pooled_results(results)
    active_registry = registry or pooled[0].semantic_registry
    selection = _select_versions(pooled, asset_id=asset_id, case_tags=case_tags, variants=variants)
    versions = _lineage_order(selection.parents)
    comparisons = [
        _parent_comparison(
            selection,
            version=version,
            metrics=metrics,
            registry=active_registry,
            case_tags=case_tags,
        )
        for version in versions
    ]
    return AssetTrajectoryReport(
        asset_id=asset_id,
        experiment_ids=tuple(result.experiment_id for result in pooled),
        case_tags=case_tags,
        variants=variants,
        ambiguous_run_count=selection.ambiguous_run_count,
        steps=tuple(
            AssetTrajectoryStep(
                version=version,
                parent_version=selection.parents[version],
                run_count=len(selection.runs_by_version[version]),
                comparison=comparison,
                adjusted_p_values=adjusted,
            )
            for version, comparison, adjusted in zip(
                versions, comparisons, _adjusted_p_values(comparisons), strict=True
            )
        ),
    )


def compare_asset_versions(
    results: ExperimentResult | Sequence[ExperimentResult],
    *,
    asset_id: str,
    baseline: str,
    candidate: str,
    metrics: tuple[MetricAggregation, ...] = DEFAULT_LEADERBOARD_METRICS,
    registry: SemanticRegistry | None = None,
    case_tags: tuple[str, ...] = (),
    variants: tuple[str, ...] = (),
    name: str | None = None,
) -> ComparisonReport:
    """Compare two versions of one asset over the cases both ran.

    ``baseline`` and ``candidate`` name recorded versions, or ``"root"`` for the one version
    without an evaluated parent and ``"leaf"`` for the one version without an evaluated child.
    A side that names no version, a selector that matches several, or two sides that resolve to
    the same version leave the comparison without runs. The report's diagnostics explain why,
    and a hypothesis on the comparison is untestable.

    Args:
        results: Experiment evidence to analyze, from one record or several.
        asset_id: Tracked asset whose versions are compared.
        baseline: Baseline version or selector.
        candidate: Candidate version or selector.
        metrics: Metrics to aggregate and pair by case.
        registry: Semantic registry used to resolve metric values; defaults to the registry of
            the first record.
        case_tags: Restrict both versions to cases carrying every listed tag.
        variants: Restrict both versions to runs of the listed variants.
        name: Optional comparison label.

    Returns:
        Aggregate deltas, case-paired outcomes, and exact paired tests for every metric.

    Raises:
        ValueError: If no record is given, or the pooled records disagree on the benchmark or on
            the content of a case.
    """
    pooled = _pooled_results(results)
    selection = _select_versions(pooled, asset_id=asset_id, case_tags=case_tags, variants=variants)
    baseline_version, baseline_problem = _resolve_version(selection, asset_id, baseline)
    candidate_version, candidate_problem = _resolve_version(selection, asset_id, candidate)
    diagnostics = [
        problem for problem in (baseline_problem, candidate_problem) if problem is not None
    ]
    if baseline_version is not None and baseline_version == candidate_version:
        # A version compared with itself only produces ties, which is no evidence either way.
        diagnostics.append(
            f"baseline and candidate both resolve to version {baseline_version!r} "
            f"of asset {asset_id!r}"
        )
        baseline_version = candidate_version = None
    comparison = compare_runs(
        baseline=baseline_version or baseline,
        candidate=candidate_version or candidate,
        baseline_runs=selection.runs_for(baseline_version),
        candidate_runs=selection.runs_for(candidate_version),
        metrics=metrics,
        registry=registry or pooled[0].semantic_registry,
        case_tags=case_tags,
        name=name,
    )
    return comparison.model_copy(update={"asset_id": asset_id, "diagnostics": tuple(diagnostics)})


def summarize_tracked_assets(
    results: ExperimentResult | Sequence[ExperimentResult],
) -> tuple[TrackedAssetSummary, ...]:
    """List every asset the runs recorded, with its version and run counts.

    Args:
        results: Experiment evidence to inspect, from one record or several.

    Returns:
        One summary per asset id, sorted by asset id.

    Raises:
        ValueError: If no record is given, or the pooled records disagree on the benchmark or on
            the content of a case.
    """
    versions: dict[str, set[str]] = {}
    runs: dict[str, set[tuple[str, str]]] = {}
    for result in _pooled_results(results):
        for run in result.runs:
            for item in run.asset_versions:
                versions.setdefault(item.asset_id, set()).add(item.version)
                runs.setdefault(item.asset_id, set()).add((result.experiment_id, run.run_id))
    return tuple(
        TrackedAssetSummary(
            asset_id=asset_id,
            version_count=len(versions[asset_id]),
            run_count=len(runs[asset_id]),
        )
        for asset_id in sorted(versions)
    )


@dataclass(frozen=True, slots=True)
class _VersionSelection:
    runs_by_version: dict[str, list[RunResult]]
    parents: dict[str, str | None]
    ambiguous_run_count: int

    def runs_for(self, version: str | None) -> list[RunResult]:
        return [] if version is None else self.runs_by_version[version]


def _pooled_results(
    results: ExperimentResult | Sequence[ExperimentResult],
) -> tuple[ExperimentResult, ...]:
    pooled = (results,) if isinstance(results, ExperimentResult) else tuple(results)
    if not pooled:
        raise ValueError("at least one experiment result is required")
    benchmark_ids = sorted({result.benchmark_id for result in pooled})
    if len(benchmark_ids) > 1:
        raise ValueError(
            f"pooled records must share one benchmark, got: {', '.join(benchmark_ids)}"
        )
    if len(pooled) > 1:
        cases: dict[str, Case] = {}
        for result in pooled:
            for run in result.runs:
                if cases.setdefault(run.case_id, run.case) != run.case:
                    raise ValueError(f"case {run.case_id!r} differs between the pooled records")
    return pooled


def _select_versions(
    results: Sequence[ExperimentResult],
    *,
    asset_id: str,
    case_tags: tuple[str, ...],
    variants: tuple[str, ...],
) -> _VersionSelection:
    runs_by_version: dict[str, list[RunResult]] = {}
    parents: dict[str, str | None] = {}
    ambiguous_run_count = 0
    for result in results:
        for run in result.runs:
            if not has_case_tags(run, case_tags) or (variants and run.variant_id not in variants):
                continue
            used = {item.version: item for item in run.asset_versions if item.asset_id == asset_id}
            if len(used) > 1:
                ambiguous_run_count += 1
                continue
            for version in used.values():
                parents.setdefault(version.version, version.parent_version)
                runs_by_version.setdefault(version.version, []).append(run)
    return _VersionSelection(
        runs_by_version=runs_by_version,
        parents=parents,
        ambiguous_run_count=ambiguous_run_count,
    )


def _parent_comparison(
    selection: _VersionSelection,
    *,
    version: str,
    metrics: tuple[MetricAggregation, ...],
    registry: SemanticRegistry,
    case_tags: tuple[str, ...],
) -> ComparisonReport | None:
    parent = selection.parents[version]
    if parent is None or parent not in selection.runs_by_version:
        return None
    return compare_runs(
        baseline=parent,
        candidate=version,
        baseline_runs=selection.runs_by_version[parent],
        candidate_runs=selection.runs_by_version[version],
        metrics=metrics,
        registry=registry,
        case_tags=case_tags,
    )


def _adjusted_p_values(
    comparisons: Sequence[ComparisonReport | None],
) -> list[dict[str, float]]:
    tested: dict[str, list[tuple[int, float]]] = {}
    for index, comparison in enumerate(comparisons):
        for metric in () if comparison is None else comparison.metric_results:
            if metric.sign_test is not None:
                tested.setdefault(metric.name, []).append((index, metric.sign_test.two_sided_p))
    adjusted: list[dict[str, float]] = [{} for _ in comparisons]
    for metric_name, entries in tested.items():
        values = holm_adjusted([p_value for _, p_value in entries])
        for (index, _), value in zip(entries, values, strict=True):
            adjusted[index][metric_name] = value
    return adjusted


def _resolve_version(
    selection: _VersionSelection,
    asset_id: str,
    selector: str,
) -> tuple[str | None, str | None]:
    if selector not in (ROOT_VERSION, LEAF_VERSION):
        if selector in selection.runs_by_version:
            return selector, None
        return None, f"version {selector!r} of asset {asset_id!r} has no selected runs"
    evaluated = selection.runs_by_version
    if selector == ROOT_VERSION:
        matches = [
            version for version, parent in selection.parents.items() if parent not in evaluated
        ]
    else:
        children_of = set(selection.parents.values())
        matches = [version for version in selection.parents if version not in children_of]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"asset {asset_id!r} has no {selector} version among the selected runs"
    return None, (
        f"asset {asset_id!r} has {len(matches)} {selector} versions among the selected runs: "
        f"{', '.join(matches)}"
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


__all__ = (
    "LEAF_VERSION",
    "ROOT_VERSION",
    "build_asset_trajectory",
    "compare_asset_versions",
    "summarize_tracked_assets",
)
