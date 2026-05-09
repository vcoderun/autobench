from __future__ import annotations as _annotations

from math import isnan
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autobench.reports.reporting import (
    BenchmarkReport,
    CaseMatrixVisualSpec,
    ComparisonReport,
    ComparisonVisualSpec,
    DistributionVisualSpec,
    LeaderboardVisualSpec,
    MetricDistribution,
    ReportSpec,
    StatusVisualSpec,
    VariantConfigVisualSpec,
    VisualizationSpec,
    build_report,
)
from autobench.runtime.pipeline import ExperimentResult


def export_png_report(
    result: ExperimentResult,
    path: Path,
    *,
    report_spec: ReportSpec | None = None,
) -> str:
    report = build_report(result, report_spec=report_spec)
    active_report_spec = report_spec
    if active_report_spec is None and result.report_spec_data is not None:
        active_report_spec = ReportSpec.model_validate(result.report_spec_data)
    if active_report_spec is None:
        active_report_spec = ReportSpec()
    visuals = active_report_spec.visuals or default_visual_specs(report)
    figure = render_report_figure(report, visuals=visuals)
    try:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)
    return str(path)


def export_png_report_set(
    result: ExperimentResult,
    directory: Path,
    *,
    report_spec: ReportSpec | None = None,
) -> list[str]:
    report = build_report(result, report_spec=report_spec)
    active_report_spec = _active_report_spec(result, report_spec=report_spec)
    visuals = active_report_spec.visuals or default_visual_specs(report)
    directory.mkdir(parents=True, exist_ok=True)
    exported_paths: list[str] = []
    for index, visual in enumerate(visuals, start=1):
        figure = render_report_figure(report, visuals=(visual,))
        path = directory / f"{index:02d}-{_visual_slug(visual)}.png"
        try:
            figure.savefig(path, dpi=200, bbox_inches="tight")
        finally:
            plt.close(figure)
        exported_paths.append(str(path))
    return exported_paths


def _active_report_spec(
    result: ExperimentResult,
    *,
    report_spec: ReportSpec | None,
) -> ReportSpec:
    if report_spec is not None:
        return report_spec
    if result.report_spec_data is not None:
        return ReportSpec.model_validate(result.report_spec_data)
    return ReportSpec()


def default_visual_specs(report: BenchmarkReport) -> tuple[VisualizationSpec, ...]:
    visuals: list[VisualizationSpec] = [
        VariantConfigVisualSpec(title="Variant Configurations"),
        StatusVisualSpec(render_as="pie", title="Run Status"),
        LeaderboardVisualSpec(render_as="table", title="Leaderboard"),
        LeaderboardVisualSpec(render_as="grouped_bar", title="Leaderboard Metrics"),
        CaseMatrixVisualSpec(
            render_as="heatmap", title=f"Case Matrix ({report.case_matrix.metric})"
        ),
        CaseMatrixVisualSpec(render_as="grouped_bar", title="Per-Case Variant Scores"),
    ]
    for metric_name in _numeric_leaderboard_metric_names(report):
        visuals.append(
            LeaderboardVisualSpec(
                render_as="bar",
                metric=metric_name,
                title=f"Leaderboard: {metric_name}",
            )
        )
    visuals.extend(
        ComparisonVisualSpec(
            baseline=comparison.baseline,
            candidate=comparison.candidate,
            render_as="delta_bar",
            title=f"Comparison: {comparison.baseline} vs {comparison.candidate}",
        )
        for comparison in report.comparisons
    )
    visuals.extend(
        DistributionVisualSpec(
            name=distribution.name,
            render_as="boxplot",
            title=f"Distribution: {distribution.name}",
        )
        for distribution in report.distributions
    )
    return tuple(visuals)


def render_report_figure(
    report: BenchmarkReport, *, visuals: tuple[VisualizationSpec, ...]
) -> Figure:
    count = max(len(visuals), 1)
    figure, axes = plt.subplots(count, 1, figsize=(14, max(4, count * 3.5)))
    axes_list = [axes] if isinstance(axes, Axes) else list(axes)
    figure.suptitle(f"{report.benchmark_id} ({report.experiment_id})", fontsize=16, y=0.995)

    for axis, visual in zip(axes_list, visuals, strict=False):
        _render_visual(axis, report, visual)
    for axis in axes_list[len(visuals) :]:
        axis.axis("off")

    figure.tight_layout()
    return figure


def _render_visual(axis: Axes, report: BenchmarkReport, visual: VisualizationSpec) -> None:
    if isinstance(visual, VariantConfigVisualSpec):
        _render_variant_config_visual(axis, report, visual)
        return
    if isinstance(visual, StatusVisualSpec):
        _render_status_visual(axis, report, visual)
        return
    if isinstance(visual, LeaderboardVisualSpec):
        _render_leaderboard_visual(axis, report, visual)
        return
    if isinstance(visual, CaseMatrixVisualSpec):
        _render_case_matrix_visual(axis, report, visual)
        return
    if isinstance(visual, ComparisonVisualSpec):
        _render_comparison_visual(axis, report, visual)
        return
    _render_distribution_visual(axis, report, visual)


def _render_variant_config_visual(
    axis: Axes,
    report: BenchmarkReport,
    visual: VariantConfigVisualSpec,
) -> None:
    axis.axis("off")
    factor_names = _ordered_variant_factor_names(report)
    headers = ["variant", "label", *factor_names]
    rows = [
        [
            config.variant_id,
            config.label or "-",
            *[_display_value(config.factors.get(factor_name)) for factor_name in factor_names],
        ]
        for config in report.variant_configs
    ]
    _render_table(axis, headers, rows, title=visual.title or "Variant Configurations")


def _render_status_visual(
    axis: Axes,
    report: BenchmarkReport,
    visual: StatusVisualSpec,
) -> None:
    labels = list(report.status_counts)
    values = [report.status_counts[label] for label in labels]
    if visual.render_as == "pie" and values and sum(values) > 0:
        axis.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
        axis.set_title(visual.title or "Run Status")
        return
    if visual.render_as == "bar" and values:
        axis.bar(labels, values, color="#72B7B2")
        axis.set_title(visual.title or "Run Status")
        axis.set_ylabel("runs")
        return
    axis.axis("off")
    _render_table(
        axis,
        ["status", "count"],
        [[label, str(report.status_counts[label])] for label in labels],
        title=visual.title or "Run Status",
    )


def _render_leaderboard_visual(
    axis: Axes,
    report: BenchmarkReport,
    visual: LeaderboardVisualSpec,
) -> None:
    if visual.render_as == "bar":
        metric_name = _select_numeric_metric_name(report, visual.metric)
        if metric_name is None:
            _render_leaderboard_table(axis, report, title=visual.title or "Leaderboard")
            return
        labels = [row.variant_id for row in report.leaderboard]
        values = [_numeric_value(row.metrics.get(metric_name)) or 0.0 for row in report.leaderboard]
        axis.bar(labels, values, color="#4C78A8")
        axis.set_title(visual.title or f"Leaderboard: {metric_name}")
        axis.set_ylabel(metric_name)
        axis.tick_params(axis="x", rotation=15)
        return
    if visual.render_as == "grouped_bar":
        _render_leaderboard_grouped_bar(axis, report, title=visual.title or "Leaderboard Metrics")
        return
    if visual.render_as == "line":
        _render_leaderboard_line(axis, report, visual)
        return
    if visual.render_as == "pie":
        _render_leaderboard_pie(axis, report, visual)
        return
    _render_leaderboard_table(axis, report, title=visual.title or "Leaderboard")


def _render_case_matrix_visual(
    axis: Axes,
    report: BenchmarkReport,
    visual: CaseMatrixVisualSpec,
) -> None:
    if visual.render_as == "heatmap" and _case_matrix_is_numeric(report):
        _render_case_matrix_heatmap(axis, report, title=visual.title or "Case Matrix")
        return
    if visual.render_as == "grouped_bar" and _case_matrix_is_numeric(report):
        _render_case_matrix_grouped_bar(axis, report, title=visual.title or "Case Matrix")
        return
    if visual.render_as == "line" and _case_matrix_is_numeric(report):
        _render_case_matrix_line(axis, report, title=visual.title or "Case Matrix")
        return
    _render_case_matrix_table(axis, report, title=visual.title or "Case Matrix")


def _render_comparison_visual(
    axis: Axes,
    report: BenchmarkReport,
    visual: ComparisonVisualSpec,
) -> None:
    comparison = _find_comparison(report, visual)
    if comparison is None:
        axis.axis("off")
        axis.set_title(
            visual.title or f"Comparison missing: {visual.baseline} vs {visual.candidate}"
        )
        return
    if visual.render_as in {"bar", "delta_bar"}:
        metric_names = _numeric_metric_delta_names(comparison, selected_metric=visual.metric)
        if metric_names:
            if visual.render_as == "delta_bar":
                deltas = [
                    _numeric_value(comparison.metric_deltas[name].get("delta")) or 0.0
                    for name in metric_names
                ]
                colors = ["#54A24B" if value >= 0 else "#E45756" for value in deltas]
                positions = list(range(len(metric_names)))
                axis.bar(positions, deltas, color=colors)
                axis.axhline(0, color="#333333", linewidth=0.8)
                axis.set_xticks(positions)
                axis.set_xticklabels(metric_names, rotation=15)
                axis.set_title(
                    visual.title or f"Delta: {comparison.baseline} -> {comparison.candidate}"
                )
                axis.set_ylabel("candidate - baseline")
                return
            baseline_values = [
                _numeric_value(comparison.metric_deltas[name].get("baseline")) or 0.0
                for name in metric_names
            ]
            candidate_values = [
                _numeric_value(comparison.metric_deltas[name].get("candidate")) or 0.0
                for name in metric_names
            ]
            positions = list(range(len(metric_names)))
            width = 0.35
            axis.bar(
                [position - width / 2 for position in positions],
                baseline_values,
                width=width,
                label=comparison.baseline,
            )
            axis.bar(
                [position + width / 2 for position in positions],
                candidate_values,
                width=width,
                label=comparison.candidate,
            )
            axis.set_xticks(positions)
            axis.set_xticklabels(metric_names, rotation=15)
            axis.legend()
            axis.set_title(
                visual.title or f"Comparison: {comparison.baseline} vs {comparison.candidate}"
            )
            return
    _render_comparison_table(
        axis,
        comparison,
        title=visual.title or f"Comparison: {comparison.baseline} vs {comparison.candidate}",
    )


def _render_distribution_visual(
    axis: Axes,
    report: BenchmarkReport,
    visual: DistributionVisualSpec,
) -> None:
    distribution = _find_distribution(report, visual.name)
    if distribution is None:
        axis.axis("off")
        axis.set_title(visual.title or f"Distribution missing: {visual.name}")
        return
    if visual.render_as == "boxplot":
        samples_by_variant = {
            variant: [_numeric_value(value) for value in values]
            for variant, values in distribution.by_variant.items()
        }
        numeric_samples = {
            variant: [value for value in values if value is not None]
            for variant, values in samples_by_variant.items()
        }
        numeric_samples = {variant: values for variant, values in numeric_samples.items() if values}
        if numeric_samples:
            axis.boxplot(
                [numeric_samples[variant] for variant in numeric_samples],
                tick_labels=list(numeric_samples),
                patch_artist=True,
            )
            axis.set_title(visual.title or f"Distribution: {distribution.name}")
            axis.set_ylabel(distribution.semantic_type)
            axis.tick_params(axis="x", rotation=15)
            return
    if visual.render_as == "line":
        summaries = distribution.summaries
        median_values = [
            _numeric_value(summaries.get(variant, {}).get("median"))
            for variant in distribution.by_variant
        ]
        if any(value is not None for value in median_values):
            labels = list(distribution.by_variant)
            axis.plot(
                labels,
                [0.0 if value is None else value for value in median_values],
                marker="o",
                color="#4C78A8",
            )
            axis.set_title(visual.title or f"Distribution Trend: {distribution.name}")
            axis.set_ylabel(distribution.semantic_type)
            axis.tick_params(axis="x", rotation=15)
            return
    _render_distribution_table(axis, distribution, title=visual.title or distribution.name)


def _render_leaderboard_table(axis: Axes, report: BenchmarkReport, *, title: str) -> None:
    axis.axis("off")
    metric_names = _ordered_metric_names(report)
    headers = ["variant", "runs", *metric_names]
    rows = [
        [
            row.variant_id,
            str(row.run_count),
            *[_display_value(row.metrics.get(name)) for name in metric_names],
        ]
        for row in report.leaderboard
    ]
    _render_table(axis, headers, rows, title=title)


def _render_case_matrix_table(axis: Axes, report: BenchmarkReport, *, title: str) -> None:
    axis.axis("off")
    variant_names = sorted(
        {variant_name for row in report.case_matrix.rows.values() for variant_name in row}
    )
    headers = ["case", *variant_names]
    rows = [
        [case_id, *[_display_value(values.get(variant_name)) for variant_name in variant_names]]
        for case_id, values in sorted(report.case_matrix.rows.items())
    ]
    _render_table(axis, headers, rows, title=title)


def _render_comparison_table(axis: Axes, comparison: ComparisonReport, *, title: str) -> None:
    axis.axis("off")
    headers = ["metric", comparison.baseline, comparison.candidate, "delta"]
    rows = [
        [
            name,
            _display_value(payload.get("baseline")),
            _display_value(payload.get("candidate")),
            _display_value(payload.get("delta")),
        ]
        for name, payload in sorted(comparison.metric_deltas.items())
    ]
    if not rows:
        rows = [["no metric deltas", "-", "-", "-"]]
    _render_table(axis, headers, rows, title=title)


def _render_distribution_table(
    axis: Axes,
    distribution: MetricDistribution,
    *,
    title: str,
) -> None:
    axis.axis("off")
    summary_names = _ordered_summary_names(distribution)
    headers = ["variant", "samples", *summary_names]
    rows = [
        [
            variant,
            str(len(values)),
            *[
                _display_value(distribution.summaries.get(variant, {}).get(name))
                for name in summary_names
            ],
        ]
        for variant, values in distribution.by_variant.items()
    ]
    _render_table(axis, headers, rows, title=title)


def _render_case_matrix_heatmap(axis: Axes, report: BenchmarkReport, *, title: str) -> None:
    variant_names = sorted(
        {variant_name for row in report.case_matrix.rows.values() for variant_name in row}
    )
    case_ids = sorted(report.case_matrix.rows)
    matrix = [
        [
            _nan_if_none(_numeric_value(report.case_matrix.rows[case_id].get(variant_name)))
            for variant_name in variant_names
        ]
        for case_id in case_ids
    ]
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xticks(range(len(variant_names)))
    axis.set_xticklabels(variant_names, rotation=15)
    axis.set_yticks(range(len(case_ids)))
    axis.set_yticklabels(case_ids)
    for row_index, case_id in enumerate(case_ids):
        for column_index, variant_name in enumerate(variant_names):
            value = report.case_matrix.rows[case_id].get(variant_name)
            axis.text(
                column_index,
                row_index,
                _display_value(value),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    axis.figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)


def _render_leaderboard_grouped_bar(
    axis: Axes,
    report: BenchmarkReport,
    *,
    title: str,
) -> None:
    metric_names = _numeric_leaderboard_metric_names(report)
    if not metric_names:
        _render_leaderboard_table(axis, report, title=title)
        return
    variant_labels = [row.variant_id for row in report.leaderboard]
    positions = list(range(len(variant_labels)))
    width = min(0.8 / max(len(metric_names), 1), 0.18)
    offset_start = -width * (len(metric_names) - 1) / 2
    for metric_index, metric_name in enumerate(metric_names):
        values = [_numeric_value(row.metrics.get(metric_name)) or 0.0 for row in report.leaderboard]
        axis.bar(
            [position + offset_start + metric_index * width for position in positions],
            values,
            width=width,
            label=metric_name,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(variant_labels, rotation=15)
    axis.set_title(title)
    axis.legend(fontsize=8)


def _render_leaderboard_line(
    axis: Axes,
    report: BenchmarkReport,
    visual: LeaderboardVisualSpec,
) -> None:
    metric_name = _select_numeric_metric_name(report, visual.metric)
    if metric_name is None:
        _render_leaderboard_table(axis, report, title=visual.title or "Leaderboard")
        return
    labels = [row.variant_id for row in report.leaderboard]
    values = [_numeric_value(row.metrics.get(metric_name)) or 0.0 for row in report.leaderboard]
    axis.plot(labels, values, marker="o", color="#4C78A8")
    axis.set_title(visual.title or f"Leaderboard Trend: {metric_name}")
    axis.set_ylabel(metric_name)
    axis.tick_params(axis="x", rotation=15)


def _render_leaderboard_pie(
    axis: Axes,
    report: BenchmarkReport,
    visual: LeaderboardVisualSpec,
) -> None:
    metric_name = _select_numeric_metric_name(report, visual.metric)
    if metric_name is None:
        _render_leaderboard_table(axis, report, title=visual.title or "Leaderboard")
        return
    labels = [row.variant_id for row in report.leaderboard]
    values = [
        max(_numeric_value(row.metrics.get(metric_name)) or 0.0, 0.0) for row in report.leaderboard
    ]
    if not values or sum(values) <= 0:
        _render_leaderboard_table(axis, report, title=visual.title or "Leaderboard")
        return
    axis.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
    axis.set_title(visual.title or f"Leaderboard Share: {metric_name}")


def _render_case_matrix_grouped_bar(
    axis: Axes,
    report: BenchmarkReport,
    *,
    title: str,
) -> None:
    variant_names = sorted(
        {variant_name for row in report.case_matrix.rows.values() for variant_name in row}
    )
    case_ids = sorted(report.case_matrix.rows)
    positions = list(range(len(case_ids)))
    width = min(0.8 / max(len(variant_names), 1), 0.22)
    offset_start = -width * (len(variant_names) - 1) / 2
    for variant_index, variant_name in enumerate(variant_names):
        values = [
            _numeric_value(report.case_matrix.rows[case_id].get(variant_name)) or 0.0
            for case_id in case_ids
        ]
        axis.bar(
            [position + offset_start + variant_index * width for position in positions],
            values,
            width=width,
            label=variant_name,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(case_ids, rotation=20)
    axis.set_title(title)
    axis.set_ylabel(report.case_matrix.metric)
    axis.legend(fontsize=8)


def _render_case_matrix_line(
    axis: Axes,
    report: BenchmarkReport,
    *,
    title: str,
) -> None:
    variant_names = sorted(
        {variant_name for row in report.case_matrix.rows.values() for variant_name in row}
    )
    case_ids = sorted(report.case_matrix.rows)
    for variant_name in variant_names:
        values = [
            _numeric_value(report.case_matrix.rows[case_id].get(variant_name)) or 0.0
            for case_id in case_ids
        ]
        axis.plot(case_ids, values, marker="o", label=variant_name)
    axis.set_title(title)
    axis.set_ylabel(report.case_matrix.metric)
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=8)


def _render_table(axis: Axes, headers: list[str], rows: list[list[str]], *, title: str) -> None:
    axis.set_title(title)
    if not rows:
        rows = [["-" for _ in headers]]
    table = axis.table(cellText=rows, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    axis.axis("off")


def _find_comparison(
    report: BenchmarkReport, visual: ComparisonVisualSpec
) -> ComparisonReport | None:
    for comparison in report.comparisons:
        if comparison.baseline == visual.baseline and comparison.candidate == visual.candidate:
            return comparison
    return None


def _find_distribution(report: BenchmarkReport, name: str) -> MetricDistribution | None:
    for distribution in report.distributions:
        if distribution.name == name:
            return distribution
    return None


def _ordered_metric_names(report: BenchmarkReport) -> list[str]:
    names: list[str] = []
    for row in report.leaderboard:
        for metric_name in row.metrics:
            if metric_name not in names:
                names.append(metric_name)
    return names


def _ordered_summary_names(distribution: MetricDistribution) -> list[str]:
    names: list[str] = []
    for summaries in distribution.summaries.values():
        for summary_name in summaries:
            if summary_name not in names:
                names.append(summary_name)
    return names


def _ordered_variant_factor_names(report: BenchmarkReport) -> list[str]:
    names: list[str] = []
    for config in report.variant_configs:
        for factor_name in config.factors:
            if factor_name not in names:
                names.append(factor_name)
    return names


def _select_numeric_metric_name(report: BenchmarkReport, selected_metric: str | None) -> str | None:
    metric_names = _ordered_metric_names(report)
    if selected_metric is not None:
        return selected_metric if _report_metric_is_numeric(report, selected_metric) else None
    for metric_name in metric_names:
        if _report_metric_is_numeric(report, metric_name):
            return metric_name
    return None


def _numeric_leaderboard_metric_names(report: BenchmarkReport) -> list[str]:
    return [
        metric_name
        for metric_name in _ordered_metric_names(report)
        if _report_metric_is_numeric(report, metric_name)
    ]


def _report_metric_is_numeric(report: BenchmarkReport, metric_name: str) -> bool:
    for row in report.leaderboard:
        value = row.metrics.get(metric_name)
        if _numeric_value(value) is not None:
            return True
    return False


def _numeric_metric_delta_names(
    comparison: ComparisonReport,
    *,
    selected_metric: str | None,
) -> list[str]:
    if selected_metric is not None:
        payload = comparison.metric_deltas.get(selected_metric)
        return [selected_metric] if payload is not None and _numeric_metric_delta(payload) else []
    return [
        metric_name
        for metric_name, payload in comparison.metric_deltas.items()
        if _numeric_metric_delta(payload)
    ]


def _numeric_metric_delta(payload: dict[str, Any]) -> bool:
    return (
        _numeric_value(payload.get("baseline")) is not None
        and _numeric_value(payload.get("candidate")) is not None
    )


def _case_matrix_is_numeric(report: BenchmarkReport) -> bool:
    for values in report.case_matrix.rows.values():
        for value in values.values():
            if _numeric_value(value) is not None:
                return True
    return False


def _display_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
        return None if isnan(numeric) else numeric
    return None


def _nan_if_none(value: float | None) -> float:
    if value is None:
        return float("nan")
    return value


def _visual_slug(visual: VisualizationSpec) -> str:
    title = getattr(visual, "title", None)
    base = str(title or getattr(visual, "kind", "visual"))
    slug = "".join(character.lower() if character.isalnum() else "-" for character in base)
    return "-".join(part for part in slug.split("-") if part) or "visual"


__all__ = (
    "default_visual_specs",
    "export_png_report",
    "export_png_report_set",
    "render_report_figure",
)
