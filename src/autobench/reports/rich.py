from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autobench.instrumentation.registry import InstrumentorStatus
from autobench.protocol.traces import Trace
from autobench.reports.exporting import CSV_METRICS
from autobench.reports.reporting import (
    BenchmarkReport,
    ComparisonReport,
    build_report,
    metric_value,
)
from autobench.runtime.pipeline import ExperimentResult, RunResult


def render_validation_summary(console: Console, summary: Mapping[str, Any]) -> None:
    console.print(
        Panel.fit(
            Text(str(summary["benchmark_id"]), style="bold green"),
            title="Autobench Spec Valid",
            border_style="green",
        )
    )
    table = _kv_table("Validation Summary")
    table.add_row("Path", str(summary["path"]))
    description = summary.get("description")
    if isinstance(description, str) and description:
        table.add_row("Description", description)
    table.add_row("Cases", str(summary["case_count"]))
    table.add_row("Variants", str(summary["variant_count"]))
    table.add_row("Planned runs", str(summary["planned_run_count"]))
    console.print(table)
    warnings = summary.get("warnings")
    if isinstance(warnings, Sequence) and warnings:
        console.print(_warnings_panel([str(warning) for warning in warnings]))


def render_experiment_result(
    console: Console,
    result: ExperimentResult,
    *,
    title: str,
    record_path: Path | None = None,
) -> None:
    console.print(
        Panel.fit(
            Text(result.benchmark_id, style="bold cyan"),
            title=title,
            border_style="cyan",
        )
    )
    table = _kv_table("Run Summary")
    table.add_row("Experiment", result.experiment_id)
    table.add_row("Planned runs", str(result.plan.planned_run_count))
    table.add_row("Runs", str(result.total_count))
    table.add_row("Passed", str(result.passed_count))
    table.add_row("Failed", str(result.failed_count))
    table.add_row("Errored", str(result.errored_count))
    table.add_row("Skipped", str(result.skipped_count))
    if record_path is not None:
        table.add_row("Recorded to", str(record_path))
    console.print(table)
    _render_report_tables(console, build_report(result))
    if any(run.trace is not None for run in result.runs):
        render_trace_summary(console, result)


def render_report(console: Console, report: BenchmarkReport) -> None:
    console.print(
        Panel.fit(
            Text(report.benchmark_id, style="bold cyan"),
            title="Benchmark Report",
            border_style="cyan",
        )
    )
    table = _kv_table("Overview")
    table.add_row("Experiment", report.experiment_id)
    table.add_row("Runs", str(report.run_count))
    console.print(table)
    _render_report_tables(console, report)


def render_comparison(console: Console, comparison: ComparisonReport) -> None:
    console.print(
        Panel.fit(
            Text(f"{comparison.baseline} vs {comparison.candidate}", style="bold magenta"),
            title="Variant Comparison",
            border_style="magenta",
        )
    )
    summary = _kv_table("Comparison Summary")
    summary.add_row("Baseline", comparison.baseline)
    summary.add_row("Candidate", comparison.candidate)
    summary.add_row("Paired runs", str(comparison.run_count))
    summary.add_row("Confounded", _format_value(comparison.confounded))
    console.print(summary)

    factor_table = _data_table("Factor Deltas")
    factor_table.add_column("Factor", style="bold")
    factor_table.add_column("Baseline")
    factor_table.add_column("Candidate")
    if comparison.factor_deltas:
        for factor_name, payload in sorted(comparison.factor_deltas.items()):
            factor_table.add_row(
                _display_column_name(factor_name),
                _format_value(payload.get("baseline")),
                _format_value(payload.get("candidate")),
            )
    else:
        factor_table.add_row("No factor differences", "", "")
    console.print(factor_table)

    metric_table = _data_table("Metric Deltas")
    metric_table.add_column("Metric", style="bold")
    metric_table.add_column("Baseline", justify="right")
    metric_table.add_column("Candidate", justify="right")
    metric_table.add_column("Delta", justify="right")
    for metric_name, payload in sorted(comparison.metric_deltas.items()):
        metric_table.add_row(
            _display_column_name(metric_name),
            _format_value(payload.get("baseline")),
            _format_value(payload.get("candidate")),
            _format_value(payload.get("delta")),
        )
    console.print(metric_table)


def render_export_preview(
    console: Console,
    result: ExperimentResult,
    *,
    export_format: str,
    output_path: Path,
) -> None:
    console.print(
        Panel.fit(
            Text(str(output_path), style="bold green"),
            title=f"Exported {export_format.upper()}",
            border_style="green",
        )
    )
    if export_format == "csv":
        _render_runs_preview(console, result.runs)
        return
    render_report(console, build_report(result))


def render_model_configurations(
    console: Console,
    model_pairs: Sequence[tuple[str, str]],
    *,
    title: str = "Model Configurations",
) -> None:
    table = _data_table(title)
    table.add_column("Configuration", style="bold")
    table.add_column("Spec Model", style="cyan")
    table.add_column("Exploration Model", style="green")
    for index, (spec_model, exploration_model) in enumerate(model_pairs, start=1):
        table.add_row(
            f"model_pair_{index}",
            _model_short(spec_model),
            _model_short(exploration_model),
        )
    console.print(table)


def render_instrumentor_statuses(
    console: Console,
    statuses: Sequence[InstrumentorStatus],
) -> None:
    """Render native integration availability, compatibility, and capture defaults."""

    table = _data_table("ABP Instrumentation Doctor")
    table.add_column("Integration", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Target", no_wrap=True)
    table.add_column("Layer", no_wrap=True)
    table.add_column("Mechanism", no_wrap=True)
    for status in statuses:
        compatibility = status.compatibility
        info = status.info
        target_version = compatibility.target_version or "not installed"
        target = f"{info.target_distribution or '-'} {target_version}"
        table.add_row(
            status.name,
            compatibility.status.value,
            target,
            info.layer.value,
            info.mechanism.value,
        )
    console.print(table)

    capture_table = _data_table("Capture Defaults")
    capture_table.add_column("Integration", style="bold", no_wrap=True)
    capture_table.add_column("Policy")
    for status in statuses:
        capture_table.add_row(status.name, status.capture_mode)
    console.print(capture_table)

    capabilities_table = _data_table("Instrumentation Capabilities")
    capabilities_table.add_column("Integration", style="bold", no_wrap=True)
    capabilities_table.add_column("Runtime", no_wrap=True)
    capabilities_table.add_column("Span kinds")
    capabilities_table.add_column("Semantic families")
    for status in statuses:
        info = status.info
        capabilities = ["sync"]
        if info.capabilities.async_:
            capabilities.append("async")
        if info.capabilities.streaming:
            capabilities.append("stream")
        if info.capabilities.native_hooks:
            capabilities.append("native hook")
        capabilities_table.add_row(
            status.name,
            ", ".join(capabilities),
            ", ".join(info.span_kinds),
            ", ".join(info.semantic_families),
        )
    console.print(capabilities_table)

    diagnostics = _data_table("Compatibility Diagnostics")
    diagnostics.add_column("Integration", style="bold", no_wrap=True)
    diagnostics.add_column("Extra", no_wrap=True)
    diagnostics.add_column("Details")
    for status in statuses:
        messages = (
            *status.compatibility.diagnostics,
            *status.compatibility.conflicts,
        )
        diagnostics.add_row(
            status.name,
            Text(f"autobench[{status.extra}]"),
            "\n".join(messages) or "No compatibility issues detected.",
        )
    console.print(diagnostics)


def render_trace_summary(console: Console, result: ExperimentResult) -> None:
    """Render recorded ABP trace shape, lifecycle state, and diagnostics."""

    table = _data_table("ABP Trace Summary")
    table.add_column("Case", style="bold", no_wrap=True)
    table.add_column("Variant", style="bold", no_wrap=True)
    table.add_column("Spans", justify="right", no_wrap=True)
    table.add_column("Roots", justify="right", no_wrap=True)
    table.add_column("Partial", no_wrap=True)
    table.add_column("Diagnostics", justify="right", no_wrap=True)
    traces: list[Trace] = []
    for run in result.runs:
        if run.trace is None:
            continue
        traces.append(run.trace)
        table.add_row(
            run.case_id,
            run.variant_id,
            str(len(run.trace.spans)),
            str(len(run.trace.root_span_ids)),
            "yes" if run.trace.partial else "no",
            str(len(run.trace.diagnostics)),
        )
    console.print(table)

    kinds: dict[str, int] = {}
    scopes: dict[str, int] = {}
    for trace in traces:
        for span in trace.spans:
            kinds[span.kind] = kinds.get(span.kind, 0) + 1
            name = span.scope.instrumentor_name
            scopes[name] = scopes.get(name, 0) + 1
    shape = _data_table("Trace Composition")
    shape.add_column("Dimension", style="bold", no_wrap=True)
    shape.add_column("Value")
    shape.add_column("Spans", justify="right", no_wrap=True)
    for kind, count in sorted(kinds.items()):
        shape.add_row("kind", kind, str(count))
    for scope, count in sorted(scopes.items()):
        shape.add_row("instrumentor", scope, str(count))
    console.print(shape)


def render_recorded_runs(console: Console, runs: Sequence[RunResult]) -> None:
    _render_runs_preview(console, runs)


def _render_report_tables(
    console: Console,
    report: BenchmarkReport,
) -> None:
    console.print(_status_table(report))
    console.print(_variant_config_table(report))
    for table in _leaderboard_tables(report):
        console.print(table)
    console.print(_case_matrix_table(report))
    for table in _run_metric_tables(report):
        console.print(table)
    for comparison in report.comparisons:
        render_comparison(console, comparison)
    for distribution in report.distributions:
        table = _data_table(f"Distribution: {distribution.name} ({distribution.semantic_type})")
        table.add_column("Variant", style="bold", no_wrap=True)
        table.add_column("Samples", justify="right", no_wrap=True)
        summary_names = _ordered_keys(distribution.summaries)
        for summary_name in summary_names:
            table.add_column(summary_name, justify="right", no_wrap=True)
        if not distribution.by_variant:
            table.add_row("No samples", "0", *["-" for _ in summary_names])
        for variant_id, values in distribution.by_variant.items():
            row = [variant_id, str(len(values))]
            summaries = distribution.summaries.get(variant_id, {})
            row.extend(_format_value(summaries.get(summary_name)) for summary_name in summary_names)
            table.add_row(*row)
        console.print(table)


def _leaderboard_tables(report: BenchmarkReport) -> list[Table]:
    metric_names = _ordered_metric_names(report)
    return [
        _leaderboard_table(report, title=title, metric_names=metric_group)
        for title, metric_group in _metric_groups(
            metric_names,
            default_title="Leaderboard",
            groups=(
                (
                    "Leaderboard: Effectiveness",
                    (
                        "pass_rate",
                        "avg_coverage",
                    ),
                ),
                (
                    "Leaderboard: Time and Cost",
                    (
                        "median_latency_s",
                        "p95_latency_s",
                        "total_cost_usd",
                        "avg_cost_usd",
                    ),
                ),
                (
                    "Leaderboard: LLM Totals",
                    (
                        "total_input_tokens",
                        "total_output_tokens",
                        "total_request_count",
                    ),
                ),
                (
                    "Leaderboard: LLM Averages",
                    (
                        "avg_input_tokens",
                        "avg_output_tokens",
                        "avg_request_count",
                    ),
                ),
                (
                    "Leaderboard: Diagnostics",
                    (
                        "avg_case_count",
                        "avg_raises_count",
                        "avg_snippet_count",
                        "avg_error_snippet_count",
                        "avg_refinement_rounds",
                    ),
                ),
            ),
        )
    ]


def _leaderboard_table(
    report: BenchmarkReport,
    *,
    title: str,
    metric_names: Sequence[str],
) -> Table:
    table = _data_table(title)
    table.add_column("Variant", style="bold", no_wrap=True)
    table.add_column("Runs", justify="right", no_wrap=True)
    for metric_name in metric_names:
        table.add_column(_display_column_name(metric_name), justify="right", no_wrap=True)
    for row in report.leaderboard:
        rendered = [row.variant_id, str(row.run_count)]
        rendered.extend(_format_value(row.metrics.get(metric_name)) for metric_name in metric_names)
        table.add_row(*rendered)
    return table


def _status_table(report: BenchmarkReport) -> Table:
    table = _data_table("Run Status")
    table.add_column("Status", style="bold", no_wrap=True)
    table.add_column("Count", justify="right", no_wrap=True)
    table.add_column("Share", justify="right", no_wrap=True)
    total = sum(report.status_counts.values()) or 1
    for status, count in report.status_counts.items():
        table.add_row(status, str(count), f"{count / total:.1%}")
    return table


def _variant_config_table(report: BenchmarkReport) -> Table:
    table = _data_table("Variant Configurations")
    table.add_column("Variant", style="bold", no_wrap=True)
    table.add_column("Label", overflow="fold")
    factor_names = _ordered_variant_factor_names(report)
    for factor_name in factor_names:
        table.add_column(_display_column_name(factor_name), overflow="fold")
    for config in report.variant_configs:
        table.add_row(
            config.variant_id,
            config.label or "-",
            *[_format_value(config.factors.get(factor_name)) for factor_name in factor_names],
        )
    return table


def _case_matrix_table(report: BenchmarkReport) -> Table:
    table = _data_table(f"Case Matrix ({report.case_matrix.metric})")
    table.add_column("Case", style="bold", no_wrap=True)
    variant_names = sorted(
        {variant_name for row in report.case_matrix.rows.values() for variant_name in row}
    )
    for variant_name in variant_names:
        table.add_column(variant_name, justify="right", no_wrap=True)
    for case_id, values in sorted(report.case_matrix.rows.items()):
        table.add_row(
            case_id,
            *[_format_value(values.get(variant_name)) for variant_name in variant_names],
        )
    return table


def _run_metric_tables(report: BenchmarkReport) -> list[Table]:
    metric_names = _ordered_run_metric_names(report)
    return [
        _run_metric_table(report, title=title, metric_names=metric_group)
        for title, metric_group in _metric_groups(
            metric_names,
            default_title="Run Metrics",
            groups=(
                (
                    "Run Metrics: Core",
                    (
                        "success",
                        "coverage",
                        "latency",
                        "cost_usd",
                    ),
                ),
                (
                    "Run Metrics: LLM Usage",
                    (
                        "input_tokens",
                        "output_tokens",
                        "request_count",
                    ),
                ),
                (
                    "Run Metrics: Eval Shape",
                    (
                        "case_count",
                        "raises_count",
                    ),
                ),
                (
                    "Run Metrics: Exploration",
                    (
                        "snippet_count",
                        "error_snippet_count",
                        "refinement_rounds",
                        "function_count",
                        "cost_per_function_usd",
                    ),
                ),
            ),
        )
    ]


def _run_metric_table(
    report: BenchmarkReport,
    *,
    title: str,
    metric_names: Sequence[str],
) -> Table:
    table = _data_table(title)
    table.add_column("Case", style="bold", no_wrap=True)
    table.add_column("Variant", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    for metric_name in metric_names:
        table.add_column(_display_column_name(metric_name), justify="right", no_wrap=True)
    for row in report.run_metrics:
        table.add_row(
            row.case_id,
            row.variant_id,
            row.status,
            *[_format_value(row.metrics.get(metric_name)) for metric_name in metric_names],
        )
    return table


def _render_runs_preview(console: Console, runs: Sequence[RunResult]) -> None:
    table = _data_table("Recorded Runs Preview")
    table.add_column("Run", style="dim", no_wrap=True)
    table.add_column("Case", style="bold", no_wrap=True)
    table.add_column("Variant", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    for metric_name, _ in CSV_METRICS:
        table.add_column(_display_column_name(metric_name), justify="right", no_wrap=True)
    for run in runs:
        row = [run.run_id, run.case_id, run.variant_id, run.status.value]
        row.extend(
            _format_value(metric_value(run, semantic_type)) for _, semantic_type in CSV_METRICS
        )
        table.add_row(*row)
    console.print(table)


def _warnings_panel(warnings: Sequence[str]) -> Panel:
    text = Text()
    for warning in warnings:
        text.append(f"- {warning}\n")
    return Panel(text, title="Warnings", border_style="yellow")


def _kv_table(title: str) -> Table:
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    return table


def _data_table(title: str) -> Table:
    return Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        title_style="bold",
    )


_COLUMN_ALIASES: Mapping[str, str] = {
    "all_passed": "pass",
    "avg_case_count": "cases",
    "avg_cost_usd": "avg cost",
    "avg_coverage": "coverage",
    "avg_error_snippet_count": "err snippets",
    "avg_input_tokens": "avg in tok",
    "avg_output_tokens": "avg out tok",
    "avg_raises_count": "raises",
    "avg_refinement_rounds": "rounds",
    "avg_request_count": "avg req",
    "avg_snippet_count": "snippets",
    "case_count": "cases",
    "cost": "cost",
    "cost_usd": "cost",
    "coverage": "coverage",
    "error_snippet_count": "err snippets",
    "exploration_model": "exploration model",
    "function_count": "functions",
    "input_tokens": "input toks",
    "latency": "latency",
    "median_latency_s": "med latency",
    "model_slug": "model slug",
    "output_tokens": "output toks",
    "p95_latency_s": "p95 latency",
    "pass_rate": "pass rate",
    "raises_count": "raises",
    "refinement_rounds": "rounds",
    "request_count": "requests",
    "snippet_count": "snippets",
    "spec_model": "spec model",
    "success": "success",
    "total_cost": "total cost",
    "total_cost_usd": "total cost",
    "total_input_tokens": "input toks",
    "total_output_tokens": "output toks",
    "total_request_count": "requests",
}


def _display_column_name(name: str) -> str:
    base_name = _metric_base_name(name)
    return _COLUMN_ALIASES.get(base_name, base_name)


def _metric_base_name(name: str) -> str:
    return name.split(" (", 1)[0]


def _metric_groups(
    metric_names: Sequence[str],
    *,
    default_title: str,
    groups: Sequence[tuple[str, Sequence[str]]],
) -> list[tuple[str, list[str]]]:
    if not metric_names:
        return [(default_title, [])]
    remaining = list(metric_names)
    grouped: list[tuple[str, list[str]]] = []
    for title, base_names in groups:
        selected = [
            metric_name for metric_name in remaining if _metric_base_name(metric_name) in base_names
        ]
        if selected:
            grouped.append((title, selected))
            remaining = [metric_name for metric_name in remaining if metric_name not in selected]
    if remaining:
        grouped.append((f"{default_title}: Other", remaining))
    return grouped


def _ordered_metric_names(report: BenchmarkReport) -> list[str]:
    names: list[str] = []
    for row in report.leaderboard:
        for metric_name in row.metrics:
            if metric_name not in names:
                names.append(metric_name)
    return names


def _ordered_variant_factor_names(report: BenchmarkReport) -> list[str]:
    names: list[str] = []
    for config in report.variant_configs:
        for factor_name in config.factors:
            if factor_name not in names:
                names.append(factor_name)
    return names


def _ordered_run_metric_names(report: BenchmarkReport) -> list[str]:
    preferred_prefixes = (
        "success ",
        "coverage ",
        "latency ",
        "cost_usd ",
        "input_tokens ",
        "output_tokens ",
        "request_count ",
        "case_count ",
        "raises_count ",
        "snippet_count ",
        "error_snippet_count ",
        "refinement_rounds ",
    )
    names: list[str] = []
    for row in report.run_metrics:
        for metric_name in row.metrics:
            if metric_name not in names:
                names.append(metric_name)
    preferred = [
        metric_name
        for prefix in preferred_prefixes
        for metric_name in names
        if metric_name.startswith(prefix)
    ]
    return [*preferred, *[metric_name for metric_name in names if metric_name not in preferred]]


def _ordered_keys(values: Mapping[str, Mapping[str, Any]]) -> list[str]:
    keys: list[str] = []
    for item in values.values():
        for key in item:
            if key not in keys:
                keys.append(key)
    return keys


def _model_short(model_name: str) -> str:
    return model_name.split("/")[-1]


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = (
    "render_comparison",
    "render_experiment_result",
    "render_export_preview",
    "render_model_configurations",
    "render_recorded_runs",
    "render_report",
    "render_validation_summary",
)
