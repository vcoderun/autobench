from __future__ import annotations as _annotations

import hashlib
import html
import json
import os
import re
import shutil
from collections.abc import Iterable, Sequence
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import quote

from pydantic import JsonValue

from autobench.errors import AutobenchError
from autobench.records.files import (
    ExperimentFile,
    RecordFileKind,
    atomic_write_text,
    create_temporary_record_directory,
    hash_and_size,
    remove_temporary_record_directory,
)
from autobench.records.models import ExperimentRecord
from autobench.reports.charts import (
    render_case_score_chart,
    render_dimension_chart,
    render_quality_gate_chart,
)
from autobench.reports.models import (
    AssetVersionReport,
    BenchmarkReport,
    EvaluationCaseReport,
    EvaluationMetricReport,
    EvaluationSummaryReport,
    MarkdownReportPublication,
    MetricComparisonReport,
    PublishedReportFile,
    ReportLayout,
    ReportSpec,
    RunDetailReport,
)
from autobench.runtime.models import ExperimentResult

_AUTO_BUNDLE_RUNS = 50
_AUTO_BUNDLE_DETAILS = 100
_AUTO_BUNDLE_MATRIX_CELLS = 1_000


class ReportPublicationError(AutobenchError):
    """Raised when a Markdown report cannot be published safely."""


class MarkdownExperimentPublisher:
    def __call__(
        self,
        result: ExperimentResult,
        record: ExperimentRecord,
        experiment_root: Path,
    ) -> tuple[ExperimentFile, ...]:
        from autobench.reports.reporting import build_report

        if result.report_spec_data is None:
            return ()
        report_spec = ReportSpec.model_validate(result.report_spec_data)
        output = report_spec.markdown.output
        if output is None:
            return ()
        report = build_report(
            result,
            report_spec=report_spec,
            experiment_record=record,
            experiment_root=experiment_root,
        )
        layout = _select_layout(report, report_spec.markdown.layout)
        if layout == "single":
            link_prefix = _record_link_prefix(
                (experiment_root / output).parent,
                experiment_root,
            )
            return (
                ExperimentFile(
                    path=output.as_posix(),
                    content=render_markdown_report(
                        report,
                        record_link_prefix=link_prefix,
                    ).encode(),
                    kind=RecordFileKind.OTHER,
                    identity=f"report:{report.experiment_id}:index",
                ),
            )
        link_prefix = _record_link_prefix(experiment_root / output, experiment_root)
        return tuple(
            ExperimentFile(
                path=(output / relative_path).as_posix(),
                content=content.encode(),
                kind=RecordFileKind.OTHER,
                identity=f"report:{report.experiment_id}:{relative_path}",
            )
            for relative_path, content in render_markdown_bundle(
                report,
                record_link_prefix=link_prefix,
            ).items()
        )


def render_markdown_report(
    report: BenchmarkReport,
    *,
    record_link_prefix: str = "",
) -> str:
    return _render_markdown_document(
        report,
        record_link_prefix=record_link_prefix,
        bundle_index=False,
    )


def _render_markdown_document(
    report: BenchmarkReport,
    *,
    record_link_prefix: str,
    bundle_index: bool,
) -> str:
    lines = [f"# {_text(report.benchmark_id)}", ""]
    _render_report_context(lines, report)
    _render_executive_summary(lines, report)
    _render_evaluation(lines, report)
    _render_variants(lines, report)
    _render_leaderboard(lines, report)
    _render_comparisons(lines, report)
    _render_regressions(lines, report)
    _render_policies(lines, report)
    _render_execution_issues(lines, report)
    _render_optimizations(lines, report)

    if report.markdown.profile != "summary":
        _render_methodology(lines, report)
        _render_evaluator_feedback(lines, report)
        _render_case_matrix(lines, report)
        _render_distributions(lines, report)
    if report.markdown.profile == "audit":
        _render_audit_identity(lines, report)
        _render_design(lines, report)
        _render_metric_catalog(lines, report)
        _render_health(lines, report)
        if not bundle_index:
            _render_runs(lines, report)
            _render_failures(lines, report)
            _render_audit_evidence(lines, report)
        _render_traces(lines, report)
        if not bundle_index:
            _render_assets(lines, report)
        _render_artifacts(lines, report, record_link_prefix=record_link_prefix)
        _render_provenance(lines, report)
    _render_limitations(lines, report)
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(
    report: BenchmarkReport,
    path: Path,
    *,
    layout: ReportLayout | None = None,
    overwrite: bool = False,
    immutable_root: Path | None = None,
) -> MarkdownReportPublication:
    requested_layout = report.markdown.layout if layout is None else layout
    selected_layout = _select_layout(report, requested_layout)
    _validate_destination(path, immutable_root=immutable_root)
    if path.is_symlink():
        raise ReportPublicationError(f"Markdown report destination cannot be a symlink: {path}")
    if selected_layout == "single":
        if path.exists() and (path.is_dir() or not overwrite):
            raise ReportPublicationError(f"Markdown report destination already exists: {path}")
        content = render_markdown_report(
            report,
            record_link_prefix=_record_link_prefix(path.parent, immutable_root),
        )
        atomic_write_text(path, content)
        digest, byte_count = hash_and_size(path)
        files = (PublishedReportFile(path=path, sha256=digest, byte_count=byte_count),)
    else:
        files = _write_bundle(
            report,
            path,
            overwrite=overwrite,
            record_link_prefix=_record_link_prefix(path, immutable_root),
        )
    return MarkdownReportPublication(
        profile=report.markdown.profile,
        requested_layout=requested_layout,
        layout=selected_layout,
        destination=path,
        files=files,
    )


def render_markdown_bundle(
    report: BenchmarkReport,
    *,
    record_link_prefix: str = "",
) -> dict[str, str]:
    pages: dict[str, str] = {}
    include_technical_pages = report.markdown.profile == "audit"
    case_paths = {
        case_id: f"cases/{_page_slug(case_id)}.md"
        for case_id in sorted({run.case_id for run in report.run_details})
    }
    variant_paths = {
        variant.variant_id: f"variants/{_page_slug(variant.variant_id)}.md"
        for variant in sorted(report.variant_configs, key=lambda item: item.variant_id)
    }
    run_paths = (
        {
            run.run_id: f"runs/{_page_slug(run.run_id)}.md"
            for run in sorted(report.run_details, key=lambda item: item.run_id)
        }
        if include_technical_pages
        else {}
    )
    asset_ids = (
        () if report.assets is None else sorted({item.asset_id for item in report.assets.versions})
    )
    asset_paths = (
        {asset_id: f"assets/{_page_slug(asset_id)}.md" for asset_id in asset_ids}
        if include_technical_pages
        else {}
    )
    all_paths = [
        "index.md",
        *case_paths.values(),
        *variant_paths.values(),
        *run_paths.values(),
        *asset_paths.values(),
    ]
    if len(all_paths) != len(set(all_paths)):
        raise ReportPublicationError("Normalized Markdown bundle paths collide.")

    index = _render_markdown_document(
        report,
        record_link_prefix=record_link_prefix,
        bundle_index=True,
    ).rstrip()
    page_groups = (
        ("Cases", case_paths),
        ("Variants", variant_paths),
        ("Runs", run_paths),
        ("Assets", asset_paths),
    )
    index_lines = [index, "", "## Report Pages", ""]
    for title, paths in page_groups:
        if not paths:
            continue
        index_lines.extend((f"### {title}", ""))
        index_lines.extend(
            f"- [{_text(identity)}]({_link_target(page)})" for identity, page in paths.items()
        )
        index_lines.append("")
    pages["index.md"] = "\n".join(index_lines).rstrip() + "\n"
    pages.update(
        (page, _render_case_page(report, case_id, run_paths))
        for case_id, page in case_paths.items()
    )
    pages.update(
        (page, _render_variant_page(report, variant_id, run_paths))
        for variant_id, page in variant_paths.items()
    )
    if run_paths:
        run_link_prefix = (PurePosixPath("..") / record_link_prefix).as_posix()
        pages.update(
            (page, _render_run_page(report, run_id, record_link_prefix=run_link_prefix))
            for run_id, page in run_paths.items()
        )
    pages.update(
        (page, _render_asset_page(report, asset_id)) for asset_id, page in asset_paths.items()
    )
    return dict(sorted(pages.items()))


def _select_layout(
    report: BenchmarkReport,
    requested: ReportLayout,
) -> Literal["single", "bundle"]:
    if requested != "auto":
        return requested
    if report.markdown.profile == "audit":
        detail_count = len(report.run_details) + len(report.failures)
        detail_count += 0 if report.assets is None else len(report.assets.versions)
        detail_count += 0 if report.artifacts is None else len(report.artifacts.artifacts)
    else:
        detail_count = 0 if report.evaluation is None else len(report.evaluation.cases)
    matrix_cells = sum(len(row) for row in report.case_matrix.rows.values())
    if (
        report.run_count > _AUTO_BUNDLE_RUNS
        or detail_count > _AUTO_BUNDLE_DETAILS
        or matrix_cells > _AUTO_BUNDLE_MATRIX_CELLS
    ):
        return "bundle"
    return "single"


def _validate_destination(path: Path, *, immutable_root: Path | None) -> None:
    if immutable_root is None:
        return
    destination = path.resolve()
    root = immutable_root.resolve()
    if destination == root or destination.is_relative_to(root):
        raise ReportPublicationError(
            "Post-hoc Markdown reports must be written outside the immutable experiment record."
        )


def _write_bundle(
    report: BenchmarkReport,
    destination: Path,
    *,
    overwrite: bool,
    record_link_prefix: str,
) -> tuple[PublishedReportFile, ...]:
    if destination.exists() and (not destination.is_dir() or not overwrite):
        raise ReportPublicationError(f"Markdown report destination already exists: {destination}")
    pages = render_markdown_bundle(report, record_link_prefix=record_link_prefix)
    backup = destination.with_name(f".{destination.name}.replaced")
    if backup.exists() or backup.is_symlink():
        raise ReportPublicationError(f"Markdown report backup path already exists: {backup}")
    staging = create_temporary_record_directory(destination)
    try:
        for relative_path, content in pages.items():
            atomic_write_text(staging / relative_path, content)
        replaced = False
        if destination.exists():
            os.replace(destination, backup)
            replaced = True
        try:
            os.replace(staging, destination)
        except BaseException:
            if replaced:
                os.replace(backup, destination)
            raise
        if replaced:
            shutil.rmtree(backup)
    finally:
        remove_temporary_record_directory(staging)
    return tuple(
        PublishedReportFile(
            path=path,
            sha256=digest,
            byte_count=byte_count,
        )
        for path in sorted(item for item in destination.rglob("*") if item.is_file())
        for digest, byte_count in (hash_and_size(path),)
    )


def _render_case_page(
    report: BenchmarkReport,
    case_id: str,
    run_paths: dict[str, str],
) -> str:
    lines = [f"# Case: {_text(case_id)}", "", "[Back to report](../index.md)", ""]
    evaluation = report.evaluation
    if evaluation is not None:
        evaluation_cases = tuple(case for case in evaluation.cases if case.case_id == case_id)
        if evaluation_cases:
            metric_by_name = {metric.name: metric for metric in evaluation.metrics}
            _append_table(
                lines,
                ("Variant", "Result", "Score", "Key diagnostics"),
                [
                    (
                        case.variant_id,
                        "Pass"
                        if case.quality_pass is True
                        else "Fail"
                        if case.quality_pass is False
                        else "Not evaluated",
                        _percentage_or_value(case.score),
                        _case_diagnostics(case, metric_by_name),
                    )
                    for case in evaluation_cases
                ],
                numeric_columns={2},
            )
            feedback = tuple(message for case in evaluation_cases for message in case.feedback)
            if feedback:
                lines.extend(["", "## Evaluator Feedback", ""])
                lines.extend(f"- {_text(message)}" for message in feedback)
            lines.append("")
    matrix = report.case_matrix.rows.get(case_id, {})
    if matrix and any(value is not None for value in matrix.values()):
        lines.extend(["## Configured Metrics", ""])
        _append_table(
            lines,
            ("Variant", report.case_matrix.metric),
            [(variant_id, _format_value(value)) for variant_id, value in sorted(matrix.items())],
            numeric_columns={1},
        )
        lines.append("")
    runs = sorted(
        (run for run in report.run_details if run.case_id == case_id),
        key=lambda item: item.run_id,
    )
    if run_paths and runs:
        lines.extend(["## Technical Runs", ""])
        _append_table(
            lines,
            ("Run", "Variant", "Status", "Partial"),
            [
                (
                    _link(run.run_id, f"../{run_paths[run.run_id]}"),
                    run.variant_id,
                    run.status,
                    "yes" if run.partial else "",
                )
                for run in runs
            ],
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_variant_page(
    report: BenchmarkReport,
    variant_id: str,
    run_paths: dict[str, str],
) -> str:
    lines = [f"# Variant: {_text(variant_id)}", "", "[Back to report](../index.md)", ""]
    variant = next(item for item in report.variant_configs if item.variant_id == variant_id)
    if variant.factor_details:
        _append_table(
            lines,
            ("factor", "value", "semantic type", "optimize"),
            [
                (
                    factor.name,
                    _format_value(factor.value),
                    factor.semantic_type or "",
                    "yes" if factor.optimize else "",
                )
                for factor in variant.factor_details
            ],
        )
        lines.append("")
    leaderboard = next((row for row in report.leaderboard if row.variant_id == variant_id), None)
    if leaderboard is not None and leaderboard.metric_details:
        _append_table(
            lines,
            ("metric", "value", "samples", "missing", "direction", "best"),
            [
                (
                    metric.name,
                    _format_value(metric.value, unit=metric.unit),
                    str(metric.sample_count),
                    str(metric.missing_count),
                    metric.direction or "",
                    "yes" if metric.best else "",
                )
                for metric in leaderboard.metric_details
            ],
            numeric_columns={1, 2, 3},
        )
        lines.append("")
    evaluation_cases = (
        ()
        if report.evaluation is None
        else tuple(case for case in report.evaluation.cases if case.variant_id == variant_id)
    )
    if evaluation_cases:
        lines.extend(["## Case Outcomes", ""])
        _append_table(
            lines,
            ("Case", "Result", "Score"),
            [
                (
                    case.case_id,
                    "Pass"
                    if case.quality_pass is True
                    else "Fail"
                    if case.quality_pass is False
                    else "Not evaluated",
                    _percentage_or_value(case.score),
                )
                for case in evaluation_cases
            ],
            numeric_columns={2},
        )
        lines.append("")
    runs = sorted(
        (run for run in report.run_details if run.variant_id == variant_id),
        key=lambda item: item.run_id,
    )
    if runs and run_paths:
        lines.extend(["## Technical Runs", ""])
        _append_table(
            lines,
            ("Run", "Case", "Status", "Partial"),
            [
                (
                    _link(run.run_id, f"../{run_paths[run.run_id]}"),
                    run.case_id,
                    run.status,
                    "yes" if run.partial else "",
                )
                for run in runs
            ],
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_run_page(
    report: BenchmarkReport,
    run_id: str,
    *,
    record_link_prefix: str,
) -> str:
    run = next(item for item in report.run_details if item.run_id == run_id)
    lines = [f"# Run: {_text(run_id)}", "", "[Back to report](../index.md)", ""]
    _append_table(
        lines,
        ("property", "value"),
        [
            ("case", run.case_id),
            ("variant", run.variant_id),
            ("status", run.status),
            ("evaluation", run.evaluation_status),
            ("partial", "yes" if run.partial else "no"),
            ("end reason", run.end_reason),
            ("parent run", run.parent_run_id or "none"),
            ("spans", str(run.span_count)),
            ("assets", str(run.asset_count)),
            ("artifacts", str(run.artifact_count)),
        ],
        numeric_columns={1},
    )
    if run.metrics:
        lines.extend(["", "## Metrics", ""])
        _append_table(
            lines,
            ("metric", "value"),
            [(name, _format_value(value)) for name, value in sorted(run.metrics.items())],
            numeric_columns={1},
        )
    failures = [failure for failure in report.failures if failure.run_id == run_id]
    if failures:
        lines.extend(["", "## Failures", ""])
        _append_table(
            lines,
            ("stage", "error", "message", "span"),
            [
                (failure.stage, failure.error_type, failure.message, failure.span_id or "")
                for failure in failures
            ],
        )
        for failure in failures:
            if failure.traceback is None:
                continue
            lines.extend(
                [
                    "",
                    "<details>",
                    f"<summary>{_text(failure.stage)} traceback</summary>",
                    "",
                    "```text",
                    _fenced_text(failure.traceback),
                    "```",
                    "</details>",
                ]
            )
    if report.artifacts is not None:
        artifacts = [
            artifact for artifact in report.artifacts.artifacts if artifact.run_id == run_id
        ]
        if artifacts:
            lines.extend(["", "## Artifacts", ""])
            _append_table(
                lines,
                ("artifact", "name", "state", "bytes", "path"),
                [
                    (
                        artifact.artifact_id,
                        artifact.name,
                        artifact.state,
                        _format_value(artifact.byte_count),
                        (
                            _link(
                                artifact.path,
                                _record_link(artifact.path, record_link_prefix),
                            )
                            if artifact.path is not None
                            else "unavailable"
                        ),
                    )
                    for artifact in artifacts
                ],
                numeric_columns={3},
            )
    _render_run_audit_evidence(lines, run)
    return "\n".join(lines).rstrip() + "\n"


def _render_asset_page(report: BenchmarkReport, asset_id: str) -> str:
    assert report.assets is not None
    versions = [item for item in report.assets.versions if item.asset_id == asset_id]
    lines = [f"# Asset: {_text(asset_id)}", "", "[Back to report](../index.md)", ""]
    _append_table(
        lines,
        (
            "version",
            "parent",
            "kind",
            "sensitivity",
            "changes",
            "content",
            "diff",
            "variants",
            "runs",
        ),
        [
            (
                version.version,
                version.parent_version or "",
                version.kind or "",
                version.sensitivity or "",
                ", ".join(version.changed_fields),
                version.content_state,
                version.diff_state,
                ", ".join(version.variant_ids),
                str(len(version.run_ids)),
            )
            for version in versions
        ],
        numeric_columns={8},
    )
    for version in versions:
        _render_asset_details(lines, version)
    return "\n".join(lines).rstrip() + "\n"


def _page_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "item"
    suffix = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{normalized[:64]}-{suffix}"


def _link_target(path: str) -> str:
    return quote(path, safe="/-._~")


def _render_report_context(lines: list[str], report: BenchmarkReport) -> None:
    design = report.design
    if design is not None and design.description:
        lines.extend([_text(design.description), ""])
    context = [f"{report.run_count} recorded runs"]
    if design is not None:
        context[0] = f"{design.case_count} cases"
        context.append(f"{design.variant_count} variant{'s' if design.variant_count != 1 else ''}")
    if report.correlation is not None and report.correlation.phase:
        context.append(report.correlation.phase.replace("-", " "))
    lines.extend([" · ".join(f"**{_text(item)}**" for item in context), ""])


def _render_executive_summary(lines: list[str], report: BenchmarkReport) -> None:
    summary = report.summary
    if summary is None:
        return
    lines.extend(["## Executive Summary", "", _text(summary.health), ""])
    for finding in summary.findings:
        if finding.kind == "evaluation":
            continue
        lines.append(f"- **{_text(finding.title)}:** {_text(finding.statement)}")
    lines.append("")


def _render_evaluation(lines: list[str], report: BenchmarkReport) -> None:
    evaluation = report.evaluation
    if evaluation is None:
        return
    lines.extend(["## Benchmark Outcome", ""])
    if evaluation.evaluated_count:
        lines.append(
            f"{evaluation.passed_count} of {evaluation.evaluated_count} evaluated cases met the "
            f"recorded quality gate ({_percentage(evaluation.pass_rate)})."
        )
    else:
        lines.append("No case-level quality gate result was recorded.")
    lines.append("")

    strongest = next(
        (
            case
            for case in sorted(
                evaluation.cases,
                key=lambda item: (
                    item.score is None,
                    -(item.score if item.score is not None else 0),
                    item.case_id,
                ),
            )
            if case.score is not None
        ),
        None,
    )
    weakest = next(
        (
            case
            for case in sorted(
                evaluation.cases,
                key=lambda item: (
                    item.score is None,
                    item.score if item.score is not None else 0,
                    item.case_id,
                ),
            )
            if case.score is not None
        ),
        None,
    )
    _append_table(
        lines,
        ("Measure", "Result"),
        (
            ("Quality gate", _percentage(evaluation.pass_rate)),
            ("Average score", _percentage_or_value(evaluation.mean_score)),
            ("Median score", _percentage_or_value(evaluation.median_score)),
            (
                "Best case",
                "Not scored"
                if strongest is None
                else f"{strongest.case_id} ({_percentage_or_value(strongest.score)})",
            ),
            (
                "Lowest case",
                "Not scored"
                if weakest is None
                else f"{weakest.case_id} ({_percentage_or_value(weakest.score)})",
            ),
            (
                "Cases evaluated",
                f"{evaluation.evaluated_count} of {evaluation.case_count}",
            ),
        ),
        numeric_columns={1},
    )
    lines.append("")
    for chart in (
        render_quality_gate_chart(evaluation),
        render_case_score_chart(evaluation),
    ):
        if chart is not None:
            lines.extend([chart, ""])

    _render_case_results(lines, report, evaluation)
    score_metrics = tuple(metric for metric in evaluation.metrics if metric.kind == "score")
    count_metrics = tuple(metric for metric in evaluation.metrics if metric.kind == "count")
    if score_metrics:
        lines.extend(["### Quality Dimensions", ""])
        chart = render_dimension_chart(evaluation)
        if chart is not None:
            lines.extend([chart, ""])
        _append_table(
            lines,
            ("Dimension", "Average", "Median", "Range", "Cases"),
            [
                (
                    metric.label,
                    _percentage_or_value(metric.mean),
                    _percentage_or_value(metric.median),
                    (
                        f"{_percentage_or_value(metric.minimum)}–"
                        f"{_percentage_or_value(metric.maximum)}"
                    ),
                    str(metric.sample_count),
                )
                for metric in score_metrics
            ],
            numeric_columns={1, 2, 3, 4},
        )
        lines.append("")
    visible_diagnostics = tuple(
        metric for metric in count_metrics if metric.total > 0 and _is_quality_issue(metric.name)
    )
    if visible_diagnostics:
        lines.extend(["### Recorded Quality Issues", ""])
        _append_table(
            lines,
            ("Issue", "Total", "Cases affected", "Worst case"),
            [
                (
                    metric.label,
                    _format_value(metric.total),
                    str(_affected_case_count(evaluation.cases, metric.name)),
                    _format_value(metric.maximum),
                )
                for metric in visible_diagnostics
            ],
            numeric_columns={1, 2, 3},
        )
        lines.append("")


def _render_case_results(
    lines: list[str],
    report: BenchmarkReport,
    evaluation: EvaluationSummaryReport,
) -> None:
    if not evaluation.cases:
        return
    limit = min(
        report.markdown.limits.table_rows, 20 if report.markdown.profile == "summary" else 50
    )
    visible = tuple(
        sorted(
            evaluation.cases,
            key=lambda case: (
                case.quality_pass is not False,
                case.score is None,
                case.score if case.score is not None else 0,
                case.case_id,
                case.variant_id,
            ),
        )[:limit]
    )
    metric_by_name = {metric.name: metric for metric in evaluation.metrics}
    headers = ["Case"]
    if len(report.variant_configs) > 1:
        headers.append("Variant")
    headers.extend(("Result", "Score", "Key diagnostics"))
    rows: list[tuple[str, ...]] = []
    for case in visible:
        row = [case.case_id]
        if len(report.variant_configs) > 1:
            row.append(case.variant_id)
        row.extend(
            (
                "Pass"
                if case.quality_pass is True
                else "Fail"
                if case.quality_pass is False
                else "Not evaluated",
                _percentage_or_value(case.score),
                _case_diagnostics(case, metric_by_name),
            )
        )
        rows.append(tuple(row))
    lines.extend(["### Case Results", ""])
    _append_table(
        lines,
        tuple(headers),
        rows,
        numeric_columns={len(headers) - 2},
    )
    if len(evaluation.cases) > len(visible):
        lines.append(f"\n_Showing {len(visible)} of {len(evaluation.cases)} case results._")
    lines.append("")


def _case_diagnostics(
    case: EvaluationCaseReport,
    metric_by_name: dict[str, EvaluationMetricReport],
) -> str:
    issues: list[tuple[int, float, str]] = []
    for name, value in case.metrics.items():
        metric = metric_by_name.get(name)
        numeric = _number(value)
        if metric is None or numeric is None:
            continue
        if metric.kind == "count" and numeric > 0 and _is_quality_issue(name):
            issues.append((0, -numeric, f"{metric.label}: {_format_value(value)}"))
    if issues:
        return "; ".join(item[2] for item in sorted(issues)[:3])
    if case.quality_pass is False:
        return "Quality gate not met"
    return "No recorded concerns"


def _is_quality_issue(name: str) -> bool:
    normalized = name.casefold()
    return any(
        term in normalized
        for term in (
            "error",
            "fail",
            "forbidden",
            "invented",
            "leak",
            "mismatch",
            "missing",
            "omission",
            "violation",
        )
    )


def _actionable_feedback(feedback: tuple[str, ...]) -> tuple[str, ...]:
    actionable = tuple(
        message
        for message in feedback
        if any(
            term in message.casefold()
            for term in (
                "avoid",
                "do not",
                "however",
                "incorrect",
                "missing",
                "omit",
                "remove",
                "should",
                "stale",
                "unsupported",
            )
        )
    )
    return actionable or feedback


def _render_evaluator_feedback(lines: list[str], report: BenchmarkReport) -> None:
    evaluation = report.evaluation
    if evaluation is None:
        return
    candidates = tuple(
        sorted(
            (case for case in evaluation.cases if case.feedback and case.quality_pass is not True),
            key=lambda case: (
                case.score is None,
                case.score if case.score is not None else 0,
                case.case_id,
            ),
        )
    )
    limit = report.markdown.limits.failure_details
    if report.markdown.profile == "full":
        limit = min(limit, 5)
    cases = candidates[:limit]
    if not cases:
        return
    lines.extend(
        [
            "## Where Quality Broke Down",
            "",
            "Priority evaluator feedback from the lowest-scoring cases that did not meet the "
            "quality gate:",
            "",
        ]
    )
    if report.markdown.profile == "full":
        _append_table(
            lines,
            ("Case", "Score", "Priority feedback"),
            [
                (
                    case.case_id,
                    _percentage_or_value(case.score),
                    "\n".join(
                        f"- {message}" for message in _actionable_feedback(case.feedback)[:3]
                    ),
                )
                for case in cases
            ],
            numeric_columns={1},
        )
        lines.append("")
        return
    for case in cases:
        lines.extend(
            [
                f"### {_text(case.case_id)} · {_percentage_or_value(case.score)}",
                "",
                *(f"- {_text(message)}" for message in case.feedback),
                "",
            ]
        )


def _render_methodology(lines: list[str], report: BenchmarkReport) -> None:
    design = report.design
    if design is None:
        return
    lines.extend(["## Benchmark Setup", ""])
    rows = [
        ("Cases", str(design.case_count)),
        ("Variants", str(design.variant_count)),
        ("Planned evaluations", str(design.planned_run_count)),
    ]
    if design.dataset_id is not None:
        dataset = design.dataset_id
        if design.dataset_version is not None:
            dataset += f" · {design.dataset_version}"
        rows.append(("Dataset", dataset))
    if report.correlation is not None and report.correlation.phase is not None:
        rows.append(("Evaluation phase", report.correlation.phase.replace("-", " ")))
    _append_table(lines, ("Property", "Value"), rows, numeric_columns={1})
    lines.append("")


def _render_execution_issues(lines: list[str], report: BenchmarkReport) -> None:
    health = report.health
    if health is None or report.markdown.profile == "audit":
        return
    unsuccessful = sum(
        health.status_counts.get(status, 0) for status in ("failed", "errored", "cancelled")
    )
    if not health.partial and not health.missing_count and not unsuccessful:
        return
    lines.extend(
        [
            "## Execution Issues",
            "",
            "Some planned evaluations did not produce complete benchmark evidence.",
            "",
        ]
    )
    _append_table(
        lines,
        ("State", "Count"),
        (
            ("Planned", str(health.planned_count)),
            ("Recorded", str(health.recorded_count)),
            ("Missing", str(health.missing_count)),
            ("Failed or errored", str(unsuccessful)),
        ),
        numeric_columns={1},
    )
    lines.append("")


def _render_audit_identity(lines: list[str], report: BenchmarkReport) -> None:
    lines.extend(["## Technical Evidence", ""])
    rows = [
        ("Experiment", report.experiment_id),
        ("Recorded runs", str(report.run_count)),
        ("Report schema", str(report.report_version)),
    ]
    if report.correlation is not None:
        rows.extend(
            (key.replace("_", " ").capitalize(), _format_value(value))
            for key, value in report.correlation.model_dump(mode="json", exclude_none=True).items()
        )
    _append_table(lines, ("Property", "Value"), rows)
    lines.append("")


def _render_health(lines: list[str], report: BenchmarkReport) -> None:
    health = report.health
    if health is None:
        return
    lines.extend(["## Run Health", ""])
    _append_table(
        lines,
        ("state", "count"),
        [
            ("planned", str(health.planned_count)),
            ("recorded", str(health.recorded_count)),
            ("missing", str(health.missing_count)),
            ("partial runs", str(health.partial_run_count)),
            *[(status, str(count)) for status, count in sorted(health.status_counts.items())],
        ],
        numeric_columns={1},
    )
    lines.extend(
        [
            "",
            f"experiment status: {_code(health.experiment_status)}",
            f"cross-run derivation complete: {_code(str(health.cross_run_derivation_complete))}",
            f"policies complete: {_code(str(health.policies_complete))}",
            "",
        ]
    )
    if health.status_by_variant:
        statuses = sorted(
            {status for counts in health.status_by_variant.values() for status in counts}
        )
        lines.extend(["### Status By Variant", ""])
        _append_table(
            lines,
            ("variant", *statuses),
            [
                (
                    variant_id,
                    *(str(counts.get(status, 0)) for status in statuses),
                )
                for variant_id, counts in sorted(health.status_by_variant.items())
            ],
            numeric_columns=set(range(1, len(statuses) + 1)),
        )
        lines.append("")
    if health.errors:
        lines.extend(["### Grouped Errors", ""])
        _append_table(
            lines,
            ("error type", "count", "runs"),
            [
                (error.error_type, str(error.count), ", ".join(error.run_ids))
                for error in health.errors
            ],
            numeric_columns={1},
        )
        lines.append("")
    if health.missing_run_ids:
        lines.extend(
            [
                "### Missing Planned Runs",
                "",
                *(_code(run_id) for run_id in health.missing_run_ids),
                "",
            ]
        )


def _render_variants(lines: list[str], report: BenchmarkReport) -> None:
    if not report.variant_configs:
        return
    if len(report.variant_configs) == 1:
        variant = report.variant_configs[0]
        if not variant.factor_details and not variant.factors and variant.label is None:
            return
        title = "Configuration Tested"
    else:
        title = "Variants Compared"
    lines.extend([f"## {title}", ""])
    rows = []
    for variant in report.variant_configs:
        factors = ", ".join(
            f"{factor.name}={_format_value(factor.value)}" for factor in variant.factor_details
        ) or ", ".join(
            f"{name}={_format_value(value)}" for name, value in sorted(variant.factors.items())
        )
        description = factors
        if variant.label is not None:
            description = f"{variant.label} · {factors}" if factors else variant.label
        rows.append((variant.variant_id, description or "Recorded default"))
    _append_table(lines, ("Variant", "Configuration"), rows)
    lines.append("")


def _render_leaderboard(lines: list[str], report: BenchmarkReport) -> None:
    if len(report.variant_configs) < 2:
        return
    metric_names = list(dict.fromkeys(name for row in report.leaderboard for name in row.metrics))
    if not metric_names:
        return
    lines.extend(["## Leaderboard", ""])
    rows = [
        (
            row.variant_id,
            str(row.run_count),
            *(_format_value(row.metrics.get(name)) for name in metric_names),
        )
        for row in report.leaderboard
    ]
    _append_table(
        lines,
        ("Variant", "Cases", *metric_names),
        rows,
        numeric_columns=set(range(1, len(metric_names) + 2)),
    )
    lines.append("")
    details = [detail for row in report.leaderboard for detail in row.metric_details]
    if details and report.markdown.profile == "audit":
        lines.extend(["### Metric Coverage", ""])
        _append_table(
            lines,
            (
                "variant",
                "metric",
                "semantic type",
                "value",
                "samples",
                "missing",
                "direction",
                "best",
            ),
            [
                (
                    row.variant_id,
                    detail.name,
                    detail.semantic_type,
                    _format_value(detail.value, unit=detail.unit),
                    str(detail.sample_count),
                    str(detail.missing_count),
                    detail.direction or "",
                    "yes" if detail.best else "",
                )
                for row in report.leaderboard
                for detail in row.metric_details
            ],
            numeric_columns={3, 4, 5},
        )
        lines.append("")


def _render_policies(lines: list[str], report: BenchmarkReport) -> None:
    if not report.policies:
        return
    lines.extend(["## Policy Outcomes", ""])
    if report.markdown.profile != "audit":
        names = sorted({policy.name for policy in report.policies})
        _append_table(
            lines,
            ("Policy", "Passed", "Failed", "Outcome"),
            [
                (
                    name,
                    str(sum(policy.name == name and policy.passed for policy in report.policies)),
                    str(
                        sum(policy.name == name and not policy.passed for policy in report.policies)
                    ),
                    (
                        "Pass"
                        if all(policy.passed for policy in report.policies if policy.name == name)
                        else "Fail"
                    ),
                )
                for name in names
            ],
            numeric_columns={1, 2},
        )
        lines.append("")
        return
    _append_table(
        lines,
        ("policy", "run", "case", "variant", "passed", "metric", "actual", "reason"),
        [
            (
                policy.name,
                policy.run_id,
                policy.case_id,
                policy.variant_id,
                "yes" if policy.passed else "no",
                policy.metric or "",
                _format_value(policy.actual),
                policy.reason or "",
            )
            for policy in report.policies[: report.markdown.limits.table_rows]
        ],
    )
    if len(report.policies) > report.markdown.limits.table_rows:
        lines.append(f"\n_Truncated to {report.markdown.limits.table_rows} policy outcome rows._")
    lines.append("")


def _render_comparisons(lines: list[str], report: BenchmarkReport) -> None:
    if not report.comparisons:
        return
    lines.extend(["## Comparisons", ""])
    for comparison in report.comparisons:
        lines.extend(
            [
                f"### {_text(comparison.baseline)} vs {_text(comparison.candidate)}",
                "",
                f"paired cases: {_code(str(comparison.paired_count))}",
                f"missing pairs: {_code(str(comparison.missing_pair_count))}",
                f"confounded: {_code(str(comparison.confounded))}",
                "",
            ]
        )
        if comparison.factor_deltas:
            _append_table(
                lines,
                ("changed factor", "baseline", "candidate"),
                [
                    (
                        name,
                        _format_value(values.get("baseline")),
                        _format_value(values.get("candidate")),
                    )
                    for name, values in sorted(comparison.factor_deltas.items())
                ],
            )
            lines.append("")
        if comparison.metric_results:
            _append_table(
                lines,
                (
                    "metric",
                    "baseline",
                    "candidate",
                    "delta",
                    "relative",
                    "outcome",
                    "paired",
                    "W/T/L",
                ),
                [_comparison_row(metric) for metric in comparison.metric_results],
                numeric_columns={1, 2, 3, 4, 6},
            )
        else:
            _append_table(
                lines,
                ("metric", "baseline", "candidate", "delta"),
                [
                    (
                        name,
                        _format_value(values.get("baseline")),
                        _format_value(values.get("candidate")),
                        _format_value(values.get("delta")),
                    )
                    for name, values in sorted(comparison.metric_deltas.items())
                ],
                numeric_columns={1, 2, 3},
            )
        lines.append("")


def _comparison_row(metric: MetricComparisonReport) -> tuple[str, ...]:
    relative = (
        "undefined"
        if metric.relative_delta is None
        else _format_value(metric.relative_delta * 100, unit="%")
    )
    return (
        metric.name,
        _format_value(metric.baseline, unit=metric.unit),
        _format_value(metric.candidate, unit=metric.unit),
        _format_value(metric.delta, unit=metric.unit),
        relative,
        metric.outcome,
        f"{metric.paired_count}/{metric.paired_count + metric.missing_pair_count}",
        f"{metric.wins}/{metric.ties}/{metric.losses}",
    )


def _render_regressions(lines: list[str], report: BenchmarkReport) -> None:
    if not report.regressions:
        return
    lines.extend(["## Improvements And Regressions", ""])
    _append_table(
        lines,
        ("baseline", "candidate", "metric", "outcome", "delta", "relative"),
        [
            (
                item.baseline,
                item.candidate,
                item.metric,
                item.outcome,
                _format_value(item.delta),
                (
                    "undefined"
                    if item.relative_delta is None
                    else _format_value(item.relative_delta * 100, unit="%")
                ),
            )
            for item in report.regressions
        ],
        numeric_columns={4, 5},
    )
    lines.append("")


def _render_design(lines: list[str], report: BenchmarkReport) -> None:
    design = report.design
    if design is None:
        return
    lines.extend(["## Experiment Design", ""])
    if design.description is not None:
        lines.extend([_text(design.description), ""])
    _append_table(
        lines,
        ("property", "value"),
        [
            ("dataset", design.dataset_id or "inline"),
            ("dataset version", design.dataset_version or "unknown"),
            ("dataset hash", design.dataset_hash or "unknown"),
            ("cases", str(design.case_count)),
            ("variants", str(design.variant_count)),
            ("planned runs", str(design.planned_run_count)),
            ("scorers", str(design.scorer_count)),
            ("derivers", str(design.deriver_count + design.post_deriver_count)),
            ("policies", str(design.policy_count)),
            ("capture", "enabled" if design.capture_enabled else "disabled"),
            ("instrumentation", ", ".join(design.instrumentation) or "none"),
        ],
    )
    if design.case_tags:
        lines.extend(["", "### Case Tags", ""])
        _append_table(
            lines,
            ("case", "tags"),
            [
                (case_id, ", ".join(tags) or "none")
                for case_id, tags in sorted(design.case_tags.items())
            ],
        )
    if design.warnings:
        lines.extend(["", "### Design Warnings", ""])
        lines.extend(f"- {_text(warning)}" for warning in design.warnings)
    lines.append("")


def _render_metric_catalog(lines: list[str], report: BenchmarkReport) -> None:
    if not report.metric_catalog:
        return
    lines.extend(["## Metric Catalog", ""])
    _append_table(
        lines,
        (
            "name",
            "semantic type",
            "unit",
            "direction",
            "role",
            "aggregation",
            "sources",
            "observed",
            "missing",
        ),
        [
            (
                metric.name,
                metric.semantic_type,
                metric.unit or "",
                metric.direction or "",
                metric.role or "",
                metric.aggregation or "",
                ", ".join(metric.sources),
                str(metric.observed_count),
                str(metric.missing_count),
            )
            for metric in report.metric_catalog
        ],
        numeric_columns={7, 8},
    )
    lines.append("")


def _render_case_matrix(lines: list[str], report: BenchmarkReport) -> None:
    if not any(
        value is not None for row in report.case_matrix.rows.values() for value in row.values()
    ):
        return
    lines.extend(["## Configured Case Matrix", ""])
    variants = sorted({variant for row in report.case_matrix.rows.values() for variant in row})
    rows = [
        (
            case_id,
            *(_optional_value(values.get(variant)) for variant in variants),
        )
        for case_id, values in sorted(report.case_matrix.rows.items())[
            : report.markdown.limits.table_rows
        ]
    ]
    _append_table(
        lines,
        ("case", *variants),
        rows,
        numeric_columns=set(range(1, len(variants) + 1)),
    )
    if len(report.case_matrix.rows) > len(rows):
        lines.append(f"\n_Truncated to {len(rows)} case rows._")
    lines.append("")


def _render_distributions(lines: list[str], report: BenchmarkReport) -> None:
    if not report.distributions:
        return
    lines.extend(["## Distributions", ""])
    for distribution in report.distributions:
        lines.extend([f"### {_text(distribution.name)} ({_code(distribution.semantic_type)})", ""])
        summary_names = list(
            dict.fromkeys(name for values in distribution.summaries.values() for name in values)
        )
        _append_table(
            lines,
            ("variant", "samples", *summary_names),
            [
                (
                    variant_id,
                    str(len(values)),
                    *(
                        _format_value(distribution.summaries.get(variant_id, {}).get(name))
                        for name in summary_names
                    ),
                )
                for variant_id, values in sorted(distribution.by_variant.items())
            ],
            numeric_columns=set(range(1, len(summary_names) + 2)),
        )
        lines.append("")


def _render_runs(lines: list[str], report: BenchmarkReport) -> None:
    if not report.run_details:
        return
    lines.extend(["## Run Results", ""])
    visible = report.run_details[: report.markdown.limits.run_details]
    _append_table(
        lines,
        (
            "run",
            "case",
            "variant",
            "status",
            "evaluation",
            "partial",
            "scores",
            "spans",
            "assets",
            "artifacts",
            "error",
        ),
        [
            (
                run.run_id,
                run.case_id,
                run.variant_id,
                run.status,
                run.evaluation_status,
                "yes" if run.partial else "",
                str(run.score_count),
                str(run.span_count),
                str(run.asset_count),
                str(run.artifact_count),
                run.error_type or "",
            )
            for run in visible
        ],
        numeric_columns={6, 7, 8, 9},
    )
    if len(report.run_details) > len(visible):
        lines.append(f"\n_Truncated to {len(visible)} run rows._")
    lines.append("")


def _render_failures(lines: list[str], report: BenchmarkReport) -> None:
    if not report.failures:
        return
    lines.extend(["## Failures And Diagnostics", ""])
    visible = report.failures[: report.markdown.limits.failure_details]
    _append_table(
        lines,
        ("run", "case", "variant", "stage", "error", "message", "span"),
        [
            (
                failure.run_id,
                failure.case_id,
                failure.variant_id,
                failure.stage,
                failure.error_type,
                failure.message,
                failure.span_id or "",
            )
            for failure in visible
        ],
    )
    for failure in visible:
        if failure.traceback is not None:
            lines.extend(
                [
                    "",
                    "<details>",
                    f"<summary>{_text(failure.run_id)} traceback</summary>",
                    "",
                    "```text",
                    _fenced_text(failure.traceback),
                    "```",
                    "</details>",
                ]
            )
    lines.append("")


def _render_audit_evidence(lines: list[str], report: BenchmarkReport) -> None:
    visible = [
        run
        for run in report.run_details[: report.markdown.limits.run_details]
        if _run_has_audit_evidence(run)
    ]
    if not visible:
        return
    lines.extend(["## Captured Audit Evidence", ""])
    for run in visible:
        lines.extend([f"### {_text(run.run_id)}", ""])
        _render_run_audit_evidence(lines, run)
        lines.append("")


def _run_has_audit_evidence(run: RunDetailReport) -> bool:
    return any(
        value is not None for value in (run.input_excerpt, run.expected_excerpt, run.output_excerpt)
    ) or bool(run.score_evidence)


def _render_run_audit_evidence(lines: list[str], run: RunDetailReport) -> None:
    if not _run_has_audit_evidence(run):
        return
    values = (
        ("Input", run.input_excerpt),
        ("Expected", run.expected_excerpt),
        ("Output", run.output_excerpt),
    )
    for label, value in values:
        if value is None:
            continue
        lines.extend(
            [
                "",
                "<details>",
                f"<summary>{label}</summary>",
                "",
                "```json",
                _fenced_text(value),
                "```",
                "</details>",
            ]
        )
    if run.score_evidence:
        lines.extend(["", "#### Score Inputs", ""])
        _append_table(
            lines,
            ("score", "actual", "expected"),
            [
                (
                    evidence.name,
                    evidence.actual_excerpt or "",
                    evidence.expected_excerpt or "",
                )
                for evidence in run.score_evidence
            ],
        )


def _render_asset_details(lines: list[str], version: AssetVersionReport) -> None:
    lines.extend(
        [
            "",
            f"### {_text(version.asset_id)}@{_text(version.version)} Metadata",
            "",
        ]
    )
    _append_table(
        lines,
        ("property", "value"),
        [
            ("content hash", version.content_hash),
            ("source hash", version.source_hash or "unknown"),
            ("source path", version.source_path or "unknown"),
            ("git commit", version.git_commit or "unknown"),
            ("representations", ", ".join(version.representations) or "none"),
            ("scopes", ", ".join(version.scopes) or "none"),
            ("source locators", ", ".join(version.source_locators) or "none"),
            ("definition assets", ", ".join(version.definition_asset_ids) or "none"),
            ("provenance", ", ".join(version.provenance) or "none"),
            ("runs", ", ".join(version.run_ids) or "none"),
        ],
    )
    excerpts = (
        ("content", "json", version.content_excerpt),
        ("diff", "diff", version.diff_excerpt),
    )
    for label, language, excerpt in excerpts:
        if excerpt is None:
            continue
        lines.extend(
            [
                "",
                "<details>",
                (f"<summary>{_text(version.asset_id)}@{_text(version.version)} {label}</summary>"),
                "",
                f"```{language}",
                _fenced_text(excerpt),
                "```",
                "</details>",
            ]
        )


def _render_traces(lines: list[str], report: BenchmarkReport) -> None:
    trace = report.traces
    if trace is None:
        return
    lines.extend(["## ABP Trace Analysis", ""])
    _append_table(
        lines,
        ("property", "count"),
        [
            ("traced runs", str(trace.traced_run_count)),
            ("untraced runs", str(trace.untraced_run_count)),
            ("complete traces", str(trace.complete_trace_count)),
            ("partial traces", str(trace.partial_trace_count)),
            ("spans", str(trace.span_count)),
            ("root spans", str(trace.root_span_count)),
            ("orphan spans", str(trace.orphan_span_count)),
            ("error spans", str(trace.error_span_count)),
            ("cancelled spans", str(trace.cancelled_span_count)),
        ],
        numeric_columns={1},
    )
    if trace.spans_by_kind:
        lines.extend(["", "### Span Composition", ""])
        _append_table(
            lines,
            ("kind", "count"),
            [(kind, str(count)) for kind, count in trace.spans_by_kind.items()],
            numeric_columns={1},
        )
    if trace.spans_by_instrumentor:
        lines.extend(["", "### Instrumentation Coverage", ""])
        _append_table(
            lines,
            ("instrumentor", "spans"),
            [
                (instrumentor, str(count))
                for instrumentor, count in trace.spans_by_instrumentor.items()
            ],
            numeric_columns={1},
        )
    if trace.diagnostics_by_code:
        lines.extend(["", "### Trace Diagnostics", ""])
        _append_table(
            lines,
            ("diagnostic", "count"),
            [(diagnostic, str(count)) for diagnostic, count in trace.diagnostics_by_code.items()],
            numeric_columns={1},
        )
    if trace.slow_spans:
        lines.extend(["", "### Slowest Spans", ""])
        _append_table(
            lines,
            ("run", "operation", "kind", "instrumentor", "duration", "status"),
            [
                (
                    span.run_id,
                    span.operation,
                    span.kind,
                    span.instrumentor,
                    _duration(span.duration_ns),
                    span.status,
                )
                for span in trace.slow_spans
            ],
            numeric_columns={4},
        )
    lines.append("")


def _render_assets(lines: list[str], report: BenchmarkReport) -> None:
    assets = report.assets
    if assets is None:
        return
    lines.extend(["## Asset Lineage", ""])
    _append_table(
        lines,
        (
            "asset",
            "version",
            "kind",
            "semantic type",
            "sensitivity",
            "parent",
            "changes",
            "content",
            "diff",
            "variants",
            "runs",
            "source",
        ),
        [
            (
                version.asset_id,
                version.version,
                version.kind or "",
                version.semantic_type or "",
                version.sensitivity or "",
                version.parent_version or "",
                ", ".join(version.changed_fields),
                version.content_state,
                version.diff_state,
                ", ".join(version.variant_ids),
                str(len(version.run_ids)),
                version.source_path or "",
            )
            for version in assets.versions[: report.markdown.limits.table_rows]
        ],
        numeric_columns={10},
    )
    for version in assets.versions[: report.markdown.limits.table_rows]:
        _render_asset_details(lines, version)
    lines.append("")


def _render_artifacts(
    lines: list[str],
    report: BenchmarkReport,
    *,
    record_link_prefix: str,
) -> None:
    inventory = report.artifacts
    if inventory is None:
        return
    lines.extend(["## Artifact Inventory", ""])
    _append_table(
        lines,
        ("state", "count"),
        [
            ("all", str(inventory.artifact_count)),
            ("complete", str(inventory.complete_count)),
            ("partial", str(inventory.partial_count)),
            ("truncated", str(inventory.truncated_count)),
        ],
        numeric_columns={1},
    )
    lines.extend(["", "### Artifacts", ""])
    _append_table(
        lines,
        (
            "artifact",
            "name",
            "run",
            "span",
            "source",
            "media type",
            "state",
            "bytes",
            "hash",
            "path",
        ),
        [
            (
                artifact.artifact_id,
                artifact.name,
                artifact.run_id,
                artifact.span_id or "",
                artifact.source,
                artifact.media_type or "",
                artifact.state,
                _format_value(artifact.byte_count),
                artifact.sha256 or "",
                (
                    _link(
                        artifact.path,
                        _record_link(artifact.path, record_link_prefix),
                    )
                    if artifact.path is not None
                    else "unavailable"
                ),
            )
            for artifact in inventory.artifacts[: report.markdown.limits.table_rows]
        ],
        numeric_columns={7},
    )
    lines.append("")


def _record_link_prefix(document_directory: Path, experiment_root: Path | None) -> str:
    if experiment_root is None:
        return ""
    relative = os.path.relpath(experiment_root.resolve(), start=document_directory.resolve())
    return "" if relative == "." else PurePosixPath(relative).as_posix()


def _record_link(path: str, prefix: str) -> str:
    return (PurePosixPath(prefix) / path).as_posix() if prefix else path


def _render_optimizations(lines: list[str], report: BenchmarkReport) -> None:
    if report.markdown.profile != "audit":
        if report.optimizations:
            lines.extend(["## Optimization Outcome", ""])
            _append_table(
                lines,
                ("Case", "Variant", "Status", "Final score", "Evaluations", "Total cost"),
                [
                    (
                        optimization.case_id,
                        optimization.variant_id,
                        optimization.execution.status,
                        _percentage_or_value(optimization.execution.final_score),
                        _budget(
                            optimization.execution.evaluations_used,
                            optimization.execution.evaluations_limit,
                        ),
                        _optional_value(optimization.execution.total_cost_used),
                    )
                    for optimization in report.optimizations
                ],
                numeric_columns={3, 4, 5},
            )
            lines.append("")
        if report.optimization_warnings:
            lines.extend(
                [
                    "Some optimizer evidence was incomplete:",
                    *(f"- {_text(warning)}" for warning in report.optimization_warnings),
                    "",
                ]
            )
        return
    if report.optimizations:
        lines.extend(["## Pydantic-GEPA Optimizations", ""])
        _append_table(
            lines,
            (
                "run",
                "case",
                "variant",
                "execution",
                "backend",
                "engine/composition",
                "status",
                "final score",
                "evaluations",
                "optimizer cost",
                "evaluator cost",
                "total cost",
            ),
            [
                (
                    optimization.benchmark_run_id,
                    optimization.case_id,
                    optimization.variant_id,
                    optimization.execution.execution_id,
                    optimization.execution.backend or "",
                    optimization.execution.engine or optimization.execution.composition or "",
                    optimization.execution.status,
                    _optional_value(optimization.execution.final_score),
                    _budget(
                        optimization.execution.evaluations_used,
                        optimization.execution.evaluations_limit,
                    ),
                    _budget(
                        optimization.execution.optimizer_cost_used,
                        optimization.execution.optimizer_cost_limit,
                    ),
                    _optional_value(optimization.execution.evaluation_cost_used),
                    _optional_value(optimization.execution.total_cost_used),
                )
                for optimization in report.optimizations
            ],
            numeric_columns={7, 8, 9, 10, 11},
        )
        lines.extend(["", "### Execution Context", ""])
        _append_table(
            lines,
            (
                "execution",
                "objective",
                "direction",
                "semantic type",
                "datasets",
                "seed",
                "best",
                "final",
                "events",
                "diagnostics",
                "stop reason",
            ),
            [
                (
                    optimization.execution.execution_id,
                    (
                        ""
                        if optimization.execution.objective is None
                        else optimization.execution.objective.name
                    ),
                    (
                        ""
                        if optimization.execution.objective is None
                        else optimization.execution.objective.direction or ""
                    ),
                    (
                        ""
                        if optimization.execution.objective is None
                        else optimization.execution.objective.semantic_type or ""
                    ),
                    (
                        ""
                        if optimization.execution.datasets is None
                        else (
                            f"train={optimization.execution.datasets.train_count}, "
                            f"validation={optimization.execution.datasets.validation_count}, "
                            f"test={optimization.execution.datasets.test_count}"
                        )
                    ),
                    optimization.execution.seed_candidate_id or "",
                    optimization.execution.best_candidate_id or "",
                    optimization.execution.final_candidate_id or "",
                    str(optimization.execution.event_count),
                    str(optimization.execution.diagnostic_count),
                    optimization.execution.stop_reason or "",
                )
                for optimization in report.optimizations
            ],
            numeric_columns={8, 9},
        )
        checkpoint_rows = [
            (optimization.execution.execution_id, checkpoint)
            for optimization in report.optimizations
            for checkpoint in optimization.execution.checkpoint_paths
        ]
        if checkpoint_rows:
            lines.extend(["", "### Checkpoints", ""])
            _append_table(lines, ("execution", "path"), checkpoint_rows)
        engines = [
            engine
            for optimization in report.optimizations
            for engine in optimization.execution.engines
        ]
        if engines:
            lines.extend(["", "### Engine Runs", ""])
            _append_table(
                lines,
                (
                    "execution",
                    "engine",
                    "step",
                    "branch",
                    "status",
                    "score",
                    "evaluations",
                    "optimizer cost",
                    "evaluator cost",
                    "total cost",
                ),
                [
                    (
                        engine.execution_id,
                        engine.engine or "",
                        engine.step_id or "",
                        engine.branch_id or "",
                        engine.status,
                        _optional_value(engine.score),
                        _budget(engine.evaluations_used, engine.evaluations_limit),
                        _budget(engine.optimizer_cost_used, engine.optimizer_cost_limit),
                        _optional_value(engine.evaluation_cost_used),
                        _optional_value(engine.total_cost_used),
                    )
                    for engine in engines
                ],
                numeric_columns={5, 6, 7, 8, 9},
            )
        candidates = [
            (optimization.execution.execution_id, candidate)
            for optimization in report.optimizations
            for candidate in optimization.execution.candidates
        ]
        if candidates:
            lines.extend(["", "### Candidate Lineage", ""])
            _append_table(
                lines,
                (
                    "execution",
                    "candidate",
                    "lifecycle",
                    "parents",
                    "generation",
                    "iteration",
                    "score",
                    "fingerprint",
                    "components",
                ),
                [
                    (
                        execution_id,
                        candidate.id,
                        " -> ".join(candidate.statuses or (candidate.status,)),
                        ", ".join(candidate.parent_ids),
                        _optional_value(candidate.generation),
                        _optional_value(candidate.iteration),
                        _optional_value(candidate.score),
                        candidate.fingerprint or "",
                        ", ".join(
                            f"{component}={version}"
                            for component, version in sorted(candidate.component_versions.items())
                        ),
                    )
                    for execution_id, candidate in candidates
                ],
                numeric_columns={4, 5, 6},
            )
        selections = [
            (optimization.execution.execution_id, selection)
            for optimization in report.optimizations
            for selection in optimization.execution.selections
        ]
        if selections:
            lines.extend(["", "### Selection Decisions", ""])
            _append_table(
                lines,
                (
                    "execution",
                    "method",
                    "selected",
                    "contenders",
                    "scores",
                    "score",
                    "reason",
                ),
                [
                    (
                        execution_id,
                        selection.method,
                        selection.selected_execution_id,
                        ", ".join(selection.contender_execution_ids),
                        ", ".join(_format_value(score) for score in selection.contender_scores),
                        _format_value(selection.score),
                        selection.reason or "",
                    )
                    for execution_id, selection in selections
                ],
                numeric_columns={5},
            )
    if report.optimization_warnings:
        lines.extend(["", "### Optimization Evidence Warnings", ""])
        lines.extend(f"- {_text(warning)}" for warning in report.optimization_warnings)
    if report.optimizations or report.optimization_warnings:
        lines.append("")


def _render_provenance(lines: list[str], report: BenchmarkReport) -> None:
    provenance = report.provenance
    source = report.source
    if provenance is None and source is None:
        return
    title = (
        "Reproducibility And Provenance" if report.markdown.profile != "summary" else "Provenance"
    )
    lines.extend([f"## {title}", ""])
    rows: list[tuple[str, str]] = []
    if source is not None:
        rows.extend(
            [
                ("report version", str(source.report_version)),
                ("record version", _format_value(source.record_version)),
                ("spec hash", source.spec_hash or "unknown"),
                ("dataset hash", source.dataset_hash or "unknown"),
                ("manifest", source.manifest_path or "in-memory"),
            ]
        )
    if provenance is not None:
        rows.extend(
            [
                ("Python", provenance.python_version),
                ("platform", provenance.platform),
                ("working directory", provenance.working_directory),
                ("semantic registry", str(provenance.semantic_registry_version)),
                ("source maps", ", ".join(provenance.source_maps) or "none"),
            ]
        )
    _append_table(lines, ("property", "value"), rows)
    if source is not None and source.file_hashes:
        lines.extend(["", "### Recorded File Hashes", ""])
        _append_table(
            lines,
            ("path", "sha256"),
            [(item.path, item.sha256) for item in source.file_hashes],
        )
    if source is not None and source.run_paths:
        lines.extend(["", "### Recorded Run Files", ""])
        lines.extend(f"- {_code(path)}" for path in source.run_paths)
    if source is not None and source.correlation is not None:
        lines.extend(["", "### Execution Correlation", ""])
        _append_table(
            lines,
            ("property", "value"),
            [
                (key, _format_value(value))
                for key, value in source.correlation.model_dump(
                    mode="json",
                    exclude_none=True,
                ).items()
            ],
        )
    lines.append("")


def _render_limitations(lines: list[str], report: BenchmarkReport) -> None:
    limitations = (
        [notice.message for notice in report.notices] if report.markdown.profile == "audit" else []
    )
    if report.health is not None:
        if report.health.missing_count:
            noun = "run is" if report.health.missing_count == 1 else "runs are"
            limitations.append(f"{report.health.missing_count} planned {noun} missing.")
        if report.health.partial:
            limitations.append("The experiment is partial.")
    limitations.extend(
        finding.statement
        for finding in (() if report.summary is None else report.summary.findings)
        if finding.kind == "limitation"
    )
    if not limitations:
        return
    lines.extend(["## Limitations And Missing Evidence", ""])
    lines.extend(f"- {_text(message)}" for message in dict.fromkeys(limitations))
    lines.append("")


def _append_table(
    lines: list[str],
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    numeric_columns: set[int] | None = None,
) -> None:
    numeric = numeric_columns or set()
    lines.append("| " + " | ".join(_cell(header) for header in headers) + " |")
    lines.append(
        "| "
        + " | ".join("---:" if index in numeric else "---" for index in range(len(headers)))
        + " |"
    )
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)


def _number(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if isfinite(value) else None


def _percentage(value: float | None) -> str:
    return "Not recorded" if value is None else f"{value:.1%}"


def _percentage_or_value(value: float | None) -> str:
    if value is None:
        return "Not scored"
    return f"{value:.1%}" if 0 <= value <= 1 else _format_value(value)


def _affected_case_count(cases: Sequence[EvaluationCaseReport], metric_name: str) -> int:
    return sum(
        numeric > 0
        for case in cases
        if (numeric := _number(case.metrics.get(metric_name))) is not None
    )


def _format_value(value: JsonValue, *, unit: str | None = None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not isfinite(value):
            return "unsupported"
        if unit in {"USD", "EUR", "GBP", "currency"}:
            rendered = f"{value:.10f}".rstrip("0").rstrip(".")
            return rendered if rendered else "0"
        if unit == "%":
            return f"{value:.4f}".rstrip("0").rstrip(".") + "%"
        return f"{value:.8g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _duration(duration_ns: int) -> str:
    if duration_ns < 1_000:
        return f"{duration_ns} ns"
    if duration_ns < 1_000_000:
        return f"{duration_ns / 1_000:.3f} us"
    if duration_ns < 1_000_000_000:
        return f"{duration_ns / 1_000_000:.3f} ms"
    return f"{duration_ns / 1_000_000_000:.3f} s"


def _budget(used: int | float | None, limit: int | float | None) -> str:
    rendered = _optional_value(used)
    if limit is not None:
        return f"{rendered}/{_format_value(limit)}"
    return rendered


def _optional_value(value: JsonValue) -> str:
    return "" if value is None else _format_value(value)


def _text(value: str) -> str:
    return html.escape(value, quote=False).replace("\x00", "")


def _cell(value: str) -> str:
    return _text(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _code(value: str) -> str:
    escaped = _text(value).replace("\r", " ").replace("\n", " ")
    fence = "``" if "`" in escaped else "`"
    return f"{fence}{escaped}{fence}"


def _fenced_text(value: str) -> str:
    return _text(value).replace("```", "` ` `")


def _link(label: str, path: str) -> str:
    return f"[{_text(label)}]({quote(path, safe='/._-')})"


__all__ = (
    "MarkdownExperimentPublisher",
    "ReportPublicationError",
    "render_markdown_bundle",
    "render_markdown_report",
    "write_markdown_report",
)
