from __future__ import annotations as _annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

from autobench.io import dump_yaml
from autobench.metrics.semantics import Semantic
from autobench.reports.reporting import (
    BenchmarkReport,
    ReportSpec,
    build_report,
    metric_value,
    render_markdown_report,
)
from autobench.runtime.pipeline import ExperimentResult

CSV_METRICS: tuple[tuple[str, str], ...] = (
    ("success", Semantic.RESULT_SUCCESS),
    ("coverage", Semantic.COVERAGE_RATIO),
    ("cost", Semantic.MONEY_COST),
    ("input_tokens", Semantic.LLM_TOKENS_INPUT),
)


def export_summary_yaml(
    result: ExperimentResult,
    path: Path | None = None,
    *,
    report_spec: ReportSpec | None = None,
) -> str:
    report = build_report(result, report_spec=report_spec)
    return dump_yaml(report_to_yaml_view(report), path, schema_name="report")


def report_to_yaml_view(report: BenchmarkReport) -> dict[str, Any]:
    variants = {
        row.variant_id: {
            **({"label": row.label} if row.label is not None else {}),
            **({"factors": row.factors} if row.factors else {}),
        }
        for row in report.variant_configs
    }
    leaderboard = {
        row.variant_id: {
            "runs": row.run_count,
            "metrics": row.metrics,
        }
        for row in report.leaderboard
    }
    cases: dict[str, dict[str, Any]] = {}
    for row in report.run_metrics:
        case_rows = cases.setdefault(row.case_id, {})
        case_rows[row.variant_id] = {
            "status": row.status,
            "metrics": row.metrics,
        }
    comparisons = {
        f"{comparison.baseline} -> {comparison.candidate}": {
            "runs": comparison.run_count,
            **({"confounded": True} if comparison.confounded else {}),
            **({"factors": comparison.factor_deltas} if comparison.factor_deltas else {}),
            **({"metrics": comparison.metric_deltas} if comparison.metric_deltas else {}),
        }
        for comparison in report.comparisons
    }
    distributions = {
        distribution.name: {
            "semantic": distribution.semantic_type,
            "variants": distribution.by_variant,
            "summaries": distribution.summaries,
        }
        for distribution in report.distributions
    }
    return {
        "record": {
            "type": "report",
            "version": 1,
        },
        "report": {
            "benchmark": report.benchmark_id,
            "experiment": report.experiment_id,
            "runs": report.run_count,
            "status": report.status_counts,
            "variants": variants,
            "leaderboard": leaderboard,
            "cases": cases,
            "matrix": {
                "metric": report.case_matrix.metric,
                "cases": report.case_matrix.rows,
            },
            "compare": comparisons,
            "distributions": distributions,
        },
    }


def export_runs_csv(result: ExperimentResult, path: Path | None = None) -> str:
    output = StringIO()
    fieldnames = [
        "run_id",
        "case_id",
        "variant_id",
        "status",
        *[name for name, _ in CSV_METRICS],
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for run in result.runs:
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "case_id": run.case_id,
            "variant_id": run.variant_id,
            "status": run.status.value,
        }
        for name, semantic_type in CSV_METRICS:
            row[name] = metric_value(run, semantic_type)
        writer.writerow(row)

    rendered = output.getvalue()
    if path is not None:
        path.write_text(rendered, encoding="utf-8")
    return rendered


def export_markdown_report(
    result: ExperimentResult,
    path: Path | None = None,
    *,
    report_spec: ReportSpec | None = None,
) -> str:
    rendered = render_markdown_report(build_report(result, report_spec=report_spec))
    if path is not None:
        path.write_text(rendered, encoding="utf-8")
    return rendered


__all__ = (
    "CSV_METRICS",
    "export_markdown_report",
    "export_runs_csv",
    "export_summary_yaml",
    "report_to_yaml_view",
)
