from __future__ import annotations as _annotations

from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape

from autobench.errors import SpecLoadError, SpecValidationError
from autobench.instrumentation import InstrumentationError, instrumentor_statuses
from autobench.records.recording import RecordingError, record_experiment
from autobench.records.replay import replay_experiment
from autobench.reports.exporting import (
    export_markdown_report,
    export_runs_csv,
    export_summary_yaml,
)
from autobench.reports.reporting import build_report, compare_variants
from autobench.reports.rich import (
    render_comparison,
    render_experiment_result,
    render_export_preview,
    render_instrumentor_statuses,
    render_report,
    render_trace_summary,
    render_validation_summary,
)
from autobench.runtime.awaitables import run_sync
from autobench.runtime.pipeline import run_benchmark_spec
from autobench.spec import collect_benchmark_source_files, load_benchmark_spec
from autobench.spec import render_validation_summary as build_validation_summary

_console = Console()
_EXPORTERS = ("csv", "markdown", "yaml")


@click.group()
def cli() -> None:
    """Autobench CLI."""


@cli.command("validate")
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(spec_path: Path) -> None:
    """Validate a YAML Autobench spec."""
    try:
        spec = load_benchmark_spec(spec_path)
    except SpecLoadError as exc:
        _print_spec_error(exc)
        raise SystemExit(1) from exc
    except SpecValidationError as exc:
        _console.print(f"[red]Spec validation failed:[/red] {exc}")
        raise SystemExit(1) from exc
    summary = build_validation_summary(spec_path, spec)
    render_validation_summary(_console, summary)


@cli.command("run")
@click.argument("spec_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Maximum number of runs to execute concurrently.",
)
@click.option(
    "--record",
    "record_path",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write immutable YAML run records to this directory.",
)
@click.option(
    "--no-record",
    is_flag=True,
    default=False,
    help="Do not persist evidence for this run.",
)
def run(
    spec_path: Path,
    concurrency: int,
    record_path: Path | None,
    no_record: bool,
) -> None:
    """Run a YAML Autobench spec."""
    try:
        spec = load_benchmark_spec(spec_path)
        result = run_sync(run_benchmark_spec(spec, concurrency_limit=concurrency))
    except SpecLoadError as exc:
        _print_spec_error(exc)
        raise SystemExit(1) from exc
    except SpecValidationError as exc:
        _console.print(f"[red]Spec validation failed:[/red] {exc}")
        raise SystemExit(1) from exc
    except InstrumentationError as exc:
        _console.print(f"[red]Instrumentation failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc

    active_record_path = (
        None
        if no_record
        else record_path or spec_path.parent / ".autobench" / spec_path.stem / result.experiment_id
    )
    if active_record_path is not None:
        try:
            source_files = list(collect_benchmark_source_files(spec_path))
            record_experiment(
                result,
                active_record_path,
                source_files=source_files,
                path_root=Path.cwd(),
            )
        except RecordingError as exc:
            _console.print(f"[red]Recording failed:[/red] {exc}")
            raise SystemExit(1) from exc

    render_experiment_result(
        _console,
        result,
        title="Benchmark Run Complete",
        record_path=active_record_path,
    )


@cli.command("replay")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def replay(run_dir: Path) -> None:
    """Replay recorded YAML evidence without importing benchmark tasks."""
    result = replay_experiment(run_dir)
    render_experiment_result(_console, result, title="Replay Loaded")


@cli.command("report")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report(run_dir: Path) -> None:
    """Render the Rich terminal report from recorded evidence."""
    result = replay_experiment(run_dir)
    render_report(_console, build_report(result))


@cli.command("export")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--format",
    "export_format",
    type=click.Choice(sorted(_EXPORTERS)),
    required=True,
    help="Export format.",
)
@click.option(
    "--path",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Write the export to this file.",
)
def export(run_dir: Path, export_format: str, output_path: Path) -> None:
    """Export recorded evidence to a file and show a Rich terminal preview."""
    result = replay_experiment(run_dir)
    if export_format == "csv":
        export_runs_csv(result, output_path)
    elif export_format == "markdown":
        export_markdown_report(result, output_path)
    else:
        export_summary_yaml(result, output_path)
    render_export_preview(
        _console,
        result,
        export_format=export_format,
        output_path=output_path,
    )


@cli.command("compare")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--baseline", required=True, help="Baseline variant id.")
@click.option("--candidate", required=True, help="Candidate variant id.")
def compare(
    run_dir: Path,
    baseline: str,
    candidate: str,
) -> None:
    """Compare two recorded variants without claiming causality."""
    result = replay_experiment(run_dir)
    comparison = compare_variants(result, baseline=baseline, candidate=candidate)
    render_comparison(_console, comparison)


@cli.group("instrumentation")
def instrumentation() -> None:
    """Inspect native ABP instrumentation and recorded traces."""


@instrumentation.command("doctor")
def instrumentation_doctor() -> None:
    """Report installed integrations, compatibility, and capture defaults."""

    render_instrumentor_statuses(_console, instrumentor_statuses())


@instrumentation.command("trace")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def instrumentation_trace(run_dir: Path) -> None:
    """Summarize ABP traces from recorded evidence without optional SDK imports."""

    render_trace_summary(_console, replay_experiment(run_dir))


def _print_spec_error(exc: SpecLoadError) -> None:
    location = ""
    if exc.line is not None and exc.column is not None:
        location = f" (line {exc.line}, column {exc.column})"
    _console.print(f"[red]Spec load failed:[/red] {exc}{location}")


def main() -> int:
    cli.main(standalone_mode=False)
    return 0


__all__ = ("cli", "main")
