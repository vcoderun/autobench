from __future__ import annotations as _annotations

from pathlib import Path
from types import TracebackType
from typing import TypedDict

import click
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from autobench.data.generation import (
    CaseGeneratorInput,
    generate_dataset,
    load_generation_request,
    resolve_case_generator,
    write_generation_result,
)
from autobench.errors import (
    AutobenchError,
    GenerationError,
    SpecLoadError,
    SpecValidationError,
    TaskResolutionError,
)
from autobench.exporters.otlp import OTLPSettings, export_record_otlp
from autobench.instrumentation import InstrumentationError, instrumentor_statuses
from autobench.records.recording import RecordingError
from autobench.records.replay import replay_experiment
from autobench.records.staging import (
    FileRecorder,
    archive_staging,
    discard_staging,
    finalize_staging,
    inspect_staging,
)
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
    render_generation_result,
    render_instrumentor_statuses,
    render_otlp_export,
    render_report,
    render_staging_inspection,
    render_trace_summary,
    render_validation_summary,
)
from autobench.runtime.awaitables import ProcessSignalInterrupt, run_sync_cooperatively
from autobench.runtime.models import ExecutionCorrelation
from autobench.runtime.pipeline import generate_experiment_id, run_benchmark_spec
from autobench.runtime.progress import (
    ProgressErrorPolicy,
    ProgressEvent,
    ProgressEventKind,
    ProgressHandlerFailure,
)
from autobench.spec import collect_benchmark_source_files, load_benchmark_spec
from autobench.spec import render_validation_summary as build_validation_summary

_console = Console()
_EXPORTERS = ("csv", "markdown", "yaml")


class CorrelationOverridePayload(TypedDict, total=False):
    group_id: str
    attempt: int
    phase: str
    parent_experiment_id: str
    resumed_from_experiment_id: str
    labels: dict[str, str]


class _CliProgress:
    def __init__(self, console: Console) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=not console.is_terminal,
        )
        self._task_id: TaskID | None = None

    def __enter__(self) -> _CliProgress:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self._progress.stop()

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind is ProgressEventKind.BENCHMARK_STARTED:
            planned = event.data.get("planned_run_count")
            total = planned if isinstance(planned, int) else None
            self._task_id = self._progress.add_task(
                f"Running {event.benchmark_id or 'benchmark'}",
                total=total,
            )
            return
        if self._task_id is None:
            return
        if event.kind is ProgressEventKind.RUN_FINISHED:
            self._progress.advance(self._task_id)
        elif event.kind is ProgressEventKind.BENCHMARK_FINISHED:
            self._progress.update(
                self._task_id,
                description=f"Benchmark {event.experiment_status or 'finished'}",
            )


def _report_progress_failure(failure: ProgressHandlerFailure) -> None:
    click.echo(
        (
            f"Progress renderer failed during {failure.event_kind} "
            f"(sequence {failure.sequence}): {failure.error}"
        ),
        err=True,
    )


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


@cli.group("dataset")
def dataset() -> None:
    """Prepare benchmark datasets outside active matrix execution."""


@dataset.command("generate")
@click.argument("target")
@click.option(
    "--request",
    "request_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML CaseGeneratorInput request.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Dataset YAML path to publish after complete generation.",
)
@click.option("--id", "dataset_id", required=True, help="Stable generated dataset id.")
@click.option("--version", help="Generated dataset version.")
@click.option("--force", is_flag=True, help="Replace existing generation outputs.")
def dataset_generate(
    target: str,
    request_path: Path | None,
    output_path: Path,
    dataset_id: str,
    version: str | None,
    force: bool,
) -> None:
    """Run a typed generator and publish a frozen dataset plus provenance."""

    search_paths = (
        (str(request_path.parent), str(Path.cwd()))
        if request_path is not None
        else (str(Path.cwd()),)
    )
    try:
        request = (
            CaseGeneratorInput() if request_path is None else load_generation_request(request_path)
        )
        generator = resolve_case_generator(target, search_paths=search_paths)
        result = run_sync_cooperatively(
            generate_dataset(
                generator,
                request,
                generator_id=target,
                dataset_id=dataset_id,
                version=version,
            )
        )
        written = write_generation_result(result, output_path, force=force)
    except (GenerationError, SpecLoadError, TaskResolutionError) as exc:
        _console.print(f"[red]Dataset generation failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc
    render_generation_result(_console, result, written)
    if not result.batch.complete:
        raise SystemExit(2)


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
@click.option("--group-id", help="Group this invocation with related experiments.")
@click.option("--attempt", type=click.IntRange(min=1), help="Positive invocation attempt number.")
@click.option("--phase", help="Named execution phase, such as train or validation.")
@click.option("--parent-experiment-id", help="Experiment that proposed this invocation.")
@click.option("--resumed-from-experiment-id", help="Earlier experiment this invocation resumes.")
@click.option(
    "--correlation-label",
    "correlation_labels",
    type=(str, str),
    multiple=True,
    metavar="KEY VALUE",
    help="Attach a repeatable string correlation label.",
)
def run(
    spec_path: Path,
    concurrency: int,
    record_path: Path | None,
    no_record: bool,
    group_id: str | None,
    attempt: int | None,
    phase: str | None,
    parent_experiment_id: str | None,
    resumed_from_experiment_id: str | None,
    correlation_labels: tuple[tuple[str, str], ...],
) -> None:
    """Run a YAML Autobench spec."""
    try:
        spec = load_benchmark_spec(spec_path)
    except SpecLoadError as exc:
        _print_spec_error(exc)
        raise SystemExit(1) from exc
    except SpecValidationError as exc:
        _console.print(f"[red]Spec validation failed:[/red] {exc}")
        raise SystemExit(1) from exc
    experiment_id = generate_experiment_id(spec.benchmark.id)
    correlation_payload: CorrelationOverridePayload = {}
    if group_id is not None:
        correlation_payload["group_id"] = group_id
    if attempt is not None:
        correlation_payload["attempt"] = attempt
    if phase is not None:
        correlation_payload["phase"] = phase
    if parent_experiment_id is not None:
        correlation_payload["parent_experiment_id"] = parent_experiment_id
    if resumed_from_experiment_id is not None:
        correlation_payload["resumed_from_experiment_id"] = resumed_from_experiment_id
    if correlation_labels:
        correlation_payload["labels"] = dict(correlation_labels)
    correlation = (
        ExecutionCorrelation.model_validate(correlation_payload) if correlation_payload else None
    )
    active_record_path = (
        None
        if no_record
        else record_path or spec_path.parent / ".autobench" / spec_path.stem / experiment_id
    )
    recorder = (
        None
        if active_record_path is None
        else FileRecorder(
            active_record_path,
            source_files=collect_benchmark_source_files(spec_path),
            path_root=Path.cwd(),
        )
    )
    try:
        with _CliProgress(_console) as progress:
            result = run_sync_cooperatively(
                run_benchmark_spec(
                    spec,
                    experiment_id=experiment_id,
                    correlation=correlation,
                    concurrency_limit=concurrency,
                    recorder=recorder,
                    progress_handlers=(progress,),
                    progress_error_policy=ProgressErrorPolicy.BEST_EFFORT,
                    progress_error_handler=_report_progress_failure,
                )
            )
    except ProcessSignalInterrupt as exc:
        _console.print("[yellow]Benchmark interrupted by SIGTERM.[/yellow]")
        _render_staging_path(recorder)
        raise SystemExit(128 + exc.signal_number) from exc
    except KeyboardInterrupt as exc:
        _console.print("[yellow]Benchmark interrupted.[/yellow]")
        _render_staging_path(recorder)
        raise SystemExit(130) from exc
    except InstrumentationError as exc:
        _console.print(f"[red]Instrumentation failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc
    except RecordingError as exc:
        _console.print(f"[red]Recording failed:[/red] {escape(str(exc))}")
        _render_staging_path(recorder)
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


@cli.group("telemetry")
def telemetry() -> None:
    """Export immutable Autobench evidence to optional telemetry backends."""


@telemetry.command("export")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--endpoint", help="OTLP HTTP/protobuf traces endpoint; otherwise use OTel env.")
@click.option(
    "--header",
    "headers",
    type=(str, str),
    multiple=True,
    metavar="NAME VALUE",
    help="Repeatable OTLP request header.",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=10.0,
    show_default=True,
    help="Exporter request timeout in seconds.",
)
@click.option("--service-name", default="autobench", show_default=True)
@click.option("--service-namespace")
@click.option(
    "--certificate-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--include-captured-content",
    is_flag=True,
    help="Include ABP bodies, outputs, actual/expected values, and retained source facts.",
)
def telemetry_export(
    run_dir: Path,
    endpoint: str | None,
    headers: tuple[tuple[str, str], ...],
    timeout_seconds: float,
    service_name: str,
    service_namespace: str | None,
    certificate_file: Path | None,
    include_captured_content: bool,
) -> None:
    """Replay a record into OTLP spans without changing its ABP evidence."""

    try:
        result = export_record_otlp(
            run_dir,
            settings=OTLPSettings(
                endpoint=endpoint,
                headers=dict(headers),
                timeout_seconds=timeout_seconds,
                certificate_file=certificate_file,
                service_name=service_name,
                service_namespace=service_namespace,
                include_captured_content=include_captured_content,
            ),
        )
    except AutobenchError as exc:
        _console.print(f"[red]OTLP export failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc
    render_otlp_export(_console, result)


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


@cli.group("recording")
def recording() -> None:
    """Inspect and recover durable recording sessions."""


@recording.command("inspect")
@click.argument("staging_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def recording_inspect(staging_dir: Path) -> None:
    """Inspect committed and incomplete evidence in a staging directory."""

    try:
        inspection = inspect_staging(staging_dir)
    except RecordingError as exc:
        _console.print(f"[red]Inspection failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc
    render_staging_inspection(_console, inspection)


@recording.command("finalize")
@click.argument("staging_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option("--allow-partial", is_flag=True, help="Publish explicit incomplete evidence.")
def recording_finalize(
    staging_dir: Path,
    output_dir: Path,
    allow_partial: bool,
) -> None:
    """Publish staged evidence as an immutable experiment record."""

    try:
        record = finalize_staging(
            staging_dir,
            output_dir,
            allow_partial=allow_partial,
        )
    except RecordingError as exc:
        _console.print(f"[red]Finalization failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc
    _console.print(f"[green]Published[/green] {record.experiment_id} to {escape(str(output_dir))}")


@recording.command("archive")
@click.argument("staging_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
def recording_archive(staging_dir: Path, output_dir: Path) -> None:
    """Copy staging evidence to a separate archive without publishing it as a record."""

    try:
        archive_staging(staging_dir, output_dir)
    except RecordingError as exc:
        _console.print(f"[red]Archive failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc
    _console.print(f"[green]Archived staging to[/green] {escape(str(output_dir))}")


@recording.command("discard")
@click.argument("staging_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--yes", is_flag=True, help="Confirm permanent deletion of staged evidence.")
def recording_discard(staging_dir: Path, yes: bool) -> None:
    """Permanently delete a validated Autobench staging directory."""

    if not yes:
        raise click.UsageError("Pass --yes to discard staged evidence.")
    try:
        discard_staging(staging_dir)
    except RecordingError as exc:
        _console.print(f"[red]Discard failed:[/red] {escape(str(exc))}")
        raise SystemExit(1) from exc
    _console.print(f"[green]Discarded[/green] {escape(str(staging_dir))}")


def _print_spec_error(exc: SpecLoadError) -> None:
    location = ""
    if exc.line is not None and exc.column is not None:
        location = f" (line {exc.line}, column {exc.column})"
    _console.print(f"[red]Spec load failed:[/red] {exc}{location}")


def _render_staging_path(recorder: FileRecorder | None) -> None:
    if recorder is not None:
        _console.print(f"[yellow]Staging path:[/yellow] {recorder.staging_dir}")


def main() -> int:
    cli.main(standalone_mode=False)
    return 0


__all__ = ("cli", "main")
