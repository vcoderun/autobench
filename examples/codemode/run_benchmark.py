from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

from autobench import (
    collect_benchmark_source_files,
    load_benchmark_spec,
    record_experiment,
    run_benchmark_spec,
)
from autobench.records.recording import ExperimentRecord
from autobench.reports.rich import render_experiment_result

EXAMPLE_ROOT = Path(__file__).resolve().parent
SPEC_PATH = EXAMPLE_ROOT / "autobench.yaml"


@click.command()
@click.option("--only", multiple=True, help="Run only the selected scenario ids.")
@click.option(
    "--record",
    "record_path",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write immutable run evidence to this directory.",
)
def main(only: tuple[str, ...], record_path: Path | None) -> None:
    spec = load_benchmark_spec(SPEC_PATH)
    if only:
        selected = set(only)
        cases = [case for case in spec.dataset.cases if case.id in selected]
        missing = selected.difference(case.id for case in cases)
        if missing:
            raise click.ClickException(f"Unknown scenarios: {', '.join(sorted(missing))}")
        spec = spec.model_copy(update={"dataset": spec.dataset.model_copy(update={"cases": cases})})

    result = asyncio.run(run_benchmark_spec(spec))
    active_record_path = record_path or EXAMPLE_ROOT / ".autobench" / result.experiment_id
    record: ExperimentRecord = record_experiment(
        result,
        active_record_path,
        source_files=list(collect_benchmark_source_files(SPEC_PATH)),
        path_root=EXAMPLE_ROOT.parent.parent,
    )
    render_experiment_result(
        Console(),
        result,
        title="CodeMode Benchmark Complete",
        record_path=active_record_path,
    )
    if record.errored_count:
        raise click.ClickException(f"{record.errored_count} runs errored")


if __name__ == "__main__":
    main()
