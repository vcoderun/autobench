# Getting Started

## Install

```bash
uv sync --extra dev
```

## Validate A Spec

Create a benchmark spec in the current DSL format, then validate it:

```bash
uv run autobench validate path/to/benchmark.yaml
```

Expected output includes:

- benchmark id
- case count
- variant count
- planned run count

## Run And Record

```bash
uv run autobench run path/to/benchmark.yaml --record runs/benchmark
```

This executes the task matrix and writes:

- `experiment.yaml`
- `summary.yaml`
- per-run `cases/<case_id>/<variant_id>/run.yaml`
- artifact payloads under `artifacts/`

## Replay, Report, Export

```bash
uv run autobench replay runs/benchmark
uv run autobench report runs/benchmark
uv run autobench export runs/benchmark --format yaml --path runs/benchmark/report.yaml
uv run autobench export runs/benchmark --format csv --path runs/benchmark/runs.csv
uv run autobench export runs/benchmark --format png --path runs/benchmark/report.png
```

Replay does not import or execute the original task target. It only reads recorded evidence.
Report and compare render Rich terminal tables. Export writes a file and shows a Rich preview.

## Quality Gates

```bash
make prod
make pre-commit
make docs
```

`make prod` covers tests, `100%` source line and branch coverage, formatting, linting, typing, docs, and the Python validation matrix.
