# Getting Started

## Install

```bash
uv sync --extra dev
```

## Validate A Spec

Validate the smallest complete benchmark:

```bash
uv run autobench validate examples/minimal/autobench.yaml
```

Expected output includes:

- benchmark id
- case count
- variant count
- planned run count

## Run And Record

```bash
uv run autobench run examples/minimal/autobench.yaml --record /tmp/autobench-minimal
```

This executes the task matrix and writes:

- `experiment.yaml`
- `summary.yaml`
- per-run `cases/<case_id>/<variant_id>/run.yaml`
- artifact payloads under `artifacts/`

## Replay, Report, Export

```bash
uv run autobench replay /tmp/autobench-minimal
uv run autobench report /tmp/autobench-minimal
uv run autobench export /tmp/autobench-minimal --format yaml --path /tmp/minimal-report.yaml
uv run autobench export /tmp/autobench-minimal --format csv --path /tmp/minimal-runs.csv
```

Replay does not import or execute the original task target. It only reads recorded evidence.
Report and compare render Rich terminal tables. Export writes a file and shows a Rich preview.

## Quality Gates

```bash
make prod
make pre-commit
make docs
make examples
```

`make prod` covers tests, `100%` source line and branch coverage, formatting checks, linting,
typing, docs, the Python validation matrix, and all offline examples.
