# Autobench

Autobench is a YAML-first benchmark and evidence framework for replayable, semantic experiment data.

It is built for the problem behind most hand-written `run_benchmark.py` files:

- define cases and variants once
- run deterministic case x variant matrices
- collect semantic metrics, factors, spans, and artifacts
- derive metrics such as token cost or paired baseline speedup
- record immutable YAML run records
- replay, report, export, and compare without re-running tasks

Autobench is general-purpose. AI and agent systems are a first-class use case, but the runtime is not tied to LLM-only semantics.

## Install

```bash
uv sync --extra dev
```

## Quickstart

Validate a spec:

```bash
uv run autobench validate examples/minimal.yaml
```

Run and record it:

```bash
uv run autobench run examples/minimal.yaml --record runs/minimal
```

Replay and report without executing the task again:

```bash
uv run autobench replay runs/minimal
uv run autobench report runs/minimal
uv run autobench export runs/minimal --format yaml --path runs/minimal/report.yaml
uv run autobench export runs/minimal --format png --path runs/minimal/report.png
uv run autobench export runs/minimal --format png-set --path runs/minimal/figures
```

## What 0.1.0 Includes

- YAML-first `BenchmarkSpec`
- datasets, cases, case defaults, and file-backed dataset loading
- variants and factor matrices
- Python task runtime with sync and async support
- `RunContext`, spans, artifacts, measurements, and semantic observations
- scoring via output, pass/fail, exact, schema, and Python scorers
- token-cost derivation
- paired-baseline post-derivation
- policy checks
- immutable YAML run records and replay
- Rich terminal tables for run, replay, report, export, and compare
- Markdown, YAML, CSV, composite PNG, and per-panel PNG-set exports
- deterministic offline examples plus an optional real CodeMode integration example

## Example Set

- `examples/minimal.yaml`
  Smallest runnable spec for quick validation and recording.
- `examples/basic/`
  Ticket-routing benchmark with semantic scoring and report configuration.
- `examples/mid/`
  Measurement, token-cost derivation, policies, custom reports, and paired-baseline comparison.
- `examples/codemode/`
  Advanced dogfood example that mirrors a real benchmark script shape.

## Development

Primary quality gates:

```bash
make prod
make pre-commit
make docs
```

Source coverage for `src/autobench` is enforced at `100%` line and branch coverage.

## Documentation

The MkDocs site lives under `docs/` and can be built locally with:

```bash
uv run --extra dev mkdocs build --strict
```
