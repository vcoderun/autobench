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

Run the smallest complete example:

```bash
uv run autobench validate examples/minimal/autobench.yaml
uv run autobench run examples/minimal/autobench.yaml --record /tmp/autobench-minimal
```

Run and record it:

```bash
uv run autobench run examples/basic/autobench.yaml --record /tmp/autobench-basic
```

Replay and report without executing the task again:

```bash
uv run autobench replay /tmp/autobench-basic
uv run autobench report /tmp/autobench-basic
uv run autobench export /tmp/autobench-basic --format yaml --path /tmp/basic-report.yaml
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
- Markdown, YAML, and CSV exports
- portable source-file provenance for CLI-recorded evidence
- offline minimal, basic, mid, and advanced end-to-end examples
- a real, optional CodeMode integration example

## Examples

- `examples/minimal/`: inline cases, variants, exact scoring, reporting, and comparison.
- `examples/basic/`: file-backed support tickets, spans, artifacts, and Rich reports.
- `examples/mid/`: semantic token usage, pricing, cost derivation, policies, and distributions.
- `examples/advanced/`: repeated measurement and paired-baseline speedup derivation.
- `examples/codemode/`: live Vowel CodeMode generation and generated-spec replay.

The first four run offline and are enforced by `make examples`. CodeMode is a live integration that
requires its external runtime, model credentials, and network access.

## Development

Primary quality gates:

```bash
make prod
make pre-commit
make docs
make examples
```

Source coverage for `src/autobench` is enforced at `100%` line and branch coverage.

## Documentation

The documentation is published at [vcoderun.github.io/autobench](https://vcoderun.github.io/autobench/).
It is built with Zensical's modern theme:

```bash
make docs
make docs-serve
```
