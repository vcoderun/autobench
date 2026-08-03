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

## What Autobench Includes

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
- ABP traces with privacy-aware capture and replayable semantic evidence
- native Pydantic AI, OpenAI Python, OpenAI Agents, and HTTPX instrumentors
- typed Python/YAML instrumentation settings, automatic compatible-integration discovery, and Rich
  compatibility diagnostics

## Examples

- `examples/minimal/`: inline cases, variants, exact scoring, reporting, and comparison.
- `examples/basic/`: file-backed support tickets, spans, artifacts, and Rich reports.
- `examples/mid/`: semantic token usage, pricing, cost derivation, policies, and distributions.
- `examples/advanced/`: repeated measurement and paired-baseline speedup derivation.
- `examples/codemode/`: live Vowel CodeMode generation and generated-spec replay.
- `examples/pydantic_ai/`: provider-neutral and live OpenRouter Pydantic AI instrumentation.
- `examples/abp_manual/`: manual spans and method instrumentation through ABP.
- `examples/abp_concurrent/`: task-local concurrent trace parentage.
- `examples/abp_openai/`: offline real OpenAI streaming over HTTPX.
- `examples/abp_openai_agents/`: offline native OpenAI Agents trace processing.
- `examples/abp_replay/`: provider-free trace replay and evidence extraction.

The minimal, basic, mid, advanced, ABP manual, and ABP concurrent examples run offline and are
enforced by `make examples`. CodeMode and the OpenRouter Pydantic AI program are live integrations
that require their model credentials and network access.

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
