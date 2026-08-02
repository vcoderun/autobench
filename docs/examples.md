# Examples

The release examples are applications of the public framework, not alternate runtimes or mock-only
snippets. Every offline example executes the complete `run -> record -> replay -> report -> export`
workflow through `make examples`.

## Minimal

```bash
uv run autobench run examples/minimal/autobench.yaml --record /tmp/autobench-minimal
```

Demonstrates inline cases, deterministic variants, exact scoring, a case matrix, and comparison.

## Basic

```bash
uv run autobench run examples/basic/autobench.yaml --record /tmp/autobench-basic
```

Routes file-backed support tickets and records workflow spans plus decision artifacts. The second
variant fixes an enterprise-outage routing failure, making the comparison visible in terminal tables.

## Mid

```bash
uv run autobench run examples/mid/autobench.yaml --record /tmp/autobench-mid
```

Records semantic token usage and latency, derives request cost from a local pricing DSL, applies
success and cost policies, and renders cost distributions.

## Advanced

```bash
uv run autobench run examples/advanced/autobench.yaml --record /tmp/autobench-advanced
```

Uses repeated measurements and sample artifacts, then derives per-case speedup with a paired baseline.
Correctness remains a constraint while speed is the optimization objective.

## CodeMode

```bash
uv run python examples/codemode/run_benchmark.py --only parse_cron
```

This is a live integration with the runtime that provides `vowel.codemode`. It generates evaluation
specs with configured models, replays each generated spec against the source function, and records
coverage, latency, generated specs, and exploration artifacts as Autobench evidence. It requires the
external CodeMode runtime, an `OPENROUTER_API_KEY`, and network access.

CodeMode-specific calls remain in the example task. Autobench core only owns the generic dataset,
variant, task, observation, artifact, scoring, recording, replay, and reporting seams.
