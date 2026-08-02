# Autobench

**Turn one-off benchmark scripts into replayable semantic experiment data.**

Autobench is a YAML-first benchmark and evidence framework. It runs deterministic case and
variant matrices, records typed observations and artifacts, and lets you replay or compare the
evidence without executing the subject again.

```bash
uv add autobench
autobench validate autobench.yaml
autobench run autobench.yaml --record runs/latest
```

The core loop is:

```text
Dataset / Cases
  x Variants / Factors
    -> Task executes the subject
    -> Context and spans collect observations
    -> Scorers evaluate outputs
    -> Derivers add semantic metrics
    -> Recorder writes immutable YAML evidence
    -> Replay, report, export, and compare operate on recorded runs
```

Autobench is designed for AI-heavy systems, but the runtime itself is generic. If you can express
a case, a variant, a task, and semantic outcomes, Autobench can benchmark it.

## Why It Exists

Most benchmark codebases keep re-implementing the same machinery:

- scenario loading
- case x variant expansion
- task orchestration
- metrics and derived metrics
- artifacts and replay
- comparison and reporting

Autobench provides those utilities as framework primitives so users describe the benchmark instead of rebuilding the runner.

## What 0.1.0 Ships

- YAML-first benchmark specs
- cases, datasets, variants, and factors
- sync and async Python task execution
- semantic observations, spans, and artifacts
- output, pass/fail, exact, schema, and Python scorers
- token-cost derivation
- paired-baseline post-derivation
- policies, replay, export, and comparison
- Rich terminal reports and YAML, CSV, and Markdown exports
- tracked prompt, tool, type, and configuration assets

## Choose A Path

| Goal | Start here |
| --- | --- |
| Run the smallest complete benchmark | [Getting Started](getting-started.md) |
| Learn the evidence model | [Concepts](concepts.md) |
| Adapt a working integration | [Examples](examples.md) |
| Define a benchmark declaratively | [YAML Spec](yaml-spec.md) |
| Build custom tasks and scorers | [Python API](python-api.md) |
| Replay and compare recorded evidence | [Recording And Reporting](recording-and-reporting.md) |

## Start Here

- [Getting Started](getting-started.md)
- [Examples](examples.md)
- [YAML Spec](yaml-spec.md)
- [Python API](python-api.md)
