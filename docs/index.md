# Autobench

Autobench turns one-off benchmark scripts into replayable semantic experiment data.

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

Autobench is designed for AI-heavy systems, but the runtime itself is generic. If you can express a case, a variant, a task, and metrics, Autobench can benchmark it.

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

## Start Here

- [Getting Started](getting-started.md)
- [YAML Spec](yaml-spec.md)
- [Python API](python-api.md)
