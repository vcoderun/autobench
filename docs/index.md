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

## What Autobench Owns

Autobench is more than a matrix runner. It owns the evidence lifecycle from benchmark definition
to optimization-ready records:

| Layer | Capabilities |
| --- | --- |
| Definition | YAML DSL, Python builder, datasets, case defaults, variants, factors, schema hints |
| Execution | deterministic matrix planning, sync/async tasks, concurrency, failure isolation, progress events |
| Evidence | semantic observations, spans, artifacts, checks, diagnostics, errors, trace envelopes |
| Evaluation | six scorer kinds, expected-action evaluation, policies, metric packs, custom scorers |
| Derivation | token cost, tiered pricing, paired baselines, verdicts, measurement statistics |
| Lineage | prompt/tool/type/config tracking, structured schemas, source hashes, versions, diffs |
| Persistence | immutable YAML RunRecords, source hashes, environment metadata, portable artifacts |
| Analysis | replay, Rich reports, leaderboards, case matrices, comparisons, distributions, exports |
| Optimization | compact feedback records and semantic evidence for pydantic-gepa and autoptimize |

See the [Capability Map](capabilities.md) for the complete feature inventory and ownership
boundaries.

## Choose A Path

| Goal | Start here |
| --- | --- |
| Run the smallest complete benchmark | [Getting Started](getting-started.md) |
| See everything Autobench supports | [Capability Map](capabilities.md) |
| Learn the evidence model | [Core Concepts](concepts.md) |
| Adapt a working integration | [Examples](examples.md) |
| Define a benchmark declaratively | [YAML Spec](yaml-spec.md) |
| Instrument an existing application | [Instrumentation And Traces](instrumentation-and-traces.md) |
| Track prompts, tools, and schemas | [Asset Tracking](asset-tracking.md) |
| Evaluate agent behavior | [Agentic Evaluation](agentic-evaluation.md) |
| Replay and compare recorded evidence | [Recording And Reporting](recording-and-reporting.md) |

## Start Here

- [Getting Started](getting-started.md)
- [Capability Map](capabilities.md)
- [Examples](examples.md)
- [YAML Spec](yaml-spec.md)
- [Python API](python-api.md)
