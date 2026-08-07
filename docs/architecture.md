# Architecture

Autobench separates application execution from experiment infrastructure. This is the central
design constraint: the framework can benchmark any system because it does not own the system.

## Layered Model

| Layer | Owns | Does not own |
| --- | --- | --- |
| Definition | benchmark, dataset, variants, evaluation and report configuration | application implementation |
| Runtime | matrix planning, context, task invocation, concurrency, statuses | provider event loops or business orchestration |
| ABP | trace context, signals, spans, measurements, capture, source provenance | OpenTelemetry or hosted trace storage |
| Evaluation | scorers, derivation, policies, paired comparisons | domain truth that only the application can supply |
| Tracking | behavioral asset identity, versions, representations, diffs, uses | source control or deployment promotion |
| Records | immutable run and experiment evidence, artifacts, source hashes | mutable operational databases |
| Reports | semantic projections, aggregation, comparison and exports | causal inference from uncontrolled changes |
| Outbound adapters | projections such as ABP-to-OTLP delivery | canonical evidence or application instrumentation |

## One Canonical Spec

YAML and the Python builder converge on `BenchmarkSpec`:

```text
YAML DSL -----------+
                    +--> BenchmarkSpec --> BenchmarkPlan --> ExperimentResult
Benchmark builder --+
```

The builder is ergonomic composition; it is not a second runtime. `Benchmark.to_spec()` returns the
same model loaded by `load_benchmark_spec()`.

## Execution Lifecycle

For each case x variant pair, Autobench:

1. creates a stable run ID and `RunContext`;
2. activates task-local ABP context;
3. invokes `task(ctx, case)` synchronously or asynchronously;
4. preserves evidence even when the task fails;
5. evaluates built-in and Python scorers;
6. projects scores into semantic observations;
7. derives per-run metrics such as token cost;
8. finalizes status and trace state.

The context tracks these phases explicitly. With durable recording, `await ctx.checkpoint(name)`
commits a frozen partial snapshot through the active record session. Cooperative cancellation
finalizes partial trace state, commits a reserved terminal checkpoint, and propagates the same
`CancelledError`; concurrent cancellation drains sibling cleanup within a fixed bound before the
experiment session aborts.

As runs finish, an optional recorder commits their execution snapshots. After all runs finish, the
runtime applies cross-run derivation and policy evaluation, then
materializes immutable YAML records and referenced artifacts.

Hard termination is intentionally weaker: `SIGKILL` and power loss preserve only the last staging
manifest commit under the selected durability mode. Autobench checkpoints evidence, while the
subject application remains responsible for resumable execution state.

## Evidence Model

`Observation` is the common query and aggregation unit. Its local `name` explains the metric in the
application; `semantic_type` explains what the value means across applications. Source and
provenance distinguish task observations, scores, derived values, and trace extraction.

ABP preserves richer execution evidence as an ordered signal stream and a materialized `Trace`.
Useful trace values can be extracted into observations without discarding their span or source-map
lineage.

```text
SDK call
  -> native instrumentor
  -> ABP signals
  -> canonical Trace
  -> semantic extraction
  -> Observation
  -> report / policy / optimizer
```

## Definition And Effective Assets

Behavioral components have two useful representations:

- **definition**: what application code configured, such as a prompt template or Python tool;
- **effective**: what an SDK sent to a model or downstream system after normalization.

Automatic asset discovery can record both and link their versions. This makes lineage explain not
only that a tool changed, but also how its model-facing schema changed.

## Immutability And Replay

A `RunRecord` is evidence, not a cache entry. Replay never mutates it and never silently executes
the task. Rescoring, recanonicalization, or trace extraction creates new derived records with parent
lineage.

This supports three distinct workflows:

- **report replay**: render new views over unchanged evidence;
- **evidence replay**: run a versioned extractor or canonicalizer over stored ABP data;
- **execution rerun**: intentionally execute a new experiment against the current application.

## Extension Seams

Choose the narrowest seam that owns the behavior:

| Need | Extension |
| --- | --- |
| Call application code | Python task |
| Evaluate domain output | Python scorer |
| Compute from same-run observations | Deriver |
| Compare matched runs | Post-deriver |
| Enforce an acceptance rule | Policy |
| Collect a stable SDK boundary | Instrumentor |
| Map vendor fields to semantics | Source map / extractor |
| Add domain defaults | Metric pack |
| Version a behavioral component | Tracking or asset discovery |

Application-specific logic belongs in tasks and scorers. Generic SDK behavior belongs in an
instrumentor. This prevents core Autobench from accumulating one-off integrations disguised as
framework concepts.

An external framework may retain ownership of its telemetry backend. Its adapter can use
`InstrumentationRuntime.span()` to join the active ABP tree, while the external package owns backend
multiplexing and conditional restoration. Autobench core therefore supplies context and evidence
semantics without importing the framework or replacing an existing telemetry destination.

After recording, the optional OTLP adapter can map immutable experiment, run, trace, and span
evidence to an external telemetry backend. This happens outside benchmark execution and never
turns OTLP into storage, replay lineage, or canonical semantics.

## Optimization Boundary

Autobench produces optimization-grade evidence: objectives, constraints, diagnostics, factors,
asset versions, candidate feedback, and replayable run lineage. It deliberately does not select
mutation strategies or promote candidates. Consumers such as pydantic-gepa and autoptimize can use
the records without Autobench becoming coupled to one optimizer.
