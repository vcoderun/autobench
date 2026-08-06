# Core Concepts

Autobench models a benchmark as a deterministic experiment over cases and variants. The concepts
below appear in both the YAML DSL and Python API.

## BenchmarkSpec

The canonical definition of one benchmark. It contains metadata, capture policy, dataset, task,
variants, scoring, derivation, policies, instrumentation, report configuration, and a semantic
registry.

## Case And Dataset

A `Case` is one input and its optional expectation:

```python
from autobench import Case

case = Case(
    id="refund-request",
    input={"message": "Refund order 42"},
    expected={"route": "billing"},
    tags=["regression", "routing"],
    metadata={"language": "en"},
)
```

A `DatasetSpec` adds identity, version, defaults, source provenance, and attachments to a case
collection.

## Variant And Factor

A variant is one concrete configuration of the subject. Factors are independent variables such as
model, prompt version, implementation strategy, or feature flag.

```text
case: refund-request
variant: candidate
factors: model=gpt-x, prompt=refund-v4, temperature=0
```

Autobench records factors and their semantics. It does not assume that changing several factors at
once proves which one caused a metric delta.

## Task

The task adapts a case and variant to the system being benchmarked:

```python
def run(ctx, case):
    model = ctx.factor("model")
    return application.execute(case.input, model=model)
```

The task owns application calls. Autobench owns invocation, timing context, evidence preservation,
and status classification.

## Observation

An `Observation` is a typed fact produced during a run. It has a kind, name, value, optional
semantic type and unit, source, role, direction, and provenance.

Kinds include metrics, factors, events, diagnostics, outcomes, checks, and artifacts. Sources let
projection distinguish a task-emitted metric from a scorer or derived value with the same semantic
type.

## Semantic Type

A semantic type is a stable string such as `quality.correctness`, `llm.tokens.input`, or
`time.latency`. It lets generic components consume meaning rather than application-local names.

The registry carries aliases, parent relationships, aggregation hints, cardinality, privacy, and
stability metadata. Applications may extend it without replacing built-ins.

## Score

A `ScoreRecord` is evaluator output. Scores can be objectives, constraints, or diagnostics and can
include reasons, errors, and selected span provenance. They are projected into observations with
score precedence for reporting and policies.

## Derivation

A deriver computes a metric from evidence:

- per-run derivation uses one run, such as tokens + model pricing -> cost;
- post-derivation uses the experiment, such as matched baseline/candidate latency -> speedup.

Derived observations preserve their input references and source.

## Policy

A policy is a pass/fail requirement over semantic metrics. Operators are explicit fields such as
`must_equal`, `must_greater_equal`, `must_less_equal`, and `must_be_between`. Policies affect
evaluation status without hiding the underlying metric.

## ABP Trace

The Autobench Protocol (ABP) is the native execution evidence model. Instrumentors and manual spans
emit immutable signals that materialize into a trace containing spans, measurements, events, links,
references, errors, stream state, and diagnostics.

ABP is not an OpenTelemetry wrapper. Optional bridges may export it later, but Autobench controls
its semantic, replay, accounting, and optimization contracts.

## Tracked Asset

A tracked asset is a behavioral component whose exact version matters to a run: prompt, tool,
output schema, type, capability, agent, guardrail, handoff, policy, toolset, or arbitrary config.

Assets have stable logical IDs and content-addressed versions. Their history records normalized
state, source hashes, parent versions, changed fields, and diffs. An `AssetUse` binds the version and
representation actually used to a run and optional span.

## RunRecord And ExperimentRecord

`RunRecord` is the immutable evidence for one case x variant execution. `ExperimentRecord` describes
the plan, source hashes, environment, report configuration, run paths, and aggregate statuses for
the whole matrix.

Records are human-readable YAML views backed by strict typed models and versioned JSON Schemas.

## Report

A report is a replay-time projection, not the source of truth. Leaderboards, case matrices,
comparisons, distributions, and run tables all derive from RunRecords. New report configuration can
therefore analyze existing evidence without running the subject.

## Optimization Feedback

Feedback records compact failed scores, policy violations, task and span errors, factors, and asset
versions. They give optimizers structured evidence without forcing them to scrape terminal output or
infer semantics from metric names.
