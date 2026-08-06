# Autobench Protocol (ABP)

## Contents

- [Purpose](#purpose)
- [Protocol model](#protocol-model)
- [Manual spans](#manual-spans)
- [Signals and trace materialization](#signals-and-trace-materialization)
- [Capture policy](#capture-policy)
- [Method instrumentation](#method-instrumentation)
- [Extraction and accounting](#extraction-and-accounting)
- [Concurrency and streaming](#concurrency-and-streaming)
- [Adapter boundary](#adapter-boundary)
- [Compatibility rules](#compatibility-rules)

## Purpose

ABP is Autobench's own instrumentation and trace protocol. It collects experiment evidence without
requiring hand-written telemetry at every SDK call. It is inspired by useful tracing concepts but
is not an OpenTelemetry wrapper and does not require OpenTelemetry.

ABP is subject- and experiment-oriented. It preserves measurements, errors, events, artifacts,
usage, factors, links, and behavioral assets needed for evaluation, replay, and optimization.
Future exporters may send ABP evidence to OTLP-compatible systems, but ABP semantics remain owned
by Autobench.

## Protocol Model

Important models include:

- immutable protocol signals;
- span start, update, end, event, error, measurement, usage, and artifact evidence;
- `TraceEnvelope` and materialized span records;
- scopes and instrumentor metadata;
- references for large or binary values;
- source maps and extraction versions;
- diagnostics for malformed, partial, or incompatible traces.

Trace IDs, span IDs, parent IDs, timestamps, execution IDs, and source identity are explicit. Do
not rely on global mutable parent state.

## Manual Spans

```python
from autobench import Direction, DurationMetricSpec, RunContext, Semantic, SpanKind


def run(ctx: RunContext, case):
    with ctx.span(
        "support.workflow",
        kind=SpanKind.WORKFLOW,
        input=case.input,
        attributes={"profile": ctx.factor("profile")},
        duration_metric=DurationMetricSpec(
            name="workflow_latency",
            semantic_type=Semantic.TIME_LATENCY,
            unit="s",
            direction=Direction.MINIMIZE,
        ),
    ) as span:
        result = application(case.input)
        span.metric("steps", result.steps, semantic_type="workflow.steps")
        span.event("routed", {"queue": result.queue})
        span.set_output(result)
        return result
```

The span owns duration. Record a second timing only when it measures a different scope or timer.

Span kinds include task, workflow, model, tool, retrieval, evaluation, HTTP/client, and generic
operation families exposed by the installed version. Use the most semantically accurate kind and
add tags for cross-cutting concerns.

## Signals And Trace Materialization

Collectors accept signals during execution. Materialization reconstructs spans deterministically,
including partial traces. Diagnostics cover:

- duplicate signal IDs;
- missing parents;
- cycles;
- invalid or reversed timestamps;
- foreign execution IDs;
- signals without starts;
- child spans outside parent timing;
- bounded diagnostic overflow.

Malformed instrumentation evidence must not alter application behavior. Preserve valid signals and
surface diagnostics instead of discarding the whole trace.

## Capture Policy

Capture is explicit and path-aware. Levels are:

- `none`;
- `metadata`;
- `hash`;
- `redacted`;
- `full`.

Runtime evidence defaults to metadata. Behavioral assets default to full so exact successful
versions can be reconstructed. Presets such as `CapturePolicy.hashed()` set both defaults; a direct
`CapturePolicy()` preserves the split defaults.

Policy features include semantic/path allow and deny rules, known secret names, custom redactors,
inline/artifact thresholds, string/collection/depth limits, binary handling, HTTP body controls,
and source-attribute retention.

Rules:

- secret paths remain redacted even under full capture;
- credentials and authorization headers never become ordinary attributes;
- unknown values do not fall back to arbitrary `repr()` capture;
- large values become content-addressed artifacts when allowed;
- capture failures become diagnostics and never fail the subject call.

## Method Instrumentation

`instrument_method()` patches a trusted method behind a lifecycle handle:

```python
from autobench import InstrumentMetricSpec, SpanKind, instrument_method

handle = instrument_method(
    Client,
    "execute",
    span="client.execute",
    span_kind=SpanKind.WORKFLOW,
    metrics=[
        InstrumentMetricSpec(
            name="steps",
            semantic_type="workflow.steps",
            value_path="result.step_count",
        )
    ],
)
try:
    benchmark.run()
finally:
    handle.close()
```

Method patches preserve descriptors, sync/async signatures, iterators, context managers, streams,
results, exceptions, and cancellation. The patch manager reference-counts owners and reports
external wrapper conflicts instead of overwriting them.

Use typed `value_factory` callables in Python when path extraction is insufficient. Serializable
configuration uses importable extractor targets, never eval strings.

## Extraction And Accounting

ABP extraction turns raw protocol evidence into semantic observations. Built-in extractor concepts
include direct signal, span, usage, and composite extraction.

Source maps canonicalize provider or SDK attributes into Autobench semantics. For example, an SDK
token field can map to `llm.tokens.input` without making the SDK's naming convention canonical.

Accounting-safe extraction must distinguish:

- direct evidence from one logical operation;
- aggregate or summary evidence;
- nested layers representing the same model request;
- retries and streamed chunks;
- request totals versus component subtotals.

Do not add token or cost observations from Pydantic AI, OpenAI client, and HTTPX layers together
when they represent one request. Correlate logical operations and retain source provenance.

Replay extraction may apply a newer extractor or source map to an immutable recorded trace. It
writes new derived evidence with lineage; it does not mutate the original trace or observations.

## Concurrency And Streaming

ABP uses task-local context so concurrent sibling operations preserve correct parentage. Context
must not leak after a stream closes, a task is cancelled, or an instrumentor manager exits.

Streaming instrumentation must handle:

- normal completion;
- early close;
- iterator failure;
- context-manager close failure;
- cancellation;
- late use after manager close;
- partial usage before termination.

Finalization must be idempotent. A stream wrapper must not eagerly consume or change the stream.

## Adapter Boundary

A native instrumentor owns:

- compatibility detection and optional dependency versions;
- exact patch points in the target SDK;
- ABP span/signal extraction;
- SDK-to-Autobench semantic mapping;
- behavioral asset extraction at supported boundaries;
- lifecycle install/close;
- duplicate and suppression policy.

Core owns protocol models, collector, capture, traces, extraction contracts, records, and replay.
Do not import provider SDKs from core protocol or records modules.

## Compatibility Rules

- Replay works without optional SDK imports.
- Unknown additive protocol fields remain forward-compatible where declared.
- Version incompatibility produces a clear diagnostic before patching.
- Explicit instrumentors take precedence over automatic discovery.
- Duplicate instrumentor IDs are rejected or deterministically resolved.
- Scoped suppression prevents nested wrappers from duplicating one operation.
- Instrumentation never changes a subject result, exception identity, cancellation, or stream
  semantics.

