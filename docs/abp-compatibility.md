# ABP Compatibility Contract

This page freezes the observable behavior preserved while the Autobench
Instrumentation Protocol (ABP) replaces legacy span and instrumentation
internals. Phases 1 through 8 now satisfy this contract. The complete public
instrumentation guide is in [Instrumentation And Traces](instrumentation-and-traces.md).

## Compatibility Boundary

The following top-level imports remain available while ABP is introduced:

```python
from autobench import (
    ArtifactRef,
    AssetVersion,
    DurationMetricSpec,
    ErrorRecord,
    InstrumentationHandle,
    InstrumentCall,
    InstrumentFactorSpec,
    InstrumentMetricSpec,
    Observation,
    RunContext,
    RunRecord,
    Span,
    SpanKind,
    SpanRecord,
    TraceEnvelope,
    attach_trace,
    get_active_run_context,
    instrument_method,
    trace_to_observations,
)
```

ABP may move implementations into new packages, but these imports and their
current behavior remain compatibility facades until a separately announced
deprecation cycle.

### Manual spans

Existing manual spans preserve these guarantees:

- `ctx.span(...)` is a synchronous context manager;
- nested spans receive the active span as `parent_id`;
- every completed span has UTC start/end timestamps and a non-negative
  monotonic duration;
- a configured duration metric is linked to the span;
- metrics, factors, events, errors, and artifacts retain their span link;
- exceptions are recorded and then propagated;
- `Span.set_output`, `Span.set_attribute`, and `Span.set_usage` continue to
  update the recorded span;
- entering instrumentation without an active run context remains a no-op;
- closing the final `InstrumentationHandle` restores the original descriptor.

The tests in `tests/test_abp_compatibility.py` are the executable form of this
contract.

### Stored evidence

Legacy model-shaped RunRecord and TraceEnvelope YAML remains loadable. ABP
will add protocol data additively and preserve the existing `spans` input
during migration. Replay must not require the task module or an optional
instrumented SDK.

The frozen legacy examples are:

- `tests/fixtures/abp/legacy_run_record.yaml`
- `tests/fixtures/abp/legacy_trace_envelope.yaml`

## Concurrency Regression Contract

`RunContext` now uses task-local ABP context. The concurrency migration is
covered by passing regression tests for all of these cases:

1. concurrent sibling spans under one parent both point to that parent;
2. a nested task inherits the parent active at task creation;
3. completing one sibling does not change the other sibling's active parent;
4. out-of-order completion does not corrupt later parent selection;
5. cancellation closes only the cancelled branch and restores its context;
6. separate RunContexts never share active spans.

The old mutable-stack reproduction is retained only in design history; it is
not the current runtime behavior.

## Canonical Trace Decision

ABP will have one canonical immutable `Trace` model. `TraceEnvelope` does not
have behavior that justifies a second trace representation, so it will become
a compatibility name for `Trace` rather than a parallel model. Existing
`TraceEnvelope(...)`, `attach_trace(...)`, and `trace_to_observations(...)`
callers continue to work.

This avoids conversion drift between manually attached traces and traces
materialized from native ABP signals.

## Package Shape

ABP code is introduced only when its phase needs it. Empty placeholder modules
are not created.

```text
autobench/
  protocol/
    ids.py
    values.py
    signals.py
    traces.py
    context.py
    capture.py
    collector.py
  instrumentation/
    models.py
    manager.py
    patching.py
    streaming.py
    pydantic_ai.py
    openai.py
    openai_agents.py
    httpx.py
```

Small modules are combined when separation would only create navigation cost.
Existing unrelated modules are not moved as part of ABP.

## Optional Integration Targets

The initial integration extras are reserved as follows:

| Extra | Research baseline | First implementation phase |
| --- | ---: | ---: |
| `autobench[pydantic-ai]` | Pydantic AI 2.22.0 | 10 |
| `autobench[openai]` | OpenAI Python 2.52.0 and 2.53.0 | 11 |
| `autobench[openai-agents]` | OpenAI Agents 0.19.2 | 11 |
| `autobench[httpx]` | HTTPX 0.28.1 | 12 |
| `autobench[instrumentation]` | all integrations above | 13 |

These versions are the public-API research baseline captured on 2026-08-03,
not a compatibility claim. Dependency metadata is added only when each native
instrumentor and its version matrix exist. Autobench core remains free of
these dependencies.

## Manual Span Performance Baseline

The baseline measures a minimal completed manual span with no observations,
artifacts, or errors. Each repeat creates one RunContext and records 10,000
spans. Timing uses `timeit.repeat`; duration comes from the host monotonic
clock. The benchmark does not enforce a CI latency threshold because shared CI
timing is not stable.

Reproduce it with:

```bash
uv run python scripts/benchmark_spans.py --iterations 10000 --repeats 7
```

Baseline captured before ABP runtime changes:

| Field | Value |
| --- | ---: |
| Date | 2026-08-03 |
| Python | 3.11.13 |
| Platform | macOS 26.1 arm64 |
| Minimum | 3,227.4 ns/span |
| Median | 3,324.8 ns/span |

The raw per-repeat values were `3467.0`, `3308.4`, `3299.3`, `3394.4`,
`3394.3`, `3324.8`, and `3227.4` ns/span. Later phases compare using the same
script and workload; they do not compare unrelated machine results.

Release measurements captured after ABP materialization on the same host:

| Workload | Measurement |
| --- | ---: |
| Manual ABP span | 28,836.7 ns/span median |
| HTTPX baseline request | 35,346.2 ns/request median |
| Instrumented HTTPX request | 224,263.6 ns/request median |
| HTTPX instrumentation overhead | 188,917.4 ns/request median |
| 10,000 x 32-byte HTTP stream | 2,618.0 ns/chunk median |
| Long-stream peak allocation | 30,918 bytes |

The long-stream result is about `3.1` peak allocated bytes per emitted chunk, which confirms the
instrumentor does not retain chunk payloads as the stream grows. These numbers characterize this
machine and are not release thresholds. Reproduce transport and stream measurements with:

```bash
uv run python scripts/benchmark_spans.py --httpx --iterations 1000 --repeats 7
uv run python scripts/benchmark_spans.py --httpx-stream --chunks 10000 --chunk-size 32 --repeats 7
```
