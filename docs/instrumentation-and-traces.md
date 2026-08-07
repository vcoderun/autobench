# Instrumentation And Traces

Autobench supports four collection styles that can be mixed in one run:

1. Explicit `RunContext` and `Span` calls inside a task.
2. Lightweight method instrumentation for existing application classes.
3. Trace-envelope adapters for an external agent or workflow runtime.
4. Native Pydantic AI, OpenAI, OpenAI Agents, and HTTPX instrumentors configured from Python or
   YAML.

OpenTelemetry is not a core dependency. The optional outbound
[OTLP exporter](otlp-export.md) can replay immutable Autobench evidence to compatible backends,
while ABP remains the canonical evidence model.

See [Native Instrumentation](native-instrumentation.md) for the typed fluent API, YAML DSL,
compatibility doctor, privacy defaults, layered traces, and provider examples.

ABP is the native collection protocol underneath these APIs. It owns signal ordering, task-local
context, capture policy, instrumentation scope, trace materialization, and compatibility
diagnostics. Instrumentors emit ABP evidence directly; they do not create OpenTelemetry spans and
then convert them back into Autobench records.

## Manual Spans

```python
from autobench import DurationMetricSpec, Semantic, SpanKind


def run_case(ctx, case):
    with ctx.span(
        "support_agent",
        kind=SpanKind.AGENT,
        input=case.input,
        duration_metric=DurationMetricSpec(
            name="agent_latency",
            semantic_type=Semantic.TIME_LATENCY,
            unit="ms",
        ),
    ) as agent:
        result = call_agent(case.input)
        agent.set_output(result)
        agent.outcome(result["ok"])
        return result
```

Span duration is calculated when the context manager closes. Nested spans preserve parent-child
relationships and retain evidence emitted before an exception.

## Span Kinds

`SpanKind` includes:

- agent
- LLM
- tool
- retriever
- parser
- workflow
- custom

Kinds are semantic selectors, not restrictions. A domain can use custom kinds and tags while
generic agentic scorers continue selecting standard spans.

## Method Instrumentation

`instrument_method` is the high-level helper for one class method. It records evidence only while
a `RunContext` is active:

```python
from autobench import InstrumentMetricSpec, Semantic, instrument_method

handle = instrument_method(
    SearchClient,
    "search",
    span="search.request",
    metrics=[
        InstrumentMetricSpec(
            name="result_count",
            semantic_type="retrieval.result_count",
            value_factory=lambda call: len(call.result),
        ),
        InstrumentMetricSpec(
            name="request_count",
            semantic_type="llm.requests",
            value_path="result.usage.requests",
        ),
    ],
)

try:
    run_benchmark()
finally:
    handle.close()
```

Instrumentation supports:

- instance, static, class, and inherited methods;
- synchronous and asynchronous calls;
- iterators and generators, including `send`, `throw`, and early close;
- asynchronous iterators and generators, including `asend`, `athrow`, and `aclose`;
- synchronous and asynchronous context managers.

The wrapper preserves the original descriptor, callable signature, return value, exception
identity, and lazy streaming behavior. A stream span ends when the stream actually completes,
fails, times out, or closes, so its duration is not merely the time required to construct an
iterator.

`value_factory` is the typed Python extraction seam. It receives an `InstrumentCall` containing
the bound instance, arguments, result, error, stream item count, and last stream item. `value_path`
is the declarative alternative for trusted attribute, mapping, and zero-argument accessor paths.
Autobench does not execute arbitrary YAML expressions.

Extraction and lifecycle callback errors are recorded as evidence or compatibility diagnostics.
They do not replace the application's result or exception.

The returned `InstrumentationHandle` is also a context manager and restores the original method on
close.

## Scoped Suppression

Instrumentation can be suppressed for the current task without changing global process state:

```python
from autobench import suppress_instrumentation

with suppress_instrumentation("search.client"):
    result = client.search("internal health check")
```

Suppression keys can identify an instrumentor or an operation family. Unrelated instrumentors stay
active, nested scopes compose, and context tokens are reset even when application code raises. An
empty `suppress_instrumentation()` scope suppresses all ABP instrumentation in the current task.

## Native Instrumentors

Reusable SDK integrations implement the `Instrumentor` contract:

```python
from autobench import (
    Compatibility,
    InstrumentationHandle,
    InstrumentationRuntime,
    InstrumentorInfo,
)
from autobench.protocol import AbstractionLayer, CaptureMechanism


class ClientInstrumentor:
    info = InstrumentorInfo(
        id="example.client",
        version="1.0.0",
        target_distribution="example-client",
        supported_versions=">=2,<3",
        mechanism=CaptureMechanism.HOOK,
        layer=AbstractionLayer.CLIENT,
        span_kinds=("client.request",),
        semantic_families=("request", "response"),
    )

    def check(self) -> Compatibility:
        return Compatibility.compatible()

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        unsubscribe = register_native_callback(...)
        return InstrumentationHandle(unsubscribe, info=self.info)
```

Install instrumentors directly through one manager when building a custom integration:

```python
from autobench import InstrumentationManager

with InstrumentationManager() as manager:
    compatibility = manager.check(ClientInstrumentor())
    if compatibility.installable:
        manager.install(ClientInstrumentor())
        run_benchmark()
```

`InstrumentorInfo` declares stable identity, target package and version range, mechanism, layer,
semantic families, source convention, optional dependencies, and sync/async/streaming/native-hook
capabilities. `Compatibility` distinguishes compatible, degraded, unavailable, unsupported, and
conflicting installations. Missing or incompatible optional dependencies degrade only the feature
that needs them; a missing required target package prevents installation.

Installing the same instrumentor version twice increments an owner reference count instead of
installing duplicate hooks. Closing the final handle unregisters native callbacks or restores the
exact patched descriptor. Competing owners can instrument the same method independently, while an
external wrapper replacement produces a conflict diagnostic instead of being overwritten.

### External Backend Integration

An SDK that already exposes its own telemetry backend does not need a second method-patching layer.
Its Autobench adapter can install a backend that asks `InstrumentationRuntime` for a span:

```python
from collections.abc import Generator
from contextlib import contextmanager

from autobench import InstrumentationRuntime, InstrumentorInfo, Span


class AutobenchBackend:
    def __init__(self, runtime: InstrumentationRuntime, info: InstrumentorInfo) -> None:
        self.runtime = runtime
        self.info = info

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, bool | str | int | float],
    ) -> Generator[Span | None, None, None]:
        span = self.runtime.span(
            self.info,
            name,
            kind="workflow",
            attributes=attributes,
            target_version="1.4.0",
            suppression_keys=("example-sdk",),
        )
        if span is None:
            yield None
            return
        with span:
            yield span
```

`runtime.span()` returns an unentered span only during an active Autobench run. It returns `None`
outside a run, in a matching suppression scope, or when the active protocol context is not owned by
a `RunContext`. Entering the span makes it the parent of nested built-in instrumentation, including
Pydantic AI, OpenAI, and HTTPX calls. Exiting it records normal completion, failure, timeout, or
cancellation without changing the host SDK result or exception.

External counters and histograms can use the matching observation seam:

```python
runtime.metric(
    info,
    "framework.jobs",
    1,
    semantic_type="workflow.jobs",
    unit="job",
    suppression_keys=("example-sdk",),
)
```

The observation is attached to the currently entered Autobench span when one exists and is marked
with `source=instrumentation`. Like `runtime.span()`, this method returns `None` outside an owned,
unsuppressed benchmark context.

Backends that enrich a span owned by another installed instrumentor can use the active-span view:

```python
current = runtime.current_span(info, suppression_keys=("example-sdk",))
if current is not None:
    current.set_attribute("framework.profile", "reviewer")
    current.set_usage("input_tokens", 420)
    current.event("framework cache hit", semantic_type="event.occurrence")
```

`CurrentSpan` can read and set attributes, set usage, attach instrumentation events, and record an
exception. It never renames the ABP operation after `span_start`; an external backend that supports
display-name updates should store that value as an attribute. Calls return `None` outside an active
legacy-backed span, including a raw ABP context that is not owned by a `RunContext`.

`runtime.is_installed(instrumentor_id)` and `runtime.installed_ids` let a composite backend choose
the native owner dynamically. For example, a framework adapter can suppress its model/tool spans
only while `autobench.pydantic_ai` is actually installed, avoiding both missing evidence and double
accounting.

The adapter remains responsible for its host's backend ownership. If a backend is already installed,
compose or multiplex it with the Autobench backend; do not replace it silently. The close callback
must restore the previous backend only when the adapter still owns the active slot. This preserves
an existing OTel, Logfire, or application backend and avoids clobbering a newer installation.

Mechanisms should be selected in this order:

1. stable native processor or callback;
2. stable native wrapper/decorator extension point;
3. public method patch;
4. explicitly version-pinned private method patch;
5. unsupported with a compatibility diagnostic.

Application benchmarks normally use the higher-level lifecycle owner instead:

```python
from autobench import Benchmark, HTTPXInstrumentation, OpenAIInstrumentation

benchmark = Benchmark("chat").instrument(
    OpenAIInstrumentation(),
    HTTPXInstrumentation(),
)
result = benchmark.run()
```

`Benchmark.instrument(...)` installs configured and custom instrumentors before any matrix item,
keeps them active through concurrent runs and streams, and closes them after execution.

## Trace Envelopes

Adapters can normalize a completed external trace into `TraceEnvelope`:

```python
from autobench import TraceEnvelope, attach_trace

trace = TraceEnvelope(
    trace_id="trace-42",
    name="checkout-agent",
    input={"cart_id": "c1"},
    output={"status": "complete"},
    spans=tuple(converted_spans),
    attributes={"framework": "custom-agent-runtime"},
)

attach_trace(ctx, trace)
```

`attach_trace` preserves spans and errors and projects known usage, model, provider, duration, and
outcome fields into semantic observations. Large native trace payloads should be written as an
artifact and referenced by `raw_artifact`.

## Pydantic AI Usage

Install the optional native instrumentor when the application uses Pydantic AI:

```bash
pip install 'autobench[pydantic-ai]'
```

The integration uses Pydantic AI's public capability hooks and only injects its
capability while an Autobench run is active:

```python
from autobench import Benchmark, PydanticAIInstrumentation

experiment = benchmark.instrument(PydanticAIInstrumentation()).run()
```

No manual span or metric calls are required. The instrumentor captures:

- agent runs and streamed execution;
- model requests, requested and response model identities, providers, and direct usage;
- tool argument validation, execution, retry, failure, approval, and deferred control flow;
- structured-output validation;
- first-chunk latency, partial streams, failures, and normal completion;
- tracked prompt, tool, and output-schema versions;
- multimodal metadata, with binary references only when the capture policy requests full content.

The instrumentor composes with user event handlers and Pydantic AI's own
`Instrumentation` capability. It does not configure, replace, or require
OpenTelemetry. Autobench supports the audited public integration seam across
Pydantic AI 2.22.x and 2.23.x; `InstrumentationManager.check()` reports incompatible
versions before installing hooks.

Application outputs and exceptions are passed through unchanged. Aggregate agent
usage and direct model usage retain distinct accounting scopes, and cost remains a
downstream derivation. Replaying recorded ABP evidence does not require Pydantic AI
to be installed.

See the [live Pydantic AI example](examples.md#pydantic-ai-live-layered-instrumentation) for a tool-using,
structured-output, streaming benchmark with a retry path.

### Usage Bridge

Pydantic AI usage can be normalized without importing Pydantic AI into core:

```python
from autobench import PydanticAIUsage, record_pydantic_ai_usage

record_pydantic_ai_usage(
    ctx,
    PydanticAIUsage(
        requests=1,
        input_tokens=420,
        output_tokens=83,
        model_name="gemini-3-flash-preview",
        provider="openrouter",
    ),
)
```

The bridge emits canonical LLM token, model, and provider observations that pricing derivation and
reports can consume.

## Trace Extraction And Accounting

Instrumentors record immutable facts. Extractors turn a completed ABP trace into semantic
observations without mutating that trace:

```python
from autobench import (
    CompositeExtractor,
    SignalExtractor,
    SpanExtractor,
    UsageExtractor,
    replay_extraction,
)

extractor = CompositeExtractor(
    SignalExtractor(),
    SpanExtractor(),
    UsageExtractor(),
)
derived_record = replay_extraction(record, extractor)
```

The extractors have separate ownership:

- `SignalExtractor` reconstructs measurements and events and preserves their accounting scope,
  abstraction layer, logical operation ID, and instrumentor identity.
- `SpanExtractor` derives generic operation counts, direct durations, maximum depth and fan-out,
  critical-path makespan, parallelism, incomplete work, retry/recovery, validation, approval,
  tool-call, message-growth, and reference evidence.
- `UsageExtractor` derives LLM request, token, requested-model, response-model, and provider
  evidence. It never derives cost.

Every extractor has a stable name and version. Replay records both in extraction evidence and
RunRecord lineage. Replaying a newer version replaces observations owned by the older version in
the derived record; the parent record remains unchanged.

### Direct And Aggregate Evidence

ABP keeps all raw measurements but prevents framework/client nesting from inflating totals:

1. Aggregate parent measurements are never added to direct child measurements.
2. Usage totals select one abstraction layer per semantic, preferring client evidence before
   framework, application, and transport evidence.
3. Equivalent direct operations with a shared logical operation ID are counted once.
4. Equal equivalent values are deduplicated. Conflicting values require a unique explicit
   authority; unresolved conflicts produce `ambiguous_direct_measurement` and are excluded from
   the derived total.
5. Aggregate values are retained as validation evidence. A disagreement with the direct total
   produces `aggregate_measurement_mismatch`.
6. Requested and response model identities remain separate factors.

Reports and `ObservationQuery.first_exact()` prefer an accounting-safe aggregate summary over
same-source per-operation direct evidence. Raw and projected queries can still inspect every
underlying observation.

Graph timing uses monotonic timestamps only. `time.critical_path` is the observed trace makespan,
and `operation.parallelism` is completed leaf work divided by that makespan. Invalid or partial
clock evidence is retained through diagnostics rather than repaired with wall-clock subtraction.

## Adapter Boundary

Core instrumentation intentionally does not know Pydantic AI, OpenAI Agents, LangChain, DSPy, or
OpenTelemetry internals. An integration should:

1. Collect from the framework's stable hooks.
2. Convert native calls or traces into Autobench spans and observations.
3. Store large raw payloads as artifacts.
4. Keep native dependencies optional.

This boundary lets applications use existing instrumentation while RunRecords remain portable.
