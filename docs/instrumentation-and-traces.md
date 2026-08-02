# Instrumentation And Traces

Autobench supports three collection styles that can be mixed in one run:

1. Explicit `RunContext` and `Span` calls inside a task.
2. Lightweight method instrumentation for existing application classes.
3. Trace-envelope adapters for an external agent or workflow runtime.

OpenTelemetry is not a core dependency. Future OTLP bridges can export Autobench spans, but the
evidence model remains owned by Autobench.

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

`instrument_method` patches one class method and records evidence only while a RunContext is
active:

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
        )
    ],
)

try:
    run_benchmark()
finally:
    handle.close()
```

Instrumentation supports instance, static, and class methods plus sync and async results. The
extractor receives an `InstrumentCall` containing call arguments, result, error, and context. An
extractor error becomes structured diagnostic evidence rather than corrupting the application
result.

The returned `InstrumentationHandle` is also a context manager and restores the original method on
close.

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

## Adapter Boundary

Core instrumentation intentionally does not know Pydantic AI, OpenAI Agents, LangChain, DSPy, or
OpenTelemetry internals. An integration should:

1. Collect from the framework's stable hooks.
2. Convert native calls or traces into Autobench spans and observations.
3. Store large raw payloads as artifacts.
4. Keep native dependencies optional.

This boundary lets applications use existing instrumentation while RunRecords remain portable.
