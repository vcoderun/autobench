# Native Instrumentation

Autobench native instrumentors collect ABP traces from supported SDKs without task-level
`ctx.span()` or `ctx.metric()` calls. They are optional adapters around public hooks or pinned,
reviewed patch points. Core benchmark, record, replay, and report imports do not require any of the
instrumented SDKs.

## Install

Install one integration or the complete set:

```bash
pip install 'autobench[pydantic-ai]'
pip install 'autobench[openai]'
pip install 'autobench[openai-agents]'
pip install 'autobench[httpx]'
pip install 'autobench[pydantic-gepa]'
pip install 'autobench[instrumentation]'
```

The integration registry is lazy. Loading a YAML spec, replaying evidence, or running
`autobench instrumentation doctor` does not import an SDK that is not installed.

## Automatic Discovery

Use `instrument_all()` when the benchmark should activate every built-in integration that is
installed and compatible in the current environment:

```python
from autobench import Benchmark

benchmark = Benchmark("support-agent").instrument_all()
```

Semantic instrumentors also discover SDK-visible behavioral assets by default. The application does
not need tracking decorators for prompts, tools, output schemas, capabilities, agents, guardrails,
handoffs, or policies already visible at those boundaries. See
[Automatic Asset Discovery](automatic-asset-discovery.md) for identity, source/effective links,
privacy, persistence, and custom SDK extraction.

Unavailable or unsupported integrations are skipped by default and recorded on each run as
`instrumentation.skipped` diagnostic evidence. Use `strict=True` when the environment must support
the complete selected set:

```python
benchmark = Benchmark("support-agent").instrument_all(
    exclude={"httpx"},
    strict=True,
)
```

Explicit settings take precedence over discovery, including an explicit `false`. A custom runtime
instrumentor with the same instrumentor ID also takes precedence, so automatic discovery does not
install a duplicate. Calling `instrument_all()` again replaces the previous automatic settings.

### Live OpenRouter Trace

The live Pydantic AI example exercises automatic discovery across all three active layers:

```bash
uv sync --extra instrumentation
export OPENROUTER_API_KEY=...
export OPENROUTER_MODEL=openrouter:openai/gpt-5.6-luna
uv run python examples/pydantic_ai/openrouter_instrument_all.py \
  --record /tmp/autobench-openrouter
```

The benchmark itself only opts in once:

```python
benchmark = Benchmark("openrouter-shopping-agent").instrument_all()
```

The example uses plain instructions, a plain tool function, and an undecorated Pydantic output type.
Automatic discovery adds Pydantic AI, OpenAI client, and HTTPX transport instrumentors. One real
request therefore records agent, model, tool, output
validation, stream, client request, and transport spans with their native parentage. It also records
model identity, token usage, durations, HTTP method/host/path/status, score observations, asset
versions, capture diagnostics, and replayable source provenance. HTTP bodies and credentials remain
redacted by the default capture policy.

The full source is `examples/pydantic_ai/openrouter_instrument_all.py`. It deliberately contains no
manual `ctx.span()` or `ctx.metric()` calls so the resulting record demonstrates native collection
rather than hand-authored benchmark telemetry.

## YAML

Instrumentation belongs to the named benchmark:

```yaml
# yaml-language-server: $schema=schemas/0.3.0/benchmark_schema.json
benchmark:
  support-agent:
    dataset:
      source: file://datasets/cases.yaml
    run:
      python: support_benchmark:run
    variants:
      baseline:
        factors:
          model.name: openrouter:openai/gpt-5.6-luna
    instrumentation:
      all:
        exclude: [httpx]
        strict: false
        assets:
          representations: [definition, effective]
          include: [prompt, tool, output_schema, capability]
      pydantic_ai: {}
      openai: {}
      httpx:
        capture:
          path: hash
          request_headers: [x-request-id]
          response_headers: [x-request-id]
          request_body: false
          response_body: false
          max_body_bytes: 65536
```

Use `false` to retain a known integration in a shared spec without installing it:

```yaml
instrumentation:
  openai_agents: false
```

Unknown integration names and unknown settings fail validation. YAML never evaluates Python
expressions.

The `all` block follows the same precedence rules as the Python builder. In this example HTTPX is
excluded from discovery but its explicit capture settings still install it; all explicit entries
remain authoritative.

## Python

The fluent API accepts typed, serializable settings:

```python
from autobench import (
    Benchmark,
    HTTPXCaptureSettings,
    HTTPXInstrumentation,
    OpenAIInstrumentation,
)

benchmark = Benchmark("streaming-chat").instrument(
    OpenAIInstrumentation(),
    HTTPXInstrumentation(
        capture=HTTPXCaptureSettings(
            path="hash",
            response_headers=("x-request-id",),
        )
    ),
)
experiment = benchmark.run()
```

It also accepts a custom `Instrumentor` instance. Runtime instances are installed for the whole
benchmark matrix and closed even when execution fails. They are intentionally not serialized into
the YAML spec:

```python
benchmark.instrument(MyNativeInstrumentor(settings))
```

Duplicate instrumentor IDs are rejected before hooks are installed. This avoids ambiguous
ownership when a typed setting and a custom instance configure the same integration.

### SDK-Owned Telemetry Backends

Some frameworks already route lifecycle spans through a process-wide or context-local backend.
Implement the Autobench integration against that stable backend contract instead of patching every
framework method. `InstrumentationRuntime.span()` creates an unentered ABP span in the active
benchmark run and returns `None` when capture is inactive or suppressed:

```python
span = runtime.span(
    instrumentor.info,
    "framework.workflow",
    kind="workflow",
    attributes={"framework.operation": "run"},
    target_version=installed_framework_version,
    suppression_keys=("framework",),
)

if span is None:
    return call_subject()

with span:
    return call_subject()
```

Use `runtime.metric()` for host counters and histograms that should become semantic observations.
It follows the same active-context and suppression rules and attaches the observation to the
currently entered ABP span.

Use `runtime.current_span()` when the host enriches a span owned by a nested native instrumentor.
The returned `CurrentSpan` updates attributes, usage, events, and errors on that active ABP span.
Check `runtime.is_installed()` before claiming native ownership so disabling a child instrumentor
does not silently remove model or tool evidence.

The entered span becomes the parent of nested Autobench instrumentors automatically. This is the
preferred integration for an agent runtime whose workflow span should contain native Pydantic AI
agent/model/tool spans. Keep provider usage on the native provider or framework span; a higher-level
workflow span should not copy the same token totals.

Backend installation must preserve existing observers:

1. read the current host backend;
2. install a host-owned composite containing the previous backend and the ABP backend;
3. retain the exact installed composite identity;
4. on close, restore the previous backend only if the composite is still current.

This policy allows existing OTel or Logfire telemetry and Autobench evidence to coexist. Autobench
does not own the host backend protocol and core does not import the external framework.

## Built-In Integrations

| Integration | Layer | Collection seam | Evidence |
| --- | --- | --- | --- |
| Pydantic AI | framework | public agent capability | spans plus agent/capability/prompt/tool/toolset/output-schema lineage |
| Pydantic-GEPA | optimizer | typed event subscription | optimization/composition/engine/evaluation spans, budgets, candidate lineage, and component asset versions |
| OpenAI Python | client | reviewed public client methods and stream types | spans plus effective prompt/tool/output-schema lineage |
| OpenAI Agents | framework | native trace processor and public Runner surface | spans plus agent/prompt/tool/output-schema/guardrail/handoff/policy lineage |
| HTTPX | transport | public transport methods | request method/host/path policy, status, selected headers, body metadata, stream lifecycle |

The pydantic-gepa integration has its own complete guide, including Optimize Anything composition,
detail modes, semantic accounting, assets, replay, and reports: [Pydantic-GEPA
Instrumentation](pydantic-gepa-instrumentation.md).

Run compatibility diagnostics before a benchmark:

```bash
autobench instrumentation doctor
```

The Rich output shows availability, installed version, supported range, abstraction layer,
mechanism, sync/async/streaming capabilities, span kinds, semantic families, capture defaults, and
degradation diagnostics.

## Layered Traces

Instrumentors compose instead of flattening one another. A Pydantic AI request using the OpenAI
client over HTTPX can produce this parent chain:

```text
task
  agent
    llm framework operation
      OpenAI client operation
        HTTP request
```

Transport spans do not emit token or cost usage. Framework aggregate usage and client direct usage
retain different accounting scopes. Trace extraction selects one authoritative direct layer and
keeps aggregate values as validation evidence, so enabling HTTPX cannot inflate LLM totals.

## Streaming Lifecycle

A stream span does not end when an SDK returns an iterator. It remains open until the stream:

- completes normally;
- raises;
- is cancelled;
- is explicitly closed early;
- is abandoned when the instrumentor manager closes.

ABP records first-chunk evidence, item/chunk counts, partial state, and the final end reason. Native
items, exceptions, iterator methods, and context-manager behavior pass through unchanged.

## HTTP Privacy Defaults

HTTPX capture defaults are deliberately conservative:

- query-free path hash, not the raw path;
- no request or response headers unless named;
- authorization, cookies, API keys, tokens, passwords, and secrets always redacted;
- no request or response body capture;
- bounded capture when bodies are explicitly enabled;
- binary bodies represented by metadata and a digest, not embedded bytes.

`path: full` is an explicit opt-in. Query strings and URL user information are not recorded by the
path setting. Capture policies apply before evidence reaches a RunRecord.

## Trace Diagnostics

Every native span records an `InstrumentationScope`: instrumentor and target versions, mechanism,
abstraction layer, and source convention. Source facts can be retained alongside canonical
Autobench semantic attributes. Unsupported library versions fail installation instead of silently
patching an unknown lifecycle.

Inspect recorded trace shape without importing task modules or optional SDKs:

```bash
autobench instrumentation trace runs/support-agent/exp_...
```

The command reports per-case span/root counts, partial traces, diagnostics, span-kind totals, and
instrumentor composition.

## Replay Without SDKs

RunRecords contain materialized ABP traces, not live provider objects. A reporting or optimization
worker can replay and re-extract evidence with only Autobench installed:

```python
from autobench import CompositeExtractor, SignalExtractor, SpanExtractor, UsageExtractor
from autobench.records.replay import load_run_record, replay_extraction

record = load_run_record(path, root_dir=run_dir)
derived = replay_extraction(
    record,
    CompositeExtractor(SignalExtractor(), SpanExtractor(), UsageExtractor()),
)
```

Extraction creates a derived record with lineage; it never mutates the original record.

## ABP And OpenTelemetry

ABP is not an OpenTelemetry wrapper and has no OTel dependency. It is Autobench's evidence protocol
for benchmark execution, semantic measurements, accounting scope, partial streams, replay, and
optimization lineage. Native instrumentors use the same kinds of stable SDK hooks that mature OTel
instrumentations validate, but emit ABP directly.

The optional [OTLP exporter](otlp-export.md) can replay immutable ABP evidence to systems such as
Logfire, Datadog, or a vendor-neutral collector. It remains an outbound adapter: ABP is the source
evidence model, and importing the base Autobench package does not require an OTel SDK or collector.

## Protocol Stability

ABP protocol version `1` is the initial public serialized contract. Autobench `0.2.x` preserves the
meaning of its signal, trace, scope, provenance, and accounting fields. Readers retain unknown
additive data through extension maps, while a breaking wire-format change requires a new protocol
version. Instrumentor patch points are compatibility-gated separately because provider SDK
lifecycles can change independently of ABP.

## Examples

- `examples/abp_manual`: explicit workflow spans plus method instrumentation.
- `examples/abp_concurrent`: concurrent sibling operations with task-local parentage.
- `examples/pydantic_ai`: tool use, retry, streaming, and structured output; OpenAI models add
  OpenAI and HTTPX layers.
- `examples/abp_openai`: offline official OpenAI streaming over an HTTPX mock transport.
- `examples/abp_openai_agents`: offline native OpenAI Agents trace-processor workflow.
- `examples/abp_replay`: trace extraction from recorded evidence without importing provider SDKs.
