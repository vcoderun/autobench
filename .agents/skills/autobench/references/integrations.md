# Native Instrumentation And Integrations

## Contents

- [Installation](#installation)
- [Automatic instrumentation](#automatic-instrumentation)
- [Explicit configuration](#explicit-configuration)
- [Pydantic AI](#pydantic-ai)
- [OpenAI Python](#openai-python)
- [OpenAI Agents](#openai-agents)
- [HTTPX](#httpx)
- [Pydantic-GEPA](#pydantic-gepa)
- [Custom SDKs](#custom-sdks)
- [Diagnostics](#diagnostics)

## Installation

Base Autobench does not require provider SDKs. Install only the integration needed:

```bash
uv add 'autobench[pydantic-ai]'
uv add 'autobench[openai]'
uv add 'autobench[openai-agents]'
uv add 'autobench[httpx]'
uv add 'autobench[pydantic-gepa]'
uv add 'autobench[instrumentation]'
```

The combined instrumentation extra includes the supported Pydantic AI, OpenAI, OpenAI Agents,
HTTPX, and pydantic-gepa versions. Check the target package's `pyproject.toml` rather than assuming
version ranges.

## Pydantic-GEPA

Use `autobench[pydantic-gepa]` to record optimizer lifecycle without a handwritten observer. YAML
configuration is:

```yaml
instrumentation:
  pydantic_gepa:
    detail: full
    assets:
      representations: [definition, effective]
```

The instrumentor subscribes to pydantic-gepa event contract `1` and records optimization,
composition, engine, evaluation, candidate, reflection, and final-rescore lifecycle. It also
records objective and dataset declarations, evaluation/optimizer budgets, selection contenders,
candidate parentage, and effective component asset versions. `summary`, `evaluations`, and `full`
control high-cardinality spans; the replay-safe `autobench.pydantic_gepa/v1` extension is retained
in every mode.

Do not write a second Autobench observer or duplicate Pydantic AI/OpenAI usage on optimizer spans.
Those nested native instrumentors own direct model/token/request evidence. Use the maintained
`examples/pydantic_gepa` project and the canonical
`docs/pydantic-gepa-instrumentation.md` page for the full contract.

The maintained offline matrix covers the standard GEPA backend, an Optimize Anything Omni
pipeline, one coupled prompt/tool/output-schema candidate, and checkpoint resume. The optional
live example proves that optimizer evidence can contain native Pydantic AI and HTTPX evidence
without assigning model usage to optimizer spans a second time.

## Automatic Instrumentation

```python
from autobench import Benchmark

benchmark = Benchmark("support-agent").instrument_all()
```

`instrument_all()` selects installed and compatible built-ins. Missing integrations are skipped
with per-run diagnostics by default. Use exclusions and strict mode when needed:

```python
benchmark.instrument_all(exclude={"httpx"}, strict=True)
```

Rules:

- explicit configuration, including explicit disabled state, overrides automatic selection;
- custom instrumentors with the same ID take precedence;
- repeated `instrument_all()` replaces previous automatic settings;
- each selected instrumentor installs once for the benchmark lifecycle;
- automatic behavioral asset discovery is enabled at semantic SDK boundaries unless configured
  otherwise.

## Explicit Configuration

Use `.instrument(...)` with typed settings for Python-only runtime instrumentors or YAML
`instrumentation` for serializable built-ins. Do not grow integration APIs through unvalidated
`**kwargs`.

Common configuration dimensions:

- enabled/disabled;
- target version compatibility;
- capture policy or HTTP capture settings;
- asset discovery include/exclude families;
- definition/effective representations;
- strict missing-dependency behavior.

## Pydantic AI

Native Pydantic AI instrumentation records supported agent, model, tool, output validation,
streaming, retry, and usage operations. Automatic asset discovery can capture:

- agent identity and instructions/system prompts;
- tools and toolsets;
- output schemas;
- capabilities and capability instructions;
- definition and effective forms visible at the call boundary.

Capability-scoped assets use stable scope information so identical instruction kinds from two
capabilities do not collide. The conceptual identity includes capability name, asset kind, and
local asset ID.

Use Pydantic AI's typed `output_type` directly in application code. Do not wrap it in an Autobench
injection object unless an optimizer is actually injecting candidate values.

## OpenAI Python

The OpenAI instrumentor records supported sync/async request and streaming lifecycles, model
identity, usage, errors, and relevant request factors. It composes with HTTPX without double
counting logical usage.

Input bodies, output bodies, and headers follow capture policy. API keys and authorization data are
never benchmark attributes.

## OpenAI Agents

The OpenAI Agents integration records native agent, model, tool, handoff, guardrail, and policy
boundaries exposed by supported SDK versions. Automatic assets may include agent instructions,
tools/toolsets, handoffs, guardrails, output schemas, and policies.

It preserves native tracing/processor behavior rather than replacing it. Autobench produces ABP
evidence for benchmark ownership.

## HTTPX

HTTPX instrumentation operates at transport boundaries and supports sync, async, redirect, error,
and stream lifecycles. Default capture is conservative:

- method, host, status, and timing are safe metadata;
- paths may be hashed;
- request/response headers are allowlisted;
- bodies are disabled unless explicitly enabled;
- body size and content type determine inline, artifact, truncated, or omitted behavior.

Do not infer LLM token usage from HTTP bodies in core. Provider-aware extraction belongs in the
provider integration.

## Custom SDKs

Use `instrument_method()` and `InstrumentAssetSpec` when an arbitrary SDK exposes stable method
boundaries:

```python
from autobench import InstrumentAssetSpec, SpanKind, instrument_method

handle = instrument_method(
    WorkflowClient,
    "execute",
    span="workflow_client.execute",
    span_kind=SpanKind.WORKFLOW,
    operation_family="workflow_client.execute",
    assets=[
        InstrumentAssetSpec(
            kind="prompt",
            local_id="instructions",
            value_path="kwargs.instructions",
        ),
        InstrumentAssetSpec(
            kind="tool",
            local_id="tools",
            value_path="kwargs.tools",
            many=True,
        ),
        InstrumentAssetSpec(
            kind="output_schema",
            local_id="output",
            value_path="kwargs.output_type",
        ),
    ],
)
try:
    benchmark.run()
finally:
    handle.close()
```

`value_path` can traverse trusted arguments, mappings, attributes, results, and supported
zero-argument accessors. Use `many=True` for independent mapping/sequence assets. Use a typed
factory for Python-only extraction or an importable extractor target for serializable settings.

Design custom integrations generically. Do not copy one application's class names or semantics
into Autobench core.

## Diagnostics

```bash
autobench instrumentation doctor
autobench instrumentation trace runs/latest
```

Doctor reports installed, unavailable, unsupported, conflicting, or degraded integrations. Trace
inspection loads recorded ABP evidence without importing provider SDKs.

When native evidence is missing:

1. confirm the optional extra is installed;
2. run doctor;
3. confirm the benchmark selected the integration;
4. inspect skipped instrumentation diagnostics in the run;
5. verify the SDK version is supported;
6. inspect trace composition for suppression or duplicate layers;
7. use manual spans only for application operations no native boundary can observe.
