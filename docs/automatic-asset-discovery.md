# Automatic Asset Discovery

Automatic asset discovery turns SDK-visible prompts, tools, output schemas, capabilities, agents,
guardrails, handoffs, and policies into versioned benchmark evidence. It is part of the common ABP
instrumentation runtime, not a Pydantic AI-specific tracking mode.

The application does not need `@track.prompt`, `@track.tool`, or `@track.type` when a supported
instrumentor can already see the behavioral component at a stable SDK boundary. Explicit tracking
still composes with discovery when the application owns a better identity, semantic type, source
path, or parent version.

## Quick Start

Install the SDK integrations used by the application:

```bash
pip install 'autobench[instrumentation]'
```

Then enable compatible integrations for the benchmark:

```python
from autobench import Benchmark

benchmark = Benchmark("support-agent").instrument_all()
```

While a benchmark run is active, Autobench now:

1. observes definitions at supported framework and client surfaces;
2. normalizes them into SDK-independent asset candidates;
3. resolves stable logical identity and aliases;
4. computes behavioral content versions;
5. attaches exact asset uses to the owning run and span;
6. persists referenced histories when the experiment is recorded.

Calls made outside an active Autobench run remain unchanged and produce no discovery evidence.

## What Counts As An Asset

An asset is a versionable component whose content or behavior can change benchmark outcomes.

| Observed value | Autobench treatment |
| --- | --- |
| static or callable instructions | prompt definition |
| rendered system/developer instructions | effective prompt |
| function, hosted, native, or MCP tool | tool definition or effective tool schema |
| Pydantic type, dataclass, or JSON Schema | output schema |
| Pydantic AI capability | scoped composite asset |
| OpenAI Agents guardrail or handoff | guardrail or handoff asset |
| routing, retry, output, or tool-use configuration | policy asset |
| composed agent or toolset | composite asset with child locators |
| model/provider/settings | factor or configuration evidence |
| user input, output, message history, tool arguments/results | evidence, not an asset |
| tokens, cost, latency, quality | metric |

Kinds are open strings. Autobench does not require every domain to fit an AI-only taxonomy.

## Definitions And Effective Representations

Definitions and model-facing values answer different questions:

```text
definition
  Which source component did the application declare?

effective
  Which resolved representation did this operation actually use?
```

A dynamic instruction callable is a definition asset. Calling it during the SDK's normal lifecycle
may produce an effective prompt for one run. Autobench records both and links the effective
`AssetUse.definition_asset_id` and `definition_version` to the source definition. Case interpolation
therefore does not create a fake source edit for every input.

Autobench discovery never invokes instruction callbacks, tools, validators, or guardrails merely to
inspect them. It observes declaration values or values already resolved by the actual SDK lifecycle.

## Pydantic AI Without Tracking Decorators

This agent has no explicit Autobench tracking:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class SupportAnswer(BaseModel):
    answer: str = Field(description="A grounded support answer.")
    queue: str


def lookup_policy(topic: str) -> str:
    """Return the active support policy."""
    return f"policy for {topic}"


agent = Agent(
    model,
    name="support-router",
    output_type=SupportAnswer,
    instructions="Use the policy tool before routing.",
    tools=[lookup_policy],
)
```

With `instrument_all()`, a run discovers the agent composite, prompt, function tool, toolset, output
schema, final request instructions, effective tool definitions, and validated output schema. The
plain Python function and Pydantic type keep their source-aware identities; no wrapper replaces
their signatures or results.

The complete offline example uses Pydantic AI's real `Agent` lifecycle and `TestModel`:

```bash
uv run python examples/automatic_assets/pydantic_ai_discovery.py \
  --record /tmp/autobench-pydantic-assets
```

## Capability Scopes

Pydantic AI capabilities are first-class scopes because multiple capabilities can expose an
`instructions` or `search` component with the same local name:

```python
from pydantic_ai.capabilities import AbstractCapability


class RetrievalCapability(AbstractCapability[None]):
    id = "retrieval"

    def get_instructions(self) -> str:
        return "Ground answers in retrieved evidence."
```

The local and global locators are both retained:

```text
retrieval:prompt:instructions
pydantic_ai:retrieval:prompt:instructions
pydantic_ai:retrieval:capability:self
```

Capability identity uses a non-empty `id`, then `get_serialization_name()`, then the stable module
and qualified class name. A shared explicitly tracked component keeps one logical version while its
uses retain each capability alias and provenance.

## OpenAI Client And OpenAI Agents

The OpenAI client instrumentor discovers provider-facing assets:

- Chat Completions system/developer messages, tools, legacy functions, and response format;
- Responses instructions, managed prompt references, tools, text format, and output schema;
- Pydantic output types passed through structured parsing surfaces.

These are effective representations because the client sees the final request rather than the
framework's source declaration. Embedding inputs remain evidence and produce no asset.

The OpenAI Agents instrumentor discovers public Agent and Runner definitions:

- instructions and prompt references;
- function, hosted, MCP, computer, and agent tools;
- output schema;
- input/output/tool guardrails;
- handoffs and routing/tool-use policy;
- the composed agent.

It does not run callback instructions, guardrails, handoffs, or tools for discovery. Native trace
processing and public Runner resolution are combined without changing callback counts.

HTTPX intentionally discovers no semantic assets. A transport request lacks ownership context and
may contain secrets. It remains transport evidence while framework/client layers provide the
authoritative asset projections.

## Selecting Asset Families

Definition and effective discovery are enabled by default for semantic instrumentors. Restrict the
surface from Python:

```python
from autobench import AssetDiscoverySettings, AssetRepresentation, Benchmark

benchmark = Benchmark("support-agent").instrument_all(
    assets=AssetDiscoverySettings(
        representations=(
            AssetRepresentation.DEFINITION,
            AssetRepresentation.EFFECTIVE,
        ),
        include=("prompt", "tool", "output_schema", "capability"),
    )
)
```

Or use the YAML DSL:

```yaml
# yaml-language-server: $schema=schemas/0.3.0/benchmark_schema.json
benchmark:
  support-agent:
    instrumentation:
      all:
        assets:
          discover: true
          representations: [definition, effective]
          include: [prompt, tool, output_schema, capability]
```

An explicit integration can use a different filter and overrides automatic selection:

```yaml
instrumentation:
  all:
    exclude: [openai]
  openai:
    assets:
      representations: [effective]
      include: [prompt, tool, output_schema]
```

Set `discover: false` to keep spans and metrics from that instrumentor while disabling only its
asset discovery.

## Privacy And Capture Policy

Asset definitions and runtime evidence share one `CapturePolicy`, but they have separate fallback
levels because they serve different purposes:

- `default_level` defaults to `metadata` for messages, model payloads, HTTP data, and other runtime
  evidence;
- `asset_default_level` defaults to `full` so a successful prompt, tool, or output schema can be
  reconstructed, diffed, replayed, and optimized later.

Configure a stricter asset policy when the experiment directory cannot retain behavioral content:

```python
from autobench import Benchmark, CaptureLevel, CapturePolicy

benchmark = Benchmark("private-agent").capture(
    CapturePolicy.hashed(
        semantic_overrides={
            "tool": CaptureLevel.FULL,
            "output_schema": CaptureLevel.FULL,
        },
        deny_paths=("assets.*:prompt:private_notes",),
    )
)
```

The equivalent YAML is validated and completed by the versioned schema:

```yaml
benchmark:
  private-agent:
    capture:
      default_level: hash
      asset_default_level: hash
      use_semantic_defaults: false
      semantic_overrides:
        tool: full
        output_schema: full
      deny_paths:
        - assets.*:prompt:private_notes
```

Capture levels are `none`, `metadata`, `hash`, `redacted`, and `full`. The preset constructors
`none()`, `metadata()`, `hashed()`, `redacted()`, and `full()` set both fallbacks together. A direct
`CapturePolicy()` keeps runtime evidence metadata-first while retaining captured asset definitions.
Semantic overrides apply to both paths, and secret field names are filtered by the capture
normalizer.

The private content fingerprint still drives behavioral versioning. Changing capture from hash to
full does not create a false asset version. Omitted values retain an omission marker and digest;
large allowed values can be stored as bounded artifact references.

## Explicit Tracking Composition

Explicit tracking wins when Autobench observes the exact Python target:

```python
from autobench import track


@track.tool(name="knowledge_search")
def search(query: str) -> list[str]:
    """Search the approved knowledge base."""
    return backend.search(query)
```

If Pydantic AI or another instrumented SDK receives `search`, automatic discovery reuses the
explicit `ToolAsset` identity and version. SDK and capability locators become aliases; Autobench
does not create a second source version. The final provider schema remains a linked effective asset.

Use explicit tracking when the application needs:

- a domain-owned ID or name;
- a manually supplied source path or parent version;
- custom semantic classification;
- lineage before an instrumented run exists;
- a component that no SDK boundary exposes.

## Custom SDK Discovery

`InstrumentAssetSpec` adds the same lineage to arbitrary methods without modifying the target SDK:

```python
from autobench import InstrumentAssetSpec, SpanKind, instrument_method

handle = instrument_method(
    WorkflowClient,
    "execute",
    span="workflow_client.execute",
    span_kind=SpanKind.WORKFLOW,
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

`value_path` traverses trusted call arguments, mappings, attributes, results, and zero-argument
accessors. Python integrations can use a typed `value_factory`. Serializable integration settings
can use `extractor_target="package.extractors:extract_assets"`; Autobench imports the callable and
passes its `InstrumentCall`. It never evaluates expression strings.

Set `many=True` when the extracted value is a sequence or mapping of independent assets. Each use
is attached to the method span. Extraction failures become run errors or instrumentation
diagnostics without replacing the SDK method's result or exception.

Run the complete offline custom SDK example:

```bash
uv run python examples/automatic_assets/custom_sdk_discovery.py \
  --record /tmp/autobench-custom-assets
```

## Identity, Aliases, And Cross-Layer Correlation

Autobench resolves identity conservatively:

1. the exact explicitly tracked Python target;
2. an explicit asset ID;
3. the target's stable module and qualified name;
4. a previously registered source locator or alias;
5. the SDK, scope, kind, and local ID locator.

Content equality alone does not merge implementations. Two tools can expose the same schema while
running different code. When one framework definition has a unique matching client projection,
Autobench links source and effective forms. Multiple possible definitions produce an
`asset_correlation_ambiguous` diagnostic instead of a silent merge.

Repeated observations of the same asset/version/source/span are deduplicated. Observations on
different spans remain separate `AssetUse` evidence because they prove separate participation.

## Persistence And Replay

`record_experiment(...)` persists only assets referenced by the experiment:

```text
recording/
  experiment.yaml
  cases/
    <case>/<variant>/run.yaml
  assets/
    index.yaml
    <safe-asset-id>.yaml
  artifacts/
    asset-content.sqlite3
```

Each asset history is a lightweight manifest containing identity, immutable versions, parent
links, changed paths, and typed `content_ref` and `diff_ref` values. Captured snapshots and readable
diffs are stored in the experiment-local `artifacts/asset-content.sqlite3` registry; prompt, tool,
and schema bodies never appear in manifests. The content-addressed store deduplicates identical
payloads, performs indexed lookups, and updates transactionally without rewriting the complete
history. A file lock coordinates manifest updates while SQLite protects registry writes.

Resolve any historical snapshot directly:

```python
from pathlib import Path

from autobench import load_asset_content, load_asset_diff

snapshot = load_asset_content(
    Path("recording/artifacts/asset-content.sqlite3"),
    asset_id="pydantic_ai:agent:support-router:prompt:instructions",
    version="595012541db0",
)
prompt = snapshot["content"]

diff = load_asset_diff(
    Path("recording/artifacts/asset-content.sqlite3"),
    asset_id="pydantic_ai:agent:support-router:prompt:instructions",
    version="595012541db0",
    parent_version="21477c4a101a",
)
```

Every run record contains:

- `assets`: exact `AssetVersion` references used by the run;
- `asset_uses`: representation, source locator, scope, span, provenance, aliases, and source link;
- ABP references on the participating spans;
- capture and conflict diagnostics in the materialized trace.

Replay loads those values without importing Pydantic AI, OpenAI, OpenAI Agents, or the application
task module. Rescoring and report replay never mutate the original asset history.

## Compatibility And Failure Behavior

| Integration | Discovery | Representations | Default asset families |
| --- | --- | --- | --- |
| Pydantic AI `>=2.22,<2.23` | yes | definition + effective | agent, capability, prompt, tool, toolset, output schema, policy |
| OpenAI Python `>=2.52,<2.54` | yes | effective | prompt, tool, output schema |
| OpenAI Agents `>=0.19.2,<0.20` | yes | definition | agent, prompt, tool, output schema, guardrail, handoff, policy, toolset |
| HTTPX `>=0.28,<0.29` | no | none | transport evidence only |

Run `autobench instrumentation doctor` to inspect installed compatibility and declared asset
families. Unsupported versions are not patched silently.

Discovery failure is non-fatal by default. Autobench emits typed diagnostics for normalization,
correlation, callback, capture, or persistence problems while preserving the host call's return
value, exception identity, streaming lifecycle, and callback count.

## Choosing The Right Surface

Use automatic discovery for SDK-visible behavioral components. Use explicit tracking for
application-owned identity and components that never cross an SDK boundary. Use
`InstrumentAssetSpec` for a custom SDK or framework. Keep user inputs and outputs as evidence,
models as factors, and measured outcomes as metrics.

That separation makes the resulting RunRecords suitable for reporting today and controlled
candidate optimization later without coupling Autobench core to one AI framework.
