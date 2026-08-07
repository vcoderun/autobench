# Example Projects

The repository examples use the public Autobench runtime. They are ordered by the amount of
framework surface they demonstrate, not by whether the subject is AI-based.

## Offline Release Matrix

These examples are credential-free and run in `make examples`:

| Example | Subject | Main features |
| --- | --- | --- |
| `minimal` | text transformation | inline cases, variants, exact score, matrix, comparison |
| `basic` | support routing | file dataset, spans, checks, artifacts, Rich reports |
| `mid` | response generation | semantic usage, pricing, cost, policies, distributions |
| `advanced` | search implementations | repeated samples, noise, paired speedup |
| `abp_manual` | ticket router | manual span plus method instrumentation |
| `abp_concurrent` | async workers | task-local trace context and concurrent runs |
| `automatic_assets` | Pydantic AI and custom SDK | automatic behavioral asset lineage |
| `generated_dataset` | support-routing case preparation | typed generator, request YAML, review state, frozen dataset and provenance manifest |
| `otlp_export` | immutable ABP record | offline OTLP hierarchy mapping through an injected exporter |

Run all offline examples:

```bash
make examples
```

## Minimal: Learn The Matrix

```bash
autobench validate examples/minimal/autobench.yaml
autobench run examples/minimal/autobench.yaml --record /tmp/autobench-minimal
autobench replay /tmp/autobench-minimal
```

Read `examples/minimal/autobench.yaml` together with `minimal_benchmark.py`. This is the shortest
complete `case x variant -> task -> score -> record -> report` implementation.

## Generated Dataset: Prepare Before Planning

```bash
cd examples/generated_dataset
autobench dataset generate generator:generate_routing_cases \
  --request request.yaml \
  --output generated-cases.yaml \
  --id routing-generated \
  --version v1
```

This example is deterministic and credential-free. It writes a normal dataset and a separate
generation manifest, showing the boundary between data preparation and benchmark execution.

## Basic: Application Evidence

```bash
autobench run examples/basic/autobench.yaml --record /tmp/autobench-basic
```

The task validates typed input, reads a factor, opens a workflow span, stores its output as an
artifact, and lets declarative scorers evaluate correctness and handling. The candidate fixes a
known routing failure, so the case matrix and comparison contain a visible behavioral delta.

## Mid: Quality, Cost, And Constraints

```bash
autobench run examples/mid/autobench.yaml --record /tmp/autobench-mid
```

This example records input/output tokens and latency, resolves a local model pricing table, derives
`money.cost`, checks success and cost policies, and configures leaderboard and distribution views.
It is the best starting point for an LLM benchmark that already has a task implementation.

## Advanced: Measurement And Paired Baselines

```bash
autobench run examples/advanced/autobench.yaml --record /tmp/autobench-advanced
```

The task uses `measure_callable()` and `ctx.record_measurement()` instead of custom timing loops.
The post-deriver matches runs by case and computes candidate speedup against the baseline while
correctness remains a constraint.

## Pydantic AI: Live Layered Instrumentation

```bash
uv sync --extra instrumentation
export OPENROUTER_API_KEY=...
export OPENROUTER_MODEL=openrouter:openai/gpt-5.6-luna
uv run python examples/pydantic_ai/openrouter_instrument_all.py \
  --record /tmp/autobench-openrouter
```

The program makes a real OpenRouter request through Pydantic AI, uses a tool, streams structured
output, and calls `Benchmark.instrument_all()`. The task has no manual metrics, spans, or tracking
decorators. Autobench collects layered Pydantic AI, OpenAI client, and HTTPX evidence plus prompt,
tool, output-schema, and agent versions.

Inspect it afterward:

```bash
autobench instrumentation trace /tmp/autobench-openrouter
autobench replay /tmp/autobench-openrouter
```

`examples/pydantic_ai/agent_benchmark.py` is provider-neutral and accepts any configured Pydantic AI
model identifier through `PYDANTIC_AI_MODEL`.

## Automatic Asset Discovery

```bash
uv run python examples/automatic_assets/pydantic_ai_discovery.py \
  --record /tmp/autobench-pydantic-assets

uv run python examples/automatic_assets/custom_sdk_discovery.py \
  --record /tmp/autobench-custom-assets
```

Both are offline. The first uses a real Pydantic AI `Agent`, `AbstractCapability`, tool, and Pydantic
output model with no explicit tracking. The second adds prompt, tools, and output-schema extraction
to an arbitrary method with `InstrumentAssetSpec`.

## ABP Manual And Concurrent

```bash
autobench run examples/abp_manual/autobench.yaml --record /tmp/abp-manual
autobench run examples/abp_concurrent/autobench.yaml \
  --concurrency 2 \
  --record /tmp/abp-concurrent
```

Use the manual example to learn `RunContext.span()` and `instrument_method()`. Use the concurrent
example to inspect sibling span parentage and task-local context under async execution.

## OpenAI Streaming

```bash
uv sync --extra instrumentation
uv run python examples/abp_openai/run_openai_streaming.py
```

This uses the official OpenAI client and a real streaming parser over an offline HTTPX mock
transport. It demonstrates first-chunk and stream-completion evidence without network access.

## OpenAI Agents

```bash
uv sync --extra openai-agents
uv run python examples/abp_openai_agents/run_openai_agents.py
```

The example sends real OpenAI Agents workflow/function/custom trace events through the Autobench
trace processor. It requires no model request.

## Replay And Extraction

```bash
uv run python examples/abp_replay/replay_and_extract.py /tmp/recorded-experiment
```

The script loads records without provider SDKs and creates extraction-derived records with explicit
parent lineage.

## Offline OTLP Export

```bash
uv run autobench run examples/abp_manual/autobench.yaml --record /tmp/abp-manual
uv run python examples/otlp_export/export_record.py /tmp/abp-manual
```

The example maps a real recorded experiment to OTel SDK spans through an injected in-memory
exporter, so it verifies hierarchy and delivery without a collector or network request. Production
delivery uses `autobench telemetry export`; see [OTLP Export](otlp-export.md).

## CodeMode: Migrating A Real Benchmark Runner

```bash
export OPENROUTER_API_KEY=...
uv run python examples/codemode/run_benchmark.py --only parse_cron \
  --record /tmp/autobench-codemode
```

The CodeMode example replaces a bespoke benchmark script with cases, model-pair factors, a task,
semantic coverage/success/latency evidence, generated-spec artifacts, recording, and reports. Its
task still owns Vowel CodeMode calls; Autobench remains generic. The external CodeMode runtime and
network credentials are required.

## What To Copy

Copy the pattern, not generated run directories:

- task signature and typed input/output from `minimal` or `basic`;
- pricing and policies from `mid`;
- measurement and paired comparison from `advanced`;
- automatic SDK setup from `pydantic_ai`;
- custom instrumentation from `automatic_assets`;
- replay processing from `abp_replay`.

For combinations not represented by one project, use [Use Cases](use-cases.md) and the
[Capability Map](capabilities.md).
