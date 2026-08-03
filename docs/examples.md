# Examples

The release examples are applications of the public framework, not alternate runtimes or mock-only
snippets. Every offline example executes the complete `run -> record -> replay -> report -> export`
workflow through `make examples`.

## Minimal

```bash
uv run autobench run examples/minimal/autobench.yaml --record /tmp/autobench-minimal
```

Demonstrates inline cases, deterministic variants, exact scoring, a case matrix, and comparison.

## Basic

```bash
uv run autobench run examples/basic/autobench.yaml --record /tmp/autobench-basic
```

Routes file-backed support tickets and records workflow spans plus decision artifacts. The second
variant fixes an enterprise-outage routing failure, making the comparison visible in terminal tables.

## Mid

```bash
uv run autobench run examples/mid/autobench.yaml --record /tmp/autobench-mid
```

Records semantic token usage and latency, derives request cost from a local pricing DSL, applies
success and cost policies, and renders cost distributions.

## Advanced

```bash
uv run autobench run examples/advanced/autobench.yaml --record /tmp/autobench-advanced
```

Uses repeated measurements and sample artifacts, then derives per-case speedup with a paired baseline.
Correctness remains a constraint while speed is the optimization objective.

## CodeMode

```bash
uv run python examples/codemode/run_benchmark.py --only parse_cron
```

This is a live integration with the runtime that provides `vowel.codemode`. It generates evaluation
specs with configured models, replays each generated spec against the source function, and records
coverage, latency, generated specs, and exploration artifacts as Autobench evidence. It requires the
external CodeMode runtime, an `OPENROUTER_API_KEY`, and network access.

CodeMode-specific calls remain in the example task. Autobench core only owns the generic dataset,
variant, task, observation, artifact, scoring, recording, replay, and reporting seams.

## Pydantic AI

```bash
uv sync --extra instrumentation
export OPENROUTER_API_KEY=...
export OPENROUTER_MODEL=openrouter:openai/gpt-5.6-luna
uv run python examples/pydantic_ai/openrouter_instrument_all.py \
  --record /tmp/autobench-openrouter
```

This live benchmark makes a real OpenRouter request through Pydantic AI's OpenAI-compatible model.
It uses a tracked prompt, a catalog tool, streaming execution, and structured Pydantic output.
`Benchmark.instrument_all()` discovers Pydantic AI, OpenAI, and HTTPX automatically, producing a
layered framework/client/transport trace plus semantic token, model, latency, tool, validation,
streaming, HTTP, scoring, and asset-version evidence. The task contains no manual `ctx.span()` or
`ctx.metric()` calls. The complete experiment is recorded as replayable YAML under the supplied
directory.

`agent_benchmark.py` remains the provider-neutral variant for any configured Pydantic AI model.

## ABP Manual And Method

```bash
uv run autobench run examples/abp_manual/autobench.yaml --record /tmp/abp-manual
```

Combines a manual workflow span with automatic `TicketRouter.route` instrumentation. The method
instrumentor emits a nested span, one metric, and one factor while preserving the method signature
and result.

## ABP Concurrent

```bash
uv run autobench run examples/abp_concurrent/autobench.yaml --concurrency 2 \
  --record /tmp/abp-concurrent
```

Runs asynchronous worker siblings under a workflow span. It demonstrates task-local ABP context,
correct sibling parentage, and concurrent benchmark matrix execution.

## OpenAI Streaming

```bash
uv sync --extra instrumentation
uv run python examples/abp_openai/run_openai_streaming.py
```

Uses the official OpenAI client with a real streaming parser and an offline HTTPX mock transport.
The resulting trace contains client and transport spans, first-chunk evidence, and normal stream
completion without network access or credentials.

## OpenAI Agents

```bash
uv sync --extra openai-agents
uv run python examples/abp_openai_agents/run_openai_agents.py
```

Runs a real OpenAI Agents workflow, function span, and custom span through the native trace
processor. No model or network call is required.

## Replay And Extraction

```bash
uv run python examples/abp_replay/replay_and_extract.py /tmp/recorded-experiment
```

Loads each recorded RunRecord and creates an immutable extraction-derived record with signal, span,
and usage observations. The script imports no provider SDK.
