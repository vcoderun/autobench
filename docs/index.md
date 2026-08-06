# Autobench

**Define what better means once. Autobench evaluates every version by the same rules, compares the
results, and keeps the full history ready for inspection.**

Applications change: a team may switch models, revise a prompt, replace a tool, tune an algorithm,
or ship a new configuration. To decide whether the change is actually better, they commonly write
a benchmark script. That script runs representative inputs, checks the outputs, records values
such as correctness, latency, token usage, or cost, and compares one version with another.

The script works, but every project tends to build this machinery again. Results use incompatible
formats, measurement and scoring logic become mixed with application code, and an old result is
often impossible to inspect without rerunning the original program.

We built Autobench to solve this problem. You describe the inputs to test, the variants to compare,
the application task, and the meaning of success in YAML or Python. Autobench then runs the full
matrix, collects measurements and traces, evaluates each result, records the application assets
that affected it, and stores an immutable experiment record. That same record can be replayed,
reported, compared, or exported later without calling the application again.

```bash
uv add autobench
autobench validate autobench.yaml
autobench run autobench.yaml --record runs/latest
```

After the run, inspect the recorded experiment without executing the application again:

```bash
autobench replay runs/latest
autobench report runs/latest
autobench compare runs/latest --baseline current --candidate proposed
autobench export runs/latest --format csv --path analysis/runs.csv
```

## The Framework Loop

```text
BenchmarkSpec
  Dataset[Case] x Variant[Factor]
    -> task(ctx, case)
    -> observations + ABP trace + artifacts + asset versions
    -> scorers + per-run derivation
  -> cross-run derivation + policies
  -> immutable RunRecords
  -> replay + Rich reports + comparison + exports + optimization feedback
```

The task is the only application-specific part. Autobench owns the repeated infrastructure around
it: matrix planning, context propagation, instrumentation, scoring, derivation, persistence,
reporting, and replay.

## Why Semantic Evidence Matters

Raw names such as `prompt_tokens`, `input_tokens`, `accuracy`, and `answer_quality` are local
conventions. Autobench observations can also declare stable meaning:

```text
llm.tokens.input
llm.tokens.output
llm.model.name
quality.correctness
time.latency
money.cost
agent.tool.argument.correctness
```

That semantic layer lets reports, pricing derivation, policy checks, and optimization systems use
evidence from different applications without guessing what every local metric name means.

## What You Can Benchmark

Autobench is optimized for AI systems but does not require one:

| System | Cases | Variants | Evidence |
| --- | --- | --- | --- |
| LLM application | prompts and expected answers | model, prompt, temperature | quality, tokens, latency, cost |
| Agent | user goals and expected actions | instructions, tools, model | action selection, arguments, sequence, completion |
| Search or retrieval | queries and relevant items | index, reranker, limits | recall, precision, latency |
| Service/API | requests and expected responses | release, configuration | correctness, errors, throughput, SLA |
| Algorithm | input fixtures | implementation | correctness, repeated timings, speedup |
| Data pipeline | source batches | parser or policy | coverage, validity, loss, runtime |

See [Use Cases](use-cases.md) for complete patterns.

## Core Capabilities

| Area | Included |
| --- | --- |
| Definition | Human-readable YAML DSL, typed Python builder, JSON Schema completion |
| Data | Inline/file/glob datasets, defaults, attachments, generated and production cases |
| Execution | Sync/async tasks, deterministic matrices, bounded concurrency, failure isolation |
| Evidence | Semantic observations, checks, events, artifacts, measurements, ABP traces |
| Evaluation | Built-in and custom scorers, expected actions, policies, metric packs |
| Derivation | Token cost, tiered pricing, paired baselines, comparison verdicts |
| Instrumentation | Manual spans, method instrumentation, Pydantic AI, OpenAI, Agents, HTTPX |
| Lineage | Explicit and automatic prompt/tool/schema/capability/agent asset versioning |
| Persistence | Immutable YAML records, source hashes, environment metadata, portable artifacts |
| Analysis | Replay, Rich reports, leaderboards, matrices, distributions, comparisons, exports |

## Choose A Starting Point

| Goal | Read |
| --- | --- |
| Install the right extras | [Installation](installation.md) |
| Run a complete benchmark | [First Benchmark](getting-started.md) |
| Find a pattern for your system | [Use Cases](use-cases.md) |
| Understand ownership and data flow | [Architecture](architecture.md) |
| Author the full DSL | [YAML Spec](yaml-spec.md) |
| Compose benchmarks in Python | [Python API](python-api.md) |
| Instrument an existing SDK application | [Native Instrumentation](native-instrumentation.md) |
| Collect prompt/tool/schema lineage automatically | [Automatic Asset Discovery](automatic-asset-discovery.md) |
| Inspect all shipped features | [Capability Map](capabilities.md) |

## Project Boundaries

Autobench records and evaluates evidence. It does not own your application, make causal claims from
confounded runs, keep provider pricing permanently current, or choose an optimization algorithm.
Those boundaries keep the core usable for arbitrary systems while allowing pydantic-gepa,
autoptimize, or another consumer to build on stable experiment records.
