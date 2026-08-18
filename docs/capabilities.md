# Capability Map

This page is the inventory of what Autobench owns today. Every public feature belongs to one of
the layers below; application-specific behavior stays in tasks, scorers, adapters, and metric
packs.

## End-To-End Lifecycle

```text
BenchmarkSpec
  -> Dataset x Variants
  -> BenchmarkPlan
  -> Task(ctx, case)
  -> Observations + Spans + Artifacts + Errors
  -> Scores + Derived Metrics + Policies
  -> Cross-run Derivation
  -> Immutable RunRecord / ExperimentRecord
  -> Replay -> Report -> Compare -> Export -> Optimization Feedback
```

The same lifecycle is available through the YAML DSL, Python models, the `Benchmark` builder, and
the CLI. YAML is the portable authoring format; Python remains the extension surface for
application execution and custom evaluation logic.

## Definition And Data

| Capability | What it provides |
| --- | --- |
| `BenchmarkSpec` | Validated benchmark metadata, dataset, task, variants, scoring, derivation, policies, and reports |
| Dataset | Inline cases, file-backed datasets, glob-backed case files, defaults, tags, metadata, attachments, and versions |
| Cases | Arbitrary input and expected payloads with stable IDs and artifact references |
| Variants | Named factor combinations with labels, semantic types, and `optimize` hints |
| Generated datasets | Separate sync/async preparation API and CLI, typed requests/batches, review state, provenance, usage/cost, content hashes, complete publication, and incomplete sidecars |
| YAML schemas | Versioned JSON schemas and `yaml-language-server` headers for completion and validation |
| Source discovery | Hash collection for specs, datasets, pricing files, task modules, and scorer modules |

See [Datasets And Variants](datasets-and-variants.md) and [YAML Spec](yaml-spec.md).

## Planning And Execution

| Capability | What it provides |
| --- | --- |
| Matrix planning | Deterministic case x variant expansion and stable run IDs |
| Task runtime | Sync and async Python callables with `ctx` first and `case` second |
| Concurrency | Bounded async execution while preserving deterministic result ordering |
| Failure isolation | One task, scorer, derivation, or policy failure does not erase other runs |
| Progress events | Typed lifecycle events for runners and future UI integrations |
| Execution correlation | Immutable cross-invocation group, attempt, phase, association, and scalar-label metadata across Python, YAML, CLI, records, replay, and reports |
| Optional Pydantic Evals bridge | Internal conversion to Pydantic Evals-compatible case and dataset payloads |

See [Tasks And Runtime](tasks-and-runtime.md).

## Evidence Collection

| Capability | What it provides |
| --- | --- |
| Observations | Metrics, factors, events, diagnostics, artifacts, roles, units, directions, tags, and sources |
| Semantic registry | Canonical semantic types, aliases, parent relationships, and custom extensions |
| Projection | Source precedence and duplicate detection for one canonical metric view |
| Context spans | Nested agent, LLM, tool, retriever, parser, workflow, and custom spans |
| Automatic duration | Span timing and optional duration metrics owned by the runtime |
| Artifacts | Structured values and files materialized outside the main record payload |
| Errors | Structured task, scorer, trace, and policy errors with traceback capture |
| Measurement | Warmup, repetitions, time budgets, samples, median, p95, standard deviation, and noise |

See [Observations And Semantics](observations-and-semantics.md) and
[Instrumentation And Traces](instrumentation-and-traces.md).

Native Pydantic AI, pydantic-gepa, OpenAI, OpenAI Agents, and HTTPX integrations can be selected
through typed Python settings or the YAML `instrumentation` section. They emit ABP directly,
compose across optimizer/framework/client/transport layers, preserve lifecycle, and remain optional
for replay. See [Native Instrumentation](native-instrumentation.md) and
[Pydantic-GEPA Instrumentation](pydantic-gepa-instrumentation.md).

Semantic instrumentors automatically discover SDK-visible prompt, tool, output-schema, capability,
agent, guardrail, handoff, policy, and toolset versions. Definition/effective relationships,
capability scopes, aliases, privacy-controlled content, and span-local `AssetUse` evidence survive
recording and replay. HTTPX remains transport evidence and performs no semantic asset inference.
See [Automatic Asset Discovery](automatic-asset-discovery.md).

Immutable records can also be replayed through the optional outbound
[OTLP exporter](otlp-export.md). Experiment/run/ABP hierarchy, semantic events, record identity,
partial state, links, and source provenance are preserved without making OTel canonical or a base
dependency.

## Scoring And Constraints

| Scorer | Purpose |
| --- | --- |
| `output` | Project an output path into a semantic score |
| `pass_fail` | Turn a boolean output path into a pass/fail score |
| `exact` | Compare actual and expected paths |
| `schema` | Validate output against a schema/model |
| `python` | Run a sync or async custom scorer using `ScoringCall` |
| `expected_action` | Evaluate action/tool selection, arguments, or sequence from spans |

Scores declare semantic type, unit, direction, role, and optional failure behavior. Policies add
typed requirements including equality, membership, numeric bounds, and inclusive ranges.

See [Scoring And Derivation](scoring-and-derivation.md) and
[Agentic Evaluation](agentic-evaluation.md).

## Derivation And Cost

| Capability | What it provides |
| --- | --- |
| Token cost | Derive `money.cost` from input/output tokens and normalized model/provider factors |
| Pricing DSL | Static YAML pricing, aliases, provider maps, cache prices, token tiers, and model normalization |
| Price sources | Optional llm-prices and genai-prices importers that normalize external data into `PricingTable` |
| Paired baseline | Per-case or factor-matched speedup, delta, percent change, diagnostics, and verdicts |
| Comparison classifier | Improved, regressed, unchanged, or inconclusive outcomes with relative noise thresholds |

External price sources are convenience importers, not runtime dependencies or Autobench's source
of truth. A local pricing YAML remains fully supported.

## Agentic Evidence

Autobench records agent behavior without requiring OpenTelemetry:

- typed trace envelopes and nested span records
- expected tool/action selection, argument, and sequence checks
- span selectors by kind, name, tag, path, or semantic type
- Pydantic AI usage normalization
- metric packs for agentic, structured-output, LLM-usage, and performance defaults
- compact feedback records for optimization systems

See [Agentic Evaluation](agentic-evaluation.md).

## Asset Lineage

The tracking registry understands:

- text prompts from inline text or files
- arbitrary assets and configuration values
- callable tools, signatures, parameters, docs, and return types
- Pydantic models, standard dataclasses, and typed classes
- field names, annotations, descriptions, aliases, defaults, requirements, constraints, and examples
- source hashes, structured-schema hashes, versions, parent versions, and diffs
- persistent human-readable YAML asset histories
- automatic SDK-boundary discovery without tracking decorators
- source/effective representation links, capability scopes, provenance, and cross-layer aliases
- automatic experiment persistence and replayable span-local asset uses

Decorators preserve the original callable or class type so tracking does not degrade static
typing. See [Asset Tracking](asset-tracking.md) and
[Automatic Asset Discovery](automatic-asset-discovery.md).

## Records, Replay, And Analysis

| Capability | What it provides |
| --- | --- |
| `RunRecord` | Immutable case x variant evidence including output, scores, observations, spans, factors, assets, artifacts, and errors |
| `ExperimentRecord` | Plan, environment, semantic registry, report config, source hashes, and run paths |
| Correlated reports | Field filters and `group_id` report groups across independent experiment results |
| Replay | Load records without importing task or scorer modules |
| Rich terminal reports | Status, variant configuration, leaderboard, run metrics, case matrix, comparisons, and distributions |
| Markdown reports | Decision-facing quality gates, case outcomes, evaluator feedback, purposeful inline SVG, paired comparisons, summary/full/audit profiles, audit-only traces/assets/provenance, single/bundle/auto layouts, and atomic publication |
| Optimizer reports | pydantic-gepa outcome/resources, engine branches, candidate lineage, component versions, selections, and diagnostics |
| Exports | Human-readable YAML summary, CSV run projection, and Markdown report |
| Optimization feedback | Failure category, score, reasons, factors, asset versions, and selected evidence |

See [Recording And Reporting](recording-and-reporting.md) and
[Markdown Reports](markdown-reports.md).

## Ownership Boundaries

Autobench deliberately does not own:

- application or model execution
- hosted tracing or observability storage
- model-specific pricing as an always-current service
- causal claims from confounded comparisons
- optimizer search strategies or candidate promotion
- large catalogs of domain-specific LLM judges

Tasks and adapters own application execution. Optional integrations may import traces, pricing, or
evaluator results, but the core contract remains semantic, generic, and replayable.
