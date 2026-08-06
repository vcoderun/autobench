# Repository And Documentation Map

## Contents

- [Authority order](#authority-order)
- [Public documentation](#public-documentation)
- [Canonical examples](#canonical-examples)
- [Source packages](#source-packages)
- [Tests](#tests)
- [Change maps](#change-maps)
- [External package boundaries](#external-package-boundaries)

## Authority Order

When this skill is used inside the Autobench repository, resolve uncertainty in this order:

1. current source and public exports;
2. tests and versioned schemas;
3. current public docs and canonical examples;
4. this skill's references;
5. historical plans or external projects.

The skill provides workflow knowledge, not permission to override a newer installed API. Check
`src/autobench/_version.py`, `pyproject.toml`, `src/autobench/__init__.py`, and the target spec's
schema version.

## Public Documentation

Read only the page relevant to the task:

| Need | Canonical page |
| --- | --- |
| Product model and start | `docs/index.md`, `docs/getting-started.md` |
| Installation/extras | `docs/installation.md` |
| Real scenarios | `docs/use-cases.md`, `docs/examples.md` |
| Ownership and lifecycle | `docs/architecture.md`, `docs/concepts.md` |
| Cases and factors | `docs/datasets-and-variants.md` |
| Task/runtime behavior | `docs/tasks-and-runtime.md` |
| Metrics and semantics | `docs/observations-and-semantics.md` |
| Scoring/derivation | `docs/scoring-and-derivation.md` |
| Agent evaluation | `docs/agentic-evaluation.md` |
| Records/reports | `docs/recording-and-reporting.md` |
| Explicit assets | `docs/asset-tracking.md` |
| Native asset discovery | `docs/automatic-asset-discovery.md` |
| ABP protocol | `docs/instrumentation-and-traces.md`, `docs/abp-compatibility.md` |
| SDK integrations | `docs/native-instrumentation.md` |
| Complete YAML | `docs/yaml-spec.md` |
| Complete Python surface | `docs/python-api.md`, `docs/api-reference.md` |
| CLI and failures | `docs/cli.md`, `docs/troubleshooting.md` |
| Shipped inventory | `docs/capabilities.md` |
| Development/release | `docs/development.md` |

`docs/llms.txt` is the narrow LLM index. `docs/llms-full.txt` is generated from public pages in
navigation order. Do not edit generated staged `_markdown` copies directly.

## Canonical Examples

| Directory | Purpose |
| --- | --- |
| `examples/minimal` | inline cases, variants, exact score, report, compare |
| `examples/basic` | file data, spans, checks, artifacts, failure visibility |
| `examples/mid` | semantic tokens, pricing, cost, policies, distributions |
| `examples/advanced` | repeated measurement and paired speedup |
| `examples/abp_manual` | manual spans plus method instrumentation |
| `examples/abp_concurrent` | task-local concurrent trace parentage |
| `examples/abp_openai` | offline OpenAI streaming over HTTPX mock transport |
| `examples/abp_openai_agents` | offline native Agents trace processing |
| `examples/abp_replay` | extraction from recorded traces without SDK imports |
| `examples/pydantic_ai` | live layered instrumentation and OpenRouter flow |
| `examples/automatic_assets` | Pydantic AI and arbitrary SDK asset discovery |
| `examples/codemode` | migration of a real bespoke benchmark runner |

Use repository examples as the most accurate executable source. Skill examples are smaller
copy-ready patterns.

## Source Packages

| Package | Ownership |
| --- | --- |
| `builders` | fluent benchmark components |
| `data` | cases, datasets, ingestion, generation, variants |
| `evaluation` | scoring, derivation, pricing, policies, measurements, feedback |
| `instrumentation` | lifecycle, patching, SDK adapters, configuration, streaming |
| `metrics` | observations, semantics, mappings, queries, projections, packs |
| `protocol` | ABP signals, collector, capture, trace materialization |
| `records` | artifacts, immutable recording, storage, replay |
| `reports` | projections, aggregation, Rich rendering, exports |
| `runtime` | task execution, context, pipeline, optional eval runtime |
| `specs` | benchmark/task/report typed specs and YAML transforms |
| `tracking` | explicit/automatic asset models, registry, history, content store |

Public exports are deliberate in `src/autobench/__init__.py` and subpackage `__init__.py` files.
Do not export internal helpers simply to make a test import easier.

## Tests

Tests are behavior-oriented rather than phase-numbered. Key suites include benchmark/spec loading,
runtime, metrics/semantics, evaluation, reporting, recording/replay, tracking, ABP, each native
instrumentor, streaming, compatibility, examples, and package support.

Coverage must be 100% for source lines and branches, with real behavioral cases. Do not add tests
that execute a branch only to satisfy coverage without asserting its contract.

## Change Maps

### Add or change a YAML field

Update typed spec, authoring transform, serialization view, schema, loader validation, docs,
examples, round-trip tests, and compatibility fixtures.

### Add a semantic type

Update semantic literal/registry, parent/alias mapping, any native source maps, report grouping,
docs, and semantic tests.

### Add an instrumentor

Update optional dependency metadata, compatibility registry, typed/YAML settings, lifecycle,
patching/extraction, suppression/accounting, automatic selection, doctor output, docs, examples,
SDK-version CI, and replay-without-SDK tests.

### Add an asset kind

Update literals/models, canonical content normalization, hashing/versioning, YAML manifest view,
content registry, capture semantics, native/custom extraction, docs, and history/diff tests.

### Add a report/export feature

Keep records unchanged. Add typed config, projection, Rich renderer or file exporter, CLI wiring,
docs, and replay-only tests.

## External Package Boundaries

- `autobench`: benchmark/evidence runtime and records;
- `pydantic-gepa`: general Pydantic AI + Pydantic Evals adapter for GEPA;
- `autoptimize`: strategy, experiment planning, candidate validation, and promotion.

Autobench may expose optimization-grade feedback but does not own candidate mutation or search.
Pydantic Evals may be an internal runtime adapter; normal Autobench users should not be forced to
build its `Dataset`/`Case` types.

External references such as DeepEval, Promptfoo, Vowel, OpenTelemetry instrumentors, or bespoke
benchmarks are discovery sources. Borrow useful feature concepts, not their product ownership or
entire abstraction stack.

