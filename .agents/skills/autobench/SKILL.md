---
name: autobench
description: Design, implement, run, debug, review, or document Autobench benchmarks and integrations. Use for Autobench YAML specs, Python Benchmark builders, task(ctx, case) functions, datasets and variants, semantic metrics, scoring and derivation, policies, ABP spans/traces, native Pydantic AI/OpenAI/Agents/HTTPX instrumentation, behavioral asset tracking, immutable records, replay, Rich reports, comparisons, exports, optional OTLP delivery, optimizer evidence, or Autobench repository development.
---

# Autobench

Use Autobench to define how an application is measured once, execute every variant by the same
rules, and preserve enough evidence to replay and compare results later. Keep this file as the
router. Load only the reference and example required by the current task.

## Start Here

1. Identify the real decision: correctness, quality, latency, cost, agent behavior, regression,
   lineage, or optimizer evidence.
2. Inspect the target project, installed Autobench version, existing specs, task import layout, and
   optional SDKs. Run `scripts/inspect_environment.py` when discovery would otherwise be manual.
3. Read [references/workflow.md](references/workflow.md) for a new benchmark or architecture
   decision.
4. Load only the topic references listed below.
5. Start from the nearest maintained example; adapt the application task and evidence, not the
   framework internals.
6. Validate before executing. Record a real run, then replay and report from that record.
7. For Autobench repository changes, run the complete repository gates before finishing.

## Non-Negotiable Contracts

- The application task signature is `task(ctx, case)`: `RunContext` first, `Case` second.
- YAML is the portable definition surface; Python owns application calls and typed extensions.
- Cases are inputs; variants are complete factor configurations; metrics are outcomes.
- Observations, scores, derived metrics, policies, and reports have separate ownership.
- Declare semantic type, direction, role, and unit when they carry meaning across applications.
- Span duration is automatic. Use `measure_callable()` for repeated samples and custom timers.
- ABP is Autobench-owned instrumentation, not an OpenTelemetry wrapper.
- OTLP is an optional outbound record projection; it is not ABP collection or persistence.
- Native instrumentation must never change subject results, exceptions, cancellation, or streams.
- Runtime capture is metadata-first; behavioral assets are full by default for reconstruction.
- Run records are immutable. Replay/report/compare/export do not execute the subject.
- Comparisons report deltas and confounding; they do not assert causality.
- Keep domain-specific behavior in the task, scorer, adapter, or example. Add only reusable generic
  primitives to Autobench core.
- Do not require normal users to construct Pydantic Evals datasets or cases.
- Every Autobench YAML file includes a versioned `yaml-language-server` schema header.

## Reference Router

Read one or more direct references based on the task:

- **Designing a benchmark or deciding boundaries**:
  [references/workflow.md](references/workflow.md)
- **Writing or changing Autobench YAML**:
  [references/yaml-dsl.md](references/yaml-dsl.md)
- **Using the builder, task context, typed specs, or custom scorer API**:
  [references/python-api.md](references/python-api.md)
- **Cases, datasets, factors, generated/production data, concurrency, or failure isolation**:
  [references/data-runtime.md](references/data-runtime.md)
- **Semantics, scoring, policies, cost, paired baselines, measurements, or agentic evaluation**:
  [references/evaluation.md](references/evaluation.md)
- **ABP spans, capture, signals, extraction, accounting, streaming, adapters, or OTLP delivery**:
  [references/abp.md](references/abp.md)
- **Pydantic AI, OpenAI, OpenAI Agents, HTTPX, `instrument_all()`, or custom SDK patching**:
  [references/integrations.md](references/integrations.md)
- **Prompts, tools, schemas, automatic assets, hashes, versions, diffs, or candidate lineage**:
  [references/assets.md](references/assets.md)
- **RunRecord, persistence, replay, Rich reports, comparison, exports, or optimizer handoff**:
  [references/records-reports.md](references/records-reports.md)
- **CLI commands, schemas, troubleshooting, quality gates, or release readiness**:
  [references/cli-operations.md](references/cli-operations.md)
- **Editing Autobench itself or locating canonical source/docs/tests/examples**:
  [references/repository-map.md](references/repository-map.md)

Do not load every reference by default. For API uncertainty inside the Autobench repository, trust
current source, tests, schemas, and canonical docs before this bundled summary.

## Example Router

- `examples/minimal`: first YAML benchmark, typed task, variants, exact scoring, matrix, comparison.
- `examples/advanced`: repeated measurements, raw sample artifact, paired speedup,
  policy, distribution.
- `examples/generated_dataset`: typed pre-run generation request, review state, frozen dataset, and
  provenance manifest.
- `examples/abp_manual`: explicit workflow span plus generic method instrumentation.
- `examples/pydantic_ai`: real Pydantic AI `TestModel`, `instrument_all()`, structured output,
  capabilities, tools, and automatic asset lineage.
- `examples/pydantic_gepa`: standard GEPA, Optimize Anything Omni, multi-component candidate,
  checkpoint/resume, and optional live Pydantic AI optimizer instrumentation.
- `examples/otlp_export`: replay a real record into an injected offline OTLP exporter.

Copy a maintained example into a new directory with:

```bash
python .agents/skills/autobench/scripts/scaffold.py minimal benchmarks/my-benchmark
```

If the skill is installed elsewhere, replace `.agents/skills/autobench` with the directory that
contains this `SKILL.md`.

## Deterministic Scripts

- `scripts/inspect_environment.py [project-root]`: report Autobench, optional SDK versions, and
  discovered benchmark specs as JSON.
- `scripts/scaffold.py <example> <new-directory>`: copy one complete example without overwriting an
  existing destination.
- `scripts/validate_workflow.py <spec> [--record <directory>]`: validate, and optionally run,
  replay, and report through the active Autobench CLI.
- `scripts/smoke_examples.py`: run all bundled offline YAML examples end to end.

Scripts orchestrate public Autobench behavior. They do not replace the CLI, duplicate the runtime,
or weaken validation.

## Implementation Workflow

### New benchmark

1. Read `workflow.md`, then `yaml-dsl.md` or `python-api.md`.
2. Choose the nearest example.
3. Define stable cases, variants, objectives, constraints, diagnostics, and report views.
4. Implement the smallest application-owned task.
5. Add native/manual evidence only where the application outcome alone is insufficient.
6. Validate, run to a fresh record directory, replay, report, compare, and inspect artifacts.

### Existing benchmark bug

1. Reproduce with the smallest failing spec/case/variant.
2. Classify the failure: authoring, import resolution, task, instrumentation, scoring, derivation,
   policy, recording, replay, or report projection.
3. Read the matching reference and inspect the record before rerunning.
4. Fix the owning layer; do not mask it in CLI rendering or add case-specific core behavior.
5. Add a behavioral regression test and rerun the real workflow.

### Autobench framework change

1. Read `repository-map.md` and the domain reference.
2. Preserve public ownership boundaries and optional dependency isolation.
3. Update all coupled surfaces: typed model, YAML transform/schema, public export, records,
   compatibility, docs, examples, and tests as applicable.
4. Maintain 100% meaningful source line and branch coverage.
5. Run `make prod` and `make pre-commit`.

## Completion Criteria

Do not stop at syntax. A completed Autobench task has:

- a validated definition and importable task;
- meaningful cases and factorized variants;
- correctly classified objective/constraint/diagnostic evidence;
- explicit capture and asset behavior when sensitive or optimization-relevant;
- a successful real run or a clearly reproduced expected failure;
- an immutable record that replays and reports without subject execution;
- tests for changed behavior and edge cases;
- current docs/examples/schema when a public surface changed;
- no stale placeholder, generated cache, local run, credential, or case-specific framework code.
