# Autobench Workflow

## Contents

- [Product model](#product-model)
- [Choose the benchmark boundary](#choose-the-benchmark-boundary)
- [Design the experiment](#design-the-experiment)
- [Implementation sequence](#implementation-sequence)
- [Evidence requirements](#evidence-requirements)
- [Review checklist](#review-checklist)
- [Anti-patterns](#anti-patterns)

## Product Model

An application change creates a practical question: did the new version become more correct,
faster, cheaper, safer, or more useful? A bespoke benchmark script usually answers that question
by running representative inputs, collecting outputs and measurements, and printing a comparison.
Autobench owns the infrastructure that otherwise gets rewritten around that application call.

The lifecycle is:

```text
BenchmarkSpec
  -> Dataset[Case] x Variant[Factor]
  -> task(ctx, case)
  -> observations + checks + artifacts + ABP trace + asset versions
  -> scorers + per-run derivation + policies
  -> cross-run derivation
  -> immutable RunRecord / ExperimentRecord
  -> replay + report + compare + export + optimization feedback
```

The application task remains application-owned. Autobench owns planning, context propagation,
instrumentation, evaluation, persistence, and analysis.

Autobench is optimized for AI systems but is not AI-only. Use the same primitives for APIs,
algorithms, data pipelines, retrieval systems, compilers, services, and hardware-neutral
performance tests.

## Choose The Benchmark Boundary

Start with the smallest subject that produces a meaningful user or system outcome:

| Subject | Case | Variant | Typical evidence |
| --- | --- | --- | --- |
| LLM application | prompt and expected response | model, prompt, temperature | quality, tokens, latency, cost |
| Agent | goal and expected actions | instructions, tools, model | completion, tool choice, arguments, sequence |
| Retrieval | query and relevant items | index, reranker, limits | recall, precision, latency |
| API/service | request and expected response | release, configuration | correctness, errors, throughput |
| Algorithm | input fixture | implementation | correctness, samples, speedup |
| Data pipeline | source batch | parser or policy | validity, coverage, loss, runtime |

Do not put benchmark infrastructure inside the subject. The task should adapt `ctx` and `case` to
one application call and return a typed or serializable output. Keep provider setup, domain
fixtures, and expected outcomes outside core Autobench modules.

## Design The Experiment

Define these before writing code:

1. **Decision**: what change will this benchmark accept, reject, rank, or diagnose?
2. **Cases**: which representative, edge, failure, and regression inputs prove the decision?
3. **Variants**: which concrete factor combinations are being compared?
4. **Objectives**: which metrics should improve?
5. **Constraints**: which metrics must remain within bounds?
6. **Diagnostics**: which evidence explains failures without becoming an objective?
7. **Artifacts**: which payloads are needed to understand or reproduce a result?
8. **Lineage**: which prompts, tools, schemas, configurations, or code assets affected the run?
9. **Replay boundary**: what must remain inspectable without executing the subject again?

Factors are independent inputs such as model, prompt version, algorithm, or configuration.
Metrics are outcomes such as correctness, latency, cost, or throughput. Do not encode factors as
metrics or infer factor identity from ad hoc metric names.

Use one-factor-at-a-time or controlled matrices when attribution matters. Autobench reports deltas
and confounding; it does not claim causality from a run where multiple factors changed together.

## Implementation Sequence

1. Inspect the target project, installed Autobench version, existing benchmark files, and task
   import layout.
2. Choose YAML for portable experiment definition and Python for application calls or custom
   evaluation logic.
3. Define stable case and variant IDs. IDs are data keys, not display labels.
4. Implement `task(ctx, case)` with `RunContext` first and `Case` second.
5. Emit raw runtime evidence through context, spans, or native instrumentation.
6. Add scoring separately from task observation. Scores evaluate outcomes; observations describe
   what happened.
7. Add derivation only when an output depends on already-recorded evidence.
8. Add policies for hard requirements and reports for human inspection.
9. Validate before running, record every meaningful execution, and use replay/report/compare on
   the record directory.
10. Test real behavior, error paths, optional dependencies, and deterministic replay.

Use `examples/minimal` for a first benchmark, `examples/performance` for repeated measurements,
`examples/abp-manual` for explicit evidence, and `examples/pydantic-ai` for native SDK discovery.

## Evidence Requirements

A useful run should answer:

- Which case and variant ran?
- Which factor values were active?
- What output or error occurred?
- Which objective, constraint, and diagnostic values were produced?
- Which spans and measurements explain runtime behavior?
- Which artifacts preserve large or structured payloads?
- Which source files and behavioral asset versions affected the result?
- Can the result be loaded without importing the original application or provider SDK?

Prefer semantic types such as `quality.correctness`, `time.latency`, `money.cost`,
`llm.tokens.input`, and `agent.tool.argument.correctness`. A local name remains useful for human
display; a semantic type gives the value stable meaning across projects.

## Review Checklist

- The benchmark answers a concrete decision, not merely “collect data.”
- Cases are representative and include known failures or boundaries.
- Variants are explicit factor sets; `optimize: true` marks only controllable factors.
- The task signature is exactly `task(ctx, case)`.
- Application code does not depend on report or storage internals.
- Raw observations, scores, derived metrics, and policies have distinct ownership.
- Semantic types, directions, roles, and units are declared where meaningful.
- Capture policy is explicit when sensitive payloads may cross an instrumented boundary.
- Behavioral assets required for later reproduction remain available or intentionally hashed.
- Run records are treated as immutable.
- Replay/report/export do not call the application.
- Comparison output avoids causal language for confounded variants.
- YAML has a versioned language-server schema header.
- The benchmark validates and its real end-to-end path runs.

## Anti-patterns

- Writing another monolithic `run_benchmark.py` that owns dataset parsing, execution, scoring,
  persistence, and reporting.
- Asking users to construct Pydantic Evals `Dataset` or `Case` objects for normal Autobench use.
- Using untyped callback dictionaries instead of `RunContext`, `ScoringCall`, or typed specs.
- Calling `perf_counter()` manually when a span or `measure_callable()` already owns duration.
- Duplicating the same semantic value as both a task observation and score without a clear source
  precedence rule.
- Treating a comparison as proof that one changed factor caused the delta.
- Hiding task import failures by modifying `sys.path` globally.
- Storing credentials, request bodies, or private prompts under a full capture policy by accident.
- Mutating old records during replay, rescore, canonicalization, or report generation.
- Adding case-specific framework helpers when the application task can own the domain behavior.

