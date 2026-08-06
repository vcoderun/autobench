# YAML DSL

## Contents

- [Authority and schema](#authority-and-schema)
- [Authoring shape](#authoring-shape)
- [Complete skeleton](#complete-skeleton)
- [Resolution rules](#resolution-rules)
- [Datasets and variants](#datasets-and-variants)
- [Evaluation sections](#evaluation-sections)
- [Instrumentation and capture](#instrumentation-and-capture)
- [Reports](#reports)
- [Rules for agents](#rules-for-agents)

## Authority And Schema

YAML is the portable authoring format. Python builders compile to the same `BenchmarkSpec`; do not
create a separate Python-only semantic model.

Every Autobench YAML artifact must begin with a versioned schema header:

```yaml
# yaml-language-server: $schema=schemas/0.3.0/benchmark_schema.json
```

Use a path that is correct relative to the actual file. Repository examples can reference the
checked-in schema. Generated projects may use a cached or copied schema. Keep the schema version
aligned with the Autobench format being authored.

YAML is data, not Python. It may declare a dotted `value_path` or an import target, but it must not
evaluate lambdas, arbitrary expressions, or executable `value_factory` strings.

## Authoring Shape

The root contains one named benchmark:

| Section | Required | Purpose |
| --- | --- | --- |
| `description` | no | Human-readable intent |
| `cases` or `dataset` | yes | Inline, file, or glob-backed cases |
| `run` | for execution | Python task target |
| `variants` | yes | Named factor combinations |
| `score` | no | Built-in or Python scorers |
| `derive` | no | Per-run derivation |
| `post_derive` | no | Cross-run derivation |
| `policies` | no | Metric requirements |
| `report` | no | Leaderboards, matrices, comparisons, distributions |
| `instrumentation` | no | Native or automatic SDK instrumentation |
| `capture` | no | Runtime and asset retention policy |
| `semantic_registry` | no | Custom semantic definitions and aliases |

## Complete Skeleton

```yaml
# yaml-language-server: $schema=schemas/0.3.0/benchmark_schema.json
benchmark:
  support-routing:
    description: Compare current and proposed routing behavior.
    dataset:
      source: file://datasets/cases.yaml
      version: "2026-08-07"
    run:
      python: benchmark_task:run
    variants:
      current:
        label: Current routing
        factors:
          profile: v3
      proposed:
        label: Proposed routing
        factors:
          profile:
            value: v4
            semantic: routing.profile
            optimize: true
    score:
      route:
        exact:
          actual: output.route
          expected: case.expected.route
        semantic: quality.correctness
        goal: maximize
        role: objective
    post_derive:
      - kind: paired_baseline
        baseline_variant: current
        metric: time.latency
        formula: baseline_over_candidate
        output:
          name: speedup
          semantic_type: performance.speedup
          unit: ratio
          direction: maximize
          role: diagnostic
    policies:
      - name: preserve-correctness
        metric: quality.correctness
        must_equal: true
    report:
      leaderboard:
        show:
          correctness:
            metric: quality.correctness
            aggregate: ratio_true
          latency:
            metric: time.latency
            aggregate: mean
      matrix:
        metric: quality.correctness
      compare:
        current -> proposed:
          show:
            correctness:
              metric: quality.correctness
              aggregate: ratio_true
      distributions:
        - name: latency
          semantic_type: time.latency
          summaries: [min, median, p95, max]
```

## Resolution Rules

- Resolve `file://` dataset, pricing, and artifact paths relative to the YAML file that declares
  them.
- Resolve task and scorer modules relative to the benchmark spec directory before relying on the
  process working directory.
- Use `module:callable` targets. The callable must be importable without global `sys.path` hacks.
- Preserve source paths and hashes in records so the exact definition can be audited later.
- Reject unknown fields and ambiguous shapes rather than silently ignoring them.
- Keep case IDs, variant IDs, scorer names, and report keys stable and machine-safe.

## Datasets And Variants

Inline cases:

```yaml
cases:
  - id: refund
    input:
      message: Refund order 42
    expected:
      route: billing
    tags: [billing, regression]
    metadata:
      source: incident-184
```

File-backed datasets use a separate schema:

```yaml
# yaml-language-server: $schema=schemas/0.3.0/dataset_schema.json
dataset:
  routing-cases:
    version: "2026-08-07"
    defaults:
      tags: [support]
    cases:
      - id: refund
        input: {message: Refund order 42}
        expected: {route: billing}
```

Factor shorthand is appropriate for simple values. Use the expanded form for labels, semantic
types, or optimization hints. A variant is a complete factor configuration, not a dataset case.

## Evaluation Sections

Scorer forms include:

```yaml
score:
  exact_route:
    exact:
      actual: output.route
      expected: case.expected.route
    semantic: quality.correctness
    goal: maximize
    role: objective
  successful:
    pass: output.success
    semantic: result.success
    role: constraint
  confidence:
    output_metric: output.confidence
    semantic: quality.confidence
    goal: maximize
    role: diagnostic
  custom:
    python: benchmark_scorers:score_result
    optional: false
```

Derivation operates on recorded evidence. Token-cost derivation needs semantic input/output token
counts plus provider/model factors and a pricing source. Paired-baseline derivation matches
baseline and candidate runs, usually by case ID, and writes a new metric such as speedup.

Policy operators are named fields, not operator strings: `must_equal`, `must_not_equal`,
`must_greater`, `must_greater_equal`, `must_less`, `must_less_equal`, and bounded requirements when
supported by the typed policy spec.

## Instrumentation And Capture

```yaml
instrumentation:
  all:
    exclude: [httpx]
    strict: false
    assets:
      discover: true
      representations: [definition, effective]
      include: [prompt, tool, output_schema, capability]
  pydantic_ai: {}
  openai: {}
  openai_agents: false
  httpx:
    capture:
      path: hash
      request_headers: [x-request-id]
      response_headers: [x-request-id]
      request_body: false
      response_body: false
      max_body_bytes: 65536
```

Explicit entries override `all`, including explicit `false`. Missing automatic integrations are
diagnostics unless `strict: true` is set.

```yaml
capture:
  default_level: metadata
  asset_default_level: full
  use_semantic_defaults: true
  deny_paths:
    - assets.*:prompt:private_notes
```

Runtime evidence defaults to metadata; behavioral assets default to full content so successful
prompts, tools, and schemas can be reconstructed. Tighten this deliberately for sensitive runs.

## Reports

Report aggregates include `count`, `mean`, `sum`, `min`, `max`, `median`, `p95`, standard
deviation, geometric mean, and `ratio_true`. Use:

- leaderboard for per-variant aggregates;
- matrix for per-case/per-variant inspection;
- compare for declared baseline/candidate deltas;
- distributions for sample shape and noise.

Report configuration changes presentation and aggregation; it must not mutate source records.

## Rules For Agents

- Prefer the human-readable authoring DSL over serialized internal model payloads.
- Keep top-level YAML ordered by workflow: description, data, run, variants, evaluation,
  instrumentation/capture, reports.
- Emit readable block mappings rather than dense machine-only YAML.
- Do not manually write exported `RunRecord` shapes as benchmark input.
- Validate the spec before execution.
- When changing a public YAML model, update the typed model, DSL transforms, versioned JSON schema,
  docs, examples, and round-trip tests together.

