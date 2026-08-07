# YAML Spec

Autobench is YAML-first. Python builders compile to the same internal `BenchmarkSpec`.
Every YAML file written by Autobench includes a `yaml-language-server` schema header that points
to the versioned schema cache under `~/.autobench/<version>/schemas/`.

## Authoring Sections

The authoring DSL places the benchmark ID under `benchmark` and keeps all behavior inside that
named benchmark:

| Section | Required | Purpose |
| --- | --- | --- |
| `description` | No | Human-readable benchmark intent |
| `execution` | No | Static cross-invocation correlation metadata |
| `dataset` | Yes | Inline or file-backed cases, defaults, version, and metadata |
| `run` | For execution | Python task target |
| `variants` | Yes | Named factor combinations |
| `score` | No | Built-in or Python scorers |
| `derive` | No | Per-run semantic derivation such as token cost |
| `post_derive` | No | Cross-run derivation such as paired baseline |
| `policies` | No | Semantic metric constraints |
| `report` | No | Leaderboard, matrix, comparisons, and distributions |
| `semantic_registry` | No | Custom semantic definitions and aliases |

## Complete Authoring Example

```yaml
# yaml-language-server: $schema=./schemas/0.3.0/benchmark_schema.json
benchmark:
  support-routing:
    description: Compare current and candidate routing behavior.
    execution:
      correlation:
        group_id: routing-proposal-42
        attempt: 1
        phase: validation
        parent_experiment_id: routing-baseline-17
        labels:
          owner: evaluation
          seed: 7
    dataset:
      source: file://datasets/cases.yaml
      version: v2
      defaults:
        tags: [regression]
    run:
      python: benchmark_tasks:run_case
    variants:
      baseline:
        factors:
          model:
            value: openrouter:openai/gpt-5.6-luna
            semantic: llm.model.name
          prompt_version:
            value: route-v3
            semantic: prompt.version
            optimize: true
      candidate:
        factors:
          model:
            value: openrouter:openai/gpt-5.6-luna
            semantic: llm.model.name
          prompt_version:
            value: route-v4
            semantic: prompt.version
            optimize: true
    score:
      route_correctness:
        exact:
          actual: output.route
          expected: case.expected.route
        semantic: quality.correctness
        goal: maximize
        role: objective
      success:
        pass: output.ok
        semantic: result.success
        role: constraint
    derive:
      - kind: token_cost
        pricing: file://pricing/models.yaml
        output:
          name: request_cost
          semantic_type: money.cost
          unit: usd
          direction: minimize
          role: constraint
    policies:
      - name: must-succeed
        metric: result.success
        must_equal: true
    report:
      leaderboard:
        show:
          accuracy:
            metric: quality.correctness
            aggregate: ratio_true
          total_cost:
            metric: money.cost
            aggregate: sum
      matrix:
        metric: quality.correctness
      compare:
        baseline -> candidate:
          show:
            accuracy:
              metric: quality.correctness
              aggregate: ratio_true
```

`execution.correlation` groups separate benchmark invocations without changing matrix identity.
`attempt` must be positive, and labels accept only stable string, integer, finite float, or boolean
values. Correlation is copied unchanged to the experiment and every run record. It is not replay
lineage and does not claim that Autobench can resume application workflow state.

Python or CLI overrides merge field by field. An omitted override keeps the YAML value; supplied
labels replace matching keys and preserve the other YAML labels. Per-case or per-variant correlation
resolvers are intentionally not part of this surface.

## Resolution Rules

- File references resolve relative to the benchmark YAML.
- Python targets use `module:callable` and receive inferred search paths from the spec directory.
- Duplicate case and variant IDs are validation errors.
- A nonempty runnable matrix requires a task.
- Scorer definitions must select exactly one scoring action.
- Remote file references are rejected; price-source URL loading is an explicit integration API.
- Custom semantics should be declared in the semantic registry.

## Shape

```yaml
benchmark:
  support-routing:
    description: Deterministic support routing benchmark.
    dataset:
      source: file://datasets/cases.yaml
      defaults:
        metadata:
          owner: docs
    run:
      python: app.benchmarks.support:run_ticket_case
    variants:
      route_v1:
        factors:
          prompt_version:
            value: route-v1
            semantic: prompt.version
            optimize: true
          routing_profile: baseline
    score:
      routing_correctness:
        exact:
          actual: output.queue
          expected: case.expected.queue
        semantic: quality.correctness
      tool_arguments:
        expected_action:
          metric: arguments
          observed_kind: tool
        span:
          kind: tool
        semantic: agent.tool.argument.correctness
    report:
      leaderboard:
        show:
          pass_rate:
            metric: result.success
            aggregate: ratio_true
```

## Exported Benchmark YAML

When Autobench renders a benchmark spec back to YAML, it uses a DSL-like shape instead of a raw
model dump:

```yaml
benchmark:
  support-routing:
    description: Route support tickets.
    dataset:
      source: datasets/cases.yaml
      cases:
        - id: ticket_1
          input:
            subject: Refund
    run:
      python: app.benchmarks.support:run_ticket_case
    variants:
      route_v1:
        factors:
          prompt_version:
            value: route-v1
            semantic: prompt.version
            optimize: true
          routing_profile: baseline
    score:
      success:
        pass: output.matched
        semantic: result.success
        goal: maximize
    report:
      leaderboard:
        show:
          pass_rate:
            metric: result.success
            aggregate: ratio_true
```

## Notes

- `dataset.source` supports local `file://` references and globs.
- task targets use `module:function`.
- variant factors accept either mapping or list form.
- YAML does not execute inline expressions.
- importable code hooks such as Python scorers remain explicit dotted targets.
- `score.<name>.span` can target component spans by kind, name, tag, path, or semantic type.
- `expected_action` scores compare `case.expected.actions` or `case.expected.tool_calls` with observed spans.

## Native Instrumentation

The optional `instrumentation` section installs ABP SDK integrations for the complete benchmark
matrix:

```yaml
benchmark:
  support-agent:
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

`all` discovers every installed, compatible built-in integration. Missing integrations are skipped
and recorded as run diagnostics unless `strict: true` is set. `exclude` accepts `pydantic_ai`,
`openai`, `openai_agents`, and `httpx`. An explicit entry, including `false`, overrides discovery;
the explicit HTTPX block above therefore remains enabled despite the discovery exclusion.

`{}` selects privacy-safe defaults. `false` disables a known integration. Unknown integration
names, settings, exclusions, or HTTP capture modes are validation errors. Optional SDKs are
imported only when their enabled integration is resolved for execution. Replay never resolves this
section.

The versioned `benchmark_schema.json` describes this surface, so YAML language servers complete
integration names and capture settings. See [Native Instrumentation](native-instrumentation.md) for
the lifecycle and privacy contract.

## Capture Policy

The benchmark-level `capture` section controls ABP evidence and discovered asset content for every
case/variant run:

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

Supported levels are `none`, `metadata`, `hash`, `redacted`, and `full`. `default_level` controls
runtime evidence and defaults to `metadata`; `asset_default_level` controls versioned behavioral
assets and defaults to `full`. Other fields include
semantic/path allow and deny lists, secret names, inline/artifact limits, collection/string/depth
limits, binary retention, and source-attribute retention. Unknown fields or levels fail validation.
See [Automatic Asset Discovery](automatic-asset-discovery.md#privacy-and-capture-policy) for the
content/version behavior.

## Safe Extensibility

YAML is intended to be shareable and replayable. For that reason:

- file references are resolved relative to the spec path
- remote URLs are rejected
- inline Python expressions are not part of the YAML surface

## Exported Run Record YAML

Run records are the immutable per-case/per-variant evidence files used by replay. The trace signal
objects below are abridged; recorded files retain their timestamps, sequence IDs, execution
references, scope provenance, and captured attributes:

```yaml
record:
  type: run
  version: 4

protocol:
  name: abp
  version: 1
  semantic_registry: 1

run:
  id: run_ticket_1_route_v1
  experiment: exp_support_routing_20260507T120000Z
  benchmark: support-routing
  case: ticket_1
  variant: route_v1
  status: passed
  outcome:
    evaluation: passed
    task: passed

case:
  id: ticket_1
  input:
    subject: Refund
  expected:
    queue: billing

variant:
  id: route_v1
  factors:
    prompt_version:
      value: route-v1
      semantic: prompt.version
      optimize: true

scores:
  routing_correctness:
    value: true
    semantic: quality.correctness
    role: objective

metrics:
  measurements:
    routing_correctness:
      id: observation_1
      name: routing_correctness
      kind: metric
      value: true
      semantic: quality.correctness
  diagnostics:
    latency_ms:
      value: 12.4
      semantic: time.latency
      unit: ms

trace:
  protocol: abp
  protocol_version: 1
  trace_id: 70d8f4b6742d412a85cb7a198db07fe1
  execution:
    benchmark_id: support-routing
    experiment_id: exp_support_routing_20260507T120000Z
    run_id: run_ticket_1_route_v1
    case_id: ticket_1
    variant_id: route_v1
  root_span_ids: [3f2f6c57b9f56a11]
  spans:
    - span_id: 3f2f6c57b9f56a11
      operation: benchmark.run
      kind: task
      scope:
        instrumentor_name: autobench.manual
        instrumentor_version: 0.3.0
        package_name: autobench
        package_version: 0.3.0
        mechanism: manual
        layer: application
      status: ok
      end_reason: completed
      measurements: []
      events: []
      links: []
      references: []
      partial: false
  links: []
  references: []
  diagnostics: []
  signals:
    - type: span_start
      protocol: abp
      protocol_version: 1
      span_id: 3f2f6c57b9f56a11
      operation: benchmark.run
      kind: task
    - type: span_end
      protocol: abp
      protocol_version: 1
      span_id: 3f2f6c57b9f56a11
      status: ok
      reason: completed
  partial: false

spans:
  call_router:
    kind: workflow
    started_at: "2026-05-07T12:00:00Z"
    duration: 0.0124
    attributes:
      component: router
  lookup_user:
    kind: tool
    parent: call_router
    input:
      user_id: u1
    output:
      tier: gold
    duration: 0.004

artifacts:
  generated_spec:
    media: application/x-yaml
    path: artifacts/run_ticket_1_route_v1/generated_spec.yaml

assets:
  prompt.router:
    version: 7c91d4d7b1af

output:
  queue: billing
```

When the serialized ABP trace exceeds the inline limit, the same section becomes a compact summary
and artifact reference:

```yaml
trace:
  id: 70d8f4b6742d412a85cb7a198db07fe1
  partial: false
  spans: 7
  signals: 31
  artifact:
    id: abp_trace
    name: ABP trace
    media: application/vnd.autobench.abp-trace+yaml
    path: artifacts/run_ticket_1_route_v1/trace.yaml
```

## Exported Dataset YAML

Dataset exports use a DSL-like shape instead of raw model dumps:

```yaml
record:
  type: dataset
  version: 1

dataset:
  id: tickets
  version: v1
  metadata:
    owner: support
  defaults:
    tags: [smoke]
  cases:
    - id: ticket_1
      input:
        subject: Refund
```

Generated datasets use this exact dataset format. Generation itself has two additional schemas. A
request is a portable input to an application-owned generator:

```yaml
# yaml-language-server: $schema=schemas/0.3.0/generation_request_schema.json
generation:
  request:
    seed: 17
    prompt:
      content: Generate privacy-safe routing cases.
      asset_version: prompt.routing-generator@v1
    settings:
      count: 20
    seed_cases:
      - id: reviewed-refund
        input: {message: Refund a duplicate charge}
        expected: {route: billing}
```

The resulting `.generation.yaml` or `.incomplete.yaml` manifest uses
`generation_schema.json`. It records generator/provider/model identity, determinism, request and
case hashes, usage, cost, review states, rejected-case reasons, output counts, and the published
dataset reference. Prompt content is represented by its hash and tracked asset version rather than
duplicated into the manifest. See [Generated Datasets](generated-datasets.md).

## Exported Semantic Registry YAML

Semantic registry exports use stable type ids with compact metadata:

```yaml
record:
  type: semantic_registry
  version: 1

semantic_registry:
  version: 1
  aliases:
    quality.answer: quality.score
  types:
    money.cost:
      unit: usd
      shape: number
    serving.cost:
      parent: money.cost
      unit: usd
      shape: number
```

## Exported Pricing YAML

Pricing tables are helper data, not a required runtime dependency. They keep provider/model aliases
and tiered token prices readable:

```yaml
record:
  type: pricing
  version: 1

pricing:
  provider: openrouter
  source: genai-prices
  updated_at: "2026-05-07"
  models:
    google/gemini-3-flash-preview:
      name: Gemini 3 Flash Preview
      aliases:
        - google:gemini-3-flash-preview
        - openrouter/google/gemini-3-flash-preview
      input:
        unit: mtok
        price: 0.3
        tiers:
          - up_to: 1000000
            price: 0.3
          - price: 0.6
      output:
        unit: mtok
        price: 2.5
      cache_read:
        unit: mtok
        price: 0.03
```

## Exported Report YAML

Report exports keep the summary under a single `report:` body:

```yaml
record:
  type: report
  version: 1

report:
  benchmark: support-routing
  experiment: exp_support_routing_20260507T120000Z
  runs: 6
  status:
    passed: 5
    failed: 1
  variants:
    baseline:
      factors:
        model.name: openrouter:openai/gpt-5.6-luna
  leaderboard:
    baseline:
      runs: 2
      metrics:
        avg_coverage: 0.82
  cases:
    ticket_1:
      baseline:
        status: passed
        metrics:
          coverage (coverage.ratio): 0.8
  matrix:
    metric: coverage.ratio
    cases:
      ticket_1:
        baseline: 0.8
  compare:
    baseline -> candidate:
      runs: 2
      confounded: true
  distributions:
    cost_distribution:
      semantic: money.cost
      variants:
        baseline: [0.01, 0.02]
```

## Exported Experiment YAML

Experiment records keep replay data structured, but the outer shape stays readable:

```yaml
record:
  type: experiment
  version: 5

experiment:
  id: exp_support_routing_20260507T120000Z
  benchmark: support-routing
  termination:
    status: completed
    partial: false
    post_processing:
      cross_run_derivation: true
      policies: true
    planned_runs: [run_ticket_1_route_v1]
    recorded_runs: [run_ticket_1_route_v1]
    missing_runs: []

benchmark:
  id: support-routing
  dataset:
    id: tickets
    version: v1
    hash: 9b5d...
  cases:
    - ticket_1
    - ticket_2
  counts:
    cases: 2
    variants: 3
    runs: 6
  warnings: []
  spec:
    hash: a13c...
    snapshot:
      benchmark:
        id: support-routing

runs:
  count: 6
  passed: 5
  failed: 1
  errored: 0
  skipped: 0
  cancelled: 0
  paths:
    - cases/ticket_1/route_v1/run.yaml

manifest: manifest.yaml

files:
  /abs/path/autobench.yaml: 3c4d...

environment:
  python: "3.11.13"
  platform: macOS-15.5-arm64-arm-64bit
  cwd: /workspace/autobench

semantic_registry:
  version: 1
  aliases:
    quality.answer: quality.score
  types:
    money.cost:
      unit: usd
      shape: number
```

The corresponding integrity manifest is also human-readable and schema-backed:

```yaml
record:
  type: manifest
  version: 1

experiment:
  id: exp_support_routing_20260507T120000Z

files:
  - path: experiment.yaml
    sha256: 58f4...
    bytes: 2148
    kind: experiment
    identity: exp_support_routing_20260507T120000Z
  - path: cases/ticket_1/route_v1/run.yaml
    sha256: 2c91...
    bytes: 4892
    kind: run
    identity: run_ticket_1_route_v1
```

Run records in format version 5 add `run.partial` and `run.end_reason`. Format version 6 adds
optional execution correlation to experiment, summary, run, staging, and checkpoint documents.
Existing records without
these fields remain loadable; Autobench infers completed, failed, deferred, or cancelled lifecycle
state from their legacy status.

## Exported Artifact YAML

Artifacts are split into metadata and payload files. Text payloads stay as text. Structured payloads
are wrapped so they remain recognizable YAML records:

```yaml
record:
  type: artifact
  version: 1

artifact:
  id: trace
  name: trace
  media_type: application/x-yaml
  span_id: call_router
  payload: artifacts/run_ticket_1_route_v1/trace.yaml
```

```yaml
record:
  type: artifact_payload
  version: 1

artifact:
  id: trace
  name: trace
  media_type: application/x-yaml

payload:
  steps:
    - tool: route_ticket
      arguments:
        queue: billing
```

## Exported Asset YAML

Tracked assets are stored as a readable index plus per-asset history files:

```yaml
record:
  type: asset_index
  version: 1

assets:
  tool.create_car:
    kind: tool
    name: create_car
    semantic: agent.tool
    current_version: 7c91d4d7b1af
    file: tool_create_car.yaml
```

```yaml
record:
  type: asset
  version: 2

asset:
  id: tool.create_car
  kind: tool
  name: create_car
  semantic: agent.tool
  current_version: 7c91d4d7b1af
  content_ref:
    asset_id: tool.create_car
    version: 7c91d4d7b1af
    path: artifacts/asset-content.sqlite3

versions:
  - version: 15aa0dbceb02
    content_ref:
      asset_id: tool.create_car
      version: 15aa0dbceb02
      path: artifacts/asset-content.sqlite3
    hashes:
      content: ...
    changes:
      fields: [initial]
  - version: 7c91d4d7b1af
    parent: 15aa0dbceb02
    content_ref:
      asset_id: tool.create_car
      version: 7c91d4d7b1af
      path: artifacts/asset-content.sqlite3
    hashes:
      content: ...
      source: ...
    source:
      path: ./vsh.py
    changes:
      fields:
        - params.year.type
      diff_ref:
        asset_id: tool.create_car
        version: 7c91d4d7b1af
        parent_version: 15aa0dbceb02
        path: artifacts/asset-content.sqlite3
```

The referenced SQLite artifact is an internal, transaction-safe content store rather than an
authoring DSL. It keeps content-addressed snapshots and readable diffs out of manifests while
supporting indexed lookup by `asset_id` and `version`. Use `load_asset_content(...)` or
`load_asset_diff(...)`; application code does not query its tables directly.
