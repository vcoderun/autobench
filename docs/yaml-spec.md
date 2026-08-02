# YAML Spec

Autobench is YAML-first. Python builders compile to the same internal `BenchmarkSpec`.
Every YAML file written by Autobench includes a `yaml-language-server` schema header that points
to the versioned schema cache under `~/.autobench/<version>/schemas/`.

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

## Safe Extensibility

YAML is intended to be shareable and replayable. For that reason:

- file references are resolved relative to the spec path
- remote URLs are rejected
- inline Python expressions are not part of the YAML surface

## Exported Run Record YAML

Run records are the immutable per-case/per-variant evidence files used by replay:

```yaml
record:
  type: run
  version: 3

run:
  id: run_ticket_1_route_v1
  experiment: exp_support_routing_20260507T120000Z
  benchmark: support-routing
  case: ticket_1
  variant: route_v1
  status: passed
  outcome:
    evaluation: passed
    task: completed

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
  objectives:
    routing_correctness:
      value: true
      semantic: quality.correctness
  diagnostics:
    latency_ms:
      value: 12.4
      semantic: time.latency
      unit: ms

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
  trace:
    media_type: application/x-yaml
    value: artifacts/run_ticket_1_route_v1/trace.yaml

assets:
  prompt.router:
    version: 7c91d4d7b1af

output:
  queue: billing
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
        model.name: gpt-4.1-mini
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
  version: 3

experiment:
  id: exp_support_routing_20260507T120000Z
  benchmark: support-routing

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
  paths:
    - cases/ticket_1/route_v1/run.yaml

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
  version: 1

asset:
  id: tool.create_car
  kind: tool
  name: create_car
  semantic: agent.tool
  current_version: 7c91d4d7b1af
  doc: Create a new car instance.
  params:
    make:
      type: Literal["audi", "bmw", "mercedes"]
      required: true
    model:
      type: str
      required: true
  returns:
    type: Car
    asset_id: type.Car

versions:
  - version: 15aa0dbceb02
    state:
      kind: tool
      name: create_car
      params:
        make:
          type: Literal["audi", "bmw", "mercedes"]
          required: true
    hashes:
      content: ...
    changes:
      fields: [initial]
  - version: 7c91d4d7b1af
    parent: 15aa0dbceb02
    state:
      kind: tool
      name: create_car
      params:
        make:
          type: Literal["audi", "bmw", "mercedes"]
          required: true
    hashes:
      content: ...
      source: ...
    source:
      path: ./vsh.py
    changes:
      fields:
        - params.year.type
      diff: |
        --- 15aa0dbceb02
        +++ 7c91d4d7b1af
        @@ ...
```
