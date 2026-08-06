# Use Cases

The same Autobench runtime supports deterministic functions, services, LLM applications, agents,
and performance experiments. The patterns below show where domain code ends and framework
infrastructure begins.

## Choose A Pattern

| Need | Core primitives |
| --- | --- |
| Compare implementations | cases, variants, exact/pass scorers, comparison |
| Measure noisy performance | `measure_callable`, sample artifacts, paired baseline |
| Compare LLM quality and cost | semantic token metrics, pricing derivation, policies |
| Evaluate agent behavior | ABP tool spans, expected actions, span selectors |
| Instrument an existing AI app | `instrument_all()`, native SDK instrumentors |
| Track prompts/tools/schemas | explicit tracking or automatic asset discovery |
| Turn production failures into regressions | `ProductionSample`, sampling policy, reviewed cases |
| Feed an optimizer | objectives, constraints, factors, asset versions, feedback records |

## Application Regression Benchmark

Use a file-backed dataset when the benchmark is a maintained regression suite:

```yaml
# yaml-language-server: $schema=./schemas/0.3.0/benchmark_schema.json
benchmark:
  support-routing:
    dataset:
      source: file://datasets/tickets.yaml
      version: "2026-08-06"
      defaults:
        tags: [regression]
    run:
      python: benchmark_tasks:route_ticket
    variants:
      production:
        factors:
          routing_profile: v3
      candidate:
        factors:
          routing_profile:
            value: v4
            optimize: true
    score:
      route:
        exact:
          actual: output.queue
          expected: case.expected.queue
        semantic: quality.correctness
        goal: maximize
        role: objective
      handled:
        pass: output.handled
        semantic: result.success
        role: constraint
```

Keep routing logic in `benchmark_tasks.py`. Autobench handles matrix expansion, status isolation,
score projection, and comparison. This pattern also fits parsers, validators, ranking functions,
API clients, and data transformations.

## Repeated Performance Measurement

Do not hand-roll warmup, repetition budgets, percentiles, or sample artifacts:

```python
from autobench import Case, RunContext, Semantic, measure_callable


def run(ctx: RunContext, case: Case) -> dict[str, bool]:
    values = list(case.input["values"])
    target = int(case.input["target"])
    strategy = str(ctx.factor("strategy"))

    def execute() -> None:
        if strategy == "linear":
            target in values
        else:
            target in set(values)

    measurement = measure_callable(
        execute,
        warmup=3,
        repetitions=25,
        max_seconds=2.0,
    )
    ctx.record_measurement(
        "lookup_latency",
        measurement,
        semantic_type=Semantic.TIME_LATENCY,
        include_samples_artifact=True,
    )
    return {"found": target in values}
```

Derive candidate speedup only after both matched runs exist:

```yaml
post_derive:
  - kind: paired_baseline
    baseline_variant: linear
    match_on: case_id
    metric: time.latency
    formula: baseline_over_candidate
    include_baseline: true
    output:
      name: speedup
      semantic_type: performance.speedup
      unit: ratio
      direction: maximize
      role: objective
```

Correctness should remain a constraint. A faster wrong implementation is not a successful
candidate.

## LLM Quality, Usage, And Cost

Instrumentors or tasks record usage as semantic observations:

```python
ctx.metric(
    "input_tokens",
    usage.input_tokens,
    semantic_type="llm.tokens.input",
    unit="token",
)
ctx.metric(
    "output_tokens",
    usage.output_tokens,
    semantic_type="llm.tokens.output",
    unit="token",
)
ctx.factor_observation("model", model_name, semantic_type="llm.model.name")
ctx.factor_observation("provider", provider, semantic_type="llm.provider")
```

Cost remains a derivation instead of being hard-coded into Autobench instrumentation:

```yaml
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
  - name: quality-floor
    metric: quality.correctness
    must_greater_equal: 0.9
  - name: per-request-budget
    metric: money.cost
    must_less_equal: 0.01
```

The pricing file can normalize provider-specific model identifiers, aliases, cache prices, and
tiered input/output rates. Price sources are convenience adapters into this format; Autobench does
not become a live pricing service.

## Existing Pydantic AI Application

For a Pydantic AI application, automatic instrumentation removes task-level telemetry:

```python
from autobench import Benchmark, Case, ExactScorer, Semantic

benchmark = (
    Benchmark("support-agent")
    .dataset(
        [
            Case(
                id="order-status",
                input="Where is order A-42?",
                expected={"status": "delayed"},
            )
        ]
    )
    .variants(
        [
            {
                "id": "luna",
                "factors": {
                    "model": "openrouter:openai/gpt-5.6-luna",
                },
            }
        ]
    )
    .task("support_benchmark:run")
    .scoring(
        [
            ExactScorer(
                name="status",
                actual="output.status",
                expected="case.expected.status",
                semantic_type=Semantic.QUALITY_CORRECTNESS,
            )
        ]
    )
    .instrument_all()
)
```

The task can contain only the agent call. Compatible instrumentors collect Pydantic AI agent/model/
tool/validation activity, the OpenAI-compatible client layer, and HTTPX transport evidence. The run
also receives automatically discovered prompt, tool, output-schema, capability, and agent versions
when those values cross supported SDK boundaries.

Use `exclude={"httpx"}` to avoid transport spans or select a narrower asset family:

```python
benchmark.instrument_all(
    exclude={"httpx"},
    assets={
        "representations": ["definition", "effective"],
        "include": ["prompt", "tool", "output_schema"],
    },
)
```

## Agent Tool Selection And Arguments

Agent evaluation should use execution evidence, not only final text. Declare expected actions in
the case:

```yaml
cases:
  - id: refund-order
    input:
      message: Refund order A-42
    expected:
      actions:
        - tool: lookup_order
          args:
            order_id: A-42
          order: 1
        - tool: issue_refund
          args:
            order_id: A-42
          order: 2
```

Then score the tool spans:

```yaml
score:
  tool_selection:
    expected_action:
      metric: selection
      span:
        kind: tool
    semantic: agent.tool.selection.correctness
    goal: maximize
    role: objective
  tool_arguments:
    expected_action:
      metric: arguments
      span:
        kind: tool
    semantic: agent.tool.argument.correctness
    goal: maximize
    role: objective
  tool_sequence:
    expected_action:
      metric: sequence
      span:
        kind: tool
    semantic: agent.tool.sequence.correctness
    goal: maximize
    role: constraint
```

This works with manually recorded tool spans and native SDK traces. It does not require an LLM judge
for deterministic action contracts.

## Custom SDK Without Application Changes

When an SDK is not built in, instrument a stable method and declare both evidence and assets:

```python
from autobench import (
    InstrumentAssetSpec,
    InstrumentMetricSpec,
    Semantic,
    SpanKind,
    instrument_method,
)

instrument_method(
    WorkflowClient,
    "execute",
    span="workflow.execute",
    span_kind=SpanKind.WORKFLOW,
    metrics=[
        InstrumentMetricSpec(
            name="confidence",
            semantic_type=Semantic.QUALITY_SCORE,
            value_path="result.confidence",
        ),
    ],
    assets=[
        InstrumentAssetSpec(
            kind="prompt",
            local_id="instructions",
            value_path="kwargs.instructions",
            name="routing_instructions",
        ),
        InstrumentAssetSpec(
            kind="tool",
            local_id="tools",
            value_path="kwargs.tools",
            many=True,
        ),
        InstrumentAssetSpec(
            kind="output_schema",
            local_id="output",
            value_path="kwargs.output_type",
            name="routing_output",
        ),
    ],
)
```

Serializable configurations use `value_path` or an import target. Typed Python integrations may use
`value_factory` for extraction that cannot be represented as a path. Keep domain computation in the
application; instrumentation should describe stable boundaries and evidence extraction.

## Production Failures As Regression Cases

Convert selected production samples into cases without coupling the benchmark to a production
database:

```python
from autobench import (
    ProductionSample,
    SampleReason,
    SamplingPolicy,
    samples_to_cases,
)

samples = [
    ProductionSample(
        id="trace-1842",
        input={"message": "Refund order A-42"},
        expected={"route": "billing"},
        reason=SampleReason.FAILURE_ONLY,
        privacy_tags=("customer_text",),
    )
]

cases = samples_to_cases(
    samples,
    policy=SamplingPolicy(
        reasons=(SampleReason.FAILURE_ONLY,),
        max_samples=100,
    ),
)
```

Review state, source reason, timestamp, trace identity, and privacy tags become metadata. Promote
reviewed cases into a versioned YAML dataset before using them as a release gate.

## Synthetic Case Generation With Provenance

Autobench does not own a model-based generator, but it preserves generated-data lineage:

```python
from autobench import Case, generated_batch_from_cases

batch = generated_batch_from_cases(
    [Case(id="edge-1", input={"message": "..."})],
    generator_asset_version="prompt.generate_cases@82ab39",
    model_provider="openrouter",
    model_name="openai/gpt-5.6-luna",
)
```

Generated cases remain candidates until reviewed. This keeps generator behavior and benchmark truth
from collapsing into the same untracked process.

## CI Regression Gate

A typical CI job validates, executes, stores artifacts, and checks policy state:

```bash
set -e
autobench validate benchmarks/release.yaml
autobench run benchmarks/release.yaml \
  --concurrency 4 \
  --record artifacts/autobench-release
autobench report artifacts/autobench-release
autobench export artifacts/autobench-release \
  --format csv \
  --path artifacts/autobench-runs.csv
```

Persist the whole record directory, not only the CSV. The CSV is a projection; the immutable YAML
records and artifacts contain replay, lineage, source, and diagnostic evidence.

## Optimization Handoff

Autobench marks metrics by role and direction:

- objective: improve this metric;
- constraint: do not violate this boundary;
- diagnostic: explain behavior without becoming an objective.

Factors can set `optimize: true`, and tracked assets identify the exact prompt/tool/schema versions
used. Convert a recorded run into compact feedback:

```python
from pathlib import Path

from autobench import build_optimization_feedback_input, load_run_record

record = load_run_record(
    Path("runs/latest/cases/refund-order/candidate/run.yaml"),
    root_dir=Path("runs/latest"),
)
feedback = build_optimization_feedback_input(record)
```

An optimizer should propose candidates and run controlled validation experiments. Autobench supplies
evidence and comparison; it does not claim that independently best assets can be mixed safely.

## Replay-Only Analysis

Recorded evidence supports analysis in an environment without the application or provider SDKs:

```python
from pathlib import Path

from autobench import build_report, replay_experiment

experiment = replay_experiment(Path("runs/latest"))
report = build_report(experiment)
```

This is the correct boundary for dashboards, offline reports, audits, post-hoc extraction, and
optimizer data ingestion.
