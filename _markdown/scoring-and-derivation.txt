# Scoring And Derivation

Scorers evaluate one run. Derivers compute new metrics from collected evidence. Post-derivers work
across runs after the complete experiment exists. Policies turn semantic metrics into explicit
requirements.

## Scoring Contract

Every scorer declares:

- a local score name
- semantic type
- optional unit
- optimization direction
- role: objective, constraint, or diagnostic
- whether scorer failure is optional

Scores are stored as `ScoreRecord` values and projected into observations with score-source
precedence. The original task observations remain available.

## Output Metric

Project an output value directly:

```yaml
score:
  coverage:
    value: output.coverage
    semantic: coverage.ratio
    goal: maximize
    role: objective
```

Use this when the task already computes a trustworthy metric.

## Pass/Fail

```yaml
score:
  success:
    pass: output.ok
    semantic: result.success
    role: constraint
```

The path must resolve to a boolean-like success value.

## Exact Match

```yaml
score:
  route_correctness:
    exact:
      actual: output.queue
      expected: case.expected.queue
    semantic: quality.correctness
    goal: maximize
```

Paths can address `output`, `case.input`, `case.expected`, factors, and structured values.

## Schema Validation

`SchemaScorer` validates a selected output path against a JSON Schema mapping. It is appropriate
for contracts where structural validity is separate from domain correctness.

```python
from autobench import SchemaScorer, Semantic

scorer = SchemaScorer(
    name="output_schema",
    path="output",
    schema={
        "type": "object",
        "required": ["customer_name", "id"],
        "properties": {
            "customer_name": {"type": "string"},
            "id": {"type": "string"},
        },
    },
    semantic_type=Semantic.AGENT_OUTPUT_STRUCTURE_VALIDITY,
)
```

## Python Scorers

Custom scorers receive `ScoringCall`, not loose callback dictionaries:

```python
from autobench import ScoreRecord, ScoringCall


def field_accuracy(call: ScoringCall) -> ScoreRecord:
    expected = call.case.expected
    output = call.output
    fields = ("name", "id", "pocket_id")
    matches = sum(output[field] == expected[field] for field in fields)
    return ScoreRecord(
        name="field_accuracy",
        semantic_type="quality.field_accuracy",
        value=matches / len(fields),
    )
```

`ScoringCall` exposes the case, variant, task output/result, observations, spans, and selected spans.
Python scorers may be sync or async. Optional scorers record errors without failing the run.

## Expected Actions

`ExpectedActionScorer` deterministically evaluates action/tool selection, arguments, or ordered
sequence from spans. See [Agentic Evaluation](agentic-evaluation.md).

## Dotted Paths

`resolve_dotted_path` is the shared structured-path resolver used by built-in scorers. Missing
paths produce explicit scorer errors instead of silently returning `None`.

## Per-Run Derivation

`derive` runs after task observations and scores are available for one run. `TokenCostDeriver` is
the built-in per-run deriver.

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
```

By default it reads:

- `llm.tokens.input`
- `llm.tokens.output`
- `llm.provider`
- `llm.model.name`

Input semantics and output metadata can be overridden through `TokenCostInputs` and
`DerivedMetricOutput` in the Python API.

Unknown models, missing usage, missing rates, and ambiguous inputs produce diagnostics; Autobench
does not invent a zero cost.

## Pricing DSL

Pricing is normalized into a `PricingTable` keyed by stable model IDs. Provider-specific aliases
allow input forms such as `provider:model`, `provider/model`, or application-specific model slugs
to resolve to the same entry.

```yaml
pricing:
  version: 1
  provider: openai
  models:
    openai/gpt-demo:
      aliases: [openai:gpt-demo, gpt-demo]
      input:
        unit: mtok
        price_per_million_tokens: 1.0
      output:
        unit: mtok
        tiers:
          - up_to_tokens: 100000
            price_per_million_tokens: 4.0
          - price_per_million_tokens: 6.0
      cache_read:
        unit: mtok
        price_per_million_tokens: 0.1
```

Supported fields include input, output, cache-read, and cache-write prices plus token-count tiers.
`StaticPriceSource`, `LLMPricesSource`, and `GenAIPricesSource` only import external price data into
this model. They do not make an external catalog authoritative at runtime.

## Paired Baseline Post-Derivation

`post_derive` has access to the full experiment:

```yaml
post_derive:
  - kind: paired_baseline
    baseline_variant: baseline
    match_on:
      - kind: case_id
      - kind: factor
        name: workload.size
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

Formulas:

- `baseline_over_candidate`
- `candidate_over_baseline`
- `candidate_minus_baseline`
- `baseline_minus_candidate`
- `percent_change_from_baseline`

Matching supports case IDs and factor keys. Missing matches, nonnumeric metrics, absent metrics, and
zero division can be skipped or recorded as diagnostics.

Relative-noise thresholds and `ComparisonVerdictSpec` can emit improved, regressed, unchanged, or
inconclusive verdicts. These are controlled comparisons, not automatic causal claims.

## Policies

Policies evaluate projected semantic values and append `PolicyResult` evidence:

```yaml
policies:
  - name: request-must-succeed
    metric: result.success
    must_equal: true
  - name: cost-cap
    metric: money.cost
    must_less_equal: 0.001
  - name: acceptable-latency
    metric: time.latency
    must_between:
      min: 0
      max: 500
      inclusive: true
```

Each policy declares exactly one requirement:

- `must_equal` / `must_not_equal`
- `must_greater` / `must_greater_equal`
- `must_less` / `must_less_equal`
- `must_in` / `must_not_in`
- `must_between`

A failed constraint can change final run status while preserving the successful task output and
all evidence that explains the decision.

## Repeated Measurement

`measure_callable` avoids repeating warmup and sampling loops in benchmark tasks:

```python
from autobench import MeasurementBudget, measure_callable

measurement = measure_callable(
    lambda: search(case.input["items"], case.input["query"]),
    budget=MeasurementBudget(warmup=3, repetitions=20, max_seconds=2.0),
)
ctx.record_measurement("search", measurement)
```

`Measurement` includes samples, count, min, max, mean, median, p95, standard deviation, and relative
noise. A custom timer can measure accelerators or remote systems without adding domain-specific
logic to Autobench.
