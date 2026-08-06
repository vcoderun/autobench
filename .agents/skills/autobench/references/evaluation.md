# Evaluation, Semantics, And Derivation

## Contents

- [Evidence layers](#evidence-layers)
- [Semantic types](#semantic-types)
- [Roles, directions, and units](#roles-directions-and-units)
- [Built-in scorers](#built-in-scorers)
- [Custom scorers](#custom-scorers)
- [Derived metrics](#derived-metrics)
- [Pricing and cost](#pricing-and-cost)
- [Policies](#policies)
- [Repeated measurement](#repeated-measurement)
- [Agentic evaluation](#agentic-evaluation)

## Evidence Layers

Keep these concepts separate:

| Layer | Owner | Example |
| --- | --- | --- |
| Observation | task, span, instrumentor | token count, latency, selected tool |
| Score | scorer | expected route matched |
| Derived metric | deriver | token counts converted to cost |
| Policy result | policy engine | cost remained below cap |
| Diagnostic | any evidence producer | missing price, skipped instrumentor |

Raw evidence describes what happened. Scoring judges an output against a case or rubric. Derivation
transforms existing evidence. Policies enforce requirements. Reports aggregate without changing
the source records.

If a task observation and scorer intentionally produce the same semantic type, define source
precedence or distinct names so reports do not count the value twice.

## Semantic Types

Local metric names are application-specific. Semantic types provide stable meaning:

```text
quality.correctness
quality.score
result.success
time.latency
performance.speedup
money.cost
llm.tokens.input
llm.tokens.output
llm.model.name
llm.provider
agent.tool.selection.correctness
agent.tool.argument.correctness
```

Use built-in `Semantic` literals when available. Custom semantics should follow hierarchical,
domain-readable names and may declare parents or aliases in `SemanticRegistry`.

Do not name a semantic after the current provider or benchmark unless that specificity is part of
its meaning. A CodeMode-specific coverage metric can be `coverage.ratio`; overall answer quality
should remain a distinct metric.

## Roles, Directions, And Units

- **objective**: a value an optimizer or comparison should improve;
- **constraint**: a value that must satisfy a requirement;
- **diagnostic**: explanatory evidence that is not optimized.

Directions are maximize or minimize. Boolean pass values generally maximize. Cost and latency
generally minimize. Units such as `s`, `ms`, `tokens`, `usd`, and `ratio` must match the actual
value and remain consistent across compared variants.

## Built-in Scorers

Prefer built-ins where they express the contract:

- `ExactScorer`: compare dotted actual and expected paths;
- `PassFailScorer`: convert a boolean path into score evidence;
- `OutputMetricScorer`: promote an output field to a score;
- `SchemaScorer`: validate output shape;
- `ExpectedActionScorer`: evaluate agent actions;
- `PythonScorer`: import a typed custom scorer.

Dotted paths can traverse output, case, expected data, mappings, and model fields according to the
public resolution contract. A missing path is a scoring error, not a false match.

## Custom Scorers

Custom scorers receive `ScoringCall` and return `ScoreRecord` or the documented result form. Use
them for domain evaluators that cannot be expressed by exact, pass/fail, schema, output metric, or
expected actions.

The scorer must not mutate the case, task output, observations, or traces. It may inspect selected
spans and produce reasons or diagnostic data. Support sync and async naturally rather than running
an event loop inside the scorer.

An optional scorer failure should be visible as scorer evidence while allowing the run to finish.
A required scorer failure should affect run status according to the runtime contract.

## Derived Metrics

Per-run derivation uses evidence from one run. Cross-run derivation uses a set of records.

Token cost is per-run. Paired speedup is cross-run because the candidate value depends on a matched
baseline run.

Paired baseline configuration specifies:

- baseline variant;
- matching key, commonly case ID or selected factors;
- source semantic metric;
- formula such as baseline-over-candidate;
- output name, semantic type, unit, direction, and role;
- missing-input policy;
- whether the baseline receives a neutral output value.

Missing baseline, missing metric, non-numeric value, and zero division are distinct diagnostics.
Do not silently fabricate a derived value.

## Pricing And Cost

Autobench is not a pricing database. Price sources normalize external or local prices into
`PricingTable`; cost derivation consumes that format.

Supported concepts include:

- provider and normalized model reference;
- input, output, and cache prices;
- units such as per-token or per-million-token;
- tiered prices and numeric limits;
- source identity and update time;
- static YAML tables and optional helper sources such as GenAI Prices or LiteLLM data.

Model strings may be provider-specific. Normalize them to a model reference rather than directly
matching every spelling. Let users supply custom pricing YAML when a source lacks the needed model
or tier.

Cost derivation requires semantic token counts plus model/provider factors. A missing price should
produce an explicit diagnostic, not a zero-dollar result that appears valid.

## Policies

Policies evaluate semantic observations after scoring and derivation. Use named operators:

```python
from autobench import PolicySpec

PolicySpec(
    name="quality-floor",
    metric="quality.correctness",
    must_greater_equal=0.9,
)
```

Available typed requirements include equality, inequality, greater/less comparisons, inclusive
bounds, and between-style requirements. Policy results should state which metric and requirement
failed. Policies are constraints, not replacements for scorer definitions.

## Repeated Measurement

Use `measure_callable()` for warmup, repetitions, maximum time, sample collection, median, p95,
noise, and custom timers:

```python
from autobench import measure_callable

measurement = measure_callable(
    operation,
    warmup=3,
    repetitions=25,
    max_seconds=2.0,
)
recorded = ctx.record_measurement("operation", measurement)
```

The helper is domain-neutral. It does not know CUDA, HTTP, databases, or model providers. Users may
provide a timer when wall-clock time is not the correct measure.

Store raw samples as an artifact when later distribution analysis matters. Compare noisy metrics
with enough repetitions and a declared aggregation rather than one timing.

## Agentic Evaluation

Agent evidence is represented through ABP spans and semantics rather than one required agent SDK.
Autobench supports:

- expected action/tool selection;
- argument correctness;
- action sequence;
- completion and outcome evidence;
- span selection by kind, name, path, tag, or semantic type;
- agentic, structured-output, LLM-usage, and performance metric packs;
- compact `FeedbackRecord` and `OptimizationFeedbackInput` projections.

Metric packs provide conventions and report defaults. They do not own model calls. Custom agent
runtimes can emit the same span and semantic contracts.

Use `failure_category: None` for a successful feedback item. A failure category names an actual
failure such as error, assertion failure, or low score; success is not a failure category.

