# Python API

Autobench exposes typed models and functions for every core layer. The `Benchmark` builder is a
compact convenience API; direct `BenchmarkSpec` construction provides the complete configuration
surface.

## Builder Example

```python
from autobench import Benchmark, Case, ExactScorer, FactorValue, PassFailScorer, Semantic, Variant

result = (
    Benchmark("builder-demo")
    .dataset([Case(id="case_1", expected={"answer": "ok"})])
    .variants(
        [
            Variant(id="v1", factors=[FactorValue(name="enabled", value=True)]),
            {"id": "v2", "factors": {"enabled": False}},
        ]
    )
    .task("my_app.benchmarks:run_case")
    .scoring(
        [
            PassFailScorer(
                name="success",
                path="output.success",
                semantic_type=Semantic.RESULT_SUCCESS,
            ),
            ExactScorer(
                name="answer",
                actual="output.answer",
                expected="case.expected.answer",
                semantic_type=Semantic.QUALITY_CORRECTNESS,
            ),
        ]
    )
    .run()
)
```

Builder methods cover description, dataset, variants, task, scoring, per-run derivation, spec
compilation, and sync/async execution. `to_spec()` returns the canonical `BenchmarkSpec`.

For post-derivation, policies, report configuration, or a custom semantic registry, construct or
update the typed spec before calling `run_benchmark_spec`:

```python
from autobench import BenchmarkSpec, PolicySpec, run_benchmark_spec

spec = BenchmarkSpec.model_validate(payload)
spec = spec.model_copy(
    update={
        "policies": [
            PolicySpec(
                name="quality-gate",
                metric="quality.correctness",
                must_greater_equal=0.9,
            )
        ]
    }
)
result = await run_benchmark_spec(spec, concurrency_limit=4)
```

## Task Signature

Python task targets use:

```python
def run_case(ctx, case):
    ...
```

`ctx` is always the first parameter. `case` is always the second.

Tasks may be sync or async.

## Context Utilities

`RunContext` and `Span` provide:

- `metric`
- `factor_observation`
- `event`
- `diagnostic`
- `outcome`
- `check`
- `metrics`
- `record_measurement`
- `artifact`
- `error`

Span duration is owned by Autobench. Tasks do not need to hand-roll `perf_counter` timing for benchmark spans.

## Agentic Evidence

Agent and workflow runs can record typed spans:

```python
from autobench import Semantic, SpanKind


def run_case(ctx, case):
    with ctx.span("support_agent", kind=SpanKind.AGENT) as agent:
        agent.metric("task_completed", True, semantic_type=Semantic.AGENT_TASK_COMPLETION)
        with ctx.span("lookup_user", kind=SpanKind.TOOL, input={"user_id": "u1"}) as tool:
            tool.set_output({"tier": "gold"})
```

Expected tool/action checks can be expressed as scorers:

```python
from autobench import ExpectedActionScorer, Semantic, SpanSelector

scorer = ExpectedActionScorer(
    name="tool_arguments",
    semantic_type=Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS,
    metric="arguments",
    span=SpanSelector(kind="tool"),
)
```

Cases can use either `expected.actions` or `expected.tool_calls`:

```python
Case(
    id="refund",
    expected={
        "actions": [
            {"tool": "lookup_user", "args": {"user_id": "u1"}, "order": 1},
        ]
    },
)
```

External framework traces can be attached with `TraceEnvelope`, and Pydantic AI usage can be recorded through `PydanticAIUsage` without making either OpenTelemetry or Pydantic AI a core dependency.

## Programmatic Layers

| Layer | Primary APIs |
| --- | --- |
| Data | `Case`, `CaseDefaults`, `DatasetSpec`, `Variant`, `FactorValue` |
| Spec | `BenchmarkInfo`, `BenchmarkSpec`, `TaskSpec`, `load_benchmark_spec`, `build_benchmark_plan` |
| Runtime | `RunContext`, `Span`, `run_benchmark_spec`, `run_benchmark_path`, `expand_matrix` |
| Native instrumentation | typed integration settings, `InstrumentationManager`, registry status, compatibility diagnostics |
| Evidence | `Observation`, `ObservationQuery`, `SemanticRegistry`, projection helpers |
| Scoring | built-in scorer models, `ScoringCall`, `ScoreRecord`, `SpanSelector` |
| Derivation | token cost, pricing models, paired-baseline derivation, policies, measurement |
| Tracking | `track`, `TrackingRegistry`, tracked asset models and YAML views |
| Records | `record_experiment`, record loaders, `replay_experiment`, environment capture |
| Reports | report models, builders, comparison, aggregation, rendering, and exporters |
| Feedback | `build_feedback_records`, `build_optimization_feedback_input` |

## Loading And Running YAML

```python
from pathlib import Path

from autobench import load_benchmark_spec, run_benchmark_path

spec = load_benchmark_spec(Path("autobench.yaml"))
result = await run_benchmark_path(
    Path("autobench.yaml"),
    experiment_id="candidate-42",
    concurrency_limit=4,
)
```

`load_benchmark_spec` supports authoring DSL and normalized model shapes. It merges custom semantic
registries with built-ins and resolves file-backed datasets and pricing relative to the spec.

## Recording And Replay

```python
from pathlib import Path

from autobench import record_experiment, replay_experiment

record_experiment(result, Path("runs/candidate-42"))
replayed = replay_experiment(Path("runs/candidate-42"))
```

Replay returns normal runtime result models but never imports the task target.

## Reports And Exports

```python
from pathlib import Path

from autobench import build_report, export_runs_csv, export_summary_yaml

report = build_report(replayed)
export_summary_yaml(replayed, Path("analysis/summary.yaml"))
export_runs_csv(replayed, Path("analysis/runs.csv"))
```

Report builders can also be called independently: `build_leaderboard`, `build_case_matrix`,
`compare_variants`, `build_metric_distribution`, and `build_run_metric_rows`.

## Typed Native Instrumentation

Activate every compatible built-in integration available in the current environment:

```python
from autobench import Benchmark

benchmark = Benchmark("chat").instrument_all(
    exclude={"httpx"},
    strict=False,
)
```

Discovery skips unavailable integrations and records why on each run. `strict=True` turns the
first unavailable or unsupported selected integration into an `InstrumentationError`. Explicit
typed settings and custom runtime instrumentors take precedence over their discovered equivalent.

Configure individual integrations when capture settings must be controlled directly:

```python
from autobench import (
    Benchmark,
    HTTPXCaptureSettings,
    HTTPXInstrumentation,
    OpenAIInstrumentation,
    instrumentor_statuses,
)

benchmark = Benchmark("chat").instrument(
    OpenAIInstrumentation(),
    HTTPXInstrumentation(
        capture=HTTPXCaptureSettings(path="hash", response_headers=("x-request-id",))
    ),
)

for status in instrumentor_statuses():
    print(status.name, status.compatibility.status)
```

Settings are part of `BenchmarkSpec` and round-trip through the YAML DSL. Custom `Instrumentor`
instances can use the same fluent method but remain runtime-only. See
[Native Instrumentation](native-instrumentation.md).

## Extension Rules

- Keep application execution in tasks.
- Use custom Python scorers for domain evaluation, returning `ScoreRecord`.
- Register domain semantics rather than overloading generic names.
- Use adapters to convert external traces or usage into Autobench evidence.
- Store large native payloads as artifacts.
- Do not mutate recorded evidence; produce a new derived experiment or export.

The complete signatures and model fields are available in [API Reference](api-reference.md).
