# Python API

## Contents

- [Choose a surface](#choose-a-surface)
- [Task contract](#task-contract)
- [RunContext](#runcontext)
- [Fluent builder](#fluent-builder)
- [Low-level specs](#low-level-specs)
- [Scoring extensions](#scoring-extensions)
- [Execution and persistence](#execution-and-persistence)
- [Report projection and publication](#report-projection-and-publication)
- [Typing and extension rules](#typing-and-extension-rules)

## Choose A Surface

Use YAML for portable experiment definitions. Use Python for:

- application task functions;
- custom scorers or derivers;
- runtime-only instrumentation;
- generated or production cases;
- fluent assembly when the benchmark itself is dynamic.

The Python builder compiles to `BenchmarkSpec`. Do not build a parallel runtime that bypasses the
spec, matrix, or record models.

Import public APIs from `autobench` unless a documented subpackage is intentionally required.
Inspect the installed version before relying on a newer symbol.

## Task Contract

The required signature is `task(ctx, case)`. `RunContext` is first and `Case` is second. Sync and
async callables are supported.

```python
from pydantic import BaseModel

from autobench import Case, RunContext


class Input(BaseModel):
    text: str


class Output(BaseModel):
    text: str


def run(ctx: RunContext, case: Case) -> Output:
    sample = Input.model_validate(case.input)
    mode = str(ctx.factor("mode"))
    with ctx.span("transform", kind="workflow", input=sample.model_dump()) as span:
        text = sample.text.upper() if mode == "upper" else sample.text.title()
        output = Output(text=text)
        span.set_output(output.model_dump())
        return output
```

Do not reverse the parameters. Do not construct a Pydantic Evals dataset in application code;
Autobench owns its optional evaluation-runtime adaptation internally.

## RunContext

`RunContext` owns evidence for one case x variant execution:

| Method | Purpose |
| --- | --- |
| `factor(name)` | Read a configured factor |
| `span(...)` | Time and nest an operation |
| `metric(...)`, `metrics(...)` | Record runtime metrics |
| `factor_observation(...)` | Record a runtime-discovered factor |
| `event(...)` | Record a discrete event |
| `diagnostic(...)` | Record explanatory evidence |
| `outcome(...)` | Record semantic success |
| `check(...)` | Record a boolean check and reason |
| `record_measurement(...)` | Record summaries and optional samples |
| `artifact(...)` | Attach structured, text, or binary payloads |
| `error(...)` | Preserve a structured non-fatal error |
| `attach_tracked_asset(...)` | Bind an explicit asset version |

Evidence emitted before a task exception remains in the failed run. Span duration is automatic;
do not wrap a span with a redundant `perf_counter()` unless measuring a distinct inner operation.

## Fluent Builder

```python
from autobench import (
    Benchmark,
    Case,
    Direction,
    ExactScorer,
    FactorValue,
    ObservationRole,
    Semantic,
    Variant,
)

benchmark = (
    Benchmark("routing")
    .description("Compare current and proposed routing.")
    .dataset(
        [Case(id="refund", input={"message": "Refund order 42"}, expected={"route": "billing"})],
        dataset_id="routing-regressions",
        version="v3",
    )
    .variants(
        [
            Variant(id="current", factors=[FactorValue(name="profile", value="v3")]),
            {"id": "proposed", "factors": {"profile": {"value": "v4", "optimize": True}}},
        ]
    )
    .task("benchmark_task:run")
    .scoring(
        [
            ExactScorer(
                name="route",
                actual="output.route",
                expected="case.expected.route",
                semantic_type=Semantic.QUALITY_CORRECTNESS,
                direction=Direction.MAXIMIZE,
                role=ObservationRole.OBJECTIVE,
            )
        ]
    )
)
```

Builder methods configure description, capture, dataset, variants, task, scoring, per-run
derivation, explicit instrumentation, automatic instrumentation, spec compilation, and execution.
Post-derivation, policies, reports, and custom semantic registries can be added to the compiled
`BenchmarkSpec` with a validated model update.

## Low-level Specs

Use typed models when constructing advanced behavior:

- `BenchmarkSpec`, `TaskSpec`, `DatasetSpec`, `Case`, `Variant`, `FactorValue`;
- scorer specs and `ScoreRecord`;
- `TokenCostDeriverSpec`, `PairedBaselineDeriverSpec`, `PolicySpec`;
- report specs;
- instrumentation and capture settings;
- semantic registry definitions.

Do not pass loose dictionaries across core extension boundaries when a public typed model exists.
Dictionary shorthand is acceptable at ergonomic builder/YAML boundaries where Autobench
normalizes it immediately.

## Scoring Extensions

Custom scorers receive one `ScoringCall`:

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

`ScoringCall` includes case, variant, output/result, observations, spans, and selected spans. Custom
scorers may be sync or async. An optional scorer records its own failure without failing the run.

Use built-in exact, pass/fail, schema, output-metric, and expected-action scorers before adding a
custom callback.

## Execution And Persistence

```python
from pathlib import Path

from autobench import (
    load_benchmark_spec,
    record_experiment,
    replay_experiment,
    run_benchmark_spec_sync,
)

spec = load_benchmark_spec(Path("autobench.yaml"))
experiment = run_benchmark_spec_sync(spec, concurrency_limit=4)
record_experiment(experiment, Path("runs/latest"), source_files=[Path("autobench.yaml")])
replayed = replay_experiment(Path("runs/latest"))
```

Use async execution when already inside an event loop. The sync runner preserves host-loop
behavior and should not be called as an event-loop workaround from async code.

Replay, report, compare, export, and trace inspection operate on records and should not import or
execute the benchmark subject.

## Report Projection And Publication

```python
from pathlib import Path

from autobench import (
    build_report,
    load_experiment_record,
    replay_experiment,
    write_markdown_report,
)

record_dir = Path("runs/latest")
result = replay_experiment(record_dir)
record = load_experiment_record(record_dir)
report = build_report(
    result,
    experiment_record=record,
    experiment_root=record_dir,
)
publication = write_markdown_report(
    report,
    Path("analysis/report"),
    layout="bundle",
    immutable_root=record_dir,
)
```

`BenchmarkReport` is the typed analysis IR. Markdown is a deterministic projection of that model.
Use `summary`, `full`, or explicitly content-enabled `audit`; use `single`, `bundle`, or `auto` for
publication. Post-hoc reports cannot write inside the immutable record. Configured in-run reports
use `ExperimentPublisher(result, record, experiment_root)` and return `ExperimentFile` values for
the recorder to validate and seal.

`full` is decision-facing. It projects task-output `hard_pass`, `score`, `metrics`, and `feedback`
into quality KPIs, case outcomes, purposeful inline SVG, and priority feedback. Use `audit` for run
inventories, traces, assets, hashes, artifacts, and captured values.

## Typing And Extension Rules

- Prefer exact types, then constrained generics, then `Any` only for genuinely unconstrained data.
- Do not use `object` as a type-checking escape hatch.
- Preserve decorated callable signatures with `ParamSpec` and return type variables.
- Use `asyncio.isawaitable()` for dynamic awaitable results.
- Use `Protocol` only for meaningful external seams; protocol class names should describe the
  capability without a `Protocol` suffix.
- Avoid one-use helpers that only rename a direct expression.
- Keep optional SDK imports behind integration resolution. Core, records, and replay must remain
  importable without provider extras.
- Add a generic primitive only when multiple applications need the behavior. Keep domain-specific
  logic in tasks, scorers, adapters, or examples.
