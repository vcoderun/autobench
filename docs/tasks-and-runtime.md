# Tasks And Runtime

The task is the only application-specific execution boundary required by Autobench. It receives a
runtime context and a case, invokes the subject, records evidence, and returns the output that
scorers evaluate.

## Task Contract

```python
from autobench import Case, RunContext


def run_case(ctx: RunContext, case: Case) -> dict[str, object]:
    model = ctx.factor("model")
    result = call_application(case.input, model=model)
    ctx.outcome(result.ok)
    return {"answer": result.answer, "ok": result.ok}
```

The positional contract is always `task(ctx, case)`. Tasks may be synchronous or asynchronous:

```python
async def run_case(ctx: RunContext, case: Case) -> dict[str, object]:
    result = await call_application(case.input)
    return {"answer": result.answer}
```

YAML resolves the callable relative to the benchmark file before falling back to import paths:

```yaml
run:
  python: benchmark_tasks:run_case
```

## RunContext

`RunContext` owns evidence for one case x variant run:

| Method | Use |
| --- | --- |
| `factor(name)` | Read a variant factor |
| `span(...)` | Open a timed nested operation |
| `metric(...)` / `metrics(...)` | Record one or many metrics |
| `factor_observation(...)` | Record a runtime-discovered factor |
| `event(...)` | Record a discrete event |
| `diagnostic(...)` | Record non-objective diagnostic evidence |
| `outcome(...)` | Record semantic run success |
| `check(...)` | Record a boolean correctness check with an optional reason |
| `record_measurement(...)` | Record summary statistics and optional sample artifact |
| `artifact(...)` | Attach a structured or file-like payload |
| `error(...)` | Attach a structured error without losing collected evidence |
| `attach_tracked_asset(...)` | Bind a tracked asset version to the run |

Context evidence remains available even when the task raises. The runtime captures the exception,
preserves observations and artifacts already emitted, and records a structured error.

## Matrix Execution

`build_benchmark_plan` validates and counts the matrix before execution. `expand_matrix` produces
one `MatrixRunSpec` per case x variant pair. The CLI renders the same plan during validation.

```bash
autobench validate autobench.yaml
autobench run autobench.yaml --concurrency 4 --record runs/latest
```

Concurrency bounds the number of active runs. Result ordering stays deterministic even when task
completion order differs.

## Failure And Status Model

Autobench separates three status layers:

- `TaskStatus`: whether application execution completed, failed, or was skipped.
- `EvaluationStatus`: whether scoring and constraints completed.
- `RunStatus`: final passed, failed, errored, or skipped state.

This distinction prevents a policy failure from looking like an application exception and lets
reports separate execution reliability from evaluation quality.

## Progress Events

`ProgressEvent` and `ProgressEventKind` provide typed lifecycle notifications. Known event fields
remain stable while event-specific data is carried in the payload. This is the extension surface
for terminal progress, service runners, and future UIs without coupling the core runtime to one
frontend.

## Python Builder

The builder compiles to the same `BenchmarkSpec` used by YAML:

```python
from autobench import Benchmark, Case, FactorValue, PassFailScorer, Semantic, Variant

result = (
    Benchmark("routing")
    .dataset([Case(id="refund", input={"message": "Refund order 42"})])
    .variants(
        [
            Variant(
                id="baseline",
                factors=[FactorValue(name="route", value="v1")],
            )
        ]
    )
    .task("benchmark_tasks:run_case")
    .scoring(
        [
            PassFailScorer(
                name="success",
                path="output.ok",
                semantic_type=Semantic.RESULT_SUCCESS,
            )
        ]
    )
    .run()
)
```

Use YAML for portable benchmark definitions and the builder when a Python application needs to
compose specs programmatically. Both execute through the same planner and runtime.
