# First Benchmark

This guide builds a complete deterministic benchmark. It compares two text transformations, scores
their outputs, records every case and variant, and replays the result.

## Project Layout

```text
text-benchmark/
  autobench.yaml
  benchmark_task.py
```

The task remains ordinary application code. The YAML file describes how Autobench should execute
and evaluate it.

## Write The Task

Create `benchmark_task.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, TypeAdapter

from autobench import Case, RunContext

Transform = Literal["upper", "title_upper"]
TRANSFORM = TypeAdapter(Transform)


class TextInput(BaseModel):
    text: str


class TextOutput(BaseModel):
    text: str


def run(ctx: RunContext, case: Case) -> TextOutput:
    sample = TextInput.model_validate(case.input)
    transform = TRANSFORM.validate_python(ctx.factor("transform"))

    with ctx.span(
        "transform_text",
        kind="workflow",
        input=sample.model_dump(),
        attributes={"transform": transform},
    ) as span:
        text = sample.text.upper()
        if transform == "title_upper":
            text = sample.text.title().upper()
        output = TextOutput(text=text)
        span.set_output(output.model_dump())
        return output
```

The required task signature is `task(ctx, case)`: `RunContext` is always first and `Case` is always
second. Sync and async functions are both supported. Span duration is measured by Autobench.

## Define The Benchmark

Create `autobench.yaml`:

```yaml
# yaml-language-server: $schema=./schemas/0.3.0/benchmark_schema.json
benchmark:
  text-transform:
    description: Compare deterministic text transformations.
    cases:
      - id: greeting
        input:
          text: hello autobench
        expected:
          text: HELLO AUTOBENCH
      - id: whitespace
        input:
          text: release ready
        expected:
          text: RELEASE READY
    run:
      python: benchmark_task:run
    variants:
      current:
        label: Current implementation
        factors:
          transform: upper
      proposed:
        label: Proposed implementation
        factors:
          transform:
            value: title_upper
            optimize: true
    score:
      exact_text:
        exact:
          actual: output.text
          expected: case.expected.text
        semantic: quality.correctness
        goal: maximize
        role: objective
    report:
      leaderboard:
        show:
          correctness:
            metric: quality.correctness
            aggregate: ratio_true
      matrix:
        metric: quality.correctness
      compare:
        current -> proposed:
          show:
            correctness:
              metric: quality.correctness
              aggregate: ratio_true
```

This produces four runs: two cases multiplied by two variants.

## Validate Before Running

From `text-benchmark/`:

```bash
autobench validate autobench.yaml
```

Validation parses the DSL, resolves the task and source files relative to the spec, loads external
datasets and pricing files, verifies unique IDs, and displays the planned matrix. It does not call
the task.

## Run And Record

```bash
autobench run autobench.yaml --record runs/text-transform
```

Autobench renders Rich terminal tables and writes:

```text
runs/text-transform/
  experiment.yaml
  summary.yaml
  cases/
    greeting/current/run.yaml
    greeting/proposed/run.yaml
    whitespace/current/run.yaml
    whitespace/proposed/run.yaml
  artifacts/
```

The actual run filenames use stable run IDs inside the case and variant directories. The records
include the case snapshot, factors, output, observations, score, trace, source hashes, environment,
and status.

## Replay And Analyze

```bash
autobench replay runs/text-transform
autobench report runs/text-transform
autobench compare runs/text-transform --baseline current --candidate proposed
```

These commands load recorded evidence. They do not import `benchmark_task.py` and do not execute the
subject again. Comparison reports factor changes and metric deltas but does not claim that a
confounded difference is causal.

## Export A Projection

```bash
autobench export runs/text-transform \
  --format yaml \
  --path analysis/text-transform.yaml

autobench export runs/text-transform \
  --format csv \
  --path analysis/text-transform.csv
```

Terminal output stays human-oriented and uses Rich tables. YAML, CSV, and Markdown are file export
formats.

## Add Runtime Evidence

Tasks can emit evidence that is not part of the return value:

```python
ctx.metric(
    "characters",
    len(output.text),
    semantic_type="text.characters",
    unit="count",
)
ctx.check("not_empty", bool(output.text), reason="The transformed text must not be empty.")
ctx.artifact("output", output.model_dump(), media_type="application/yaml")
```

Use scores for evaluation results, observations for runtime facts, and artifacts for payloads that
must remain inspectable.

## Run Concurrently

```bash
autobench run autobench.yaml \
  --concurrency 4 \
  --record runs/text-transform-concurrent
```

The matrix order and run IDs remain deterministic. ABP context is task-local, so concurrent runs do
not share parent spans or evidence.

## Next Steps

- Move cases to a file: [Datasets And Variants](datasets-and-variants.md)
- Add quality, cost, and policy gates: [Scoring And Derivation](scoring-and-derivation.md)
- Instrument an SDK automatically: [Native Instrumentation](native-instrumentation.md)
- Track prompt and tool versions: [Automatic Asset Discovery](automatic-asset-discovery.md)
- Select a complete pattern: [Use Cases](use-cases.md)
