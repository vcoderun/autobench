# Python API

Autobench also exposes a small programmatic builder.

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
