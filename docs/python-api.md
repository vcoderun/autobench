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
