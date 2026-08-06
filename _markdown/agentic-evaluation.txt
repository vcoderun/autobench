# Agentic Evaluation

Autobench evaluates agents as traced systems rather than treating only the final text as evidence.
The same primitives also work for workflow engines, retrievers, and tool-using applications.

## Record Agent Behavior

```python
from autobench import Semantic, SpanKind


def run_case(ctx, case):
    with ctx.span("support_agent", kind=SpanKind.AGENT, input=case.input) as agent:
        with ctx.span(
            "lookup_user",
            kind=SpanKind.TOOL,
            input={"user_id": case.input["user_id"]},
        ) as tool:
            profile = lookup_user(case.input["user_id"])
            tool.set_output(profile)

        answer = compose_answer(profile, case.input["message"])
        agent.set_output(answer)
        agent.metric(
            "task_completed",
            True,
            semantic_type=Semantic.AGENT_TASK_COMPLETION,
        )
        return answer
```

Spans preserve selection, arguments, output, order, duration, errors, tags, and hierarchy.

## Declare Expected Actions

Cases can use generic `actions` or the tool-oriented `tool_calls` compatibility shape:

```yaml
cases:
  - id: refund
    input:
      user_id: u1
      message: Refund order 42
    expected:
      actions:
        - id: lookup
          kind: tool
          target: lookup_user
          input:
            user_id: u1
          order: 1
          required: true
```

Expected input matching is subset-based, so a tool may receive additional nonessential arguments.
Actions may also declare expected output, tolerance metadata, optional status, and explicit order.

## Score Selection, Arguments, And Sequence

```yaml
score:
  tool_selection:
    expected_action:
      metric: selection
      observed_kind: tool
      span:
        kind: tool
    semantic: agent.tool.selection.correctness
    goal: maximize

  tool_arguments:
    expected_action:
      metric: arguments
      observed_kind: tool
      span:
        kind: tool
    semantic: agent.tool.argument.correctness
    goal: maximize

  tool_sequence:
    expected_action:
      metric: sequence
      observed_kind: tool
      span:
        kind: tool
    semantic: agent.tool.sequence.correctness
    goal: maximize
```

These scorers are deterministic and do not require an LLM judge. They produce normal scores and
semantic observations, so policies and reports consume them like any other metric.

## Span Selection

`SpanSelector` filters spans by:

- kind
- name
- tags
- nested path
- emitted semantic type

Selectors can be composed with positive and negative report/evaluation filters. A scorer receives
the selected spans through `ScoringCall`, allowing custom component-level evaluators without
parsing raw traces.

## Agentic Semantic Types

Built-in semantics include:

- task completion and goal accuracy
- plan quality and plan adherence
- step efficiency and orchestration quality
- tool name and version
- tool selection, argument, and sequence correctness
- tool-call quality
- output correctness and structure validity
- agent version and serving volume

Applications may add more specific child semantics through the registry.

## Metric Packs

The `agentic` metric pack contributes standard semantic definitions and report defaults. Metric
packs are optional: they provide conventions, not a required agent SDK. A custom agent runtime can
emit the same evidence through spans or a trace adapter.

## Optimization Feedback

`build_feedback_records` compacts run evidence into one record per case. It captures:

- score and evaluator reasons
- task, scorer, policy, and span errors
- `failure_category` only when a failure exists
- factor values and tracked asset versions
- selected observations and trace context

`build_optimization_feedback_input` packages those records with benchmark identity and semantic
context. pydantic-gepa or autoptimize can consume this structured evidence without scraping Rich
tables or replay YAML.

Autobench reports association and comparison evidence; it does not claim causal attribution when
multiple factors changed together. Controlled experiment planning belongs to the optimizer layer.
