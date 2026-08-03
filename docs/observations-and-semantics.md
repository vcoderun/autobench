# Observations And Semantics

An observation is Autobench's atomic evidence unit. Raw names remain useful to humans, while
semantic types make evidence portable across applications, reports, and optimizers.

## Observation Model

An `Observation` carries:

- stable ID and local name
- kind: metric, factor, event, diagnostic, or artifact
- value and optional unit
- semantic type
- optimization direction and role
- source and optional span ID
- tags, case ID, and variant ID

```python
from autobench import Direction, ObservationRole, Semantic

ctx.metric(
    "answer_accuracy",
    0.94,
    semantic_type=Semantic.QUALITY_CORRECTNESS,
    direction=Direction.MAXIMIZE,
    role=ObservationRole.OBJECTIVE,
)
```

The local name may be `answer_accuracy`, `judge_score`, or `coverage`; the semantic type tells the
framework whether those values share meaning.

## Built-In Semantic Families

| Family | Examples |
| --- | --- |
| LLM | `llm.tokens.input`, `llm.tokens.output`, `llm.request.count`, `llm.model.requested`, `llm.model.response`, `llm.provider.name` |
| Cost | `money.cost`, `serving.cost`, `optimization.cost`, `lifetime.cost` |
| Time | `time.latency`, `time.first_chunk`, `time.critical_path` |
| Result | `result.success` |
| Quality | `quality.score`, `quality.correctness`, `coverage.ratio` |
| Agent | task completion, plan quality/adherence, step efficiency, tool selection/arguments/sequence, output correctness |
| Assets | `prompt.version`, `agent.tool.version`, `agent.version`, `dataset.version` |
| Operations | count, maximum depth/fan-out, incomplete work, parallelism, retries, recovered retries, first-attempt success |
| Workflow | validation failures, approval count/wait, tool-call success/failure, message growth, evidence-reference counts |

`Semantic` exposes completion-friendly constants. `SemanticType` remains extensible so domain
metrics can use names such as `retrieval.recall` or `business.conversion`.

## Registry And Aliases

`SemanticRegistry` stores definitions, aliases, and parent relationships. A custom registry can be
embedded in a benchmark spec and is merged with built-ins:

```yaml
semantic_registry:
  version: 1
  types:
    business.conversion:
      description: Whether the workflow produced a qualified conversion.
      parent: result.success
      unit: boolean
  aliases:
    conversion: business.conversion
```

Parent relationships let a query request a broad semantic category while preserving specific
metrics. Aliases prevent local naming differences from fragmenting evidence.

## Roles And Directions

Roles describe how a metric participates in evaluation:

- objective: something to optimize
- constraint: something that must remain acceptable
- diagnostic: explanatory evidence

Directions are `maximize` or `minimize`. Factors, events, and artifacts cannot declare an
optimization direction because they are not outcomes.

## Sources And Projection

The same semantic metric can be emitted by a task, scorer, deriver, policy, or adapter. Raw
observations are never discarded. Projection chooses a canonical value using explicit source
priority and ABP accounting scope. A derived aggregate summary is preferred to same-source direct
measurements for single-value reporting, while direct observations remain queryable. Logical
operation IDs correlate equivalent framework/client evidence; equal-priority disagreements are
marked ambiguous instead of silently picking one.

Use `ObservationQuery` for raw or projected lookup and `filter_observations` for selectors such as
semantic type, role, source, or span.

```python
from autobench import ObservationQuery

query = ObservationQuery(observations=list(result.observations))
costs = query.values("money.cost", projected=False)
```

Reports, policies, and derivation use this semantic projection layer rather than relying on local
metric names.

## Metric Packs

A `MetricPack` bundles reusable semantic defaults without forcing every metric into core:

- semantic registry additions
- scorer factory references
- default report metrics
- feedback extractors

Built-in packs cover `agentic`, `structured_output`, `llm_usage`, and `performance`. Applications
can register their own packs through `MetricPackRegistry` while keeping the RunRecord contract
unchanged.
