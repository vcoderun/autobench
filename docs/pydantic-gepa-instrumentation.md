# Pydantic-GEPA Instrumentation

Autobench can record a pydantic-gepa optimization as structured experiment evidence without
handwritten optimizer spans, metric calls, asset decorators, or an Autobench-specific observer.
The optimization continues to use the normal pydantic-gepa API. Autobench subscribes to its typed
event contract only while a benchmark run is active.

This integration preserves:

- optimization, composition, engine, stage, iteration, reflection, evaluation, candidate, and
  final-rescore lifecycles;
- train, validation, and test dataset declarations;
- objective identity, direction, role, semantic type, and unit;
- evaluation-call and optimizer-cost limits, usage, and remaining budget;
- every candidate lifecycle, parent relationship, component value, and tracked asset version;
- BestOf, Vote, AdaptiveSequential, Sequential, Parallel, Single, and Pipeline structure;
- selection method, all contenders, their scores, and the selected engine execution;
- errors, cancellation, checkpoints, diagnostics, and partial evidence;
- a compact, versioned projection that remains readable after replay without pydantic-gepa.

Autobench does not run GEPA itself, replace pydantic-gepa checkpoints, or copy arbitrary live
optimizer state into a record.

## Install

Install the dedicated extra when only this integration is needed:

```bash
uv add 'autobench[pydantic-gepa]'
```

The combined native instrumentation environment also includes it on supported Python versions:

```bash
uv add 'autobench[instrumentation]'
```

Check the installed event contract before a benchmark:

```bash
autobench instrumentation doctor
```

The pydantic-gepa instrumentor requires event contract version `1`. An unavailable or unsupported
package is reported independently; base Autobench, record replay, and report rendering continue to
work without importing pydantic-gepa.

## Run The Offline Examples

The repository includes standard GEPA, Optimize Anything Omni, multi-component, and checkpoint
resume benchmarks. They use deterministic local tasks and real optimizer runtimes.

```bash
uv run autobench run examples/pydantic_gepa/autobench.yaml \
  --record /tmp/autobench-pydantic-gepa

uv run autobench instrumentation trace /tmp/autobench-pydantic-gepa
uv run autobench report /tmp/autobench-pydantic-gepa
uv run autobench replay /tmp/autobench-pydantic-gepa
```

Run the remaining contracts with `standard.yaml`, `multi_component.yaml`, and `resume.yaml`. Each
spec records, replays, reports, and exports independently in `make examples`.

The task in `examples/pydantic_gepa/optimizer_benchmark.py` contains no Autobench span or metric
calls. Its relevant shape is:

```python
from pydantic_gepa import Optimization
from pydantic_gepa.experimental.optimize_anything import (
    BestOf,
    OptimizeAnythingConfig,
    Pipeline,
    Single,
)


def run(ctx, case):
    del ctx
    optimization = Optimization.from_examples(...)
    return optimization.optimize(
        config=OptimizeAnythingConfig(
            composition=Pipeline(
                steps=(
                    BestOf(engines=(weak_engine, strong_engine)),
                    Single(engine=continuation_engine),
                )
            )
        )
    )
```

Instrumentation is declared once in YAML:

```yaml
instrumentation:
  pydantic_gepa:
    detail: full
    assets:
      discover: true
      representations: [definition, effective]
      include: [prompt]
```

## Python Configuration

Use the typed configuration with a fluent benchmark:

```python
from autobench import (
    AssetDiscoverySettings,
    Benchmark,
    PydanticGEPAInstrumentation,
)

benchmark = Benchmark("optimizer-evaluation").instrument(
    PydanticGEPAInstrumentation(
        detail="evaluations",
        assets=AssetDiscoverySettings(include=("prompt", "tool_schema")),
    )
)
```

Use automatic discovery when every compatible installed SDK integration should compose:

```python
benchmark = Benchmark("optimizer-evaluation").instrument_all()
```

`instrument_all()` installs the pydantic-gepa observer once. Explicit `pydantic_gepa` settings
override automatic selection, including an explicit disabled state.

## Detail Modes

`detail` controls high-cardinality spans, not durable summary correctness.

| Mode | Always retained | Additional evidence |
| --- | --- | --- |
| `summary` | optimization, composition/engine lifecycle, objective, budgets, selections, terminal result, projection | no per-case, candidate, iteration, or reflection spans |
| `evaluations` | everything in `summary` | candidate, evaluation, case, and metric spans and observations |
| `full` | everything in `evaluations` | GEPA iterations, reflection/proposal detail, Pareto and backend progress events |

All modes preserve the typed optimization projection and candidate summaries. Select `summary` for
large production optimization jobs, `evaluations` for score debugging, and `full` when reflection
and proposal behavior matters.

## ABP Trace Shape

A full Pipeline can produce this hierarchy:

```text
task
  pydantic_gepa.optimization                 kind=optimization
    pydantic_gepa.composition_step           kind=workflow
      pydantic_gepa.engine                   kind=workflow
      pydantic_gepa.engine                   kind=workflow
      pydantic_gepa.candidate                kind=candidate
      pydantic_gepa.evaluation               kind=evaluation
        pydantic_gepa.case                   kind=evaluation
          pydantic_gepa.metric               kind=scorer
    pydantic_gepa.composition_step           kind=workflow
      pydantic_gepa.engine                   kind=workflow
    pydantic_gepa.final_rescore              kind=evaluation
```

GEPA-backed engines may add iteration and reflection spans. AutoResearch, Meta-Harness, Best-of-N,
and custom engines still produce the common engine/evaluation/budget/selection lifecycle even when
they do not expose GEPA-specific reflection callbacks.

Parallel engine events can arrive on worker threads. pydantic-gepa assigns stable pipeline, step,
branch, engine-execution, candidate, and parent IDs before dispatch. Autobench correlates from those
IDs and reuses the run context captured at optimization start; callback arrival order is not used
as parentage.

## Metrics And Accounting

The integration classifies optimizer evidence with semantic observations:

| Evidence | Semantic type |
| --- | --- |
| raw metric and candidate/final score | `evaluation.score` or the metric's declared semantic type |
| candidate status | `evaluation.label` |
| evaluator feedback | `evaluation.explanation` |
| calls used, limit, remaining | `optimization.evaluations.used`, `.limit`, `.remaining` |
| optimizer cost used, limit, remaining | `optimization.optimizer_cost.used`, `.limit`, `.remaining` |
| evaluator cost used | `optimization.evaluation_cost.used` |
| optimizer plus evaluator aggregate cost | `optimization.cost.used` |

Objective scores retain their direction and role. A minimizing domain metric is not silently
replaced by an optimizer's transformed selection scalar.

Native Pydantic AI, OpenAI, OpenAI Agents, and HTTPX instrumentors remain authoritative for direct
model, token, request, transport, and serving-cost evidence. The pydantic-gepa layer records
optimizer semantics and does not duplicate child SDK token totals. This setup is valid:

```yaml
instrumentation:
  pydantic_gepa:
    detail: full
  pydantic_ai: {}
  openai: {}
  httpx: {}
```

Native model calls retain their Agent -> provider -> HTTP transport hierarchy in the same benchmark
trace as the optimizer lifecycle. Accounting stays separated by source layer: pydantic-gepa owns
optimization evidence, while the native SDK instrumentors own direct model, token, request, and
transport evidence.

## Candidate Assets And Lineage

pydantic-gepa declares optimization components independently of Autobench. The native adapter maps
those components to logical tracked assets:

| Component kind | Tracked asset family |
| --- | --- |
| `instructions`, `system_prompt` | prompt/instruction asset |
| `tool_schema` | tool asset |
| `input_schema`, `output_schema` | schema asset |
| `field_description`, `schema_description` | schema-description asset |
| custom semantic component | optimization component with its declared semantic type |

Definition assets describe the initial component. Effective assets describe each candidate value.
Candidate IDs remain optimizer lineage IDs; asset versions remain content/version identities. They
are linked, not conflated.

With full asset capture, content is stored in the experiment's asset-content registry rather than
duplicated in every run YAML. Run evidence stores `asset_id@version`, parent version, use records,
and diffs. Capture policy still controls raw candidate text, evaluator output, feedback, and trace
payloads.

## Durable Projection

Detailed ABP signals remain the source of truth. Each run also contains a replay-friendly extension
under:

```text
autobench.pydantic_gepa/v1
```

Load it through the public typed model:

```python
from autobench import PydanticGEPAEvidence, replay_experiment

experiment = replay_experiment("/tmp/autobench-pydantic-gepa")
payload = experiment.runs[0].extensions["autobench.pydantic_gepa/v1"]
evidence = PydanticGEPAEvidence.model_validate(payload)

execution = evidence.executions[0]
print(execution.final_score)
print(execution.candidates)
print(execution.engines)
print(execution.selections)
```

The projection contains execution identity, backend/engine/composition, datasets, objective,
budgets, candidate lifecycle and component versions, engine summaries, selections, checkpoints,
stop reason, event count, and diagnostic count. Unknown or invalid future projection versions
produce report warnings rather than breaking normal record replay.

## Reports, Replay, And Export

Rich reports add dedicated sections for:

- optimization outcome and resources;
- engine executions and branch identity;
- candidate lifecycle and parent lineage;
- component asset versions;
- selection method, all contenders, winner, score, and reason;
- partial status or diagnostics.

Replay and export operate from the immutable record and do not rerun an optimizer:

```bash
autobench replay /tmp/autobench-pydantic-gepa
autobench report /tmp/autobench-pydantic-gepa
autobench export /tmp/autobench-pydantic-gepa \
  --format yaml \
  --path /tmp/pydantic-gepa-report.yaml
```

YAML and Markdown report views include the same typed optimization projection. CSV remains the
flat run-metric view.

## Failure And Cancellation

Instrumentation observer failures are isolated and become bounded diagnostics. They do not alter
the optimizer result, exception, cancellation, or checkpoint behavior.

- completed optimizations close normalized-but-unselected candidate spans normally;
- explicit candidate accepted/rejected events retain their lifecycle labels;
- a failed optimization closes the root with error evidence;
- cancellation closes the root as cancelled and marks unfinished operations partial;
- genuinely unmatched starts/ends produce ABP diagnostics;
- duplicate event delivery is bounded and cannot overwrite another run's state;
- calls outside an active Autobench run are no-ops.

## Ownership Boundary

pydantic-gepa owns typed events, upstream GEPA compatibility, engine/composition correlation,
candidate values, and normalized evaluation evidence. Autobench owns ABP conversion, semantic
classification, capture, tracked assets, immutable records, replay, reports, and exports.

Autoptimize can later consume these records for planning and promotion. This integration does not
make causal claims, choose the next optimization strategy, or promote a candidate.
