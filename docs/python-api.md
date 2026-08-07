# Python API

Autobench exposes the same runtime through a fluent builder, typed specification models, and
lower-level extension seams. Use the highest-level surface that can express the benchmark clearly.

## Surface Selection

| Surface | Use it when |
| --- | --- |
| `Benchmark` | Application code composes a benchmark dynamically |
| `BenchmarkSpec` | You need the complete typed configuration surface |
| YAML + `load_benchmark_spec` | Humans or agents author portable benchmark definitions |
| Runtime/evaluation functions | You are building an adapter, service, or custom runner |

All three authoring paths execute through `run_benchmark_spec()`.

## Fluent Builder

```python
from autobench import (
    Benchmark,
    Case,
    Direction,
    ExactScorer,
    FactorValue,
    ObservationRole,
    PassFailScorer,
    Semantic,
    Variant,
)

benchmark = (
    Benchmark("builder-demo")
    .description("Compare current and candidate behavior.")
    .correlation(
        group_id="routing-proposal-42",
        attempt=1,
        phase="validation",
        labels={"owner": "evaluation"},
    )
    .dataset(
        [
            Case(
                id="refund",
                input={"message": "Refund order 42"},
                expected={"route": "billing"},
            )
        ],
        dataset_id="routing-regressions",
        version="v3",
    )
    .variants(
        [
            Variant(
                id="current",
                factors=[FactorValue(name="routing_profile", value="v3")],
            ),
            {
                "id": "candidate",
                "factors": {
                    "routing_profile": {
                        "value": "v4",
                        "optimize": True,
                    }
                },
            },
        ]
    )
    .task("my_app.benchmarks:run_case")
    .scoring(
        [
            ExactScorer(
                name="route",
                actual="output.route",
                expected="case.expected.route",
                semantic_type=Semantic.QUALITY_CORRECTNESS,
                direction=Direction.MAXIMIZE,
                role=ObservationRole.OBJECTIVE,
            ),
            PassFailScorer(
                name="success",
                path="output.success",
                semantic_type=Semantic.RESULT_SUCCESS,
                role=ObservationRole.CONSTRAINT,
            ),
        ]
    )
)

result = benchmark.run(experiment_id="routing-candidate-42", concurrency_limit=4)
```

Attach live lifecycle observers without changing the benchmark definition:

```python
from autobench import ProgressEvent


def observe(event: ProgressEvent) -> None:
    print(event.sequence, event.kind, event.run_id, event.run_status)


result = benchmark.run(progress_handlers=(observe,))
```

### Builder Methods

| Method | Configures |
| --- | --- |
| `description(value)` | Benchmark description |
| `correlation(...)` | Static invocation group, attempt, phase, ancestry hints, and scalar labels |
| `capture(policy)` | ABP and asset capture policy |
| `dataset(...)` | Inline cases or a typed dataset source |
| `variants(items)` | Typed variants or normalized dictionaries |
| `task(target, kind="python")` | Task target |
| `scoring(items)` | Built-in or Python scorer specs |
| `derive(items)` | Per-run derivers |
| `instrument(*items)` | Typed built-ins or runtime custom instrumentors |
| `instrument_all(...)` | Compatible built-in discovery |
| `to_spec()` | Canonical `BenchmarkSpec` |
| `run(...)` / `run_async(...)` | Sync or async execution |

Post-derivation, policies, report views, and custom semantic registries currently live on the full
`BenchmarkSpec`. Extend the compiled spec rather than inventing builder-only state:

```python
import asyncio

from autobench import PolicySpec, run_benchmark_spec

spec = benchmark.to_spec().model_copy(
    update={
        "policies": [
            PolicySpec(
                name="quality-floor",
                metric=Semantic.QUALITY_CORRECTNESS,
                must_greater_equal=0.9,
            )
        ]
    }
)
result = asyncio.run(run_benchmark_spec(spec, concurrency_limit=4))
```

## Task Contract

```python
from autobench import Case, RunContext


def run_case(ctx: RunContext, case: Case) -> Result:
    ...
```

`ctx` is always first and `case` is always second. A task may be sync or async and may return any
serializable result. A Pydantic model is useful because scorers can resolve output fields reliably.

The runtime resolves `module:function` targets relative to the benchmark file before falling back to
normal Python import paths.

## RunContext

`RunContext` owns evidence for one case x variant run:

| Method | Purpose |
| --- | --- |
| `factor(name)` | Read a configured factor value |
| `span(...)` | Time and nest an operation |
| `metric(...)` / `metrics(...)` | Record numeric, boolean, or structured metrics |
| `factor_observation(...)` | Record a factor discovered at runtime |
| `event(...)` | Record a discrete event |
| `diagnostic(...)` | Record non-objective evidence |
| `outcome(...)` | Record semantic success |
| `check(...)` | Record a correctness constraint and reason |
| `record_measurement(...)` | Record summaries plus optional raw samples |
| `artifact(...)` | Attach a payload |
| `error(...)` | Preserve a structured error |
| `attach_tracked_asset(...)` | Bind an explicit tracked asset version |
| `await checkpoint(name)` | Commit all currently available evidence to durable staging |

Evidence emitted before an exception remains in the failed run.

`phase` reports whether the run is resolving, executing, scoring, deriving, post-processing, or
finalizing. Checkpoints preserve that phase automatically. Application tasks normally only call
`checkpoint()`:

```python
async def run_case(ctx: RunContext, case: Case) -> Result:
    draft = await build_draft(case.input)
    ctx.artifact("draft", draft)
    await ctx.checkpoint("draft-built")
    return await validate_draft(draft)
```

The method requires an active `Recorder`; without one it raises `RuntimeError`. It is cancellation
aware: if the caller is cancelled while persistence is active, Autobench gives the atomic write
bounded time to finish and then propagates cancellation. Names beginning with `autobench.` belong
to framework lifecycle checkpoints and are rejected for application calls.

## Load And Run YAML

```python
import asyncio
from pathlib import Path

from autobench import (
    ExecutionCorrelation,
    load_benchmark_spec,
    run_benchmark_path,
    run_benchmark_spec,
)

path = Path("benchmarks/routing.yaml")
spec = load_benchmark_spec(path)

sync_result = run_benchmark_path(
    path,
    experiment_id="routing-42",
    concurrency_limit=4,
)

async_result = asyncio.run(
    run_benchmark_spec(
        spec,
        experiment_id="routing-43",
        concurrency_limit=4,
        correlation=ExecutionCorrelation(attempt=2, labels={"review": "holdout"}),
        progress_handlers=(observe,),
    )
)
```

Import `ExecutionCorrelation` for invocation-level overrides. Explicit fields merge with
`spec.execution.correlation`; omitted fields remain unchanged and label maps merge by key. The
resolved value is immutable and identical on `ExperimentResult`, every `RunResult`, durable
records, and replayed results. `parent_experiment_id` and `resumed_from_experiment_id` are grouping
metadata only. Replay ancestry remains `RecordLineage` / `parent_run_id`, and Autobench does not
infer workflow resume from either correlation field.

`run_benchmark_path()`, `run_benchmark_spec()`, `Benchmark.run()`, and `Benchmark.run_async()` share
the same `progress_handlers`, `progress_error_policy`, and `progress_error_handler` contract. Bare
handlers are strict by default; see [Tasks and Runtime](tasks-and-runtime.md#progress-events) for
ordering, terminal status, and backpressure guarantees.

Loading resolves dataset, pricing, task, and Python scorer references relative to the YAML file.

## Record And Replay

```python
from pathlib import Path

from autobench import (
    collect_benchmark_source_files,
    record_experiment,
    replay_experiment,
)

record_dir = Path("runs/routing-42")
record = record_experiment(
    async_result,
    record_dir,
    source_files=list(collect_benchmark_source_files(path)),
    path_root=Path.cwd(),
    durability="atomic",
)
replayed = replay_experiment(record_dir)
```

`record_experiment()` refuses to overwrite a non-empty experiment directory. It assembles the
record in a temporary sibling, validates its integrity manifest, and publishes it atomically.
Referenced tracked-asset histories and large trace artifacts are persisted automatically. Use
`durability="synced"` when supported POSIX file and directory `fsync` calls are also required.

For durable execution, pass a recorder to the pipeline instead of waiting for the full result:

```python
from autobench import FileRecorder

durable_result = asyncio.run(
    run_benchmark_spec(
        spec,
        experiment_id="routing-durable-44",
        concurrency_limit=4,
        recorder=FileRecorder(
            Path("runs/routing-durable-44"),
            source_files=collect_benchmark_source_files(path),
            path_root=Path.cwd(),
            durability="atomic",
        ),
    )
)
```

`FileRecorder` commits complete run snapshots during execution and atomically publishes the final
directory after post-processing. Its sibling `.routing-durable-44.staging` directory survives an
interrupted process. Use `inspect_staging`, `recover_staging`, and `finalize_staging` to recover it;
use `archive_staging` or `discard_staging` for explicit lifecycle decisions.

Cooperative task cancellation, `KeyboardInterrupt`, and supported CLI `SIGTERM` handling commit a
terminal partial checkpoint before propagating. A hard process kill can preserve only the last
explicit checkpoint that had already returned; it cannot execute a final write.

```python
from autobench import finalize_staging, inspect_staging

staging = Path("runs/.routing-durable-44.staging")
inspection = inspect_staging(staging)
if inspection.recoverable and inspection.missing_run_ids:
    partial_record = finalize_staging(
        staging,
        Path("runs/routing-durable-44-partial"),
        allow_partial=True,
    )
```

`Recorder` and `RecordSession` are the typed extension contracts for another persistence backend.
The pipeline owns `open`, `stage`, `finish` or `abort`, and `close`; application code should not
open a live session just to run a normal benchmark. `ExperimentStart`, `ExecutionSnapshot`, and
`PartialRunSnapshot` are frozen transfer models. Recovery never imports the task or optional SDKs.

A custom session exposes immediate file and stream storage explicitly through
`artifact_sink: ArtifactSink | None`. The session may return itself, as `FileRecordSession` does,
or delegate to a separate local, remote, or composite sink. Returning `None` is valid, but a task
that calls `ctx.artifact_file()` or `ctx.artifact_stream()` then receives
`ArtifactSinkRequiredError` before Autobench consumes the source. The lifecycle session does not
need to proxy the complete sink protocol merely to delegate storage.

`ArtifactSink` provides synchronous file/stream methods, `prepare_file_async()`, and
`prepare_stream_async()`. Async implementations own in-flight transfers after caller cancellation
and must settle them before the associated session closes or publishes its final record.

Load one exact record when building an audit or optimizer adapter:

```python
from autobench import load_experiment_record, load_run_record

experiment = load_experiment_record(record_dir)
run = load_run_record(record_dir / experiment.run_paths[0], root_dir=record_dir)
```

## Reports And Exports

```python
from pathlib import Path

from autobench import (
    build_report,
    compare_variants,
    export_markdown_report,
    export_runs_csv,
    export_summary_yaml,
)

report = build_report(replayed)
comparison = compare_variants(
    replayed,
    baseline="current",
    candidate="candidate",
)

export_summary_yaml(replayed, Path("analysis/summary.yaml"))
export_runs_csv(replayed, Path("analysis/runs.csv"))
export_markdown_report(replayed, Path("analysis/report.md"))
```

`build_leaderboard`, `build_case_matrix`, `build_metric_distribution`, and
`build_run_metric_rows` expose individual projections.

Group or select several invocation results without changing their records:

```python
from autobench import ExecutionCorrelation, build_grouped_reports, filter_experiments

validation = filter_experiments(
    results,
    correlation=ExecutionCorrelation(group_id="routing-proposal-42", phase="validation"),
)
groups = build_grouped_reports(results)
```

Filters match only explicitly supplied fields. Grouped reports retain each experiment report and
summarize the attempts and phases present under each `group_id`.

## Optional OTLP Export

```python
from pathlib import Path

from autobench import OTLPSettings, export_record_otlp

delivery = export_record_otlp(
    Path("runs/routing-42"),
    settings=OTLPSettings(
        endpoint="https://collector.example/v1/traces",
        service_name="routing-benchmark",
    ),
)
```

Install `autobench[otlp]` on the exporting process. `export_record_otlp()` loads immutable record
models; `export_otlp()` accepts already-loaded `ExperimentRecord` and `RunRecord` values. Both
return `OTLPExportResult` and raise `OTLPExportError` without modifying evidence. Vendor settings
remain separate from `BenchmarkSpec`. See [OTLP Export](otlp-export.md).

## Native Instrumentation

```python
from autobench import Benchmark

benchmark = Benchmark("agent").instrument_all(
    exclude={"httpx"},
    strict=False,
    assets={
        "representations": ["definition", "effective"],
        "include": ["prompt", "tool", "output_schema"],
    },
)
```

Unavailable integrations become diagnostic observations. `strict=True` instead requires every
selected integration to be compatible.

Use typed settings for explicit control:

```python
from autobench import HTTPXCaptureSettings, HTTPXInstrumentation, OpenAIInstrumentation

benchmark.instrument(
    OpenAIInstrumentation(),
    HTTPXInstrumentation(
        capture=HTTPXCaptureSettings(
            path="hash",
            response_headers=("x-request-id",),
        )
    ),
)
```

Explicit settings override automatic discovery, including `enabled=False`. A custom runtime
`Instrumentor` can also be passed to `instrument()` and remains Python-only.

## Explicit Tracking

```python
from autobench import track

SYSTEM_PROMPT = track.prompt(
    name="support_system",
    source="prompts/support.md",
)


@track.tool
def lookup_order(order_id: str) -> dict[str, str]:
    """Return the current order status."""
    ...
```

`track.prompt`, `track.tool`, `track.type`, `track.dataclass`, and `track.asset` register exact
versions. `track.write_assets(path)` writes DSL-shaped manifests plus one `content.sqlite3`
registry. `load_asset_content(...)` resolves an exact historical snapshot and
`load_asset_diff(...)` resolves the corresponding readable diff. Native discovery can attach
unadorned SDK-visible components to runs. Experiment recording uses the same contract at
`artifacts/asset-content.sqlite3`.

## Production And Generated Cases

```python
from pathlib import Path

from autobench import (
    CaseGeneratorInput,
    SamplingPolicy,
    generate_dataset_sync,
    generated_batch_from_cases,
    samples_to_cases,
    write_generation_result,
)

review_cases = samples_to_cases(production_samples, policy=SamplingPolicy(max_samples=50))
generated = generated_batch_from_cases(
    synthetic_cases,
    generator_asset_version="prompt.generator@v4",
    model_provider="openrouter",
    model_name="openai/gpt-5.6-luna",
)

result = generate_dataset_sync(
    generate_cases,
    CaseGeneratorInput(seed=17, settings={"count": 20}),
    generator_id="generation:generate_cases",
    dataset_id="generated-routing",
    version="v1",
)
write_generation_result(result, Path("datasets/generated-routing.yaml"))
```

Production helpers normalize reviewed samples. Generated-dataset APIs own the typed preparation,
hashing, review projection, manifest, and safe publication boundary, while the application owns the
actual generator/provider logic. Generation finishes before normal benchmark planning. See
[Generated Datasets](generated-datasets.md).

## Extension Rules

- Put subject execution in a task.
- Put domain judgment in a Python scorer.
- Use a deriver for same-run computations and a post-deriver for matched runs.
- Use a policy for acceptance boundaries.
- Use an instrumentor for a stable SDK boundary.
- Use source maps and extractors for external field normalization.
- Use metric packs for reusable domain defaults.
- Never mutate recorded evidence; create a derived record or a new experiment.

See [API Reference](api-reference.md) for generated signatures and model fields.
