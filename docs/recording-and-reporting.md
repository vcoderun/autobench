# Recording And Reporting

Recording turns an experiment into portable evidence while it is running. Each completed matrix
item is committed to a mutable staging directory; only a validated complete or explicitly partial
experiment is published as an immutable record. Replay and analysis use the immutable record
without executing the application again.

## Record Layout

```bash
autobench run autobench.yaml --record runs/support-routing
```

The directory contains:

```text
runs/support-routing/
  experiment.yaml
  summary.yaml
  manifest.yaml
  cases/<case-id>/<variant-id>/run.yaml
  assets/index.yaml
  assets/<safe-asset-id>.yaml
  artifacts/asset-content.sqlite3
  artifacts/<other-payloads>...
```

Paths are stable and artifact references are relative so the directory can be moved or archived.
Recording is append-only: an existing run payload is never silently replaced.

The CLI and `FileRecorder` stage each completed run before the next serial run, or independently as
concurrent runs complete. Final publication preserves matrix-plan order rather than wall-clock
completion order. `record_experiment()` remains the compatible one-shot API for an already
in-memory result.

Both finalization paths build the complete record in a temporary sibling directory, write
`experiment.yaml` and `summary.yaml`, validate `manifest.yaml`, and only then publish the final
directory with one atomic rename. A normal process failure therefore leaves either no new final
directory or a complete one; readers never observe a half-written final experiment.

Each manifest entry records the relative path, SHA-256 hash, byte count, file kind, and logical
identity of one file. The manifest excludes itself to avoid a recursive hash. Replay validates the
manifest before loading runs, so changed, missing, or unexpected payloads fail explicitly.

Asset manifests contain identity, lineage, hashes, changed fields, and content references. The
versioned prompt/tool/schema snapshots and readable diffs live in the single experiment-local,
content-addressed `artifacts/asset-content.sqlite3` registry. Resolve them with
`load_asset_content(...)` and `load_asset_diff(...)`.

## Incremental Durable Recording

Use `FileRecorder` when completed runs must survive a later task, scorer, policy, recorder, or
process failure:

```python
import asyncio
from pathlib import Path

from autobench import FileRecorder, run_benchmark_spec

output = Path("runs/routing-42")
result = asyncio.run(
    run_benchmark_spec(
        spec,
        experiment_id="routing-42",
        concurrency_limit=4,
        recorder=FileRecorder(output, durability="atomic"),
    )
)
```

The pipeline owns the recording session lifecycle. It opens the recorder after constructing the
fixed experiment plan, stages each `ExecutionSnapshot`, finalizes after cross-run derivation and
policies, aborts on failure, and closes under success, failure, or cancellation. A recorder failure
is a required persistence failure: Autobench does not report a run as durably recorded when its
snapshot was not committed.

Without `recorder=`, `run_benchmark_spec()` stays purely in memory and creates no staging files.
The CLI constructs a `FileRecorder` for `--record` and for its default `.autobench/...` destination;
`--no-record` selects the in-memory path explicitly.

During execution, a sibling staging directory is used:

```text
runs/.routing-42.staging/
  staging.yaml
  staging-manifest.yaml
  cases/<case-id>/<variant-id>/run.yaml
  checkpoints/<run-id>/<name>.yaml
  artifacts/<run-id>/...
  assets/<run-id>/...
```

`staging.yaml` owns experiment identity, the immutable plan, environment, semantic registry,
source hashes, post-processing requirements, and session state. `staging-manifest.yaml` is the
commit index for run and checkpoint payloads. Both use versioned JSON Schema headers. A run is
recoverable only after all of its files and hashes appear in the manifest.

Staging is intentionally not accepted by `replay_experiment()`. Mutable execution state and
immutable experiment evidence are different formats.

## Explicit Checkpoints And Cancellation

An async task can commit the evidence collected so far without ending its run:

```python
from autobench import Case, RunContext


async def evaluate_route(ctx: RunContext, case: Case) -> dict[str, str]:
    route = await choose_route(case.input)
    ctx.metric("route_confidence", route.confidence)
    ctx.artifact("route_preview", route.model_dump(mode="json"))
    await ctx.checkpoint("route-selected")

    response = await execute_route(route)
    return {"response": response}
```

`checkpoint()` snapshots the current run phase, available task output, observations, errors,
legacy spans, canonical ABP trace, artifacts, tracked asset versions and uses, source snapshots,
and signal-sequence watermark. It returns only after the staging manifest commits the checkpoint.
Checkpoint names beginning with `autobench.` are reserved for runtime lifecycle records.

Explicit checkpoints require durable recording. Calling `ctx.checkpoint()` in an in-memory run
raises before pretending evidence was persisted. A task that does not need interruption recovery
does not pay checkpoint or staging overhead.

On cooperative cancellation Autobench:

1. records the original `CancelledError`;
2. finalizes open ABP spans as partial;
3. commits an `autobench.cancelled` checkpoint with the phase where cancellation occurred;
4. cancels active concurrent siblings and gives each bounded time to perform the same cleanup;
5. marks staging cancelled, closes the record session, and re-raises the original cancellation.

Recorder commits are cancellation-safe boundaries. Once a run has produced its complete
`RunResult`, cancellation does not discard it while `stage()` is publishing the payload or
manifest revision. Autobench keeps ownership of that commit, waits for its terminal state within
the cleanup bound, and computes `recorded_run_ids` only after outstanding recorder operations have
settled. `abort()` and `close()` are ordered after stage and checkpoint work, so they never race a
surviving file-system worker.

The same rule applies to final publication. If cancellation arrives after `finish()` has begun,
the finalization remains owned until it either commits or fails; Autobench does not report an
untracked publication and let a final directory appear later. A non-cooperative third-party
cleanup may outlive the public wait bound because Python cannot forcibly terminate an arbitrary
coroutine. Such a task is retained, its eventual exception is delivered to the event-loop exception
handler, and the cancellation receives a diagnostic that cleanup is still active.

If cancellation happens during scoring or derivation, a task output already produced by the
application remains in the partial snapshot. Recorder failures are attached to the cancellation as
notes and never replace its identity. A task timeout remains a timeout and is not reclassified as
cooperative cancellation.

The CLI maps `SIGTERM` to this cooperative path where event-loop signal handlers are supported.
`KeyboardInterrupt` also reaches pipeline cancellation before the CLI exits. `SIGKILL`, abrupt
runtime termination, and power loss cannot run cleanup; they preserve only payloads already
committed to `staging-manifest.yaml`. With `durability="synced"`, those commits receive the
documented filesystem-sync guarantee. Autobench does not claim that uncommitted in-memory evidence
survives a hard kill.

Autobench does not write on every signal and does not resume application code from a checkpoint.
The application owns executable workflow state; Autobench owns durable evidence. Automatic
periodic checkpoint policy is intentionally outside this release.

Async file artifacts use the same ownership model. `artifact_file_async()` returns cancellation
within a bounded interval even when a filesystem read is blocked. The run retains a partial
artifact reference, while the session owns the underlying transfer and settles it before
checkpoint, abort, final publication, or close. A late transfer cannot be mistaken for a complete
artifact before its payload is available.

## Inspect And Recover Staging

```bash
autobench recording inspect runs/.routing-42.staging
autobench recording finalize runs/.routing-42.staging \
  --output runs/routing-42-recovered \
  --allow-partial
autobench recording archive runs/.routing-42.staging \
  --output archives/routing-42-staging
autobench recording discard runs/.routing-42.staging --yes
```

Inspection reports health, recoverability, complete and checkpointed runs, missing identities,
corrupt or conflicting runs, orphaned files, and diagnostics. The health states are:

| Health | Meaning |
| --- | --- |
| `complete` | Every planned run has committed, valid evidence |
| `partial` | Some committed run or checkpoint evidence exists |
| `missing` | No planned run has committed evidence yet |
| `corrupt` | A manifest-committed file is missing, malformed, or has the wrong hash |
| `conflicting` | Identity, revision, or uncommitted-file state needs an explicit decision |

Not every conflict destroys committed evidence. Files left by a process failure before manifest
commit and a state/manifest revision mismatch are reported as recoverable; recovery ignores those
uncommitted files and trusts the last manifest revision. Plan, experiment, run, checkpoint, or
payload identity conflicts are not recoverable because choosing one side would silently rewrite
lineage.

Python callers can inspect or load the committed subset without application imports:

```python
from pathlib import Path

from autobench import inspect_staging, recover_staging

staging = Path("runs/.routing-42.staging")
inspection = inspect_staging(staging)
if inspection.recoverable:
    recovered = recover_staging(staging)
    print([run.run_id for run in recovered.runs])
```

`finalize_staging(..., allow_partial=False)` refuses an incomplete matrix. With
`allow_partial=True`, committed runs and the newest checkpoint per missing run become one immutable
partial experiment. Its terminal metadata lists planned, recorded, and missing run IDs and marks
incomplete cross-run derivation or policies. The source staging directory remains until it is
archived or explicitly discarded.

A finalized cancellation uses normal `RunRecord`, `ExperimentRecord`, replay, and report paths.
Recovered checkpoint runs have `RunStatus.CANCELLED`, `TaskStatus.CANCELLED`,
`EvaluationStatus.NOT_EVALUATED`, `partial=true`, and `end_reason=cancelled`; they are not rewritten
as failed evaluations.

## RunRecord

One `RunRecord` represents one case x variant execution:

- record, run, experiment, benchmark, case, and variant IDs
- final, task, and evaluation statuses
- explicit `partial` state and ABP `end_reason`, including cancelled runs
- complete case snapshot and task output
- observations and scores
- canonical ABP trace, including signals, span graph, measurements, events, links, references,
  diagnostics, and instrumentation scope provenance
- ABP protocol and semantic registry versions
- legacy span tree for records created before canonical trace storage
- materialized artifacts
- factors and tracked asset versions
- extraction and source-map replay lineage
- immutable invocation correlation: group, attempt, phase, external experiment associations, and
  scalar labels
- structured errors

The YAML view groups the data for people rather than dumping internal Pydantic fields. A schema
header points editors to the versioned Autobench JSON schema.

Small traces remain inline in `run.yaml`. Larger traces are written to
`artifacts/<run-id>/trace.yaml`; the RunRecord keeps a relative `ArtifactRef` and a compact trace
summary. Trace artifacts have their own versioned JSON Schema header and load back into the same
typed `Trace` model.

## ExperimentRecord

The experiment-level record stores:

- benchmark plan and counts
- captured environment metadata
- semantic registry
- report configuration
- normalized benchmark snapshot and hash
- hashes of resolved specs, datasets, pricing files, tasks, and scorer modules
- relative run paths and status counts
- terminal experiment state: completed, cancelled, or aborted
- planned, recorded, and missing run identities
- whether cross-run derivation and experiment policies completed
- the relative integrity-manifest path
- the same resolved execution correlation copied to every run

This is enough to explain what was planned, which files defined it, and where every run record
lives.

An experiment may be terminal and still partial. Cancelled or recovered evidence uses the same
immutable record models as complete evidence, while `EvaluationStatus.NOT_EVALUATED` distinguishes
a cancelled task from a scored failure. Records written before format version 5 load as completed,
non-partial experiments with complete post-processing. Format version 6 adds optional execution
correlation; older records load it as `None`.

Execution correlation is not replay lineage. `parent_run_id` and `RecordLineage` describe derived or
replayed evidence, while `ExecutionCorrelation` groups separately invoked experiments. Reports,
Rich tables, YAML/CSV exports, staging inspection, finalized partial records, and replay preserve
that distinction.

## Atomicity And Durability

Python callers can select the publication guarantee:

```python
record_experiment(result, Path("runs/latest"), durability="atomic")
record_experiment(result, Path("runs/durable"), durability="synced")
```

- `atomic` uses sibling temporary files/directories and `os.replace`. It protects readers from
  partially published records after an ordinary process failure.
- `synced` adds file and directory `fsync` calls before and after publication on supported POSIX
  filesystems. It is intended for callers that also require the strongest available power-loss
  durability.

`synced` fails explicitly when directory syncing is unsupported. Autobench does not silently call
an atomic record power-loss durable. Neither mode overwrites a non-empty experiment directory.

Atomicity applies to each staged payload and final-directory publication. The staging manifest is
the commit boundary: a file that exists but is absent from the manifest is uncommitted evidence,
not a completed run. `synced` also syncs staged state and manifest files where supported.

## Environment And Source Identity

`capture_environment` records reproducibility metadata such as Python, platform, package, and
working-environment details. `collect_benchmark_source_files` resolves benchmark dependencies and
records content hashes.

Source paths are stored portably when possible. Missing optional source files do not erase a run;
recording captures what was resolvable at execution time.

## Artifacts

`ctx.artifact(name, value)` adds an `ArtifactRef`. During recording, supported values are
materialized under `artifacts/` and the RunRecord keeps the relative path, media type, and tags.

Use artifacts for:

- generated specs and prompts
- traces too large for `run.yaml`
- measurement samples
- model responses and structured debug payloads
- Markdown or text reports produced by the subject

Artifact path collisions and attempts to overwrite existing payloads are recording errors.

## Replay

```bash
autobench replay runs/support-routing
```

Replay loads `ExperimentRecord` and every `RunRecord` into an `ExperimentResult`. It deliberately
does not import task or scorer modules, call models, or mutate the original directory.

This enables:

- offline report regeneration
- new exports from old evidence
- baseline/candidate comparison after execution
- future rescoring into a separate derived experiment
- optimization systems consuming stable records

Autobench distinguishes three replay modes:

- **report replay** reads stored observations without re-extracting evidence
- **extraction replay** runs a typed `TraceExtractor` against the immutable ABP trace and creates a
  derived RunRecord
- **canonicalization replay** applies newer source maps to retained source snapshots and creates a
  separate derived RunRecord

Derived records point to the original `run_id`, identify the extractor or source-map versions, and
retain the source protocol and semantic registry versions. The original record and trace bytes are
never rewritten. Replay resolves trace artifacts only inside the experiment directory and imports
neither application task modules nor optional SDK integrations.

The default `SignalExtractor` reconstructs canonical observations from stored ABP measurements and
events. `SpanExtractor` derives generic topology and workflow evidence, while `UsageExtractor`
owns LLM request/token/model accounting. `CompositeExtractor` can run them as one versioned replay
processor. Custom extractors implement the typed `TraceExtractor` interface and return
observations, diagnostics, and evidence references without mutating the trace.

When a newer version of the same extractor is replayed, its observations replace the older
version's observations in the new derived record. The previous derived record remains the lineage
parent, so extractor evolution is auditable without mixing two versions of one derived metric.

## Rich Reports

```bash
autobench report runs/support-routing
```

The terminal report can include:

- experiment overview and status counts
- variant configuration table with factor values
- semantic leaderboards
- per-run metric tables grouped by semantic family
- case x variant matrices
- baseline/candidate factor and metric deltas
- metric distributions

Reports use projected semantic metrics. They do not depend on application-specific local names.

## Report Configuration

```yaml
report:
  leaderboard:
    show:
      accuracy:
        metric: quality.correctness
        aggregate: ratio_true
      total_cost:
        metric: money.cost
        aggregate: sum
      p95_latency:
        metric: time.latency
        aggregate: p95
  matrix:
    metric: quality.correctness
  compare:
    baseline -> candidate:
      show:
        accuracy:
          metric: quality.correctness
          aggregate: ratio_true
  distributions:
    - name: request_latency
      semantic_type: time.latency
      summaries: [min, median, p95, max]
  markdown:
    profile: full
    layout: auto
    output: reports/benchmark.md
```

Aggregation functions include count, mean, sum, min, max, median, p95, standard deviation,
geometric mean, and boolean true ratio.

The default Markdown projection is decision-facing: quality gate, score range, purposeful inline
SVG, case outcomes, paired deltas, issue totals, and priority evaluator feedback. The `audit`
profile adds run health, metric coverage, ABP traces, asset lineage, artifact inventory, optimizer
evidence, hashes, and provenance. A configured `output` is staged before the record manifest is
sealed. See [Markdown Reports](markdown-reports.md) for the task-output evaluation convention,
profiles, bundle layout, audit safety, and publication.

## Comparison Semantics

```bash
autobench compare runs/support-routing --baseline baseline --candidate candidate
```

Comparison pairs runs by case, displays changed factors, aggregates requested semantic metrics, and
sets `confounded=true` when multiple relevant factors changed. It reports association and deltas;
it does not claim which factor caused the result.

Use paired-baseline post-derivation when a per-run derived metric such as speedup must be written
back into candidate evidence.

## Exports

```bash
autobench export runs/support-routing --format yaml --path report.yaml
autobench export runs/support-routing --format csv --path runs.csv
autobench export runs/support-routing --format markdown --path report.md
```

- YAML is a human-readable summary projection.
- CSV is a flat run-and-metric table for analysis tools.
- Markdown is a portable rendered report.

The CLI always writes the requested file and then renders a Rich preview. Machine exports never
replace immutable source RunRecords.
