# CLI

The CLI is human-first. Commands render Rich panels and tables; YAML, CSV, and Markdown are explicit
file exports instead of raw terminal output.

## Command Map

| Command | Executes subject? | Input | Purpose |
| --- | --- | --- | --- |
| `validate` | No | benchmark YAML | Resolve and validate the planned matrix |
| `dataset generate` | Yes | generator target + request YAML | Prepare and freeze generated cases before planning |
| `run` | Yes | benchmark YAML | Execute, record, and render an experiment |
| `replay` | No | record directory | Reconstruct recorded results |
| `report` | No | record directory | Render Rich analysis or publish Markdown |
| `compare` | No | record directory | Compare two variants without causal claims |
| `export` | No | record directory | Write a YAML, CSV, or Markdown projection |
| `recording inspect` | No | staging directory | Diagnose committed, missing, corrupt, and conflicting evidence |
| `recording finalize` | No | staging directory | Publish complete or explicitly partial immutable evidence |
| `recording archive` | No | staging directory | Copy mutable staging for retention or investigation |
| `recording discard` | No | staging directory | Permanently remove a validated staging directory |
| `instrumentation doctor` | No | environment | Inspect integration compatibility |
| `instrumentation trace` | No | record directory | Summarize ABP traces and diagnostics |
| `telemetry export` | No | record directory | Replay immutable ABP evidence to OTLP traces |

## Validate

```bash
autobench validate benchmarks/routing.yaml
```

Validation parses the DSL, loads file or glob datasets, resolves pricing and task sources relative
to the spec, checks duplicate IDs and runnable requirements, and displays case, variant, and run
counts. It does not invoke the task.

## Generate A Dataset

```bash
autobench dataset generate generator:generate_cases \
  --request generation-request.yaml \
  --output datasets/generated.yaml \
  --id generated-routing \
  --version v1
```

The target is an importable sync or async callable that accepts `CaseGeneratorInput` and returns
`GeneratedCaseBatch`. Complete generation writes normal dataset YAML and a `.generation.yaml`
provenance manifest. Explicit incomplete generation writes only `.incomplete.yaml` and exits `2`;
it never publishes or replaces the requested dataset. `--force` replaces existing complete output
files but does not weaken that incomplete-result boundary. See
[Generated Datasets](generated-datasets.md).

## Run

```bash
autobench run benchmarks/routing.yaml \
  --concurrency 4 \
  --record runs/routing-42
```

Options:

| Option | Meaning |
| --- | --- |
| `--concurrency INTEGER` | Maximum active runs; default and minimum are `1` |
| `--record DIRECTORY` | Write immutable evidence to this new directory |
| `--no-record` | Execute and display without persistence |
| `--group-id TEXT` | Override the YAML correlation group |
| `--attempt INTEGER` | Override the positive invocation attempt |
| `--phase TEXT` | Override the invocation phase |
| `--parent-experiment-id TEXT` | Associate a prior experiment without creating replay lineage |
| `--resumed-from-experiment-id TEXT` | Record an external resume association |
| `--correlation-label KEY VALUE` | Merge one repeatable scalar label |

Without either recording flag, Autobench creates
`.autobench/<spec-stem>/<experiment-id>/`. `--record` and `--no-record` are mutually exclusive.

Correlation flags override `execution.correlation` field by field. Omitted flags preserve YAML
values, while repeated labels replace matching keys and retain the rest. These values group
independent invocations for analysis; they neither resume a task nor alter replay ancestry.

The CLI records the benchmark file and resolved referenced-source hashes so evidence can explain
what was executed. Recording is incremental: every completed run is committed before whole-matrix
post-processing. If execution stops, the Rich error output includes the sibling staging path.

`Ctrl-C` and `SIGTERM` use cooperative cancellation. Active runs finalize partial ABP state and
commit cancellation checkpoints before the command exits with status 130 or `128 + SIGTERM`.
Concurrent siblings receive cancellation and bounded cleanup time before the recorder is aborted.
The staging path printed by the CLI can then be inspected or finalized explicitly.

`SIGKILL` cannot run Python cleanup. After a hard kill, only completed runs and explicit
`await ctx.checkpoint(...)` calls already present in `staging-manifest.yaml` are recoverable. This
is a deliberate guarantee boundary, not an application-resume mechanism.

Interactive terminals also receive live Rich progress through the public `ProgressEvent` observer
API. The CLI explicitly selects best-effort delivery for this display: a renderer failure is written
to stderr and never replaces benchmark execution or recorder behavior. Python callers remain strict
by default so a lost lifecycle integration cannot pass silently.

## Recording Recovery

```bash
autobench recording inspect runs/.routing-42.staging

autobench recording finalize runs/.routing-42.staging \
  --output runs/routing-42-recovered \
  --allow-partial

autobench recording archive runs/.routing-42.staging \
  --output archives/routing-42

autobench recording discard runs/.routing-42.staging --yes
```

`inspect` displays whether the staging directory is recoverable as well as complete, checkpointed,
missing, corrupt, and conflicting identities. `finalize` is strict by default; `--allow-partial`
publishes explicit missing-run and incomplete-post-processing metadata rather than pretending the
matrix completed. Finalization does not delete the source staging directory.

`archive` and `discard` are separate operations. `discard` requires `--yes` and first verifies that
the path is an Autobench staging directory. A symlink or arbitrary directory is rejected.

## Replay

```bash
autobench replay runs/routing-42
```

Replay imports neither the task nor optional provider SDKs. It reconstructs normal result models
from `experiment.yaml`, per-run records, and referenced artifacts.

## Report

```bash
autobench report runs/routing-42
```

The Rich terminal report summarizes recorded execution evidence. The Markdown report is a separate
decision-facing projection: quality gate, scores, case outcomes, purposeful charts, comparisons,
and evaluator feedback. Missing metrics remain distinct from numeric zero.

Write the richer evidence-linked Markdown projection without printing Markdown to the terminal:

```bash
autobench report runs/routing-42 \
  --format markdown \
  --profile full \
  --layout auto \
  --output analysis/routing-report
```

Profiles are `summary`, `full`, and `audit`; layouts are `single`, `bundle`, and `auto`. Use `audit`
for technical runs, traces, assets, hashes, artifacts, and provenance. Captured audit detail
additionally requires `--include-captured-content`. See
[Markdown Reports](markdown-reports.md).

## Compare

```bash
autobench compare runs/routing-42 \
  --baseline current \
  --candidate candidate
```

Both IDs must exist. The view shows changed factors, aggregate metric values and deltas, paired run
count, and whether several factors changed. `confounded=true` is a warning against causal
attribution, not a failed comparison.

## Export

```bash
autobench export runs/routing-42 \
  --format yaml \
  --path analysis/routing-summary.yaml

autobench export runs/routing-42 \
  --format csv \
  --path analysis/routing-runs.csv

autobench export runs/routing-42 \
  --format markdown \
  --path analysis/routing-report.md
```

`--format` is required and accepts `yaml`, `csv`, or `markdown`. `--path` is also required. YAML
exports include a versioned schema header; CSV is a run-level projection; Markdown is a portable
single-file report written through the same atomic publisher. The complete evidence remains the
record directory.

## OTLP Telemetry Export

```bash
autobench telemetry export runs/routing-42 \
  --endpoint https://collector.example/v1/traces \
  --header authorization 'Bearer ...' \
  --service-name routing-benchmark
```

This command requires `autobench[otlp]`. It maps the experiment, runs, ABP trace hierarchy,
semantic evidence, source provenance, links, partial state, and record identities without changing
the source record. Captured content is omitted unless `--include-captured-content` is explicit.
See [OTLP Export](otlp-export.md) for the mapping and privacy contract.

## Instrumentation Doctor

```bash
autobench instrumentation doctor
```

The compatibility table shows distribution and version state, supported range, mechanism,
abstraction layer, sync/async/streaming support, asset discovery, capture defaults, optional extra,
and diagnostics for every built-in integration.

Use it before enabling `strict=True` or when an SDK upgrade stops producing evidence.

## Trace Inspection

```bash
autobench instrumentation trace runs/routing-42
```

This is replay-only. It summarizes span roots, kinds, instrumentors, partial state, and protocol
diagnostics without loading the original SDK.

## Exit And Failure Behavior

- Invalid YAML, unresolved sources, schema errors, recording collisions, and missing records exit
  nonzero.
- YAML failures include the file and source location when available.
- Task failures are isolated to their run; already collected evidence is preserved.
- An experiment can finish with passed, failed, errored, and skipped runs. Inspect status tables and
  policies rather than assuming process completion means every run passed.
- Replay and reporting never fall back to live execution.

## CI Workflow

```bash
set -e
autobench validate benchmarks/release.yaml
autobench run benchmarks/release.yaml \
  --concurrency 4 \
  --record artifacts/autobench
autobench report artifacts/autobench
autobench export artifacts/autobench \
  --format csv \
  --path artifacts/autobench-runs.csv
```

Upload the entire `artifacts/autobench` directory so replay, traces, asset histories, and source
lineage remain available.
