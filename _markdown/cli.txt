# CLI

The CLI is human-first. Commands render Rich panels and tables; YAML, CSV, and Markdown are explicit
file exports instead of raw terminal output.

## Command Map

| Command | Executes subject? | Input | Purpose |
| --- | --- | --- | --- |
| `validate` | No | benchmark YAML | Resolve and validate the planned matrix |
| `run` | Yes | benchmark YAML | Execute, record, and render an experiment |
| `replay` | No | record directory | Reconstruct recorded results |
| `report` | No | record directory | Render configured analysis views |
| `compare` | No | record directory | Compare two variants without causal claims |
| `export` | No | record directory | Write a YAML, CSV, or Markdown projection |
| `instrumentation doctor` | No | environment | Inspect integration compatibility |
| `instrumentation trace` | No | record directory | Summarize ABP traces and diagnostics |

## Validate

```bash
autobench validate benchmarks/routing.yaml
```

Validation parses the DSL, loads file or glob datasets, resolves pricing and task sources relative
to the spec, checks duplicate IDs and runnable requirements, and displays case, variant, and run
counts. It does not invoke the task.

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

Without either recording flag, Autobench creates
`.autobench/<spec-stem>/<experiment-id>/`. `--record` and `--no-record` are mutually exclusive.

The CLI records the benchmark file and resolved referenced-source hashes so evidence can explain
what was executed.

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

The report includes experiment status, variant configuration, leaderboard values, run metrics, case
matrix, configured comparisons, and distributions. Missing metrics remain visible rather than being
silently converted to zero.

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
report. The complete evidence remains the record directory.

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
