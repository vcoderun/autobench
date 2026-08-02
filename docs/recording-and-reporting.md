# Recording And Reporting

Recording turns an in-memory experiment into portable, immutable evidence. Replay and analysis use
those records without executing the application again.

## Record Layout

```bash
autobench run autobench.yaml --record runs/support-routing
```

The directory contains:

```text
runs/support-routing/
  experiment.yaml
  summary.yaml
  cases/<case-id>/<variant-id>/run.yaml
  artifacts/...
```

Paths are stable and artifact references are relative so the directory can be moved or archived.
Recording is append-only: an existing run payload is never silently replaced.

## RunRecord

One `RunRecord` represents one case x variant execution:

- record, run, experiment, benchmark, case, and variant IDs
- final, task, and evaluation statuses
- complete case snapshot and task output
- observations and scores
- span tree
- materialized artifacts
- factors and tracked asset versions
- parent run ID for lineage
- structured errors

The YAML view groups the data for people rather than dumping internal Pydantic fields. A schema
header points editors to the versioned Autobench JSON schema.

## ExperimentRecord

The experiment-level record stores:

- benchmark plan and counts
- captured environment metadata
- semantic registry
- report configuration
- normalized benchmark snapshot and hash
- hashes of resolved specs, datasets, pricing files, tasks, and scorer modules
- relative run paths and status counts

This is enough to explain what was planned, which files defined it, and where every run record
lives.

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
```

Aggregation functions include count, mean, sum, min, max, median, p95, standard deviation,
geometric mean, and boolean true ratio.

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
