# Recording, Replay, Reporting, And Export

## Contents

- [Record model](#record-model)
- [Directory layout](#directory-layout)
- [Immutability](#immutability)
- [Recording](#recording)
- [Replay](#replay)
- [Reports](#reports)
- [Comparison](#comparison)
- [Exports](#exports)
- [Optimization handoff](#optimization-handoff)

## Record Model

`RunRecord` represents one case x variant execution. It preserves:

- run, experiment, benchmark, case, and variant identity;
- case snapshot and factors;
- task output/result and status;
- observations, scores, policies, and derived values;
- ABP trace or trace artifact reference;
- artifacts and structured errors;
- source hashes and environment metadata;
- asset versions and `AssetUse` lineage;
- duration and execution metadata.

`ExperimentRecord` owns the run set, plan metadata, source references, post-derivation, and report
configuration. Summary YAML is a projection, not the source of truth.

## Directory Layout

```text
runs/support-routing/
  experiment.yaml
  summary.yaml
  cases/<case-id>/<variant-id>/run.yaml
  assets/index.yaml
  assets/<safe-asset-id>.yaml
  artifacts/asset-content.sqlite3
  artifacts/<other-payloads>...
```

Paths inside records are relative to the experiment root. Moving or archiving the whole directory
must preserve replay. Artifact paths may not escape the record root.

All YAML files use human-readable DSL-shaped projections and versioned language-server schema
headers. They are designed for both safe deserialization and human inspection.

## Immutability

Recorded evidence is append-only. Never silently replace an existing run payload. Replay,
reporting, rescore, source-map canonicalization, and extractor replay do not mutate the original
record.

When new analysis must persist, create a new derived artifact or experiment with lineage to the
source record. This ensures old decisions remain auditable.

Immutability applies to successful, failed, errored, skipped, partial, and cancelled executions.

## Recording

CLI:

```bash
autobench run autobench.yaml --record runs/latest
```

Python:

```python
from pathlib import Path

from autobench import record_experiment

record_experiment(
    experiment,
    Path("runs/latest"),
    source_files=[Path("autobench.yaml"), Path("benchmark_task.py")],
    path_root=Path.cwd(),
)
```

Preflight path and artifact collisions before writing. Atomic or transaction-safe persistence is
required for worker merges and asset content.

## Replay

```bash
autobench replay runs/latest
```

Replay reconstructs records and Rich summaries without executing the subject. It should not import
the task module, provider SDK, custom instrumentor, or model backend merely to display evidence.

Replay failures usually indicate a missing record, escaped/corrupt artifact path, unsupported
version, or invalid payload. Do not fix replay by implicitly rerunning the benchmark.

ABP extraction replay can re-run an extractor over the recorded trace. It records extractor and
source-map lineage and replaces only matching derived evidence in a new result view.

## Reports

Default and configured Rich reports may include:

- run status summary;
- variant configurations;
- leaderboards grouped by metric family;
- per-case matrices;
- detailed run metrics;
- comparison deltas and confounding flags;
- distributions;
- ABP trace composition and diagnostics;
- asset lineage.

YAML report configuration:

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

The terminal owns prettified tables. Do not print raw Markdown or YAML as the main CLI experience.

## Comparison

```bash
autobench compare runs/latest --baseline current --candidate proposed
```

Comparison reports factor changes, metric aggregates, deltas, verdicts, and confounding. Direction
and relative thresholds determine improved, regressed, unchanged, or inconclusive outcomes.

Comparison does not prove causality. If multiple factors changed, mark the result confounded and
recommend isolated runs.

Paired-baseline derivation can write a candidate metric such as speedup by matching individual runs
on case or factors. Aggregate comparison is a report; paired derivation is evidence stored on the
derived experiment result. Keep the distinction explicit.

## Exports

```bash
autobench export runs/latest --format yaml --path analysis/report.yaml
autobench export runs/latest --format csv --path analysis/runs.csv
autobench export runs/latest --format markdown --path analysis/report.md
```

Exports are projections of recorded evidence. YAML remains human-readable and schema-addressable.
CSV is useful for tabular analysis. Markdown is a file export, not the default terminal renderer.

Visualization/image export is not part of the current v0.3 core. Do not invent PNG or Matplotlib
CLI behavior unless the installed version explicitly provides it.

## Optimization Handoff

`FeedbackRecord` and `OptimizationFeedbackInput` project runs into compact optimizer evidence:

- objectives, constraints, diagnostics;
- factor values and controllable hints;
- asset versions;
- score reasons and failure categories;
- trace or artifact references;
- candidate/run lineage.

Autobench owns evidence, not search strategy or promotion. `pydantic-gepa` adapts Pydantic AI and
Pydantic Evals to GEPA. `autoptimize` owns planning, candidate population, validation, matrix
experiments, and promotion policy.

