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
- immutable execution correlation shared by the experiment and every run.

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

Across several `ExperimentResult` values, `filter_experiments()` matches only explicitly supplied
correlation fields and `build_grouped_reports()` groups invocations by `group_id`, attempts, and
phases. Keep this separate from `parent_run_id` and `RecordLineage`, which describe replay or
derived-record ancestry.

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

## Markdown Reports

Markdown is a deterministic projection of `BenchmarkReport`, not a second analysis engine. Build
the typed report first; render or publish it afterward:

```python
from pathlib import Path

from autobench import (
    build_report,
    load_experiment_record,
    render_markdown_report,
    write_markdown_report,
)

record = load_experiment_record(record_dir)
report = build_report(result, experiment_record=record, experiment_root=record_dir)
text = render_markdown_report(report)
publication = write_markdown_report(
    report,
    Path("analysis/report"),
    layout="bundle",
    immutable_root=record_dir,
)
```

Profiles:

- `summary`: verdict, quality KPIs, meaningful charts, bounded case outcomes, comparisons, policies,
  and material limitations;
- `full`: the stakeholder report, adding benchmark setup, configured analysis, and priority
  evaluator feedback;
- `audit`: full plus lifecycle, metric coverage, runs, failures, ABP traces, assets, artifacts,
  hashes, provenance, and captured detail only when explicit permission and recording policy allow.

Execution success and benchmark quality are separate. For report-ready task output, return a mapping
or Pydantic model with `hard_pass`, `score`, `metrics`, and `feedback`, either at the top level or
under `evaluation`. Score records remain the fallback source for objective scores. Normalized
dimensions and boolean rates may be charted; neutral counts such as output length must not be
presented as quality failures.

Audit detail may include bounded case input/expected values, task output, score actual/expected
values, tracebacks, asset content, and asset diffs. `limits.value_excerpt_chars` bounds each excerpt.
`assets.diffs` is `none`, `summary`, or `full`; a full diff excerpt still requires audit plus content
permission. Sensitive assets never become inline report content.

Layouts:

- `single`: one Markdown file;
- `bundle`: linked `index.md`, case, and variant pages; run and asset pages are audit-only;
- `auto`: bundle above 50 runs, 100 profile-relevant detail rows, or 1,000 matrix cells.

CLI:

```bash
autobench report runs/latest \
  --format markdown --profile full --layout auto --output analysis/report
```

The CLI must print only the Rich publication summary. It must not print the Markdown body.

Configured YAML publication:

```yaml
report:
  markdown:
    profile: full
    layout: single
    output: reports/benchmark.md
```

With CLI recording, configured files are staged before final metadata, included in `manifest.yaml`,
and hash-validated on replay. Post-hoc writes inside a finalized record are rejected. For custom
recorders, an `ExperimentPublisher` has this contract:

```python
def publisher(
    result: ExperimentResult,
    record: ExperimentRecord,
    experiment_root: Path,
) -> Sequence[ExperimentFile]: ...
```

The staging root is read-only evidence input. Return files instead of writing them directly; the
recorder owns collision checks, writes, hashes, and sealing. Markdown uses
`MarkdownExperimentPublisher` through that generic seam.

Preserve these report contracts:

- quality-gate outcomes must never be inferred from execution status;
- normal reports prioritize verdicts, case outcomes, purposeful charts, and evaluator explanations;
- technical IDs, hashes, traces, asset internals, and raw evidence inventories are audit-only;
- direction-aware outcomes and best markers;
- paired and missing pair counts;
- factor visibility near variant IDs;
- missing distinct from zero;
- evidence references on deterministic findings;
- confounding without causal wording;
- HTML/table/link escaping and no network access;
- atomic single and bundle publication;
- policy outcome visibility and bounded captured-value excerpts;
- materialized artifact paths validated for containment, existence, hash, and size;
- malformed optional asset/run detail cannot erase valid aggregate evidence;
- only evidence-defined inline SVG charts in the core; no arbitrary chart DSL or built-in PDF
  dependency.

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

The Markdown report may contain deterministic inline SVG for quality gates, case scores, and
normalized quality dimensions. PDF remains an optional downstream conversion, not a core export
format. Do not invent PNG, Matplotlib, or arbitrary chart CLI behavior unless the installed version
explicitly provides it.

`autobench telemetry export` is a separate optional outbound projection. It maps immutable
experiment, run, and ABP evidence to OTLP spans, omits captured content by default, and never
changes the record. It requires `autobench[otlp]`; vendor configuration does not belong in the
benchmark YAML.

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
